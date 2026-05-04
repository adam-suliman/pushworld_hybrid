from dataclasses import replace
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pushworld.puzzle import Actions, PushWorldPuzzle
from pushworld.rl.architectures import (
    SkillDiscriminatorMLP,
    append_skill_one_hot,
)
from pushworld.rl.diayn import (
    DIAYNFinetuneConfig,
    DIAYNTrainingConfig,
    compute_diayn_reward,
    compute_goal_progress_penalty,
    compute_object_change_reward,
    compute_object_motion_metrics,
    compute_object_novelty_metrics,
    compute_object_novelty_reward,
    initialize_object_position_memory,
    next_finetune_skill_id,
    resample_skill,
    update_discriminator,
)
from pushworld.rl.skill_observations import (
    build_skill_discriminator_observation_config,
    encode_discriminator_state,
)


TEST_PUZZLES_PATH = os.path.join(os.path.split(__file__)[0], "puzzles")


def test_skill_one_hot_appending_shape_and_dtype():
    observations = torch.zeros((3, 5), dtype=torch.float64)
    skill_ids = torch.tensor([0, 2, 1])

    conditioned = append_skill_one_hot(observations, skill_ids, num_skills=4)

    assert conditioned.shape == (3, 9)
    assert conditioned.dtype == observations.dtype
    assert torch.equal(
        conditioned[:, -4:],
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=observations.dtype,
        ),
    )


def test_discriminator_observation_is_dynamic_finite_and_padded():
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "trivial.pwp")
    puzzle = PushWorldPuzzle(puzzle_path)
    base_config = build_skill_discriminator_observation_config(puzzle_path)
    config = replace(base_config, max_movables=base_config.max_movables + 2)

    observation = encode_discriminator_state(
        puzzle,
        puzzle.initial_state,
        steps=0,
        max_steps=32,
        config=config,
    )

    assert observation.shape == (config.observation_size,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert not hasattr(config, "max_wall_cells")
    assert not hasattr(config, "max_agent_wall_cells")
    assert not config.include_agent_position
    assert not config.include_static_context
    assert np.count_nonzero(observation[:4]) == 0

    agent_slot_start = config.global_size
    agent_slot = observation[agent_slot_start : agent_slot_start + config.object_slot_size]
    assert agent_slot[0] == 1.0
    assert agent_slot[1] == 1.0
    assert np.count_nonzero(agent_slot[2:]) == 0

    padding_start = config.global_size + puzzle.num_movables * config.object_slot_size
    assert np.count_nonzero(observation[padding_start:]) == 0


def test_object_motion_metrics_detect_push_contact_and_goal_progress():
    puzzle = PushWorldPuzzle(os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"))
    pushed_state = puzzle.get_next_state(puzzle.initial_state, Actions.RIGHT)

    push_metrics = compute_object_motion_metrics(
        puzzle,
        puzzle.initial_state,
        pushed_state,
        Actions.RIGHT,
    )

    assert push_metrics["object_displacement"] > 0.0
    assert push_metrics["moved_object_count"] == 1.0
    assert push_metrics["object_push"] == 1.0
    assert push_metrics["object_contact"] == 1.0
    assert np.isfinite(push_metrics["goal_progress"])

    blocked_state = puzzle.get_next_state(puzzle.initial_state, Actions.LEFT)
    blocked_metrics = compute_object_motion_metrics(
        puzzle,
        puzzle.initial_state,
        blocked_state,
        Actions.LEFT,
    )

    assert blocked_metrics["object_displacement"] == 0.0
    assert blocked_metrics["object_push"] == 0.0


def test_object_change_reward_rewards_object_pushes():
    config = DIAYNTrainingConfig(
        puzzle_path="unused",
        output_dir="unused",
        object_change_reward_scale=0.5,
        object_change_reward_clip=1.0,
        goal_progress_reward_scale=0.25,
    )
    reward = compute_object_change_reward(
        {
            "object_displacement": 3.0,
            "goal_progress": 2.0,
        },
        config,
    )

    assert reward == pytest.approx(1.0)


def test_object_novelty_reward_only_counts_first_time_object_positions():
    puzzle = PushWorldPuzzle(os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"))
    pushed_state = puzzle.get_next_state(puzzle.initial_state, Actions.RIGHT)
    memory = initialize_object_position_memory(puzzle.initial_state)
    config = DIAYNTrainingConfig(
        puzzle_path="unused",
        output_dir="unused",
        object_novelty_reward_scale=2.0,
        object_novelty_reward_clip=1.0,
    )

    first_metrics = compute_object_novelty_metrics(
        puzzle.initial_state,
        pushed_state,
        memory,
    )
    first_reward = compute_object_novelty_reward(first_metrics, config)
    repeated_metrics = compute_object_novelty_metrics(
        puzzle.initial_state,
        pushed_state,
        memory,
    )
    repeated_reward = compute_object_novelty_reward(repeated_metrics, config)

    assert first_metrics["novel_object_positions"] == 1.0
    assert first_metrics["object_novelty"] == 1.0
    assert first_reward == pytest.approx(2.0)
    assert repeated_metrics["novel_object_positions"] == 0.0
    assert repeated_metrics["object_novelty"] == 0.0
    assert repeated_reward == 0.0


def test_object_novelty_reward_can_be_gated_by_goal_progress():
    config = DIAYNTrainingConfig(
        puzzle_path="unused",
        output_dir="unused",
        object_novelty_reward_scale=2.0,
        object_novelty_reward_clip=2.0,
        object_novelty_requires_nonnegative_goal_progress=True,
    )
    novelty_metrics = {
        "novel_object_positions": 1.0,
        "novel_object_moves": 1.0,
        "object_novelty": 1.0,
    }

    gated_reward = compute_object_novelty_reward(
        novelty_metrics,
        config,
        {"goal_progress": -1.0},
    )
    allowed_reward = compute_object_novelty_reward(
        novelty_metrics,
        config,
        {"goal_progress": 0.0},
    )

    assert gated_reward == 0.0
    assert allowed_reward == pytest.approx(2.0)


def test_goal_progress_penalty_only_penalizes_worsening_distance():
    config = DIAYNTrainingConfig(
        puzzle_path="unused",
        output_dir="unused",
        negative_goal_progress_penalty_scale=1.5,
    )

    penalty = compute_goal_progress_penalty({"goal_progress": -2.0}, config)
    no_penalty = compute_goal_progress_penalty({"goal_progress": 0.5}, config)

    assert penalty == pytest.approx(3.0)
    assert no_penalty == 0.0


def test_diayn_reward_is_zero_for_uniform_discriminator_logits():
    logits = torch.zeros((4, 3), dtype=torch.float32)
    skill_ids = torch.tensor([0, 1, 2, 1])

    rewards = compute_diayn_reward(logits, skill_ids, num_skills=3)

    assert torch.allclose(rewards, torch.zeros_like(rewards), atol=1e-6)


def test_discriminator_update_reduces_cross_entropy_on_synthetic_batch():
    torch.manual_seed(0)
    num_skills = 4
    states = torch.eye(num_skills).repeat_interleave(8, dim=0)
    skill_ids = torch.arange(num_skills).repeat_interleave(8)
    discriminator = SkillDiscriminatorMLP(
        input_dim=num_skills,
        num_skills=num_skills,
        hidden_sizes=(16,),
    )
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-2)

    with torch.no_grad():
        initial_loss = torch.nn.functional.cross_entropy(
            discriminator(states), skill_ids
        ).item()

    update_discriminator(
        discriminator=discriminator,
        optimizer=optimizer,
        states=states,
        skill_ids=skill_ids,
        update_epochs=30,
        minibatch_size=8,
    )

    with torch.no_grad():
        final_loss = torch.nn.functional.cross_entropy(
            discriminator(states), skill_ids
        ).item()

    assert final_loss < initial_loss


def test_skill_resampling_changes_skill_when_possible():
    assert resample_skill(current_skill=1, num_skills=4) != 1
    assert resample_skill(current_skill=0, num_skills=1) == 0


def test_finetune_skill_sampling_modes():
    fixed_config = DIAYNFinetuneConfig(
        pretrained_dir="unused",
        puzzle_path="unused",
        output_dir="unused",
        num_envs=3,
        skill_sampling="fixed",
        fixed_skill_id=2,
    )
    cycle_config = DIAYNFinetuneConfig(
        pretrained_dir="unused",
        puzzle_path="unused",
        output_dir="unused",
        num_envs=3,
        skill_sampling="cycle",
    )

    assert next_finetune_skill_id(0, 0, fixed_config, num_skills=4) == 2
    assert next_finetune_skill_id(1, 1, cycle_config, num_skills=4) == 0
