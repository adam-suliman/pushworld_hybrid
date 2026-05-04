import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gym")

from pushworld.rl.architectures import SkillConditionedActorCriticMLP
from pushworld.rl.diayn import (
    DIAYNTrainingConfig,
    evaluate_skill_conditioned_policy,
    load_diayn_checkpoint,
    train_diayn_ppo,
)
from pushworld.rl.envs import load_puzzle_bank
from pushworld.rl.hierarchical import (
    HierarchicalDIAYNConfig,
    HierarchicalSkillEnv,
    train_hierarchical_diayn_ppo,
)
from pushworld.rl.observations import VectorObservationConfig, build_observation_config
from pushworld.rl.utils import resolve_device


TEST_PUZZLES_PATH = os.path.join(os.path.split(__file__)[0], "puzzles")


def test_diayn_ppo_smoke_saves_policy_discriminator_and_configs(tmp_path):
    output_dir = tmp_path / "diayn"

    result = train_diayn_ppo(
        DIAYNTrainingConfig(
            puzzle_path=os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
            output_dir=str(output_dir),
            total_timesteps=2,
            seed=5,
            num_envs=1,
            num_steps=1,
            max_steps=4,
            device="cpu",
            hidden_sizes=(16,),
            discriminator_hidden_sizes=(16,),
            update_epochs=1,
            minibatch_size=1,
            discriminator_update_epochs=1,
            discriminator_minibatch_size=1,
            save_interval=1,
            num_skills=2,
            comet_log_artifacts=False,
        )
    )

    assert result.global_step == 2
    assert os.path.exists(output_dir / "diayn_policy.pt")
    assert os.path.exists(output_dir / "discriminator.pt")
    assert os.path.exists(output_dir / "skill_config.json")
    assert os.path.exists(output_dir / "obs_config.json")
    assert os.path.exists(output_dir / "discriminator_obs_config.json")
    assert os.path.exists(output_dir / "train.csv")
    with open(output_dir / "train.csv") as file:
        header = file.readline()
    assert "mean_object_novelty_reward" in header
    assert "mean_goal_progress_penalty" in header
    assert "object_novelty_rate" in header

    checkpoint = load_diayn_checkpoint(str(output_dir), resolve_device("cpu"))
    assert checkpoint["skill_config"]["num_skills"] == 2
    assert checkpoint["global_step"] == 2


def test_hierarchical_skill_env_runs_horizon_unless_episode_ends(tmp_path):
    pretrained_dir = _make_tiny_diayn_run(tmp_path / "pretrain")
    checkpoint = load_diayn_checkpoint(str(pretrained_dir), resolve_device("cpu"))
    observation_config = VectorObservationConfig.from_dict(checkpoint["observation_config"])
    low_level_policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=4,
        num_skills=2,
        hidden_sizes=(16,),
    )
    low_level_policy.load_state_dict(checkpoint["model_state_dict"])
    low_level_policy.eval()
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "trivial.pwp")
    puzzle_bank = load_puzzle_bank(puzzle_path)
    env = HierarchicalSkillEnv(
        puzzle_bank=puzzle_bank,
        low_level_policy=low_level_policy,
        observation_config=build_observation_config(puzzle_path),
        num_skills=2,
        max_steps=20,
        skill_horizon=3,
        device=resolve_device("cpu"),
    )

    env.reset(seed=7)
    _, _, terminated, truncated, info = env.step(0)

    assert 1 <= info["primitive_steps"] <= 3
    if not (terminated or truncated):
        assert info["primitive_steps"] == 3


def test_hierarchical_diayn_ppo_smoke_saves_checkpoint(tmp_path):
    pretrained_dir = _make_tiny_diayn_run(tmp_path / "pretrain")
    output_dir = tmp_path / "hierarchical"

    train_hierarchical_diayn_ppo(
        HierarchicalDIAYNConfig(
            pretrained_dir=str(pretrained_dir),
            puzzle_path=os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
            output_dir=str(output_dir),
            total_timesteps=2,
            seed=11,
            num_envs=1,
            num_steps=1,
            max_steps=4,
            skill_horizon=2,
            device="cpu",
            hidden_sizes=(16,),
            update_epochs=1,
            minibatch_size=1,
            eval_interval=1,
            eval_episodes=1,
            save_interval=1,
            comet_log_artifacts=False,
        )
    )

    assert os.path.exists(output_dir / "latest.pt")
    assert os.path.exists(output_dir / "skill_config.json")
    assert os.path.exists(output_dir / "obs_config.json")
    assert os.path.exists(output_dir / "train.csv")


def test_skill_conditioned_eval_can_be_capped(tmp_path):
    pretrained_dir = _make_tiny_diayn_run(tmp_path / "pretrain")
    checkpoint = load_diayn_checkpoint(str(pretrained_dir), resolve_device("cpu"))
    observation_config = VectorObservationConfig.from_dict(checkpoint["observation_config"])
    policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=4,
        num_skills=2,
        hidden_sizes=(16,),
    )
    policy.load_state_dict(checkpoint["model_state_dict"])

    stats, records = evaluate_skill_conditioned_policy(
        policy=policy,
        puzzle_paths=os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
        observation_config=observation_config,
        num_skills=2,
        max_steps=4,
        device=resolve_device("cpu"),
        max_puzzles=1,
        skill_ids=[1],
        deterministic=False,
        rollouts_per_skill=3,
    )

    assert stats["puzzle_count"] == 1
    assert stats["skill_count"] == 1
    assert stats["rollouts_per_skill"] == 3
    assert stats["record_count"] == 3
    assert stats["best_of_n_record_count"] == 1
    assert len(records) == 3
    assert {record["skill_id"] for record in records} == {1}
    assert {record["rollout_id"] for record in records} == {0, 1, 2}


def _make_tiny_diayn_run(output_dir):
    train_diayn_ppo(
        DIAYNTrainingConfig(
            puzzle_path=os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
            output_dir=str(output_dir),
            total_timesteps=1,
            seed=13,
            num_envs=1,
            num_steps=1,
            max_steps=4,
            device="cpu",
            hidden_sizes=(16,),
            discriminator_hidden_sizes=(16,),
            update_epochs=1,
            minibatch_size=1,
            discriminator_update_epochs=1,
            discriminator_minibatch_size=1,
            save_interval=1,
            num_skills=2,
            comet_log_artifacts=False,
        )
    )
    return output_dir
