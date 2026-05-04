# Copyright 2026
#
# DIAYN-style skill discovery with PPO for PushWorld.

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pushworld.puzzle import NUM_ACTIONS, Actions, PushWorldPuzzle, State
from pushworld.rl.architectures import (
    SkillConditionedActorCriticMLP,
    SkillDiscriminatorMLP,
)
from pushworld.rl.envs import PushWorldVectorEnv, load_puzzle_bank
from pushworld.rl.observations import (
    VectorObservationConfig,
    build_observation_config_from_puzzles,
    resolve_puzzle_paths,
)
from pushworld.rl.ppo import PPOUpdateConfig, update_policy
from pushworld.rl.skill_observations import (
    SkillDiscriminatorObservationConfig,
    build_skill_discriminator_observation_config_from_puzzles,
    encode_discriminator_state,
)
from pushworld.rl.storage import RolloutBuffer
from pushworld.rl.tracking import CometTrackingConfig, create_comet_tracker
from pushworld.rl.utils import CSVLogger, ensure_dir, resolve_device, set_global_seeds, write_json


@dataclass(frozen=True)
class DIAYNConfig:
    num_skills: int = 8
    diayn_reward_scale: float = 1.0
    object_change_reward_scale: float = 0.5
    object_change_reward_clip: float = 1.0
    object_novelty_reward_scale: float = 1.0
    object_novelty_reward_clip: float = 2.0
    object_novelty_requires_nonnegative_goal_progress: bool = True
    goal_progress_reward_scale: float = 0.25
    negative_goal_progress_penalty_scale: float = 1.0
    discriminator_lr: float = 3e-4
    discriminator_update_epochs: int = 4
    discriminator_minibatch_size: int = 256
    entropy_coef: float = 0.05


@dataclass(frozen=True)
class DIAYNTrainingConfig:
    puzzle_path: str
    output_dir: str
    total_timesteps: int = 1_000_000
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    max_steps: int = 512
    device: str = "auto"
    learning_rate: float = 3e-4
    hidden_sizes: Tuple[int, ...] = (256, 256)
    discriminator_hidden_sizes: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    action_masking: bool = True
    save_interval: int = 10
    num_skills: int = 8
    diayn_reward_scale: float = 1.0
    object_change_reward_scale: float = 0.5
    object_change_reward_clip: float = 1.0
    object_novelty_reward_scale: float = 1.0
    object_novelty_reward_clip: float = 2.0
    object_novelty_requires_nonnegative_goal_progress: bool = True
    goal_progress_reward_scale: float = 0.25
    negative_goal_progress_penalty_scale: float = 1.0
    discriminator_lr: float = 3e-4
    discriminator_update_epochs: int = 4
    discriminator_minibatch_size: int = 256
    entropy_coef: float = 0.05
    comet_enabled: bool = False
    comet_project_name: str = "pushworld-diayn"
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: Tuple[str, ...] = ()
    comet_log_artifacts: bool = True


@dataclass(frozen=True)
class DIAYNTrainingResult:
    output_dir: str
    policy_checkpoint: str
    discriminator_checkpoint: str
    global_step: int


@dataclass(frozen=True)
class DIAYNFinetuneConfig:
    pretrained_dir: str
    puzzle_path: str
    output_dir: str
    total_timesteps: int = 1_000_000
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    max_steps: int = 512
    device: str = "auto"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    action_masking: bool = True
    eval_interval: int = 10
    eval_episodes: int = 20
    skill_sampling: str = "fixed"
    fixed_skill_id: int = 0
    save_interval: int = 10
    comet_enabled: bool = False
    comet_project_name: str = "pushworld-diayn"
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: Tuple[str, ...] = ()
    comet_log_artifacts: bool = True


def compute_diayn_reward(
    discriminator_logits: torch.Tensor,
    skill_ids: torch.Tensor,
    num_skills: int,
) -> torch.Tensor:
    """Returns log q(z | s) - log p(z) for a uniform skill prior."""
    log_probs = F.log_softmax(discriminator_logits, dim=-1)
    skill_log_probs = log_probs.gather(1, skill_ids.long().view(-1, 1)).squeeze(1)
    return skill_log_probs + math.log(num_skills)


def compute_object_change_reward(
    motion_metrics: Dict[str, float],
    config: DIAYNTrainingConfig,
) -> float:
    """Rewards object manipulation so DIAYN cannot succeed by walking only."""
    object_displacement = motion_metrics["object_displacement"]
    if config.object_change_reward_clip > 0:
        object_displacement = min(object_displacement, config.object_change_reward_clip)
    positive_goal_progress = max(0.0, motion_metrics["goal_progress"])
    return (
        config.object_change_reward_scale * object_displacement
        + config.goal_progress_reward_scale * positive_goal_progress
    )


def initialize_object_position_memory(state: State) -> List[set]:
    """Creates per-object visited-position memory for one episode."""
    return [
        {state[index]} if index > 0 else set()
        for index in range(len(state))
    ]


def compute_object_novelty_metrics(
    previous_state: State,
    next_state: State,
    object_position_memory: List[set],
) -> Dict[str, float]:
    """Updates visited object positions and reports first-time object movement."""
    novel_object_positions = 0
    novel_object_moves = 0
    for object_index in range(1, len(next_state)):
        next_position = next_state[object_index]
        moved = previous_state[object_index] != next_position
        is_novel = next_position not in object_position_memory[object_index]
        if is_novel:
            object_position_memory[object_index].add(next_position)
            novel_object_positions += 1
            if moved:
                novel_object_moves += 1
    return {
        "novel_object_positions": float(novel_object_positions),
        "novel_object_moves": float(novel_object_moves),
        "object_novelty": 1.0 if novel_object_moves > 0 else 0.0,
    }


def compute_object_novelty_reward(
    novelty_metrics: Dict[str, float],
    config: DIAYNTrainingConfig,
    motion_metrics: Optional[Dict[str, float]] = None,
) -> float:
    """Rewards first-time object positions within the current episode."""
    if (
        config.object_novelty_requires_nonnegative_goal_progress
        and motion_metrics is not None
        and motion_metrics["goal_progress"] < 0.0
    ):
        return 0.0
    novel_positions = novelty_metrics["novel_object_positions"]
    if config.object_novelty_reward_clip > 0:
        novel_positions = min(novel_positions, config.object_novelty_reward_clip)
    return config.object_novelty_reward_scale * novel_positions


def compute_goal_progress_penalty(
    motion_metrics: Dict[str, float],
    config: DIAYNTrainingConfig,
) -> float:
    """Penalizes transitions that increase object-to-goal distance."""
    negative_goal_progress = max(0.0, -motion_metrics["goal_progress"])
    return config.negative_goal_progress_penalty_scale * negative_goal_progress


def compute_object_motion_metrics(
    puzzle: PushWorldPuzzle,
    previous_state: State,
    next_state: State,
    action: int,
) -> Dict[str, float]:
    """Summarizes one transition's agent/object manipulation effects."""
    agent_displacement = _manhattan(previous_state[0], next_state[0])
    object_displacements = [
        _manhattan(previous_state[index], next_state[index])
        for index in range(1, len(previous_state))
    ]
    object_displacement = float(sum(object_displacements))
    moved_object_count = float(sum(displacement > 0 for displacement in object_displacements))
    goal_progress = _goal_distance(puzzle, previous_state) - _goal_distance(
        puzzle, next_state
    )
    return {
        "agent_displacement": float(agent_displacement),
        "agent_moved": 1.0 if agent_displacement > 0 else 0.0,
        "object_displacement": object_displacement,
        "moved_object_count": moved_object_count,
        "object_push": 1.0 if object_displacement > 0 else 0.0,
        "object_contact": 1.0
        if _action_contacts_non_agent_object(puzzle, previous_state, action)
        else 0.0,
        "goal_progress": float(goal_progress),
    }


def update_discriminator(
    discriminator: SkillDiscriminatorMLP,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    skill_ids: torch.Tensor,
    update_epochs: int,
    minibatch_size: int,
) -> Dict[str, float]:
    """Updates the DIAYN discriminator on rollout states."""
    batch_size = states.shape[0]
    stats = {"discriminator_loss": 0.0, "discriminator_accuracy": 0.0, "num_batches": 0.0}
    for _ in range(update_epochs):
        indices = torch.randperm(batch_size, device=states.device)
        for start in range(0, batch_size, min(minibatch_size, batch_size)):
            batch_indices = indices[start : start + minibatch_size]
            logits = discriminator(states[batch_indices])
            loss = F.cross_entropy(logits, skill_ids[batch_indices].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = (
                    logits.argmax(dim=-1) == skill_ids[batch_indices].long()
                ).float().mean()
            stats["discriminator_loss"] += float(loss.detach().cpu())
            stats["discriminator_accuracy"] += float(accuracy.detach().cpu())
            stats["num_batches"] += 1.0

    denominator = max(1.0, stats["num_batches"])
    stats["discriminator_loss"] /= denominator
    stats["discriminator_accuracy"] /= denominator
    return stats


def train_diayn_ppo(config: DIAYNTrainingConfig) -> DIAYNTrainingResult:
    """Pretrains skill-conditioned PPO with DIAYN intrinsic rewards."""
    _validate_diayn_training_config(config)
    ensure_dir(config.output_dir)
    set_global_seeds(config.seed)
    device = resolve_device(config.device)

    puzzle_bank = load_puzzle_bank(config.puzzle_path)
    observation_config = build_observation_config_from_puzzles(
        puzzle for _, puzzle in puzzle_bank
    )
    discriminator_observation_config = (
        build_skill_discriminator_observation_config_from_puzzles(
            puzzle for _, puzzle in puzzle_bank
        )
    )
    _write_diayn_configs(config, observation_config, discriminator_observation_config)
    tracker = _create_tracker(config)
    tracker.log_parameters(
        {
            "diayn_training": asdict(config),
            "observation": observation_config.to_dict(),
            "discriminator_observation": discriminator_observation_config.to_dict(),
            "runtime": {"device": str(device)},
        }
    )

    policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        num_skills=config.num_skills,
        hidden_sizes=config.hidden_sizes,
    ).to(device)
    discriminator = SkillDiscriminatorMLP(
        input_dim=discriminator_observation_config.observation_size,
        num_skills=config.num_skills,
        hidden_sizes=config.discriminator_hidden_sizes,
    ).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.discriminator_lr, eps=1e-5
    )
    update_config = PPOUpdateConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_coef=config.clip_coef,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        action_masking=config.action_masking,
    )
    rollout = RolloutBuffer(
        num_steps=config.num_steps,
        num_envs=config.num_envs,
        observation_dim=policy.conditioned_observation_dim,
        num_actions=NUM_ACTIONS,
        device=device,
    )
    envs, observations, action_masks, skill_ids = _make_skill_envs(
        puzzle_bank, config, observation_config
    )
    object_position_memories = [
        initialize_object_position_memory(env.current_state)
        for env in envs
    ]

    num_updates = max(
        1,
        math.ceil(config.total_timesteps / (config.num_envs * config.num_steps)),
    )
    global_step = 0
    extrinsic_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)
    recent_extrinsic_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    train_log_path = os.path.join(config.output_dir, "train.csv")

    try:
        with CSVLogger(train_log_path, _diayn_train_log_fields(config.num_skills)) as logger:
            for update in range(1, num_updates + 1):
                rollout.reset()
                disc_states = []
                disc_skill_ids = []
                intrinsic_rewards = []
                discriminator_rewards = []
                object_change_rewards = []
                object_novelty_rewards = []
                goal_progress_penalties = []
                extrinsic_rewards = []
                endpoint_states = []
                skill_counts = np.zeros((config.num_skills,), dtype=np.int64)
                per_skill_object_displacement = np.zeros(
                    (config.num_skills,), dtype=np.float64
                )
                per_skill_object_pushes = np.zeros((config.num_skills,), dtype=np.float64)
                per_skill_object_contacts = np.zeros((config.num_skills,), dtype=np.float64)
                per_skill_goal_progress = np.zeros((config.num_skills,), dtype=np.float64)
                per_skill_novel_object_positions = np.zeros(
                    (config.num_skills,), dtype=np.float64
                )
                per_skill_object_novelties = np.zeros(
                    (config.num_skills,), dtype=np.float64
                )
                object_displacements = []
                object_pushes = []
                object_contacts = []
                moved_object_counts = []
                novel_object_positions = []
                object_novelties = []
                goal_progresses = []
                agent_moves = []
                last_dones = torch.zeros(
                    (config.num_envs,), dtype=torch.bool, device=device
                )

                for _ in range(config.num_steps):
                    observation_tensor = torch.as_tensor(
                        observations, dtype=torch.float32, device=device
                    )
                    skill_tensor = torch.as_tensor(skill_ids, dtype=torch.long, device=device)
                    conditioned_observation_tensor = policy.condition_observations(
                        observation_tensor, skill_tensor
                    )
                    action_mask_tensor = torch.as_tensor(
                        action_masks, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        actions, logprobs, _, values = policy.get_action_and_value(
                            conditioned_observation_tensor,
                            action_mask=action_mask_tensor
                            if config.action_masking
                            else None,
                        )

                    next_observations = []
                    next_action_masks = []
                    intrinsic_reward_step = np.zeros((config.num_envs,), dtype=np.float32)
                    extrinsic_reward_step = np.zeros((config.num_envs,), dtype=np.float32)
                    dones = np.zeros((config.num_envs,), dtype=bool)

                    for env_index, env in enumerate(envs):
                        previous_state = env.current_state
                        current_skill_id = int(skill_ids[env_index])
                        (
                            next_observation,
                            extrinsic_reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(int(actions[env_index].cpu().item()))
                        done = bool(terminated or truncated)
                        motion_metrics = compute_object_motion_metrics(
                            env.current_puzzle,
                            previous_state,
                            info["puzzle_state"],
                            int(actions[env_index].cpu().item()),
                        )
                        novelty_metrics = compute_object_novelty_metrics(
                            previous_state,
                            info["puzzle_state"],
                            object_position_memories[env_index],
                        )
                        discriminator_state = encode_discriminator_state(
                            env.current_puzzle,
                            info["puzzle_state"],
                            steps=env.steps,
                            max_steps=config.max_steps,
                            config=discriminator_observation_config,
                        )
                        with torch.no_grad():
                            logits = discriminator(
                                torch.as_tensor(
                                    discriminator_state[None, :],
                                    dtype=torch.float32,
                                    device=device,
                                )
                            )
                            intrinsic_reward = compute_diayn_reward(
                                logits,
                                torch.as_tensor([current_skill_id], device=device),
                                config.num_skills,
                            )[0].cpu().item()

                        object_change_reward = compute_object_change_reward(
                            motion_metrics, config
                        )
                        object_novelty_reward = compute_object_novelty_reward(
                            novelty_metrics, config, motion_metrics
                        )
                        goal_progress_penalty = compute_goal_progress_penalty(
                            motion_metrics, config
                        )
                        intrinsic_reward_step[env_index] = (
                            intrinsic_reward * config.diayn_reward_scale
                            + object_change_reward
                            + object_novelty_reward
                            - goal_progress_penalty
                        )
                        extrinsic_reward_step[env_index] = extrinsic_reward
                        dones[env_index] = done
                        skill_counts[current_skill_id] += 1
                        per_skill_object_displacement[current_skill_id] += motion_metrics[
                            "object_displacement"
                        ]
                        per_skill_object_pushes[current_skill_id] += motion_metrics[
                            "object_push"
                        ]
                        per_skill_object_contacts[current_skill_id] += motion_metrics[
                            "object_contact"
                        ]
                        per_skill_goal_progress[current_skill_id] += motion_metrics[
                            "goal_progress"
                        ]
                        per_skill_novel_object_positions[
                            current_skill_id
                        ] += novelty_metrics["novel_object_positions"]
                        per_skill_object_novelties[current_skill_id] += novelty_metrics[
                            "object_novelty"
                        ]
                        disc_states.append(discriminator_state)
                        disc_skill_ids.append(current_skill_id)
                        endpoint_states.append(discriminator_state)
                        extrinsic_returns[env_index] += extrinsic_reward
                        episode_lengths[env_index] += 1
                        discriminator_rewards.append(
                            intrinsic_reward * config.diayn_reward_scale
                        )
                        object_change_rewards.append(object_change_reward)
                        object_novelty_rewards.append(object_novelty_reward)
                        goal_progress_penalties.append(goal_progress_penalty)
                        object_displacements.append(motion_metrics["object_displacement"])
                        object_pushes.append(motion_metrics["object_push"])
                        object_contacts.append(motion_metrics["object_contact"])
                        moved_object_counts.append(motion_metrics["moved_object_count"])
                        novel_object_positions.append(
                            novelty_metrics["novel_object_positions"]
                        )
                        object_novelties.append(novelty_metrics["object_novelty"])
                        goal_progresses.append(motion_metrics["goal_progress"])
                        agent_moves.append(motion_metrics["agent_moved"])

                        if done:
                            recent_extrinsic_returns.append(
                                float(extrinsic_returns[env_index])
                            )
                            recent_lengths.append(int(episode_lengths[env_index]))
                            recent_successes.append(1.0 if terminated else 0.0)
                            extrinsic_returns[env_index] = 0.0
                            episode_lengths[env_index] = 0
                            skill_ids[env_index] = resample_skill(
                                skill_ids[env_index], config.num_skills
                            )
                            next_observation, info = env.reset()
                            object_position_memories[env_index] = (
                                initialize_object_position_memory(env.current_state)
                            )

                        next_observations.append(next_observation)
                        next_action_masks.append(info["action_mask"])

                    rollout.add(
                        observations=conditioned_observation_tensor,
                        action_masks=action_mask_tensor,
                        actions=actions,
                        logprobs=logprobs,
                        rewards=torch.as_tensor(
                            intrinsic_reward_step, dtype=torch.float32, device=device
                        ),
                        dones=torch.as_tensor(dones, dtype=torch.bool, device=device),
                        values=values,
                    )
                    observations = np.stack(next_observations)
                    action_masks = np.stack(next_action_masks)
                    last_dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                    global_step += config.num_envs
                    intrinsic_rewards.extend(intrinsic_reward_step.tolist())
                    extrinsic_rewards.extend(extrinsic_reward_step.tolist())

                last_observation_tensor = torch.as_tensor(
                    observations, dtype=torch.float32, device=device
                )
                last_skill_tensor = torch.as_tensor(skill_ids, dtype=torch.long, device=device)
                with torch.no_grad():
                    last_values = policy.get_value(
                        last_observation_tensor,
                        skill_ids=last_skill_tensor,
                    )
                rollout.compute_returns_and_advantages(
                    last_values=last_values,
                    last_dones=last_dones,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                ppo_stats = update_policy(policy, policy_optimizer, rollout, update_config)
                disc_state_tensor = torch.as_tensor(
                    np.stack(disc_states), dtype=torch.float32, device=device
                )
                disc_skill_tensor = torch.as_tensor(
                    np.array(disc_skill_ids), dtype=torch.long, device=device
                )
                discriminator_stats = update_discriminator(
                    discriminator=discriminator,
                    optimizer=discriminator_optimizer,
                    states=disc_state_tensor,
                    skill_ids=disc_skill_tensor,
                    update_epochs=config.discriminator_update_epochs,
                    minibatch_size=config.discriminator_minibatch_size,
                )

                if update % config.save_interval == 0 or update == num_updates:
                    save_diayn_checkpoint(
                        output_dir=config.output_dir,
                        policy=policy,
                        discriminator=discriminator,
                        policy_optimizer=policy_optimizer,
                        discriminator_optimizer=discriminator_optimizer,
                        training_config=config,
                        observation_config=observation_config,
                        discriminator_observation_config=discriminator_observation_config,
                        global_step=global_step,
                    )

                log_row = {
                    "update": update,
                    "global_step": global_step,
                    "mean_intrinsic_reward": float(np.mean(intrinsic_rewards)),
                    "mean_discriminator_reward": float(np.mean(discriminator_rewards)),
                    "mean_object_change_reward": float(np.mean(object_change_rewards)),
                    "mean_object_novelty_reward": float(
                        np.mean(object_novelty_rewards)
                    ),
                    "mean_goal_progress_penalty": float(
                        np.mean(goal_progress_penalties)
                    ),
                    "mean_extrinsic_reward": float(np.mean(extrinsic_rewards)),
                    "mean_object_displacement": float(np.mean(object_displacements)),
                    "object_push_rate": float(np.mean(object_pushes)),
                    "object_contact_rate": float(np.mean(object_contacts)),
                    "mean_moved_object_count": float(np.mean(moved_object_counts)),
                    "mean_novel_object_positions": float(
                        np.mean(novel_object_positions)
                    ),
                    "object_novelty_rate": float(np.mean(object_novelties)),
                    "mean_goal_progress": float(np.mean(goal_progresses)),
                    "agent_move_rate": float(np.mean(agent_moves)),
                    "mean_episode_return": _mean_or_empty(recent_extrinsic_returns),
                    "mean_episode_length": _mean_or_empty(recent_lengths),
                    "mean_success": _mean_or_empty(recent_successes),
                    "skill_usage_entropy": _categorical_entropy(skill_counts),
                    "endpoint_diversity": _endpoint_diversity(endpoint_states),
                    **_per_skill_manipulation_stats(
                        skill_counts=skill_counts,
                        object_displacement=per_skill_object_displacement,
                        object_pushes=per_skill_object_pushes,
                        object_contacts=per_skill_object_contacts,
                        goal_progress=per_skill_goal_progress,
                        novel_object_positions=per_skill_novel_object_positions,
                        object_novelties=per_skill_object_novelties,
                    ),
                    **ppo_stats,
                    **discriminator_stats,
                }
                logger.write(log_row)
                tracker.log_metrics(log_row, step=global_step, prefix="diayn")

        save_diayn_checkpoint(
            output_dir=config.output_dir,
            policy=policy,
            discriminator=discriminator,
            policy_optimizer=policy_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            training_config=config,
            observation_config=observation_config,
            discriminator_observation_config=discriminator_observation_config,
            global_step=global_step,
        )
    finally:
        if config.comet_log_artifacts:
            for name in [
                "config.json",
                "skill_config.json",
                "obs_config.json",
                "discriminator_obs_config.json",
                "train.csv",
                "diayn_policy.pt",
                "discriminator.pt",
            ]:
                tracker.log_asset(os.path.join(config.output_dir, name), name=name)
        tracker.end()

    return DIAYNTrainingResult(
        output_dir=config.output_dir,
        policy_checkpoint=os.path.join(config.output_dir, "diayn_policy.pt"),
        discriminator_checkpoint=os.path.join(config.output_dir, "discriminator.pt"),
        global_step=global_step,
    )


def finetune_diayn_ppo(config: DIAYNFinetuneConfig):
    """Fine-tunes a DIAYN skill-conditioned policy on true PushWorld rewards."""
    _validate_diayn_finetune_config(config)
    ensure_dir(config.output_dir)
    set_global_seeds(config.seed)
    device = resolve_device(config.device)
    checkpoint = load_diayn_checkpoint(config.pretrained_dir, device)
    observation_config = VectorObservationConfig.from_dict(checkpoint["observation_config"])
    skill_config = checkpoint["skill_config"]
    num_skills = int(skill_config["num_skills"])
    hidden_sizes = tuple(checkpoint["training_config"].get("hidden_sizes", (256, 256)))
    policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        num_skills=num_skills,
        hidden_sizes=hidden_sizes,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    update_config = PPOUpdateConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_coef=config.clip_coef,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        action_masking=config.action_masking,
    )
    puzzle_bank = load_puzzle_bank(config.puzzle_path)
    envs, observations, action_masks, skill_ids = _make_skill_envs(
        puzzle_bank, config, observation_config, num_skills=num_skills
    )
    skill_ids = _initialize_finetune_skill_ids(config, num_skills)
    rollout = RolloutBuffer(
        num_steps=config.num_steps,
        num_envs=config.num_envs,
        observation_dim=policy.conditioned_observation_dim,
        num_actions=NUM_ACTIONS,
        device=device,
    )
    tracker = create_comet_tracker(
        CometTrackingConfig(
            enabled=config.comet_enabled,
            project_name=config.comet_project_name,
            workspace=config.comet_workspace,
            experiment_name=config.comet_experiment_name,
            tags=config.comet_tags,
            log_artifacts=config.comet_log_artifacts,
        )
    )
    tracker.log_parameters({"finetune": asdict(config), "num_skills": num_skills})
    write_json(os.path.join(config.output_dir, "config.json"), asdict(config))
    write_json(os.path.join(config.output_dir, "skill_config.json"), {"num_skills": num_skills})
    observation_config.save(os.path.join(config.output_dir, "obs_config.json"))
    train_log_path = os.path.join(config.output_dir, "train.csv")

    num_updates = max(
        1,
        math.ceil(config.total_timesteps / (config.num_envs * config.num_steps)),
    )
    global_step = 0
    best_success_rate = -1.0
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    episode_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)

    try:
        with CSVLogger(train_log_path, _finetune_log_fields()) as logger:
            for update in range(1, num_updates + 1):
                rollout.reset()
                last_dones = torch.zeros(
                    (config.num_envs,), dtype=torch.bool, device=device
                )
                for _ in range(config.num_steps):
                    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                    skill_tensor = torch.as_tensor(skill_ids, dtype=torch.long, device=device)
                    conditioned_obs = policy.condition_observations(obs_tensor, skill_tensor)
                    mask_tensor = torch.as_tensor(action_masks, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        actions, logprobs, _, values = policy.get_action_and_value(
                            conditioned_obs,
                            action_mask=mask_tensor if config.action_masking else None,
                        )
                    next_obs = []
                    next_masks = []
                    rewards = np.zeros((config.num_envs,), dtype=np.float32)
                    dones = np.zeros((config.num_envs,), dtype=bool)
                    for env_index, env in enumerate(envs):
                        observation, reward, terminated, truncated, info = env.step(
                            int(actions[env_index].cpu().item())
                        )
                        done = bool(terminated or truncated)
                        rewards[env_index] = reward
                        dones[env_index] = done
                        episode_returns[env_index] += reward
                        episode_lengths[env_index] += 1
                        if done:
                            recent_returns.append(float(episode_returns[env_index]))
                            recent_lengths.append(int(episode_lengths[env_index]))
                            recent_successes.append(1.0 if terminated else 0.0)
                            episode_returns[env_index] = 0.0
                            episode_lengths[env_index] = 0
                            skill_ids[env_index] = next_finetune_skill_id(
                                current_skill=skill_ids[env_index],
                                env_index=env_index,
                                config=config,
                                num_skills=num_skills,
                            )
                            observation, info = env.reset()
                        next_obs.append(observation)
                        next_masks.append(info["action_mask"])
                    rollout.add(
                        observations=conditioned_obs,
                        action_masks=mask_tensor,
                        actions=actions,
                        logprobs=logprobs,
                        rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                        dones=torch.as_tensor(dones, dtype=torch.bool, device=device),
                        values=values,
                    )
                    observations = np.stack(next_obs)
                    action_masks = np.stack(next_masks)
                    last_dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                    global_step += config.num_envs

                with torch.no_grad():
                    last_values = policy.get_value(
                        torch.as_tensor(observations, dtype=torch.float32, device=device),
                        skill_ids=torch.as_tensor(skill_ids, dtype=torch.long, device=device),
                    )
                rollout.compute_returns_and_advantages(
                    last_values, last_dones, config.gamma, config.gae_lambda
                )
                ppo_stats = update_policy(policy, optimizer, rollout, update_config)
                eval_success_rate = ""
                if config.eval_interval > 0 and (
                    update % config.eval_interval == 0 or update == num_updates
                ):
                    eval_stats, _ = evaluate_skill_conditioned_policy(
                        policy=policy,
                        puzzle_paths=config.puzzle_path,
                        observation_config=observation_config,
                        num_skills=num_skills,
                        max_steps=config.max_steps,
                        device=device,
                        action_masking=config.action_masking,
                        deterministic=True,
                        max_puzzles=config.eval_episodes,
                        seed=config.seed + update,
                    )
                    eval_success_rate = eval_stats["best_success_rate"]
                    if eval_success_rate > best_success_rate:
                        best_success_rate = eval_success_rate
                        save_skill_policy_checkpoint(
                            os.path.join(config.output_dir, "best_success.pt"),
                            policy,
                            optimizer,
                            config,
                            observation_config,
                            num_skills,
                            global_step,
                            best_success_rate,
                        )
                    tracker.log_metrics(eval_stats, step=global_step, prefix="finetune_eval")
                if update % config.save_interval == 0 or update == num_updates:
                    save_skill_policy_checkpoint(
                        os.path.join(config.output_dir, "latest.pt"),
                        policy,
                        optimizer,
                        config,
                        observation_config,
                        num_skills,
                        global_step,
                        best_success_rate,
                    )
                log_row = {
                    "update": update,
                    "global_step": global_step,
                    "mean_episode_return": _mean_or_empty(recent_returns),
                    "mean_episode_length": _mean_or_empty(recent_lengths),
                    "mean_success": _mean_or_empty(recent_successes),
                    "eval_success_rate": eval_success_rate,
                    **ppo_stats,
                }
                logger.write(log_row)
                tracker.log_metrics(log_row, step=global_step, prefix="finetune")
    finally:
        if config.comet_log_artifacts:
            for name in [
                "config.json",
                "skill_config.json",
                "obs_config.json",
                "train.csv",
                "latest.pt",
                "best_success.pt",
            ]:
                tracker.log_asset(os.path.join(config.output_dir, name), name=name)
        tracker.end()


def evaluate_skill_conditioned_policy(
    policy: SkillConditionedActorCriticMLP,
    puzzle_paths,
    observation_config: VectorObservationConfig,
    num_skills: int,
    max_steps: int,
    device: torch.device,
    action_masking: bool = True,
    deterministic: bool = True,
    max_puzzles: Optional[int] = None,
    skill_ids: Optional[Sequence[int]] = None,
    seed: int = 0,
    rollouts_per_skill: int = 1,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluates every skill on every puzzle and reports best/average skill success."""
    if rollouts_per_skill < 1:
        raise ValueError("rollouts_per_skill must be >= 1")
    records = []
    was_training = policy.training
    policy.eval()
    selected_puzzle_paths = _select_eval_paths(
        resolve_puzzle_paths(puzzle_paths), max_puzzles, seed
    )
    selected_skill_ids = list(range(num_skills)) if skill_ids is None else list(skill_ids)
    for skill_id in selected_skill_ids:
        _checked_fixed_skill_id(int(skill_id), num_skills)
    try:
        for puzzle_index, puzzle_path in enumerate(selected_puzzle_paths):
            for skill_id in selected_skill_ids:
                for rollout_id in range(rollouts_per_skill):
                    env = PushWorldVectorEnv(
                        puzzle_path,
                        max_steps=max_steps,
                        observation_config=observation_config,
                    )
                    rollout_seed = (
                        int(seed)
                        + 1_000_003 * (puzzle_index + 1)
                        + 1_009 * int(skill_id)
                        + int(rollout_id)
                    )
                    observation, info = env.reset(seed=rollout_seed)
                    object_position_memory = initialize_object_position_memory(
                        env.current_state
                    )
                    total_reward = 0.0
                    length = 0
                    total_object_displacement = 0.0
                    object_push_count = 0
                    object_contact_count = 0
                    total_novel_object_positions = 0.0
                    object_novelty_count = 0
                    total_goal_progress = 0.0
                    terminated = False
                    truncated = False
                    while not (terminated or truncated):
                        obs_tensor = torch.as_tensor(
                            observation[None, :], dtype=torch.float32, device=device
                        )
                        skill_tensor = torch.as_tensor(
                            [skill_id], dtype=torch.long, device=device
                        )
                        mask_tensor = torch.as_tensor(
                            info["action_mask"][None, :],
                            dtype=torch.float32,
                            device=device,
                        )
                        with torch.no_grad():
                            if deterministic:
                                logits, _ = policy.get_logits_and_value(
                                    obs_tensor, skill_tensor
                                )
                                if action_masking:
                                    mask = mask_tensor.to(dtype=torch.bool)
                                    logits = logits.masked_fill(
                                        ~mask, torch.finfo(logits.dtype).min
                                    )
                                action = torch.argmax(logits, dim=-1)
                            else:
                                action, _, _, _ = policy.get_action_and_value(
                                    obs_tensor,
                                    skill_ids=skill_tensor,
                                    action_mask=mask_tensor if action_masking else None,
                                )
                        previous_state = env.current_state
                        observation, reward, terminated, truncated, info = env.step(
                            int(action[0].cpu().item())
                        )
                        motion_metrics = compute_object_motion_metrics(
                            env.current_puzzle,
                            previous_state,
                            info["puzzle_state"],
                            int(action[0].cpu().item()),
                        )
                        novelty_metrics = compute_object_novelty_metrics(
                            previous_state,
                            info["puzzle_state"],
                            object_position_memory,
                        )
                        total_reward += reward
                        length += 1
                        total_object_displacement += motion_metrics[
                            "object_displacement"
                        ]
                        object_push_count += int(motion_metrics["object_push"])
                        object_contact_count += int(motion_metrics["object_contact"])
                        total_novel_object_positions += novelty_metrics[
                            "novel_object_positions"
                        ]
                        object_novelty_count += int(novelty_metrics["object_novelty"])
                        total_goal_progress += motion_metrics["goal_progress"]
                    records.append(
                        {
                            "puzzle_path": puzzle_path,
                            "skill_id": skill_id,
                            "rollout_id": rollout_id,
                            "return": total_reward,
                            "length": length,
                            "success": 1.0 if terminated else 0.0,
                            "truncated": 1.0 if truncated else 0.0,
                            "object_displacement": total_object_displacement,
                            "object_push_count": object_push_count,
                            "object_contact_count": object_contact_count,
                            "novel_object_positions": total_novel_object_positions,
                            "object_novelty_count": object_novelty_count,
                            "goal_progress": total_goal_progress,
                        }
                    )
    finally:
        if was_training:
            policy.train()

    by_skill = []
    for skill_id in selected_skill_ids:
        skill_records = [record for record in records if record["skill_id"] == skill_id]
        by_skill.append(
            float(np.mean([record["success"] for record in skill_records]))
            if skill_records
            else 0.0
        )
    successes = [record["success"] for record in records]
    returns = [record["return"] for record in records]
    lengths = [record["length"] for record in records]
    total_lengths = max(1.0, float(sum(lengths)))
    best_records = summarize_best_of_skill_records(records)
    best_by_skill = []
    for skill_id in selected_skill_ids:
        skill_records = [
            record for record in best_records if record["skill_id"] == skill_id
        ]
        best_by_skill.append(
            float(np.mean([record["success"] for record in skill_records]))
            if skill_records
            else 0.0
        )
    best_lengths = [record["length"] for record in best_records]
    best_total_lengths = max(1.0, float(sum(best_lengths)))
    best_puzzle_records = summarize_best_of_puzzle_records(records)
    solved_puzzles = [record for record in best_puzzle_records if record["success"] > 0]

    stats = {
        "mean_success_rate": float(np.mean(successes)) if successes else 0.0,
        "best_success_rate": float(max(by_skill) if by_skill else 0.0),
        "best_skill_id": int(selected_skill_ids[int(np.argmax(by_skill))]) if by_skill else 0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_object_displacement": float(
            np.mean([record["object_displacement"] for record in records])
        )
        if records
        else 0.0,
        "object_push_rate": float(
            sum(record["object_push_count"] for record in records) / total_lengths
        )
        if records
        else 0.0,
        "object_contact_rate": float(
            sum(record["object_contact_count"] for record in records) / total_lengths
        )
        if records
        else 0.0,
        "mean_novel_object_positions": float(
            np.mean([record["novel_object_positions"] for record in records])
        )
        if records
        else 0.0,
        "object_novelty_rate": float(
            sum(record["object_novelty_count"] for record in records) / total_lengths
        )
        if records
        else 0.0,
        "mean_goal_progress": float(
            np.mean([record["goal_progress"] for record in records])
        )
        if records
        else 0.0,
        "puzzle_count": len(selected_puzzle_paths),
        "skill_count": len(selected_skill_ids),
        "rollouts_per_skill": rollouts_per_skill,
        "record_count": len(records),
        "best_of_n_mean_success_rate": float(
            np.mean([record["success"] for record in best_records])
        )
        if best_records
        else 0.0,
        "best_of_n_best_success_rate": float(max(best_by_skill) if best_by_skill else 0.0),
        "best_of_n_best_skill_id": int(
            selected_skill_ids[int(np.argmax(best_by_skill))]
        )
        if best_by_skill
        else 0,
        "best_of_n_mean_return": float(
            np.mean([record["return"] for record in best_records])
        )
        if best_records
        else 0.0,
        "best_of_n_mean_length": float(np.mean(best_lengths)) if best_lengths else 0.0,
        "best_of_n_object_push_rate": float(
            sum(record["object_push_count"] for record in best_records)
            / best_total_lengths
        )
        if best_records
        else 0.0,
        "best_of_n_object_novelty_rate": float(
            sum(record["object_novelty_count"] for record in best_records)
            / best_total_lengths
        )
        if best_records
        else 0.0,
        "best_of_n_mean_goal_progress": float(
            np.mean([record["goal_progress"] for record in best_records])
        )
        if best_records
        else 0.0,
        "best_of_n_puzzle_coverage_rate": float(
            len(solved_puzzles) / max(1, len(selected_puzzle_paths))
        ),
        "best_of_n_solved_puzzle_count": len(solved_puzzles),
        "best_of_n_record_count": len(best_records),
    }
    return stats, records


def summarize_best_of_skill_records(
    records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Selects the best rollout for each puzzle-skill pair."""
    return _summarize_best_records(records, ("puzzle_path", "skill_id"))


def summarize_best_of_puzzle_records(
    records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Selects the best rollout across all skills for each puzzle."""
    return _summarize_best_records(records, ("puzzle_path",))


def save_diayn_checkpoint(
    output_dir: str,
    policy: SkillConditionedActorCriticMLP,
    discriminator: SkillDiscriminatorMLP,
    policy_optimizer: Optional[torch.optim.Optimizer],
    discriminator_optimizer: Optional[torch.optim.Optimizer],
    training_config: DIAYNTrainingConfig,
    observation_config: VectorObservationConfig,
    discriminator_observation_config: SkillDiscriminatorObservationConfig,
    global_step: int,
) -> None:
    policy_checkpoint = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": None
        if policy_optimizer is None
        else policy_optimizer.state_dict(),
        "training_config": asdict(training_config),
        "skill_config": {"num_skills": training_config.num_skills},
        "observation_config": observation_config.to_dict(),
        "discriminator_observation_config": discriminator_observation_config.to_dict(),
        "global_step": global_step,
    }
    torch.save(policy_checkpoint, os.path.join(output_dir, "diayn_policy.pt"))
    torch.save(
        {
            "model_state_dict": discriminator.state_dict(),
            "optimizer_state_dict": None
            if discriminator_optimizer is None
            else discriminator_optimizer.state_dict(),
            "training_config": asdict(training_config),
            "skill_config": {"num_skills": training_config.num_skills},
            "discriminator_observation_config": discriminator_observation_config.to_dict(),
            "global_step": global_step,
        },
        os.path.join(output_dir, "discriminator.pt"),
    )


def save_skill_policy_checkpoint(
    path: str,
    policy: SkillConditionedActorCriticMLP,
    optimizer: Optional[torch.optim.Optimizer],
    config,
    observation_config: VectorObservationConfig,
    num_skills: int,
    global_step: int,
    best_success_rate: float,
) -> None:
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "training_config": asdict(config),
            "skill_config": {"num_skills": num_skills},
            "observation_config": observation_config.to_dict(),
            "global_step": global_step,
            "best_success_rate": best_success_rate,
        },
        path,
    )


def load_diayn_checkpoint(pretrained_dir: str, device: torch.device) -> dict:
    path = pretrained_dir
    if os.path.isdir(path):
        path = os.path.join(path, "diayn_policy.pt")
    return torch.load(path, map_location=device)


def _make_skill_envs(
    puzzle_bank: Sequence[Tuple[str, PushWorldPuzzle]],
    config,
    observation_config: VectorObservationConfig,
    num_skills: Optional[int] = None,
):
    num_skills = config.num_skills if num_skills is None else num_skills
    envs = [
        PushWorldVectorEnv(
            [path for path, _ in puzzle_bank],
            max_steps=config.max_steps,
            observation_config=observation_config,
            preloaded_puzzles=puzzle_bank,
        )
        for _ in range(config.num_envs)
    ]
    observations = []
    action_masks = []
    skill_ids = np.zeros((config.num_envs,), dtype=np.int64)
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=config.seed + index)
        observations.append(observation)
        action_masks.append(info["action_mask"])
        skill_ids[index] = _sample_skill(num_skills)
    return envs, np.stack(observations), np.stack(action_masks), skill_ids


def _sample_skill(num_skills: int) -> int:
    return random.randrange(num_skills)


def resample_skill(current_skill: int, num_skills: int) -> int:
    """Samples a skill for a new episode, changing it when possible."""
    if num_skills <= 1:
        return 0
    next_skill = _sample_skill(num_skills - 1)
    if next_skill >= current_skill:
        next_skill += 1
    return next_skill


def next_finetune_skill_id(
    current_skill: int,
    env_index: int,
    config: DIAYNFinetuneConfig,
    num_skills: int,
) -> int:
    """Selects the next downstream fine-tuning skill for one environment."""
    if config.skill_sampling == "fixed":
        return _checked_fixed_skill_id(config.fixed_skill_id, num_skills)
    if config.skill_sampling == "cycle":
        return (int(current_skill) + config.num_envs) % num_skills
    if config.skill_sampling == "uniform":
        return resample_skill(int(current_skill), num_skills)
    raise ValueError(f"Unknown skill_sampling: {config.skill_sampling}")


def _initialize_finetune_skill_ids(
    config: DIAYNFinetuneConfig,
    num_skills: int,
) -> np.ndarray:
    if config.skill_sampling == "fixed":
        return np.full(
            (config.num_envs,),
            _checked_fixed_skill_id(config.fixed_skill_id, num_skills),
            dtype=np.int64,
        )
    if config.skill_sampling == "cycle":
        return np.array(
            [env_index % num_skills for env_index in range(config.num_envs)],
            dtype=np.int64,
        )
    if config.skill_sampling == "uniform":
        return np.array([_sample_skill(num_skills) for _ in range(config.num_envs)], dtype=np.int64)
    raise ValueError(f"Unknown skill_sampling: {config.skill_sampling}")


def _checked_fixed_skill_id(fixed_skill_id: int, num_skills: int) -> int:
    if fixed_skill_id < 0 or fixed_skill_id >= num_skills:
        raise ValueError(
            f"fixed_skill_id must be in [0, {num_skills - 1}], got {fixed_skill_id}"
        )
    return int(fixed_skill_id)


def _select_eval_paths(
    puzzle_paths: Sequence[str],
    max_puzzles: Optional[int],
    seed: int,
) -> List[str]:
    paths = list(puzzle_paths)
    if max_puzzles is None or max_puzzles <= 0 or max_puzzles >= len(paths):
        return paths
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(paths)), max_puzzles))
    return [paths[index] for index in indices]


def _summarize_best_records(
    records: Sequence[Dict[str, object]],
    group_keys: Tuple[str, ...],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for record in records:
        key = tuple(record[group_key] for group_key in group_keys)
        if key not in grouped or _record_score(record) > _record_score(grouped[key]):
            grouped[key] = dict(record)
    return sorted(
        grouped.values(),
        key=lambda record: tuple(record[group_key] for group_key in group_keys),
    )


def _record_score(record: Dict[str, object]) -> Tuple[float, float, float]:
    return (
        float(record["success"]),
        float(record["return"]),
        -float(record["length"]),
    )


def _per_skill_manipulation_stats(
    skill_counts: np.ndarray,
    object_displacement: np.ndarray,
    object_pushes: np.ndarray,
    object_contacts: np.ndarray,
    goal_progress: np.ndarray,
    novel_object_positions: np.ndarray,
    object_novelties: np.ndarray,
) -> Dict[str, float]:
    stats = {}
    for skill_id, count in enumerate(skill_counts):
        denominator = max(1, int(count))
        stats[f"skill_{skill_id}_object_displacement"] = float(
            object_displacement[skill_id] / denominator
        )
        stats[f"skill_{skill_id}_object_push_rate"] = float(
            object_pushes[skill_id] / denominator
        )
        stats[f"skill_{skill_id}_object_contact_rate"] = float(
            object_contacts[skill_id] / denominator
        )
        stats[f"skill_{skill_id}_goal_progress"] = float(
            goal_progress[skill_id] / denominator
        )
        stats[f"skill_{skill_id}_novel_object_positions"] = float(
            novel_object_positions[skill_id] / denominator
        )
        stats[f"skill_{skill_id}_object_novelty_rate"] = float(
            object_novelties[skill_id] / denominator
        )
    return stats


def _manhattan(point_a, point_b) -> int:
    return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])


def _goal_distance(puzzle: PushWorldPuzzle, state: State) -> float:
    distance = 0
    for object_index, goal in enumerate(puzzle.goal_state, start=1):
        distance += _manhattan(state[object_index], goal)
    return float(distance)


def _action_contacts_non_agent_object(
    puzzle: PushWorldPuzzle,
    state: State,
    action: int,
) -> bool:
    displacement = Actions.DISPLACEMENTS[action]
    next_agent_position = (
        state[0][0] + int(displacement[0]),
        state[0][1] + int(displacement[1]),
    )
    next_agent_cells = _absolute_object_cells(
        puzzle.movable_objects[0], next_agent_position
    )
    for object_index in range(1, puzzle.num_movables):
        object_cells = _absolute_object_cells(
            puzzle.movable_objects[object_index], state[object_index]
        )
        if next_agent_cells.intersection(object_cells):
            return True
    return False


def _absolute_object_cells(pushworld_object, position) -> set:
    return {
        (position[0] + cell[0], position[1] + cell[1])
        for cell in pushworld_object.cells
    }


def _write_diayn_configs(
    config: DIAYNTrainingConfig,
    observation_config: VectorObservationConfig,
    discriminator_observation_config: SkillDiscriminatorObservationConfig,
) -> None:
    write_json(os.path.join(config.output_dir, "config.json"), asdict(config))
    write_json(
        os.path.join(config.output_dir, "skill_config.json"),
        asdict(
            DIAYNConfig(
                num_skills=config.num_skills,
                diayn_reward_scale=config.diayn_reward_scale,
                object_change_reward_scale=config.object_change_reward_scale,
                object_change_reward_clip=config.object_change_reward_clip,
                object_novelty_reward_scale=config.object_novelty_reward_scale,
                object_novelty_reward_clip=config.object_novelty_reward_clip,
                object_novelty_requires_nonnegative_goal_progress=(
                    config.object_novelty_requires_nonnegative_goal_progress
                ),
                goal_progress_reward_scale=config.goal_progress_reward_scale,
                negative_goal_progress_penalty_scale=(
                    config.negative_goal_progress_penalty_scale
                ),
                discriminator_lr=config.discriminator_lr,
                discriminator_update_epochs=config.discriminator_update_epochs,
                discriminator_minibatch_size=config.discriminator_minibatch_size,
                entropy_coef=config.entropy_coef,
            )
        ),
    )
    observation_config.save(os.path.join(config.output_dir, "obs_config.json"))
    discriminator_observation_config.save(
        os.path.join(config.output_dir, "discriminator_obs_config.json")
    )


def _create_tracker(config):
    return create_comet_tracker(
        CometTrackingConfig(
            enabled=config.comet_enabled,
            project_name=config.comet_project_name,
            workspace=config.comet_workspace,
            experiment_name=config.comet_experiment_name,
            tags=config.comet_tags,
            log_artifacts=config.comet_log_artifacts,
        )
    )


def _validate_diayn_training_config(config: DIAYNTrainingConfig) -> None:
    if config.num_skills < 1:
        raise ValueError("num_skills must be >= 1")
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1")
    if config.num_envs < 1 or config.num_steps < 1:
        raise ValueError("num_envs and num_steps must be >= 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if config.object_change_reward_scale < 0:
        raise ValueError("object_change_reward_scale must be >= 0")
    if config.object_change_reward_clip < 0:
        raise ValueError("object_change_reward_clip must be >= 0")
    if config.object_novelty_reward_scale < 0:
        raise ValueError("object_novelty_reward_scale must be >= 0")
    if config.object_novelty_reward_clip < 0:
        raise ValueError("object_novelty_reward_clip must be >= 0")
    if config.goal_progress_reward_scale < 0:
        raise ValueError("goal_progress_reward_scale must be >= 0")
    if config.negative_goal_progress_penalty_scale < 0:
        raise ValueError("negative_goal_progress_penalty_scale must be >= 0")
    if config.save_interval < 1:
        raise ValueError("save_interval must be >= 1")
    resolve_puzzle_paths(config.puzzle_path)


def _validate_diayn_finetune_config(config: DIAYNFinetuneConfig) -> None:
    if config.skill_sampling not in {"uniform", "fixed", "cycle"}:
        raise ValueError("skill_sampling must be one of: uniform, fixed, cycle")
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1")
    if config.num_envs < 1 or config.num_steps < 1:
        raise ValueError("num_envs and num_steps must be >= 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if config.eval_interval > 0 and config.eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1 when evaluation is enabled")
    if config.save_interval < 1:
        raise ValueError("save_interval must be >= 1")
    resolve_puzzle_paths(config.puzzle_path)


def _mean_or_empty(values: deque) -> object:
    if not values:
        return ""
    return float(np.mean(values))


def _categorical_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def _endpoint_diversity(states: Sequence[np.ndarray]) -> float:
    if len(states) < 2:
        return 0.0
    stacked = np.stack(states)
    return float(np.mean(np.std(stacked, axis=0)))


def _diayn_train_log_fields(num_skills: int) -> List[str]:
    fields = [
        "update",
        "global_step",
        "mean_intrinsic_reward",
        "mean_discriminator_reward",
        "mean_object_change_reward",
        "mean_object_novelty_reward",
        "mean_goal_progress_penalty",
        "mean_extrinsic_reward",
        "mean_object_displacement",
        "object_push_rate",
        "object_contact_rate",
        "mean_moved_object_count",
        "mean_novel_object_positions",
        "object_novelty_rate",
        "mean_goal_progress",
        "agent_move_rate",
        "mean_episode_return",
        "mean_episode_length",
        "mean_success",
        "skill_usage_entropy",
        "endpoint_diversity",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "num_minibatches",
        "discriminator_loss",
        "discriminator_accuracy",
        "num_batches",
    ]
    for skill_id in range(num_skills):
        fields.extend(
            [
                f"skill_{skill_id}_object_displacement",
                f"skill_{skill_id}_object_push_rate",
                f"skill_{skill_id}_object_contact_rate",
                f"skill_{skill_id}_goal_progress",
                f"skill_{skill_id}_novel_object_positions",
                f"skill_{skill_id}_object_novelty_rate",
            ]
        )
    return fields


def _finetune_log_fields() -> List[str]:
    return [
        "update",
        "global_step",
        "mean_episode_return",
        "mean_episode_length",
        "mean_success",
        "eval_success_rate",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "num_minibatches",
    ]
