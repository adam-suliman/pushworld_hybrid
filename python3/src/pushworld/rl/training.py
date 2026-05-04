# Copyright 2026
#
# PPO training and evaluation orchestration for PushWorld.

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle
from pushworld.rl.architectures import ActorCriticMLP
from pushworld.rl.curriculum import prepare_curriculum
from pushworld.rl.envs import PushWorldVectorEnv, load_puzzle_bank
from pushworld.rl.observations import (
    VectorObservationConfig,
    build_observation_config_from_puzzles,
    resolve_puzzle_paths,
)
from pushworld.rl.ppo import PPOUpdateConfig, update_policy
from pushworld.rl.storage import RolloutBuffer
from pushworld.rl.tracking import CometTrackingConfig, create_comet_tracker
from pushworld.rl.utils import (
    CSVLogger,
    ensure_dir,
    resolve_device,
    set_global_seeds,
    write_json,
)


@dataclass(frozen=True)
class PPOTrainingConfig:
    puzzle_path: str
    output_dir: str
    total_timesteps: int = 1_000_000
    num_generated_puzzles: int = 500
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    max_steps: int = 512
    device: str = "auto"
    learning_rate: float = 3e-4
    hidden_sizes: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    action_masking: bool = True
    benchmark_levels: Tuple[str, ...] = ("level1", "level2", "level3", "level4")
    filter_generated: bool = True
    eval_interval: int = 10
    eval_episodes: int = 20
    save_interval: int = 10
    comet_enabled: bool = False
    comet_project_name: str = "pushworld-ppo"
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: Tuple[str, ...] = ()
    comet_log_artifacts: bool = True


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    latest_checkpoint: str
    best_checkpoint: Optional[str]
    global_step: int
    best_success_rate: float


def train_ppo(config: PPOTrainingConfig) -> TrainingResult:
    """Trains PPO according to `config` and writes logs/checkpoints."""
    _validate_training_config(config)
    ensure_dir(config.output_dir)
    set_global_seeds(config.seed)
    device = resolve_device(config.device)

    curriculum = prepare_curriculum(
        puzzle_path=config.puzzle_path,
        output_dir=config.output_dir,
        num_generated_puzzles=config.num_generated_puzzles,
        seed=config.seed,
        benchmark_levels=config.benchmark_levels,
        filter_generated=config.filter_generated,
    )
    for message in curriculum.messages:
        print(message)

    all_puzzle_bank = load_puzzle_bank(curriculum.all_paths)
    observation_config = build_observation_config_from_puzzles(
        puzzle for _, puzzle in all_puzzle_bank
    )
    observation_config.save(os.path.join(config.output_dir, "obs_config.json"))
    write_json(os.path.join(config.output_dir, "config.json"), asdict(config))
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
    tracker.log_parameters(
        {
            "training": asdict(config),
            "observation": observation_config.to_dict(),
            "curriculum": {
                "num_puzzles": len(all_puzzle_bank),
                "phases": [
                    {
                        "start_fraction": phase.start_fraction,
                        "num_paths": len(phase.puzzle_paths),
                    }
                    for phase in curriculum.phases
                ],
            },
            "runtime": {"device": str(device)},
        }
    )

    model = ActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        hidden_sizes=config.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

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
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        device=device,
    )

    num_updates = max(
        1,
        math.ceil(config.total_timesteps / (config.num_envs * config.num_steps)),
    )
    global_step = 0
    best_success_rate = -1.0
    best_checkpoint = None
    current_paths = None
    active_puzzle_bank = []
    envs = []
    observations = None
    action_masks = None
    last_dones = torch.zeros((config.num_envs,), dtype=torch.bool, device=device)

    episode_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    train_log_path = os.path.join(config.output_dir, "train.csv")
    try:
        with CSVLogger(train_log_path, _train_log_fields()) as train_logger:
            for update in range(1, num_updates + 1):
                progress = global_step / max(1, config.total_timesteps)
                active_paths = curriculum.paths_for_progress(progress)
                if active_paths != current_paths:
                    current_paths = active_paths
                    active_puzzle_bank = _filter_puzzle_bank(
                        all_puzzle_bank, current_paths
                    )
                    envs, observations, action_masks = _make_envs(
                        puzzle_paths=current_paths,
                        puzzle_bank=active_puzzle_bank,
                        config=config,
                        observation_config=observation_config,
                        seed=config.seed + update * 10_000,
                    )
                    last_dones = torch.zeros(
                        (config.num_envs,), dtype=torch.bool, device=device
                    )
                    episode_returns.fill(0.0)
                    episode_lengths.fill(0)

                rollout.reset()
                for _ in range(config.num_steps):
                    observation_tensor = torch.as_tensor(
                        observations, dtype=torch.float32, device=device
                    )
                    action_mask_tensor = torch.as_tensor(
                        action_masks, dtype=torch.float32, device=device
                    )

                    with torch.no_grad():
                        actions, logprobs, _, values = model.get_action_and_value(
                            observation_tensor,
                            action_mask=action_mask_tensor
                            if config.action_masking
                            else None,
                        )

                    next_observations = []
                    next_action_masks = []
                    rewards = np.zeros((config.num_envs,), dtype=np.float32)
                    dones = np.zeros((config.num_envs,), dtype=bool)

                    for env_index, env in enumerate(envs):
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(int(actions[env_index].cpu().item()))
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
                            next_observation, info = env.reset()

                        next_observations.append(next_observation)
                        next_action_masks.append(info["action_mask"])

                    rollout.add(
                        observations=observation_tensor,
                        action_masks=action_mask_tensor,
                        actions=actions,
                        logprobs=logprobs,
                        rewards=torch.as_tensor(
                            rewards, dtype=torch.float32, device=device
                        ),
                        dones=torch.as_tensor(dones, dtype=torch.bool, device=device),
                        values=values,
                    )

                    observations = np.stack(next_observations)
                    action_masks = np.stack(next_action_masks)
                    last_dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                    global_step += config.num_envs

                with torch.no_grad():
                    last_values = model.get_value(
                        torch.as_tensor(observations, dtype=torch.float32, device=device)
                    )
                rollout.compute_returns_and_advantages(
                    last_values=last_values,
                    last_dones=last_dones,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                train_stats = update_policy(model, optimizer, rollout, update_config)

                eval_stats = None
                eval_success_rate = ""
                if config.eval_interval > 0 and (
                    update % config.eval_interval == 0 or update == num_updates
                ):
                    eval_stats, _ = evaluate_policy(
                        model=model,
                        puzzle_paths=curriculum.all_paths,
                        preloaded_puzzles=all_puzzle_bank,
                        observation_config=observation_config,
                        max_steps=config.max_steps,
                        device=device,
                        num_episodes=config.eval_episodes,
                        action_masking=config.action_masking,
                        seed=config.seed + update,
                    )
                    eval_success_rate = eval_stats["success_rate"]
                    if eval_success_rate > best_success_rate:
                        best_success_rate = eval_success_rate
                        best_checkpoint = os.path.join(
                            config.output_dir, "best_success.pt"
                        )
                        save_checkpoint(
                            best_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            training_config=config,
                            observation_config=observation_config,
                            global_step=global_step,
                            best_success_rate=best_success_rate,
                        )

                latest_checkpoint = os.path.join(config.output_dir, "latest.pt")
                if update % config.save_interval == 0 or update == num_updates:
                    save_checkpoint(
                        latest_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        training_config=config,
                        observation_config=observation_config,
                        global_step=global_step,
                        best_success_rate=best_success_rate,
                    )

                log_row = {
                    "update": update,
                    "global_step": global_step,
                    "active_path_count": len(active_puzzle_bank),
                    "mean_episode_return": _mean_or_empty(recent_returns),
                    "mean_episode_length": _mean_or_empty(recent_lengths),
                    "mean_success": _mean_or_empty(recent_successes),
                    "eval_success_rate": eval_success_rate,
                    **train_stats,
                }
                train_logger.write(log_row)
                tracker.log_metrics(log_row, step=global_step, prefix="train")
                if eval_stats is not None:
                    tracker.log_metrics(eval_stats, step=global_step, prefix="eval")
    finally:
        if config.comet_log_artifacts:
            _log_final_assets(
                tracker=tracker,
                output_dir=config.output_dir,
                latest_checkpoint=os.path.join(config.output_dir, "latest.pt"),
                best_checkpoint=best_checkpoint,
                train_log_path=train_log_path,
            )
        tracker.end()

    return TrainingResult(
        output_dir=config.output_dir,
        latest_checkpoint=os.path.join(config.output_dir, "latest.pt"),
        best_checkpoint=best_checkpoint,
        global_step=global_step,
        best_success_rate=max(0.0, best_success_rate),
    )


def evaluate_policy(
    model: ActorCriticMLP,
    puzzle_paths: Sequence[str],
    observation_config: VectorObservationConfig,
    max_steps: int,
    device: torch.device,
    num_episodes: int,
    action_masking: bool = True,
    seed: int = 0,
    preloaded_puzzles: Optional[Sequence[Tuple[str, PushWorldPuzzle]]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Runs greedy policy evaluation and returns aggregate stats and records."""
    was_training = model.training
    model.eval()
    env = PushWorldVectorEnv(
        puzzle_paths,
        max_steps=max_steps,
        observation_config=observation_config,
        preloaded_puzzles=preloaded_puzzles,
    )

    records = []
    for episode in range(num_episodes):
        observation, info = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(
                observation[None, :], dtype=torch.float32, device=device
            )
            action_mask = torch.as_tensor(
                info["action_mask"][None, :], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action = greedy_actions(
                    model,
                    observation_tensor,
                    action_mask if action_masking else None,
                )[0]
            observation, reward, terminated, truncated, info = env.step(
                int(action.cpu().item())
            )
            episode_return += reward
            episode_length += 1

        records.append(
            {
                "episode": episode,
                "puzzle_path": info["puzzle_path"],
                "return": episode_return,
                "length": episode_length,
                "success": 1.0 if terminated else 0.0,
                "truncated": 1.0 if truncated else 0.0,
            }
        )

    if was_training:
        model.train()

    success_rate = float(np.mean([record["success"] for record in records]))
    mean_return = float(np.mean([record["return"] for record in records]))
    mean_length = float(np.mean([record["length"] for record in records]))
    return (
        {
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_length": mean_length,
        },
        records,
    )


def greedy_actions(
    model: ActorCriticMLP,
    observations: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    logits, _ = model.get_logits_and_value(observations)
    if action_mask is not None:
        mask = action_mask.to(dtype=torch.bool, device=logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.argmax(logits, dim=-1)


def save_checkpoint(
    path: str,
    model: ActorCriticMLP,
    optimizer: Optional[torch.optim.Optimizer],
    training_config: PPOTrainingConfig,
    observation_config: VectorObservationConfig,
    global_step: int,
    best_success_rate: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None
            if optimizer is None
            else optimizer.state_dict(),
            "training_config": asdict(training_config),
            "observation_config": observation_config.to_dict(),
            "global_step": global_step,
            "best_success_rate": best_success_rate,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def _make_envs(
    puzzle_paths: Sequence[str],
    puzzle_bank: Sequence[Tuple[str, PushWorldPuzzle]],
    config: PPOTrainingConfig,
    observation_config: VectorObservationConfig,
    seed: int,
) -> Tuple[List[PushWorldVectorEnv], np.ndarray, np.ndarray]:
    envs = [
        PushWorldVectorEnv(
            puzzle_paths,
            max_steps=config.max_steps,
            observation_config=observation_config,
            preloaded_puzzles=puzzle_bank,
        )
        for _ in range(config.num_envs)
    ]
    observations = []
    action_masks = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=seed + index)
        observations.append(observation)
        action_masks.append(info["action_mask"])
    return envs, np.stack(observations), np.stack(action_masks)


def _filter_puzzle_bank(
    puzzle_bank: Sequence[Tuple[str, PushWorldPuzzle]],
    active_paths: Sequence[str],
) -> List[Tuple[str, PushWorldPuzzle]]:
    active_file_paths = {
        os.path.normpath(path) for path in resolve_puzzle_paths(active_paths)
    }
    return [
        (path, puzzle)
        for path, puzzle in puzzle_bank
        if os.path.normpath(path) in active_file_paths
    ]


def _log_final_assets(
    tracker,
    output_dir: str,
    latest_checkpoint: str,
    best_checkpoint: Optional[str],
    train_log_path: str,
) -> None:
    tracker.log_asset(os.path.join(output_dir, "config.json"), name="config.json")
    tracker.log_asset(
        os.path.join(output_dir, "obs_config.json"), name="obs_config.json"
    )
    tracker.log_asset(train_log_path, name="train.csv")
    tracker.log_asset(latest_checkpoint, name="latest.pt")
    if best_checkpoint is not None:
        tracker.log_asset(best_checkpoint, name="best_success.pt")


def _validate_training_config(config: PPOTrainingConfig) -> None:
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1")
    if config.num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if config.num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be >= 1")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be >= 1")
    if config.num_generated_puzzles < 0:
        raise ValueError("num_generated_puzzles must be >= 0")
    if config.save_interval < 1:
        raise ValueError("save_interval must be >= 1")
    if config.eval_interval > 0 and config.eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1 when evaluation is enabled")
    resolve_puzzle_paths(config.puzzle_path)


def _mean_or_empty(values: deque) -> object:
    if not values:
        return ""
    return float(np.mean(values))


def _train_log_fields() -> List[str]:
    return [
        "update",
        "global_step",
        "active_path_count",
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
