# Copyright 2026
#
# Hierarchical PPO over frozen DIAYN skills.

from collections import deque
from dataclasses import asdict, dataclass
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle
from pushworld.rl.architectures import ActorCriticMLP, SkillConditionedActorCriticMLP
from pushworld.rl.diayn import load_diayn_checkpoint
from pushworld.rl.envs import PushWorldVectorEnv, load_puzzle_bank
from pushworld.rl.observations import VectorObservationConfig
from pushworld.rl.ppo import PPOUpdateConfig, update_policy
from pushworld.rl.storage import RolloutBuffer
from pushworld.rl.tracking import CometTrackingConfig, create_comet_tracker
from pushworld.rl.utils import CSVLogger, ensure_dir, resolve_device, set_global_seeds, write_json


@dataclass(frozen=True)
class HierarchicalDIAYNConfig:
    pretrained_dir: str
    puzzle_path: str
    output_dir: str
    total_timesteps: int = 1_000_000
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 64
    max_steps: int = 512
    skill_horizon: int = 8
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
    deterministic_low_level: bool = False
    eval_interval: int = 10
    eval_episodes: int = 20
    save_interval: int = 10
    comet_enabled: bool = False
    comet_project_name: str = "pushworld-diayn"
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: Tuple[str, ...] = ()
    comet_log_artifacts: bool = True


class HierarchicalSkillEnv:
    """Meta-environment whose actions select frozen low-level skills."""

    def __init__(
        self,
        puzzle_bank: Sequence[Tuple[str, PushWorldPuzzle]],
        low_level_policy: SkillConditionedActorCriticMLP,
        observation_config: VectorObservationConfig,
        num_skills: int,
        max_steps: int,
        skill_horizon: int,
        device: torch.device,
        deterministic_low_level: bool = False,
    ) -> None:
        self.low_level_policy = low_level_policy
        self.num_skills = num_skills
        self.skill_horizon = skill_horizon
        self.device = device
        self.deterministic_low_level = deterministic_low_level
        self.env = PushWorldVectorEnv(
            [path for path, _ in puzzle_bank],
            max_steps=max_steps,
            observation_config=observation_config,
            preloaded_puzzles=puzzle_bank,
        )

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, skill_id: int):
        total_reward = 0.0
        primitive_steps = 0
        terminated = False
        truncated = False
        info = None
        observation = None
        for _ in range(self.skill_horizon):
            observation = self.env._observe()
            info = self.env._info()
            obs_tensor = torch.as_tensor(
                observation[None, :], dtype=torch.float32, device=self.device
            )
            skill_tensor = torch.as_tensor([skill_id], dtype=torch.long, device=self.device)
            mask_tensor = torch.as_tensor(
                info["action_mask"][None, :], dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                if self.deterministic_low_level:
                    logits, _ = self.low_level_policy.get_logits_and_value(
                        obs_tensor, skill_tensor
                    )
                    mask = mask_tensor.to(dtype=torch.bool)
                    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                    action = torch.argmax(logits, dim=-1)
                else:
                    action, _, _, _ = self.low_level_policy.get_action_and_value(
                        obs_tensor,
                        skill_ids=skill_tensor,
                        action_mask=mask_tensor,
                    )
            observation, reward, terminated, truncated, info = self.env.step(
                int(action[0].cpu().item())
            )
            total_reward += reward
            primitive_steps += 1
            if terminated or truncated:
                break
        info = dict(info or {})
        info["primitive_steps"] = primitive_steps
        info["skill_id"] = skill_id
        return observation, total_reward, terminated, truncated, info


def train_hierarchical_diayn_ppo(config: HierarchicalDIAYNConfig):
    """Trains a PPO meta-controller over frozen DIAYN skills."""
    _validate_config(config)
    ensure_dir(config.output_dir)
    set_global_seeds(config.seed)
    device = resolve_device(config.device)
    checkpoint = load_diayn_checkpoint(config.pretrained_dir, device)
    observation_config = VectorObservationConfig.from_dict(checkpoint["observation_config"])
    num_skills = int(checkpoint["skill_config"]["num_skills"])
    low_level_hidden_sizes = tuple(checkpoint["training_config"].get("hidden_sizes", (256, 256)))
    low_level_policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        num_skills=num_skills,
        hidden_sizes=low_level_hidden_sizes,
    ).to(device)
    low_level_policy.load_state_dict(checkpoint["model_state_dict"])
    low_level_policy.eval()
    for parameter in low_level_policy.parameters():
        parameter.requires_grad_(False)

    puzzle_bank = load_puzzle_bank(config.puzzle_path)
    meta_policy = ActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=num_skills,
        hidden_sizes=config.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(meta_policy.parameters(), lr=config.learning_rate, eps=1e-5)
    update_config = PPOUpdateConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_coef=config.clip_coef,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        action_masking=True,
    )
    rollout = RolloutBuffer(
        num_steps=config.num_steps,
        num_envs=config.num_envs,
        observation_dim=observation_config.observation_size,
        num_actions=num_skills,
        device=device,
    )
    envs = [
        HierarchicalSkillEnv(
            puzzle_bank=puzzle_bank,
            low_level_policy=low_level_policy,
            observation_config=observation_config,
            num_skills=num_skills,
            max_steps=config.max_steps,
            skill_horizon=config.skill_horizon,
            device=device,
            deterministic_low_level=config.deterministic_low_level,
        )
        for _ in range(config.num_envs)
    ]
    observations = []
    for index, env in enumerate(envs):
        observation, _ = env.reset(seed=config.seed + index)
        observations.append(observation)
    observations = np.stack(observations)
    action_masks = np.ones((config.num_envs, num_skills), dtype=np.float32)
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
    write_json(os.path.join(config.output_dir, "config.json"), asdict(config))
    write_json(os.path.join(config.output_dir, "skill_config.json"), {"num_skills": num_skills})
    observation_config.save(os.path.join(config.output_dir, "obs_config.json"))
    tracker.log_parameters({"hierarchical": asdict(config), "num_skills": num_skills})
    train_log_path = os.path.join(config.output_dir, "train.csv")

    num_updates = max(
        1,
        math.ceil(
            config.total_timesteps
            / (config.num_envs * config.num_steps * config.skill_horizon)
        ),
    )
    global_step = 0
    best_success_rate = -1.0
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    episode_returns = np.zeros((config.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((config.num_envs,), dtype=np.int32)

    try:
        with CSVLogger(train_log_path, _hierarchical_log_fields()) as logger:
            for update in range(1, num_updates + 1):
                rollout.reset()
                last_dones = torch.zeros(
                    (config.num_envs,), dtype=torch.bool, device=device
                )
                skill_switches = 0
                for _ in range(config.num_steps):
                    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
                    mask_tensor = torch.as_tensor(action_masks, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        skills, logprobs, _, values = meta_policy.get_action_and_value(
                            obs_tensor,
                            action_mask=mask_tensor,
                        )
                    next_obs = []
                    rewards = np.zeros((config.num_envs,), dtype=np.float32)
                    dones = np.zeros((config.num_envs,), dtype=bool)
                    primitive_steps_this_meta_step = 0
                    for env_index, env in enumerate(envs):
                        observation, reward, terminated, truncated, info = env.step(
                            int(skills[env_index].cpu().item())
                        )
                        done = bool(terminated or truncated)
                        rewards[env_index] = reward
                        dones[env_index] = done
                        skill_switches += 1
                        primitive_steps_this_meta_step += int(info["primitive_steps"])
                        episode_returns[env_index] += reward
                        episode_lengths[env_index] += int(info["primitive_steps"])
                        if done:
                            recent_returns.append(float(episode_returns[env_index]))
                            recent_lengths.append(int(episode_lengths[env_index]))
                            recent_successes.append(1.0 if terminated else 0.0)
                            episode_returns[env_index] = 0.0
                            episode_lengths[env_index] = 0
                            observation, _ = env.reset()
                        next_obs.append(observation)
                    rollout.add(
                        observations=obs_tensor,
                        action_masks=mask_tensor,
                        actions=skills,
                        logprobs=logprobs,
                        rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                        dones=torch.as_tensor(dones, dtype=torch.bool, device=device),
                        values=values,
                    )
                    observations = np.stack(next_obs)
                    last_dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                    global_step += primitive_steps_this_meta_step

                with torch.no_grad():
                    last_values = meta_policy.get_value(
                        torch.as_tensor(observations, dtype=torch.float32, device=device)
                    )
                rollout.compute_returns_and_advantages(
                    last_values, last_dones, config.gamma, config.gae_lambda
                )
                ppo_stats = update_policy(meta_policy, optimizer, rollout, update_config)
                eval_success_rate = ""
                if config.eval_interval > 0 and (
                    update % config.eval_interval == 0 or update == num_updates
                ):
                    eval_stats = evaluate_hierarchical_policy(
                        meta_policy=meta_policy,
                        low_level_policy=low_level_policy,
                        puzzle_bank=puzzle_bank,
                        observation_config=observation_config,
                        num_skills=num_skills,
                        max_steps=config.max_steps,
                        skill_horizon=config.skill_horizon,
                        device=device,
                        episodes=config.eval_episodes,
                        deterministic_low_level=config.deterministic_low_level,
                    )
                    eval_success_rate = eval_stats["success_rate"]
                    tracker.log_metrics(eval_stats, step=global_step, prefix="hier_eval")
                    if eval_success_rate > best_success_rate:
                        best_success_rate = eval_success_rate
                        _save_meta_checkpoint(
                            os.path.join(config.output_dir, "best_success.pt"),
                            meta_policy,
                            optimizer,
                            config,
                            observation_config,
                            num_skills,
                            global_step,
                            best_success_rate,
                        )
                if update % config.save_interval == 0 or update == num_updates:
                    _save_meta_checkpoint(
                        os.path.join(config.output_dir, "latest.pt"),
                        meta_policy,
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
                    "skill_switches": skill_switches,
                    **ppo_stats,
                }
                logger.write(log_row)
                tracker.log_metrics(log_row, step=global_step, prefix="hier")
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


def evaluate_hierarchical_policy(
    meta_policy: ActorCriticMLP,
    low_level_policy: SkillConditionedActorCriticMLP,
    puzzle_bank: Sequence[Tuple[str, PushWorldPuzzle]],
    observation_config: VectorObservationConfig,
    num_skills: int,
    max_steps: int,
    skill_horizon: int,
    device: torch.device,
    episodes: int,
    deterministic_low_level: bool = False,
):
    records = []
    was_training = meta_policy.training
    meta_policy.eval()
    for episode in range(episodes):
        env = HierarchicalSkillEnv(
            puzzle_bank=puzzle_bank,
            low_level_policy=low_level_policy,
            observation_config=observation_config,
            num_skills=num_skills,
            max_steps=max_steps,
            skill_horizon=skill_horizon,
            device=device,
            deterministic_low_level=deterministic_low_level,
        )
        observation, _ = env.reset(seed=episode)
        total_reward = 0.0
        length = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(
                observation[None, :], dtype=torch.float32, device=device
            )
            mask_tensor = torch.ones((1, num_skills), dtype=torch.float32, device=device)
            with torch.no_grad():
                skill, _, _, _ = meta_policy.get_action_and_value(
                    obs_tensor, action_mask=mask_tensor
                )
            observation, reward, terminated, truncated, info = env.step(
                int(skill[0].cpu().item())
            )
            total_reward += reward
            length += int(info["primitive_steps"])
        records.append(
            {
                "return": total_reward,
                "length": length,
                "success": 1.0 if terminated else 0.0,
                "truncated": 1.0 if truncated else 0.0,
            }
        )
    if was_training:
        meta_policy.train()
    return {
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["length"] for record in records])),
    }


def _save_meta_checkpoint(
    path: str,
    meta_policy: ActorCriticMLP,
    optimizer: torch.optim.Optimizer,
    config: HierarchicalDIAYNConfig,
    observation_config: VectorObservationConfig,
    num_skills: int,
    global_step: int,
    best_success_rate: float,
) -> None:
    torch.save(
        {
            "model_state_dict": meta_policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(config),
            "observation_config": observation_config.to_dict(),
            "skill_config": {"num_skills": num_skills},
            "global_step": global_step,
            "best_success_rate": best_success_rate,
        },
        path,
    )


def _validate_config(config: HierarchicalDIAYNConfig) -> None:
    if config.skill_horizon < 1:
        raise ValueError("skill_horizon must be >= 1")
    if config.num_envs < 1 or config.num_steps < 1:
        raise ValueError("num_envs and num_steps must be >= 1")
    if config.total_timesteps < 1:
        raise ValueError("total_timesteps must be >= 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")


def _mean_or_empty(values: deque) -> object:
    if not values:
        return ""
    return float(np.mean(values))


def _hierarchical_log_fields() -> List[str]:
    return [
        "update",
        "global_step",
        "mean_episode_return",
        "mean_episode_length",
        "mean_success",
        "eval_success_rate",
        "skill_switches",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "num_minibatches",
    ]
