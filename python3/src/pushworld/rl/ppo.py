# Copyright 2026
#
# PPO update logic.

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from pushworld.rl.architectures import ActorCriticMLP
from pushworld.rl.storage import RolloutBuffer


@dataclass(frozen=True)
class PPOUpdateConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    action_masking: bool = True


def update_policy(
    model: ActorCriticMLP,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBuffer,
    config: PPOUpdateConfig,
) -> Dict[str, float]:
    """Runs PPO epochs over one rollout and returns scalar training stats."""
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "num_minibatches": 0.0,
    }

    for _ in range(config.update_epochs):
        for (
            observations,
            action_masks,
            actions,
            old_logprobs,
            advantages,
            returns,
            old_values,
        ) in rollout.get_minibatches(config.minibatch_size):
            del old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

            masks = action_masks if config.action_masking else None
            _, new_logprobs, entropy, new_values = model.get_action_and_value(
                observations,
                action_mask=masks,
                actions=actions,
            )

            log_ratio = new_logprobs - old_logprobs
            ratio = log_ratio.exp()
            policy_loss_unclipped = -advantages * ratio
            policy_loss_clipped = -advantages * torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            )
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
            value_loss = F.mse_loss(new_values, returns)
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                - config.entropy_coef * entropy_loss
                + config.value_coef * value_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    ((ratio - 1.0).abs() > config.clip_coef).float().mean()
                )

            stats["policy_loss"] += float(policy_loss.detach().cpu())
            stats["value_loss"] += float(value_loss.detach().cpu())
            stats["entropy"] += float(entropy_loss.detach().cpu())
            stats["approx_kl"] += float(approx_kl.detach().cpu())
            stats["clip_fraction"] += float(clip_fraction.detach().cpu())
            stats["num_minibatches"] += 1.0

    denominator = max(1.0, stats["num_minibatches"])
    for key in list(stats.keys()):
        if key != "num_minibatches":
            stats[key] /= denominator
    return stats
