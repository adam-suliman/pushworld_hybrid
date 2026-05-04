# Copyright 2026
#
# Rollout storage for PPO.

from typing import Iterator, Tuple

import torch


class RolloutBuffer:
    """Fixed-length rollout buffer with GAE-lambda support."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_dim: int,
        num_actions: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.observations = torch.zeros(
            (num_steps, num_envs, observation_dim), dtype=torch.float32, device=device
        )
        self.action_masks = torch.zeros(
            (num_steps, num_envs, num_actions), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.step = 0

    def add(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.observations[self.step].copy_(observations)
        self.action_masks[self.step].copy_(action_masks)
        self.actions[self.step].copy_(actions)
        self.logprobs[self.step].copy_(logprobs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae_lam = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def reset(self) -> None:
        self.step = 0

    def get_minibatches(
        self,
        minibatch_size: int,
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        batch_size = self.num_steps * self.num_envs
        minibatch_size = min(minibatch_size, batch_size)
        indices = torch.randperm(batch_size, device=self.device)

        flat_observations = self.observations.reshape(batch_size, -1)
        flat_action_masks = self.action_masks.reshape(batch_size, -1)
        flat_actions = self.actions.reshape(batch_size)
        flat_logprobs = self.logprobs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        for start in range(0, batch_size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            yield (
                flat_observations[batch_indices],
                flat_action_masks[batch_indices],
                flat_actions[batch_indices],
                flat_logprobs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
                flat_values[batch_indices],
            )
