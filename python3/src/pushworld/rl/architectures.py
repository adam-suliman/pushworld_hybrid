# Copyright 2026
#
# PyTorch policy/value architectures for PushWorld PPO.

from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCriticMLP(nn.Module):
    """Shared-trunk actor-critic MLP for vector PushWorld observations."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers = []
        input_dim = observation_dim
        for hidden_size in hidden_sizes:
            layers.append(_layer_init(nn.Linear(input_dim, hidden_size)))
            layers.append(nn.Tanh())
            input_dim = hidden_size

        self.trunk = nn.Sequential(*layers)
        self.policy_head = _layer_init(nn.Linear(input_dim, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(input_dim, 1), std=1.0)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        _, values = self.get_logits_and_value(observations)
        return values

    def get_logits_and_value(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.get_logits_and_value(observations)
        distribution = _masked_categorical(logits, action_mask)

        if actions is None:
            actions = distribution.sample()

        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
        )


class SkillConditionedActorCriticMLP(nn.Module):
    """Actor-critic MLP conditioned on a discrete skill id."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        num_skills: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if num_skills < 1:
            raise ValueError("num_skills must be >= 1")
        self.observation_dim = observation_dim
        self.num_skills = num_skills
        self.conditioned_observation_dim = observation_dim + num_skills
        self.actor_critic = ActorCriticMLP(
            observation_dim=self.conditioned_observation_dim,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
        )

    def condition_observations(
        self,
        observations: torch.Tensor,
        skill_ids: torch.Tensor,
    ) -> torch.Tensor:
        return append_skill_one_hot(observations, skill_ids, self.num_skills)

    def get_value(
        self,
        observations: torch.Tensor,
        skill_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.actor_critic.get_value(
            self._condition_if_needed(observations, skill_ids)
        )

    def get_logits_and_value(
        self,
        observations: torch.Tensor,
        skill_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_critic.get_logits_and_value(
            self._condition_if_needed(observations, skill_ids)
        )

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        skill_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor_critic.get_action_and_value(
            self._condition_if_needed(observations, skill_ids),
            action_mask=action_mask,
            actions=actions,
        )

    def _condition_if_needed(
        self,
        observations: torch.Tensor,
        skill_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if observations.shape[-1] == self.conditioned_observation_dim:
            return observations
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "observations must have either the base observation dimension or "
                "the skill-conditioned observation dimension."
            )
        if skill_ids is None:
            raise ValueError("skill_ids are required for unconditioned observations.")
        return self.condition_observations(observations, skill_ids)


class SkillDiscriminatorMLP(nn.Module):
    """Predicts the DIAYN skill id from a compact state summary."""

    def __init__(
        self,
        input_dim: int,
        num_skills: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(_layer_init(nn.Linear(current_dim, hidden_size)))
            layers.append(nn.Tanh())
            current_dim = hidden_size
        layers.append(_layer_init(nn.Linear(current_dim, num_skills), std=0.01))
        self.network = nn.Sequential(*layers)

    def forward(self, discriminator_states: torch.Tensor) -> torch.Tensor:
        return self.network(discriminator_states)


def append_skill_one_hot(
    observations: torch.Tensor,
    skill_ids: torch.Tensor,
    num_skills: int,
) -> torch.Tensor:
    """Appends one-hot skill ids to a batch of observations."""
    if skill_ids.ndim == 0:
        skill_ids = skill_ids.unsqueeze(0)
    skill_ids = skill_ids.to(device=observations.device, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(skill_ids, num_classes=num_skills).to(
        dtype=observations.dtype,
        device=observations.device,
    )
    if observations.ndim == 1:
        observations = observations.unsqueeze(0)
    return torch.cat([observations, one_hot], dim=-1)


def _masked_categorical(
    logits: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> Categorical:
    if action_mask is None:
        return Categorical(logits=logits)

    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    fallback_mask = torch.ones_like(mask)
    has_valid_action = mask.any(dim=-1, keepdim=True)
    mask = torch.where(has_valid_action, mask, fallback_mask)

    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


def _layer_init(layer: nn.Linear, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
