# Copyright 2026
#
# Reinforcement learning utilities for PushWorld.

from pushworld.rl.observations import (
    VectorObservationConfig,
    build_observation_config,
    compute_action_mask,
    encode_puzzle_state,
)

__all__ = [
    "DIAYNConfig",
    "DIAYNTrainingConfig",
    "HierarchicalDIAYNConfig",
    "PushWorldVectorEnv",
    "SkillConditionedActorCriticMLP",
    "SkillDiscriminatorMLP",
    "SkillDiscriminatorObservationConfig",
    "VectorObservationConfig",
    "build_observation_config",
    "compute_diayn_reward",
    "compute_object_change_reward",
    "compute_goal_progress_penalty",
    "compute_object_motion_metrics",
    "compute_object_novelty_metrics",
    "compute_object_novelty_reward",
    "compute_action_mask",
    "encode_puzzle_state",
    "encode_discriminator_state",
    "finetune_diayn_ppo",
    "summarize_best_of_skill_records",
    "train_diayn_ppo",
    "train_hierarchical_diayn_ppo",
]


def __getattr__(name):
    if name == "PushWorldVectorEnv":
        from pushworld.rl.envs import PushWorldVectorEnv

        return PushWorldVectorEnv
    if name in {"SkillConditionedActorCriticMLP", "SkillDiscriminatorMLP"}:
        from pushworld.rl.architectures import (
            SkillConditionedActorCriticMLP,
            SkillDiscriminatorMLP,
        )

        return {
            "SkillConditionedActorCriticMLP": SkillConditionedActorCriticMLP,
            "SkillDiscriminatorMLP": SkillDiscriminatorMLP,
        }[name]
    if name in {
        "DIAYNConfig",
        "DIAYNTrainingConfig",
        "compute_diayn_reward",
        "compute_object_change_reward",
        "compute_goal_progress_penalty",
        "compute_object_motion_metrics",
        "compute_object_novelty_metrics",
        "compute_object_novelty_reward",
        "finetune_diayn_ppo",
        "summarize_best_of_skill_records",
        "train_diayn_ppo",
    }:
        from pushworld.rl.diayn import (
            DIAYNConfig,
            DIAYNTrainingConfig,
            compute_diayn_reward,
            compute_object_change_reward,
            compute_goal_progress_penalty,
            compute_object_motion_metrics,
            compute_object_novelty_metrics,
            compute_object_novelty_reward,
            finetune_diayn_ppo,
            summarize_best_of_skill_records,
            train_diayn_ppo,
        )

        return {
            "DIAYNConfig": DIAYNConfig,
            "DIAYNTrainingConfig": DIAYNTrainingConfig,
            "compute_diayn_reward": compute_diayn_reward,
            "compute_object_change_reward": compute_object_change_reward,
            "compute_goal_progress_penalty": compute_goal_progress_penalty,
            "compute_object_motion_metrics": compute_object_motion_metrics,
            "compute_object_novelty_metrics": compute_object_novelty_metrics,
            "compute_object_novelty_reward": compute_object_novelty_reward,
            "finetune_diayn_ppo": finetune_diayn_ppo,
            "summarize_best_of_skill_records": summarize_best_of_skill_records,
            "train_diayn_ppo": train_diayn_ppo,
        }[name]
    if name in {"HierarchicalDIAYNConfig", "train_hierarchical_diayn_ppo"}:
        from pushworld.rl.hierarchical import (
            HierarchicalDIAYNConfig,
            train_hierarchical_diayn_ppo,
        )

        return {
            "HierarchicalDIAYNConfig": HierarchicalDIAYNConfig,
            "train_hierarchical_diayn_ppo": train_hierarchical_diayn_ppo,
        }[name]
    if name in {"SkillDiscriminatorObservationConfig", "encode_discriminator_state"}:
        from pushworld.rl.skill_observations import (
            SkillDiscriminatorObservationConfig,
            encode_discriminator_state,
        )

        return {
            "SkillDiscriminatorObservationConfig": SkillDiscriminatorObservationConfig,
            "encode_discriminator_state": encode_discriminator_state,
        }[name]
    raise AttributeError(name)
