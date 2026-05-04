import argparse
import os

from pushworld.rl.diayn import DIAYNTrainingConfig, train_diayn_ppo
from pushworld.rl.tracking import parse_tags


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUZZLE_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "benchmark", "puzzles", "level0", "all", "train")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "runs", "diayn_ppo"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain DIAYN-style skills with PPO.")
    parser.add_argument("--puzzle_path", default=DEFAULT_PUZZLE_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--hidden_sizes", default="256,256")
    parser.add_argument("--discriminator_hidden_sizes", default="256,256")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--action_masking", type=_parse_bool, default=True)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--num_skills", type=int, default=8)
    parser.add_argument("--diayn_reward_scale", type=float, default=1.0)
    parser.add_argument("--object_change_reward_scale", type=float, default=0.5)
    parser.add_argument("--object_change_reward_clip", type=float, default=1.0)
    parser.add_argument("--object_novelty_reward_scale", type=float, default=1.0)
    parser.add_argument("--object_novelty_reward_clip", type=float, default=2.0)
    parser.add_argument(
        "--object_novelty_requires_nonnegative_goal_progress",
        type=_parse_bool,
        default=True,
    )
    parser.add_argument("--goal_progress_reward_scale", type=float, default=0.25)
    parser.add_argument("--negative_goal_progress_penalty_scale", type=float, default=1.0)
    parser.add_argument("--discriminator_lr", type=float, default=3e-4)
    parser.add_argument("--discriminator_update_epochs", type=int, default=4)
    parser.add_argument("--discriminator_minibatch_size", type=int, default=256)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--comet_enabled", type=_parse_bool, default=False)
    parser.add_argument("--comet_project_name", default="pushworld-diayn")
    parser.add_argument("--comet_workspace", default="")
    parser.add_argument("--comet_experiment_name", default="")
    parser.add_argument("--comet_tags", default="")
    parser.add_argument("--comet_log_artifacts", type=_parse_bool, default=True)
    args = parser.parse_args()

    result = train_diayn_ppo(
        DIAYNTrainingConfig(
            puzzle_path=args.puzzle_path,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            max_steps=args.max_steps,
            device=args.device,
            learning_rate=args.learning_rate,
            hidden_sizes=_parse_int_tuple(args.hidden_sizes),
            discriminator_hidden_sizes=_parse_int_tuple(args.discriminator_hidden_sizes),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            action_masking=args.action_masking,
            save_interval=args.save_interval,
            num_skills=args.num_skills,
            diayn_reward_scale=args.diayn_reward_scale,
            object_change_reward_scale=args.object_change_reward_scale,
            object_change_reward_clip=args.object_change_reward_clip,
            object_novelty_reward_scale=args.object_novelty_reward_scale,
            object_novelty_reward_clip=args.object_novelty_reward_clip,
            object_novelty_requires_nonnegative_goal_progress=(
                args.object_novelty_requires_nonnegative_goal_progress
            ),
            goal_progress_reward_scale=args.goal_progress_reward_scale,
            negative_goal_progress_penalty_scale=(
                args.negative_goal_progress_penalty_scale
            ),
            discriminator_lr=args.discriminator_lr,
            discriminator_update_epochs=args.discriminator_update_epochs,
            discriminator_minibatch_size=args.discriminator_minibatch_size,
            entropy_coef=args.entropy_coef,
            comet_enabled=args.comet_enabled,
            comet_project_name=args.comet_project_name,
            comet_workspace=args.comet_workspace or None,
            comet_experiment_name=args.comet_experiment_name or None,
            comet_tags=parse_tags(args.comet_tags),
            comet_log_artifacts=args.comet_log_artifacts,
        )
    )
    print(f"policy_checkpoint={result.policy_checkpoint}")
    print(f"discriminator_checkpoint={result.discriminator_checkpoint}")
    print(f"global_step={result.global_step}")


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_int_tuple(value: str) -> tuple:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
