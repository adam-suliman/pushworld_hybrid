import argparse
import os

from pushworld.puzzle import NUM_ACTIONS
from pushworld.rl.architectures import ActorCriticMLP
from pushworld.rl.observations import VectorObservationConfig, resolve_puzzle_paths
from pushworld.rl.tracking import CometTrackingConfig, create_comet_tracker, parse_tags
from pushworld.rl.training import evaluate_policy, load_checkpoint
from pushworld.rl.utils import CSVLogger, ensure_dir, resolve_device


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUZZLE_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "benchmark", "puzzles")
)
DEFAULT_CHECKPOINT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "runs", "ppo", "latest.pt")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PushWorld PPO checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--puzzle_path", default=DEFAULT_PUZZLE_PATH)
    parser.add_argument("--obs_config", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--episodes_per_puzzle", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--action_masking", type=_parse_bool, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comet_enabled", type=_parse_bool, default=False)
    parser.add_argument("--comet_project_name", default="pushworld-ppo")
    parser.add_argument("--comet_workspace", default="")
    parser.add_argument("--comet_experiment_name", default="")
    parser.add_argument("--comet_tags", default="")
    parser.add_argument("--comet_log_artifacts", type=_parse_bool, default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    obs_config_path = args.obs_config or os.path.join(checkpoint_dir, "obs_config.json")
    if os.path.exists(obs_config_path):
        observation_config = VectorObservationConfig.load(obs_config_path)
    else:
        observation_config = VectorObservationConfig.from_dict(
            checkpoint["observation_config"]
        )

    training_config = checkpoint.get("training_config", {})
    hidden_sizes = tuple(training_config.get("hidden_sizes", (256, 256)))
    model = ActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        hidden_sizes=hidden_sizes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_csv = args.output_csv or os.path.join(checkpoint_dir, "evaluation.csv")
    ensure_dir(os.path.dirname(os.path.abspath(output_csv)))
    tracker = create_comet_tracker(
        CometTrackingConfig(
            enabled=args.comet_enabled,
            project_name=args.comet_project_name,
            workspace=args.comet_workspace or None,
            experiment_name=args.comet_experiment_name or None,
            tags=parse_tags(args.comet_tags),
            log_artifacts=args.comet_log_artifacts,
        )
    )
    tracker.log_parameters(
        {
            "evaluation": {
                "checkpoint": os.path.abspath(args.checkpoint),
                "puzzle_path": os.path.abspath(args.puzzle_path),
                "episodes_per_puzzle": args.episodes_per_puzzle,
                "max_steps": args.max_steps,
                "action_masking": args.action_masking,
                "seed": args.seed,
                "device": str(device),
            },
            "observation": observation_config.to_dict(),
        }
    )

    try:
        all_records = []
        for puzzle_index, puzzle_path in enumerate(resolve_puzzle_paths(args.puzzle_path)):
            _, records = evaluate_policy(
                model=model,
                puzzle_paths=[puzzle_path],
                observation_config=observation_config,
                max_steps=args.max_steps,
                device=device,
                num_episodes=args.episodes_per_puzzle,
                action_masking=args.action_masking,
                seed=args.seed + puzzle_index * args.episodes_per_puzzle,
            )
            all_records.extend(records)

        with CSVLogger(
            output_csv,
            ["puzzle_path", "episode", "return", "length", "success", "truncated"],
        ) as logger:
            for record in all_records:
                logger.write(record)

        success_rate = sum(record["success"] for record in all_records) / max(
            1, len(all_records)
        )
        mean_return = sum(record["return"] for record in all_records) / max(
            1, len(all_records)
        )
        mean_length = sum(record["length"] for record in all_records) / max(
            1, len(all_records)
        )
        summary = {
            "records": len(all_records),
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_length": mean_length,
        }
        tracker.log_metrics(summary, step=0, prefix="test")
        if args.comet_log_artifacts:
            tracker.log_asset(output_csv, name=os.path.basename(output_csv))

        print(f"records={len(all_records)}")
        print(f"success_rate={success_rate}")
        print(f"mean_return={mean_return}")
        print(f"mean_length={mean_length}")
        print(f"output_csv={output_csv}")
    finally:
        tracker.end()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    main()
