import argparse
import os

from pushworld.puzzle import NUM_ACTIONS
from pushworld.rl.architectures import SkillConditionedActorCriticMLP
from pushworld.rl.diayn import (
    evaluate_skill_conditioned_policy,
    load_diayn_checkpoint,
    summarize_best_of_skill_records,
)
from pushworld.rl.observations import VectorObservationConfig
from pushworld.rl.tracking import CometTrackingConfig, create_comet_tracker, parse_tags
from pushworld.rl.utils import CSVLogger, ensure_dir, resolve_device, write_json


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "runs", "diayn_ppo"))
DEFAULT_PUZZLE_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "benchmark", "puzzles", "level0", "all", "test")
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every DIAYN skill on puzzles.")
    parser.add_argument("--pretrained_dir", default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--puzzle_path", default=DEFAULT_PUZZLE_PATH)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--selection_json", default="")
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument(
        "--max_puzzles",
        type=int,
        default=0,
        help="Maximum number of puzzles to sample for evaluation. Use 0 for all.",
    )
    parser.add_argument(
        "--skill_ids",
        default="",
        help="Comma-separated skill ids to evaluate. Empty means all skills.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rollouts_per_skill",
        type=int,
        default=1,
        help="Number of rollouts per puzzle-skill pair for best-of-N stochastic evaluation.",
    )
    parser.add_argument("--action_masking", type=_parse_bool, default=True)
    parser.add_argument("--deterministic", type=_parse_bool, default=True)
    parser.add_argument("--comet_enabled", type=_parse_bool, default=False)
    parser.add_argument("--comet_project_name", default="pushworld-diayn")
    parser.add_argument("--comet_workspace", default="")
    parser.add_argument("--comet_experiment_name", default="")
    parser.add_argument("--comet_tags", default="")
    parser.add_argument("--comet_log_artifacts", type=_parse_bool, default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_diayn_checkpoint(args.pretrained_dir, device)
    observation_config = VectorObservationConfig.from_dict(checkpoint["observation_config"])
    num_skills = int(checkpoint["skill_config"]["num_skills"])
    hidden_sizes = tuple(checkpoint["training_config"].get("hidden_sizes", (256, 256)))
    policy = SkillConditionedActorCriticMLP(
        observation_dim=observation_config.observation_size,
        num_actions=NUM_ACTIONS,
        num_skills=num_skills,
        hidden_sizes=hidden_sizes,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])

    default_output_dir = (
        args.pretrained_dir
        if os.path.isdir(args.pretrained_dir)
        else os.path.dirname(os.path.abspath(args.pretrained_dir))
    )
    output_csv = args.output_csv or os.path.join(default_output_dir, "skill_eval.csv")
    summary_csv = args.summary_csv or _default_path(output_csv, "_best.csv")
    selection_json = args.selection_json or _default_path(output_csv, "_selection.json")
    ensure_dir(os.path.dirname(os.path.abspath(output_csv)))
    ensure_dir(os.path.dirname(os.path.abspath(summary_csv)))
    ensure_dir(os.path.dirname(os.path.abspath(selection_json)))
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
    try:
        stats, records = evaluate_skill_conditioned_policy(
            policy=policy,
            puzzle_paths=args.puzzle_path,
            observation_config=observation_config,
            num_skills=num_skills,
            max_steps=args.max_steps,
            device=device,
            action_masking=args.action_masking,
            deterministic=args.deterministic,
            max_puzzles=args.max_puzzles if args.max_puzzles > 0 else None,
            skill_ids=_parse_int_list(args.skill_ids) or None,
            seed=args.seed,
            rollouts_per_skill=args.rollouts_per_skill,
        )
        fieldnames = _record_fieldnames()
        with CSVLogger(output_csv, fieldnames) as logger:
            for record in records:
                logger.write(record)
        best_records = summarize_best_of_skill_records(records)
        with CSVLogger(summary_csv, fieldnames) as logger:
            for record in best_records:
                logger.write(record)
        selection = _selection_payload(
            args=args,
            output_csv=output_csv,
            summary_csv=summary_csv,
            selection_json=selection_json,
            stats=stats,
            num_skills=num_skills,
        )
        write_json(selection_json, selection)
        tracker.log_parameters(
            {
                "evaluation": {
                    "pretrained_dir": os.path.abspath(args.pretrained_dir),
                    "puzzle_path": os.path.abspath(args.puzzle_path),
                    "max_steps": args.max_steps,
                    "max_puzzles": args.max_puzzles,
                    "skill_ids": args.skill_ids,
                    "deterministic": args.deterministic,
                    "rollouts_per_skill": args.rollouts_per_skill,
                    "seed": args.seed,
                    "output_csv": os.path.abspath(output_csv),
                    "summary_csv": os.path.abspath(summary_csv),
                    "selection_json": os.path.abspath(selection_json),
                },
                "num_skills": num_skills,
            }
        )
        tracker.log_metrics(stats, step=0, prefix="skill_eval")
        if args.comet_log_artifacts:
            tracker.log_asset(output_csv, name=os.path.basename(output_csv))
            tracker.log_asset(summary_csv, name=os.path.basename(summary_csv))
            tracker.log_asset(selection_json, name=os.path.basename(selection_json))
        for key, value in stats.items():
            print(f"{key}={value}")
        print(f"records={len(records)}")
        print(f"output_csv={output_csv}")
        print(f"summary_csv={summary_csv}")
        print(f"selection_json={selection_json}")
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


def _parse_int_list(value: str) -> list:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _record_fieldnames() -> list:
    return [
        "puzzle_path",
        "skill_id",
        "rollout_id",
        "return",
        "length",
        "success",
        "truncated",
        "object_displacement",
        "object_push_count",
        "object_contact_count",
        "novel_object_positions",
        "object_novelty_count",
        "goal_progress",
    ]


def _default_path(path: str, suffix: str) -> str:
    stem, extension = os.path.splitext(path)
    if not extension:
        return path + suffix
    return stem + suffix


def _selection_payload(
    args,
    output_csv: str,
    summary_csv: str,
    selection_json: str,
    stats: dict,
    num_skills: int,
) -> dict:
    checkpoint_path = args.pretrained_dir
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "diayn_policy.pt")
    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "pretrained_dir": os.path.abspath(args.pretrained_dir),
        "puzzle_path": os.path.abspath(args.puzzle_path),
        "output_csv": os.path.abspath(output_csv),
        "summary_csv": os.path.abspath(summary_csv),
        "selection_json": os.path.abspath(selection_json),
        "num_skills": int(num_skills),
        "rollouts_per_skill": int(args.rollouts_per_skill),
        "deterministic": bool(args.deterministic),
        "max_steps": int(args.max_steps),
        "max_puzzles": int(args.max_puzzles),
        "seed": int(args.seed),
        "best_skill_id": int(
            stats.get("best_of_n_best_skill_id", stats.get("best_skill_id", 0))
        ),
        "best_success_rate": float(
            stats.get("best_of_n_best_success_rate", stats.get("best_success_rate", 0.0))
        ),
        "puzzle_coverage_rate": float(stats.get("best_of_n_puzzle_coverage_rate", 0.0)),
        "solved_puzzle_count": int(stats.get("best_of_n_solved_puzzle_count", 0)),
        "stats": stats,
    }


if __name__ == "__main__":
    main()
