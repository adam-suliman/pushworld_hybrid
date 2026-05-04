# Copyright 2026
#
# Puzzle curriculum helpers for PPO training.

from dataclasses import dataclass
import os
from typing import List, Sequence, Tuple

from pushworld.config import RGD_PLANNER_PATH
from pushworld.rl.observations import PathLike, resolve_puzzle_paths


@dataclass(frozen=True)
class CurriculumPhase:
    start_fraction: float
    puzzle_paths: Tuple[str, ...]


@dataclass(frozen=True)
class Curriculum:
    phases: Tuple[CurriculumPhase, ...]
    messages: Tuple[str, ...]

    @property
    def all_paths(self) -> Tuple[str, ...]:
        paths = []
        for phase in self.phases:
            paths.extend(phase.puzzle_paths)
        return tuple(_dedupe_paths(paths))

    def paths_for_progress(self, progress: float) -> Tuple[str, ...]:
        active = self.phases[0]
        for phase in self.phases:
            if progress >= phase.start_fraction:
                active = phase
        return active.puzzle_paths


def prepare_curriculum(
    puzzle_path: PathLike,
    output_dir: PathLike,
    num_generated_puzzles: int = 500,
    seed: int = 0,
    benchmark_levels: Sequence[str] = ("level1", "level2", "level3", "level4"),
    filter_generated: bool = True,
) -> Curriculum:
    """Creates generated puzzles when requested and returns curriculum phases."""
    output_dir = os.fspath(output_dir)
    messages = []

    generated_path = None
    if num_generated_puzzles > 0:
        generated_path = os.path.join(output_dir, "generated_level0")
        existing = []
        if os.path.isdir(generated_path):
            try:
                existing = resolve_puzzle_paths(generated_path)
            except ValueError:
                existing = []

        if existing:
            messages.append(
                f"Using {len(existing)} existing generated puzzles from {generated_path}."
            )
        else:
            rgd_available = os.path.exists(RGD_PLANNER_PATH)
            should_filter = filter_generated and rgd_available
            if filter_generated and not rgd_available:
                messages.append(
                    "RGD planner binary was not found; generated puzzles will not be "
                    "filtered for solvability."
                )
            from pushworld.generate import generate_level0_puzzles

            generate_level0_puzzles(
                save_location_path=generated_path,
                num_puzzles=num_generated_puzzles,
                random_seed=seed,
                filter_puzzles=should_filter,
            )
            messages.append(
                f"Generated level-0 puzzles in {generated_path} "
                f"(solvability_filter={should_filter})."
            )

        try:
            generated_count = len(resolve_puzzle_paths(generated_path))
        except ValueError:
            generated_count = 0
        if generated_count == 0:
            messages.append(
                "No generated level-0 puzzles are available; starting from "
                "benchmark puzzles instead."
            )
            generated_path = None

    benchmark_level_paths = _benchmark_level_paths(puzzle_path, benchmark_levels)
    if not benchmark_level_paths and generated_path is None:
        raise ValueError("No generated or benchmark puzzles are available for training.")

    generated_paths = [generated_path] if generated_path is not None else []
    early_paths = generated_paths or benchmark_level_paths
    mid_paths = generated_paths + benchmark_level_paths[:2]
    late_paths = generated_paths + benchmark_level_paths

    phases = (
        CurriculumPhase(0.0, tuple(_dedupe_paths(early_paths))),
        CurriculumPhase(0.33, tuple(_dedupe_paths(mid_paths or early_paths))),
        CurriculumPhase(0.66, tuple(_dedupe_paths(late_paths or early_paths))),
    )
    return Curriculum(phases=phases, messages=tuple(messages))


def _benchmark_level_paths(
    puzzle_path: PathLike,
    benchmark_levels: Sequence[str],
) -> List[str]:
    puzzle_path = os.path.normpath(os.fspath(puzzle_path))
    if os.path.isdir(puzzle_path):
        level_paths = [
            os.path.join(puzzle_path, level)
            for level in benchmark_levels
            if os.path.isdir(os.path.join(puzzle_path, level))
        ]
        return level_paths or [puzzle_path]
    return [puzzle_path]


def _dedupe_paths(paths: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for path in paths:
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
