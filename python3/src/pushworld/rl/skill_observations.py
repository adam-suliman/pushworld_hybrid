# Copyright 2026
#
# Compact observations for DIAYN skill discrimination.

from dataclasses import asdict, dataclass
import json
import os
from typing import Iterable, Sequence, Union

import numpy as np

from pushworld.puzzle import PushWorldPuzzle, State
from pushworld.rl.observations import (
    PathLike,
    VectorObservationConfig,
    resolve_puzzle_paths,
)


@dataclass(frozen=True)
class SkillDiscriminatorObservationConfig:
    """Fixed limits for DIAYN discriminator state summaries."""

    max_width: int
    max_height: int
    max_movables: int
    max_goals: int
    version: int = 2
    include_agent_position: bool = False
    include_static_context: bool = False

    @property
    def global_size(self) -> int:
        return 5

    @property
    def object_slot_size(self) -> int:
        if self.version <= 1:
            return 12
        return 12

    @property
    def observation_size(self) -> int:
        return self.global_size + self.max_movables * self.object_slot_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillDiscriminatorObservationConfig":
        return cls(**data)

    def save(self, path: Union[str, os.PathLike]) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "SkillDiscriminatorObservationConfig":
        with open(path) as file:
            return cls.from_dict(json.load(file))


def build_skill_discriminator_observation_config(
    paths: Union[PathLike, Sequence[PathLike]],
) -> SkillDiscriminatorObservationConfig:
    return build_skill_discriminator_observation_config_from_puzzles(
        PushWorldPuzzle(path) for path in resolve_puzzle_paths(paths)
    )


def build_skill_discriminator_observation_config_from_puzzles(
    puzzles: Iterable[PushWorldPuzzle],
) -> SkillDiscriminatorObservationConfig:
    max_width = 0
    max_height = 0
    max_movables = 0
    max_goals = 0
    seen = False
    for puzzle in puzzles:
        seen = True
        width, height = puzzle.dimensions
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_movables = max(max_movables, puzzle.num_movables)
        max_goals = max(max_goals, len(puzzle.goal_state))
    if not seen:
        raise ValueError("No puzzles provided for discriminator observation config.")
    return SkillDiscriminatorObservationConfig(
        max_width=max_width,
        max_height=max_height,
        max_movables=max_movables,
        max_goals=max_goals,
    )


def skill_config_from_vector_config(
    vector_config: VectorObservationConfig,
) -> SkillDiscriminatorObservationConfig:
    return SkillDiscriminatorObservationConfig(
        max_width=vector_config.max_width,
        max_height=vector_config.max_height,
        max_movables=vector_config.max_movables,
        max_goals=vector_config.max_goals,
    )


def encode_discriminator_state(
    puzzle: PushWorldPuzzle,
    state: State,
    steps: int,
    max_steps: int,
    config: SkillDiscriminatorObservationConfig,
) -> np.ndarray:
    """Encodes dynamic object state without static wall-layout features."""
    _validate_fits_config(puzzle, state, config)
    output = np.zeros((config.observation_size,), dtype=np.float32)

    achieved_goals = puzzle.count_achieved_goals(state)
    static_context = _static_context_features(puzzle, steps, max_steps, config)
    output[: config.global_size] = np.array(
        [
            static_context[0],
            static_context[1],
            static_context[2],
            static_context[3],
            _safe_div(achieved_goals, config.max_goals),
        ],
        dtype=np.float32,
    )

    for index, position in enumerate(state):
        start = config.global_size + index * config.object_slot_size
        _encode_object_slot(output[start : start + config.object_slot_size], puzzle, state, index, config)

    return output


def batch_encode_discriminator_states(
    puzzles: Sequence[PushWorldPuzzle],
    states: Sequence[State],
    steps: Sequence[int],
    max_steps: int,
    config: SkillDiscriminatorObservationConfig,
) -> np.ndarray:
    return np.stack(
        [
            encode_discriminator_state(puzzle, state, step, max_steps, config)
            for puzzle, state, step in zip(puzzles, states, steps)
        ]
    )


def _encode_object_slot(
    output: np.ndarray,
    puzzle: PushWorldPuzzle,
    state: State,
    index: int,
    config: SkillDiscriminatorObservationConfig,
) -> None:
    if index == 0 and not config.include_agent_position:
        _encode_mask_only_agent_slot(output)
        return

    x, y = state[index]
    init_x, init_y = puzzle.initial_state[index]
    goal = _goal_for_object(puzzle, index)
    has_goal = goal is not None
    goal_x, goal_y = goal if has_goal else (0, 0)
    moved = (x, y) != (init_x, init_y)
    on_goal = has_goal and (x, y) == (goal_x, goal_y)

    output[:] = np.array(
        [
            1.0,
            1.0 if index == 0 else 0.0,
            1.0 if 1 <= index <= len(puzzle.goal_state) else 0.0,
            1.0 if has_goal else 0.0,
            _safe_div(x, config.max_width),
            _safe_div(y, config.max_height),
            _safe_div(x - init_x, config.max_width),
            _safe_div(y - init_y, config.max_height),
            _safe_div(goal_x - x, config.max_width) if has_goal else 0.0,
            _safe_div(goal_y - y, config.max_height) if has_goal else 0.0,
            1.0 if moved else 0.0,
            1.0 if on_goal else 0.0,
        ],
        dtype=np.float32,
    )


def _static_context_features(
    puzzle: PushWorldPuzzle,
    steps: int,
    max_steps: int,
    config: SkillDiscriminatorObservationConfig,
) -> np.ndarray:
    if not config.include_static_context:
        return np.zeros((4,), dtype=np.float32)
    width, height = puzzle.dimensions
    return np.array(
        [
            _safe_div(width, config.max_width),
            _safe_div(height, config.max_height),
            min(1.0, _safe_div(steps, max_steps)),
            _safe_div(puzzle.num_movables, config.max_movables),
        ],
        dtype=np.float32,
    )


def _encode_mask_only_agent_slot(output: np.ndarray) -> None:
    output[:] = 0.0
    output[0] = 1.0
    output[1] = 1.0


def _goal_for_object(puzzle: PushWorldPuzzle, object_index: int):
    goal_index = object_index - 1
    if 0 <= goal_index < len(puzzle.goal_state):
        return puzzle.goal_state[goal_index]
    return None


def _validate_fits_config(
    puzzle: PushWorldPuzzle,
    state: State,
    config: SkillDiscriminatorObservationConfig,
) -> None:
    width, height = puzzle.dimensions
    if width > config.max_width or height > config.max_height:
        raise ValueError("Puzzle dimensions exceed discriminator observation config.")
    if len(state) != puzzle.num_movables:
        raise ValueError("State length does not match puzzle movable count.")
    if puzzle.num_movables > config.max_movables:
        raise ValueError("Puzzle movable count exceeds discriminator observation config.")
    if len(puzzle.goal_state) > config.max_goals:
        raise ValueError("Puzzle goal count exceeds discriminator observation config.")


def _safe_div(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(value) / float(denominator)
