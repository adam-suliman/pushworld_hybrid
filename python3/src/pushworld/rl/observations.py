# Copyright 2026
#
# Vector observation utilities for reinforcement learning on PushWorld.

from dataclasses import asdict, dataclass
import json
import os
from typing import Iterable, List, Sequence, Union

import numpy as np

from pushworld.config import PUZZLE_EXTENSION
from pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle, State
from pushworld.utils.filesystem import iter_files_with_extension

PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class VectorObservationConfig:
    """Fixed-size limits used to encode variable-size PushWorld puzzles."""

    max_width: int
    max_height: int
    max_movables: int
    max_goals: int
    max_shape_cells: int
    max_wall_cells: int
    max_agent_wall_cells: int
    version: int = 1

    @property
    def global_size(self) -> int:
        return 9

    @property
    def object_scalar_size(self) -> int:
        return 13

    @property
    def encoded_cell_size(self) -> int:
        return 3

    @property
    def object_slot_size(self) -> int:
        return self.object_scalar_size + self.max_shape_cells * self.encoded_cell_size

    @property
    def static_slot_size(self) -> int:
        return self.encoded_cell_size

    @property
    def observation_size(self) -> int:
        return get_observation_layout(self).observation_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VectorObservationConfig":
        return cls(**data)

    def save(self, path: PathLike) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: PathLike) -> "VectorObservationConfig":
        with open(path) as file:
            return cls.from_dict(json.load(file))


@dataclass(frozen=True)
class ObservationLayout:
    """Index layout for vectors emitted by `encode_puzzle_state`."""

    global_slice: slice
    action_mask_slice: slice
    object_start: int
    object_slot_size: int
    wall_start: int
    wall_slot_size: int
    agent_wall_start: int
    agent_wall_slot_size: int
    observation_size: int

    def object_slice(self, index: int) -> slice:
        start = self.object_start + index * self.object_slot_size
        return slice(start, start + self.object_slot_size)

    def wall_slice(self, index: int) -> slice:
        start = self.wall_start + index * self.wall_slot_size
        return slice(start, start + self.wall_slot_size)

    def agent_wall_slice(self, index: int) -> slice:
        start = self.agent_wall_start + index * self.agent_wall_slot_size
        return slice(start, start + self.agent_wall_slot_size)


def get_observation_layout(config: VectorObservationConfig) -> ObservationLayout:
    """Returns the fixed vector layout for a given observation config."""
    object_start = config.global_size + NUM_ACTIONS
    wall_start = object_start + config.max_movables * config.object_slot_size
    agent_wall_start = wall_start + config.max_wall_cells * config.static_slot_size
    observation_size = (
        agent_wall_start + config.max_agent_wall_cells * config.static_slot_size
    )
    return ObservationLayout(
        global_slice=slice(0, config.global_size),
        action_mask_slice=slice(config.global_size, config.global_size + NUM_ACTIONS),
        object_start=object_start,
        object_slot_size=config.object_slot_size,
        wall_start=wall_start,
        wall_slot_size=config.static_slot_size,
        agent_wall_start=agent_wall_start,
        agent_wall_slot_size=config.static_slot_size,
        observation_size=observation_size,
    )


def resolve_puzzle_paths(paths: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """Resolves files/directories into sorted `.pwp` file paths."""
    if isinstance(paths, (str, os.PathLike)):
        path_items = [paths]
    else:
        path_items = list(paths)

    puzzle_paths = []
    for path in path_items:
        puzzle_paths.extend(iter_files_with_extension(os.fspath(path), PUZZLE_EXTENSION))

    puzzle_paths = sorted(os.path.normpath(path) for path in puzzle_paths)
    if not puzzle_paths:
        raise ValueError(f"No PushWorld puzzles found in: {paths}")
    return puzzle_paths


def get_agent_only_wall_positions(puzzle: PushWorldPuzzle) -> set:
    """Returns agent-only wall cells without duplicating regular wall cells."""
    return set(puzzle.agent_wall_positions) - set(puzzle.wall_positions)


def build_observation_config(
    paths: Union[PathLike, Sequence[PathLike]]
) -> VectorObservationConfig:
    """Scans puzzles and returns fixed observation limits for vector encoding."""
    return build_observation_config_from_puzzles(
        PushWorldPuzzle(path) for path in resolve_puzzle_paths(paths)
    )


def build_observation_config_from_puzzles(
    puzzles: Iterable[PushWorldPuzzle],
) -> VectorObservationConfig:
    """Returns fixed observation limits for already-loaded puzzles."""
    max_width = 0
    max_height = 0
    max_movables = 0
    max_goals = 0
    max_shape_cells = 0
    max_wall_cells = 0
    max_agent_wall_cells = 0

    seen_puzzle = False
    for puzzle in puzzles:
        seen_puzzle = True
        width, height = puzzle.dimensions
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_movables = max(max_movables, puzzle.num_movables)
        max_goals = max(max_goals, len(puzzle.goal_state))
        max_shape_cells = max(
            max_shape_cells,
            max(len(obj.cells) for obj in puzzle.movable_objects),
        )
        max_wall_cells = max(max_wall_cells, len(puzzle.wall_positions))
        max_agent_wall_cells = max(
            max_agent_wall_cells,
            len(get_agent_only_wall_positions(puzzle)),
        )

    if not seen_puzzle:
        raise ValueError("No PushWorld puzzles provided for observation config.")

    return VectorObservationConfig(
        max_width=max_width,
        max_height=max_height,
        max_movables=max_movables,
        max_goals=max_goals,
        max_shape_cells=max_shape_cells,
        max_wall_cells=max_wall_cells,
        max_agent_wall_cells=max_agent_wall_cells,
    )


def compute_action_mask(puzzle: PushWorldPuzzle, state: State) -> np.ndarray:
    """Returns a boolean mask for actions that change the puzzle state."""
    return np.array(
        [puzzle.get_next_state(state, action) != state for action in range(NUM_ACTIONS)],
        dtype=bool,
    )


def encode_puzzle_state(
    puzzle: PushWorldPuzzle,
    state: State,
    steps: int,
    max_steps: int,
    config: VectorObservationConfig,
) -> np.ndarray:
    """Encodes a PushWorld state as one fixed-length `float32` vector."""
    _validate_puzzle_fits_config(puzzle, state, config)

    layout = get_observation_layout(config)
    observation = np.zeros((layout.observation_size,), dtype=np.float32)

    width, height = puzzle.dimensions
    num_goals = len(puzzle.goal_state)
    achieved_goals = puzzle.count_achieved_goals(state)
    global_features = np.array(
        [
            _safe_div(width, config.max_width),
            _safe_div(height, config.max_height),
            min(1.0, _safe_div(steps, max_steps)),
            _safe_div(puzzle.num_movables, config.max_movables),
            _safe_div(num_goals, config.max_goals),
            _safe_div(achieved_goals, config.max_goals),
            _safe_div(len(puzzle.wall_positions), config.max_wall_cells),
            _safe_div(
                len(get_agent_only_wall_positions(puzzle)),
                config.max_agent_wall_cells,
            ),
            1.0 if puzzle.is_goal_state(state) else 0.0,
        ],
        dtype=np.float32,
    )
    observation[layout.global_slice] = global_features
    observation[layout.action_mask_slice] = compute_action_mask(puzzle, state).astype(
        np.float32
    )

    for index, (obj, position) in enumerate(zip(puzzle.movable_objects, state)):
        slot = layout.object_slice(index)
        _encode_object_slot(
            output=observation[slot],
            object_cells=obj.cells,
            position=position,
            goal_position=_goal_for_object(puzzle, index),
            object_index=index,
            num_goals=num_goals,
            config=config,
        )

    _encode_static_slots(
        output=observation,
        positions=sorted(puzzle.wall_positions),
        start=layout.wall_start,
        slot_size=layout.wall_slot_size,
        config=config,
    )
    _encode_static_slots(
        output=observation,
        positions=sorted(get_agent_only_wall_positions(puzzle)),
        start=layout.agent_wall_start,
        slot_size=layout.agent_wall_slot_size,
        config=config,
    )

    return observation


def _encode_object_slot(
    output: np.ndarray,
    object_cells: Iterable[tuple],
    position: tuple,
    goal_position: Union[tuple, None],
    object_index: int,
    num_goals: int,
    config: VectorObservationConfig,
) -> None:
    x, y = position
    cells = sorted(object_cells)
    cell_xs, cell_ys = zip(*cells)
    bbox_width = max(cell_xs) - min(cell_xs) + 1
    bbox_height = max(cell_ys) - min(cell_ys) + 1
    has_goal = goal_position is not None
    goal_x, goal_y = goal_position if has_goal else (0, 0)

    output[: config.object_scalar_size] = np.array(
        [
            1.0,
            1.0 if object_index == 0 else 0.0,
            1.0 if 1 <= object_index <= num_goals else 0.0,
            1.0 if has_goal else 0.0,
            _safe_div(x, config.max_width),
            _safe_div(y, config.max_height),
            _safe_div(goal_x, config.max_width),
            _safe_div(goal_y, config.max_height),
            _safe_div(goal_x - x, config.max_width) if has_goal else 0.0,
            _safe_div(goal_y - y, config.max_height) if has_goal else 0.0,
            _safe_div(bbox_width, config.max_width),
            _safe_div(bbox_height, config.max_height),
            _safe_div(len(cells), config.max_shape_cells),
        ],
        dtype=np.float32,
    )

    shape_start = config.object_scalar_size
    for cell_index, (cell_x, cell_y) in enumerate(cells):
        start = shape_start + cell_index * config.encoded_cell_size
        output[start : start + config.encoded_cell_size] = np.array(
            [
                1.0,
                _safe_div(cell_x, config.max_width),
                _safe_div(cell_y, config.max_height),
            ],
            dtype=np.float32,
        )


def _encode_static_slots(
    output: np.ndarray,
    positions: Sequence[tuple],
    start: int,
    slot_size: int,
    config: VectorObservationConfig,
) -> None:
    for index, (x, y) in enumerate(positions):
        slot_start = start + index * slot_size
        output[slot_start : slot_start + slot_size] = np.array(
            [
                1.0,
                _safe_div(x, config.max_width),
                _safe_div(y, config.max_height),
            ],
            dtype=np.float32,
        )


def _goal_for_object(puzzle: PushWorldPuzzle, object_index: int) -> Union[tuple, None]:
    goal_index = object_index - 1
    if 0 <= goal_index < len(puzzle.goal_state):
        return puzzle.goal_state[goal_index]
    return None


def _validate_puzzle_fits_config(
    puzzle: PushWorldPuzzle,
    state: State,
    config: VectorObservationConfig,
) -> None:
    width, height = puzzle.dimensions
    if width > config.max_width or height > config.max_height:
        raise ValueError("Puzzle dimensions exceed vector observation config.")
    if len(state) != puzzle.num_movables:
        raise ValueError("State length does not match the puzzle movable count.")
    if puzzle.num_movables > config.max_movables:
        raise ValueError("Puzzle movable count exceeds vector observation config.")
    if len(puzzle.goal_state) > config.max_goals:
        raise ValueError("Puzzle goal count exceeds vector observation config.")
    if len(puzzle.wall_positions) > config.max_wall_cells:
        raise ValueError("Puzzle wall count exceeds vector observation config.")
    if len(get_agent_only_wall_positions(puzzle)) > config.max_agent_wall_cells:
        raise ValueError("Puzzle agent-wall count exceeds vector observation config.")
    for obj in puzzle.movable_objects:
        if len(obj.cells) > config.max_shape_cells:
            raise ValueError("Puzzle object shape exceeds vector observation config.")


def _safe_div(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(value) / float(denominator)
