# Copyright 2026
#
# Vector-observation Gym environments for PushWorld reinforcement learning.

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np

from pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle, State
from pushworld.rl.observations import (
    PathLike,
    VectorObservationConfig,
    build_observation_config_from_puzzles,
    compute_action_mask,
    encode_puzzle_state,
    resolve_puzzle_paths,
)

PuzzleBank = Sequence[Tuple[str, PushWorldPuzzle]]


class PushWorldVectorEnv(gym.Env):
    """A Gym environment that emits compact vector observations.

    Rewards, termination, truncation, seeding, and action semantics mirror
    `pushworld.gym_env.PushWorldEnv`. The only behavioral difference is the
    observation format.
    """

    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(
        self,
        puzzle_path: Union[PathLike, Sequence[PathLike]],
        max_steps: Optional[int] = 512,
        observation_config: Optional[VectorObservationConfig] = None,
        preloaded_puzzles: Optional[PuzzleBank] = None,
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be None or >= 1")

        if preloaded_puzzles is None:
            preloaded_puzzles = load_puzzle_bank(puzzle_path)
        if not preloaded_puzzles:
            raise ValueError(f"No PushWorld puzzles found in: {puzzle_path}")

        self._puzzle_paths = [path for path, _ in preloaded_puzzles]
        self._puzzles = [puzzle for _, puzzle in preloaded_puzzles]
        self._max_steps = max_steps
        self._observation_config = (
            observation_config
            if observation_config is not None
            else build_observation_config_from_puzzles(self._puzzles)
        )

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._observation_config.observation_size,),
            dtype=np.float32,
        )

        self._random_generator = random.Random(123)
        self._current_index = None
        self._current_puzzle = None
        self._current_state = None
        self._current_achieved_goals = 0
        self._steps = 0

    @property
    def observation_config(self) -> VectorObservationConfig:
        return self._observation_config

    @property
    def current_puzzle(self) -> Optional[PushWorldPuzzle]:
        return self._current_puzzle

    @property
    def current_state(self) -> Optional[State]:
        return self._current_state

    @property
    def current_puzzle_path(self) -> Optional[str]:
        if self._current_index is None:
            return None
        return self._puzzle_paths[self._current_index]

    @property
    def steps(self) -> int:
        return self._steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._random_generator = random.Random(seed)

        self._current_index = self._random_generator.randrange(len(self._puzzles))
        self._current_puzzle = self._puzzles[self._current_index]
        self._current_state = self._current_puzzle.initial_state
        self._current_achieved_goals = self._current_puzzle.count_achieved_goals(
            self._current_state
        )
        self._steps = 0
        return self._observe(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError("The provided action is not in the action space.")
        if self._current_state is None:
            raise RuntimeError("reset() must be called before step() can be called.")

        self._steps += 1
        previous_state = self._current_state
        self._current_state = self._current_puzzle.get_next_state(
            self._current_state, action
        )

        terminated = self._current_puzzle.is_goal_state(self._current_state)
        if terminated:
            reward = 10.0
        else:
            previous_achieved_goals = self._current_puzzle.count_achieved_goals(
                previous_state
            )
            current_achieved_goals = self._current_puzzle.count_achieved_goals(
                self._current_state
            )
            reward = current_achieved_goals - previous_achieved_goals - 0.01

        truncated = False if self._max_steps is None else self._steps >= self._max_steps
        return self._observe(), reward, terminated, truncated, self._info()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise AssertionError("mode must be rgb_array.")
        if self._current_puzzle is None or self._current_state is None:
            raise RuntimeError("reset() must be called before render() can be called.")
        return self._current_puzzle.render(self._current_state)

    def _observe(self) -> np.ndarray:
        return encode_puzzle_state(
            puzzle=self._current_puzzle,
            state=self._current_state,
            steps=self._steps,
            max_steps=0 if self._max_steps is None else self._max_steps,
            config=self._observation_config,
        )

    def _info(self) -> Dict[str, Any]:
        return {
            "puzzle_state": self._current_state,
            "action_mask": compute_action_mask(
                self._current_puzzle, self._current_state
            ).copy(),
            "puzzle_path": self.current_puzzle_path,
        }


def load_puzzle_bank(
    puzzle_path: Union[PathLike, Sequence[PathLike]]
) -> List[Tuple[str, PushWorldPuzzle]]:
    """Loads puzzle files once and returns `(path, puzzle)` pairs."""
    return [(path, PushWorldPuzzle(path)) for path in resolve_puzzle_paths(puzzle_path)]
