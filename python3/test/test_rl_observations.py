from dataclasses import replace
import os

import numpy as np

from pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle
from pushworld.rl.observations import (
    build_observation_config,
    compute_action_mask,
    encode_puzzle_state,
    get_observation_layout,
)


TEST_PUZZLES_PATH = os.path.join(os.path.split(__file__)[0], "puzzles")


def test_vector_observation_shape_dtype_determinism_and_padding():
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "trivial.pwp")
    puzzle = PushWorldPuzzle(puzzle_path)
    base_config = build_observation_config(puzzle_path)
    config = replace(
        base_config,
        max_movables=base_config.max_movables + 2,
        max_wall_cells=base_config.max_wall_cells + 1,
        max_agent_wall_cells=base_config.max_agent_wall_cells + 1,
    )
    layout = get_observation_layout(config)

    observation_1 = encode_puzzle_state(
        puzzle, puzzle.initial_state, steps=0, max_steps=32, config=config
    )
    observation_2 = encode_puzzle_state(
        puzzle, puzzle.initial_state, steps=0, max_steps=32, config=config
    )

    assert observation_1.shape == (config.observation_size,)
    assert observation_1.dtype == np.float32
    assert np.isfinite(observation_1).all()
    assert np.array_equal(observation_1, observation_2)

    inactive_object_slot = observation_1[layout.object_slice(puzzle.num_movables)]
    assert np.count_nonzero(inactive_object_slot) == 0

    padded_wall_slot = observation_1[layout.wall_slice(len(puzzle.wall_positions))]
    assert np.count_nonzero(padded_wall_slot) == 0


def test_action_mask_matches_state_transitions():
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "transitive_pushing.pwp")
    puzzle = PushWorldPuzzle(puzzle_path)
    mask = compute_action_mask(puzzle, puzzle.initial_state)

    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == bool
    for action in range(NUM_ACTIONS):
        assert mask[action] == (
            puzzle.get_next_state(puzzle.initial_state, action) != puzzle.initial_state
        )
