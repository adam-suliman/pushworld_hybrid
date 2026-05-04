import os

import pytest

pytest.importorskip("gym")

from pushworld.gym_env import PushWorldEnv
from pushworld.puzzle import Actions
from pushworld.rl.envs import PushWorldVectorEnv
from pushworld.rl.observations import build_observation_config


TEST_PUZZLES_PATH = os.path.join(os.path.split(__file__)[0], "puzzles")


def test_vector_env_matches_gym_env_rewards_and_done_flags():
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "trivial.pwp")
    config = build_observation_config(puzzle_path)
    image_env = PushWorldEnv(puzzle_path, max_steps=10)
    vector_env = PushWorldVectorEnv(
        puzzle_path,
        max_steps=10,
        observation_config=config,
    )

    vector_observation, vector_info = vector_env.reset(seed=7)
    image_env.reset(seed=7)
    assert vector_observation in vector_env.observation_space
    assert vector_info["action_mask"].shape == (4,)

    for action in [Actions.RIGHT, Actions.DOWN, Actions.RIGHT, Actions.UP]:
        _, image_reward, image_terminated, image_truncated, _ = image_env.step(action)
        (
            vector_observation,
            vector_reward,
            vector_terminated,
            vector_truncated,
            vector_info,
        ) = vector_env.step(action)

        assert vector_observation in vector_env.observation_space
        assert vector_reward == image_reward
        assert vector_terminated == image_terminated
        assert vector_truncated == image_truncated
        assert vector_info["puzzle_state"] == vector_env.current_state


def test_vector_env_truncates_at_max_steps():
    puzzle_path = os.path.join(TEST_PUZZLES_PATH, "transitive_pushing.pwp")
    env = PushWorldVectorEnv(puzzle_path, max_steps=2)
    env.reset()
    assert env.step(Actions.LEFT)[3] is False
    assert env.step(Actions.LEFT)[3] is True
