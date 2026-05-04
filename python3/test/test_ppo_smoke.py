import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gym")

from pushworld.rl.training import PPOTrainingConfig, load_checkpoint, train_ppo
from pushworld.rl.utils import resolve_device


TEST_PUZZLES_PATH = os.path.join(os.path.split(__file__)[0], "puzzles")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_ppo_training_smoke_saves_checkpoint(tmp_path):
    output_dir = tmp_path / "run"
    result = train_ppo(
        PPOTrainingConfig(
            puzzle_path=os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
            output_dir=str(output_dir),
            total_timesteps=4,
            num_generated_puzzles=0,
            seed=3,
            num_envs=2,
            num_steps=2,
            max_steps=8,
            device="cpu",
            hidden_sizes=(16,),
            update_epochs=1,
            minibatch_size=4,
            eval_interval=1,
            eval_episodes=1,
            save_interval=1,
        )
    )

    assert os.path.exists(result.latest_checkpoint)
    assert os.path.exists(output_dir / "config.json")
    assert os.path.exists(output_dir / "obs_config.json")
    assert os.path.exists(output_dir / "train.csv")

    checkpoint = load_checkpoint(result.latest_checkpoint, resolve_device("cpu"))
    assert checkpoint["global_step"] == 4
    assert "model_state_dict" in checkpoint


def test_train_ppo_cli_smoke(tmp_path):
    output_dir = tmp_path / "cli_run"
    env = os.environ.copy()
    src_path = os.path.join(REPO_ROOT, "python3", "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "python3", "scripts", "train_ppo.py"),
            "--puzzle_path",
            os.path.join(TEST_PUZZLES_PATH, "trivial.pwp"),
            "--output_dir",
            str(output_dir),
            "--total_timesteps",
            "2",
            "--num_generated_puzzles",
            "0",
            "--num_envs",
            "1",
            "--num_steps",
            "1",
            "--max_steps",
            "4",
            "--device",
            "cpu",
            "--hidden_sizes",
            "16",
            "--update_epochs",
            "1",
            "--minibatch_size",
            "1",
            "--eval_interval",
            "1",
            "--eval_episodes",
            "1",
            "--save_interval",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "latest_checkpoint=" in result.stdout
    assert os.path.exists(output_dir / "latest.pt")
