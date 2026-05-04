Python PushWorld
----------------

Prerequisites:
- Python >= v3.10 must be installed.
- Python3-venv must be installed.

To configure and install all dependencies, run the setup script from this directory:

```
./setup.sh
```

This script creates a virtual Python environment in a `venv` directory. To run Python
scripts and tests within this package, the virtual environment must be activated:

```
source venv/bin/activate
```

To run all unit tests:

```
pytest test
```

Some functions depend on [Fast Downward](https://github.com/aibasel/downward).
Once installed, you may need to update the `FAST_DOWNWARD_PATH` in
`src/pushworld/config.py`.

The RGD benchmarking functions require building the RGD planner in the [../cpp](../cpp)
directory of this repository. For build instructions, see [../cpp/README.md](../cpp/README.md).

## Scripts

The `scripts` directory contains a variety of executable scripts, each of which can
be run with the `--help` option to display usage instructions.

To render images of all benchmark puzzles:

```
./scripts/render_puzzle_previews.py --image_path=puzzle_images
```

To render videos of the solutions to all puzzles, install [ffmpeg](https://ffmpeg.org/)
and run:

```
./scripts/render_plans.py --planning_results_path=../benchmark/solutions --video_path=puzzle_solutions
```

To convert all puzzles to PDDL:

```
./scripts/convert_to_pddl.py --pddl_path=pddl_puzzles
```

To generate level 0 puzzles (use --help flag for puzzle options):

```
./scripts/generate_level0_puzzles.py --save_location_path=training_puzzles --num_puzzles=100
```

## PPO vector training

The PPO pipeline is optional and uses PyTorch in addition to the base Python
dependencies:

```
pip install -r requirements_rl.txt
```

To train a vector-observation PPO agent:

```
python scripts/train_ppo.py --puzzle_path=../benchmark/puzzles
```

Training writes configs, CSV logs, and checkpoints to `runs/ppo` by default. To
evaluate a checkpoint:

```
python scripts/evaluate_ppo.py --checkpoint=runs/ppo/latest.pt --puzzle_path=../benchmark/puzzles
```

Comet ML tracking is optional. First authenticate once with `comet login`, then
enable tracking on training or evaluation runs. The examples below assume a Unix
shell. In PowerShell, either put the command on one line or use a backtick (`) for
line continuation.

```
python scripts/train_ppo.py \
  --puzzle_path=../benchmark/puzzles/level0/base/train \
  --output_dir=runs/ppo_level0_base_comet \
  --num_generated_puzzles=0 \
  --comet_enabled=true \
  --comet_project_name=pushworld-ppo \
  --comet_experiment_name=level0-base-ppo \
  --comet_tags=ppo,level0,base
```

## DIAYN skill discovery

The DIAYN pipeline pretrains a skill-conditioned PPO policy from intrinsic reward,
then supports either downstream PPO fine-tuning or hierarchical PPO over frozen
skills:

PowerShell commands should be run from the `python3` directory:

```
cd C:\data\MIPT\heuristics final project\pushworld_hybrid\python3
$env:PYTHONPATH = "$PWD\src"

python scripts/train_diayn_ppo.py `
  --puzzle_path=..\benchmark\puzzles\level0\all\train `
  --output_dir=runs\diayn_level0_all `
  --num_skills=8

python scripts/evaluate_diayn_skills.py `
  --pretrained_dir=runs\diayn_level0_all `
  --puzzle_path=..\benchmark\puzzles\level0\all\test `
  --max_puzzles=50

python scripts/finetune_diayn_ppo.py `
  --pretrained_dir=runs\diayn_level0_all `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=runs\diayn_finetune_base `
  --skill_sampling=fixed `
  --fixed_skill_id=0

python scripts/train_hierarchical_diayn_ppo.py `
  --pretrained_dir=runs\diayn_level0_all `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=runs\diayn_hier_base `
  --skill_horizon=8
```

Unix shell equivalents:

```
python scripts/train_diayn_ppo.py \
  --puzzle_path=../benchmark/puzzles/level0/all/train \
  --output_dir=runs/diayn_level0_all \
  --num_skills=8

python scripts/evaluate_diayn_skills.py \
  --pretrained_dir=runs/diayn_level0_all \
  --puzzle_path=../benchmark/puzzles/level0/all/test \
  --max_puzzles=50

python scripts/finetune_diayn_ppo.py \
  --pretrained_dir=runs/diayn_level0_all \
  --puzzle_path=../benchmark/puzzles/level0/base/train \
  --output_dir=runs/diayn_finetune_base \
  --skill_sampling=fixed \
  --fixed_skill_id=0

python scripts/train_hierarchical_diayn_ppo.py \
  --pretrained_dir=runs/diayn_level0_all \
  --puzzle_path=../benchmark/puzzles/level0/base/train \
  --output_dir=runs/diayn_hier_base \
  --skill_horizon=8
```

These scripts write local CSV logs and checkpoints regardless of Comet settings.
Use `--comet_enabled=true` with the same Comet flags shown above to mirror metrics
and artifacts online. Fine-tuning defaults to one fixed skill because uniformly
training all skills divides sparse task reward across multiple policies. Use
`--skill_sampling=uniform` only when you specifically want all skills to adapt.
DIAYN pretraining defaults to object-centric diagnostics and rewards: the
discriminator hides static wall context and agent position, while `train.csv` logs
object displacement, push/contact rates, goal-progress deltas, and per-skill
manipulation metrics. It also tracks episodic object-position novelty and rewards
first-time object positions to discourage repeatedly farming the same push. Novelty
can be gated on non-worsening goal distance, and DIAYN pretraining can penalize
negative goal progress so object movement is biased toward useful state changes.
Skill evaluation supports best-of-N stochastic rollouts with
`--rollouts_per_skill`; the evaluator writes raw rollout rows, a best-per-skill
summary CSV, and a selection JSON with the best skill/checkpoint metadata.
