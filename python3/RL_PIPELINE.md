# PushWorld RL Pipeline

This document describes the current vector-observation reinforcement learning
pipeline end to end. It covers plain PPO, DIAYN-style skill discovery, DIAYN
skill evaluation and selection, DIAYN fine-tuning, and hierarchical PPO over
frozen DIAYN skills.

Commands below assume PowerShell and that they are run from `python3/`.

```powershell
cd C:\data\MIPT\heuristics final project\pushworld_hybrid\python3
$env:PYTHONPATH = "$PWD\src"
.\venv\Scripts\Activate.ps1
```

Install the optional RL dependencies before running training:

```powershell
python -m pip install -r requirements_rl.txt
```

The old Gym warning about Gym being unmaintained is currently expected. It is a
dependency warning, not a training failure.

## Main Components

The RL stack lives under `src/pushworld/rl/`.

| File | Role |
| --- | --- |
| `observations.py` | Fixed-size vector observations and action masks. |
| `envs.py` | `PushWorldVectorEnv`, a vector wrapper around PushWorld puzzle dynamics. |
| `architectures.py` | PPO actor-critic MLPs, skill-conditioned actor-critic, and DIAYN discriminator. |
| `ppo.py` | Rollout update logic, PPO clipped objective, GAE, entropy/value losses. |
| `storage.py` | Rollout buffer for synchronous multi-env PPO. |
| `training.py` | Plain PPO training and evaluation orchestration. |
| `curriculum.py` | Optional generated-puzzle curriculum for plain PPO. |
| `diayn.py` | DIAYN pretraining, skill evaluation, DIAYN fine-tuning, reward diagnostics. |
| `hierarchical.py` | Hierarchical PPO meta-controller over frozen DIAYN skills. |
| `tracking.py` | Optional Comet ML logging wrapper. |

## Observation Model

The RL policy does not train from rendered RGB frames. It trains from fixed-size
vectors produced by `encode_puzzle_state(...)`.

The policy observation includes:

- normalized puzzle dimensions and step fraction
- movable count, goal count, achieved-goal count
- padded per-object slots for agent and movable objects
- current object positions
- goal positions and goal deltas where available
- bounding box and area information
- exact padded shape-cell offsets
- padded wall and agent-wall coordinates
- four action-mask values

`compute_action_mask(...)` marks primitive actions that actually change state.
Training defaults to action masking. The actor still has four primitive actions:
left, right, up, down.

DIAYN uses a second, smaller discriminator observation from
`skill_observations.py`. By default it intentionally hides static wall layout and
agent position so the discriminator has less opportunity to identify skills from
trivial puzzle identity or walking-only behavior.

## Plain PPO Baseline

Use this as the scratch baseline for comparison against all skill-based methods.

```powershell
$Run = "runs\ppo_level0_base_1m"

python scripts/train_ppo.py `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=$Run `
  --num_generated_puzzles=0 `
  --total_timesteps=1000000 `
  --num_envs=8 `
  --num_steps=128 `
  --max_steps=512 `
  --device=auto `
  --action_masking=true `
  --eval_interval=10 `
  --eval_episodes=20 `
  --save_interval=10 `
  --comet_enabled=false
```

Evaluate:

```powershell
python scripts/evaluate_ppo.py `
  --checkpoint="$Run\best_success.pt" `
  --puzzle_path=..\benchmark\puzzles\level0\base\test `
  --output_csv="$Run\base_test.csv" `
  --episodes_per_puzzle=1 `
  --max_steps=512 `
  --action_masking=true
```

Plain PPO trains `ActorCriticMLP`: a shared 2-layer MLP trunk by default, Tanh
activations, orthogonal initialization, a 4-logit policy head, and a scalar value
head.

Plain PPO has optional curriculum support. If `--num_generated_puzzles > 0`, it
generates level-0 puzzles into the run directory and mixes them with selected
benchmark levels over training. If the RGD planner is available and
`--filter_generated=true`, generated puzzles can be filtered for solvability.
Set `--num_generated_puzzles=0` when you want exact benchmark-only training.

### Plain PPO Outputs

| File | Meaning |
| --- | --- |
| `config.json` | Training config. |
| `obs_config.json` | Vector observation size limits computed from selected puzzles. |
| `train.csv` | PPO update metrics and eval success. |
| `latest.pt` | Latest PPO checkpoint. |
| `best_success.pt` | Best checkpoint by eval success, when evaluation is enabled. |
| `evaluation.csv` or custom CSV | Per-episode evaluation records. |

### Plain PPO CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--puzzle_path` | `../benchmark/puzzles` | Puzzle file or directory to train on. |
| `--output_dir` | `runs/ppo` | Run directory. |
| `--total_timesteps` | `1000000` | Target primitive env steps. |
| `--num_generated_puzzles` | `500` | Number of generated level-0 puzzles for curriculum. |
| `--seed` | `0` | RNG seed. |
| `--num_envs` | `8` | Synchronous vector env count. |
| `--num_steps` | `128` | Rollout length per env before PPO update. |
| `--max_steps` | `512` | Episode time limit. |
| `--device` | `auto` | `auto`, `cpu`, or CUDA device string. |
| `--learning_rate` | `3e-4` | Adam learning rate. |
| `--hidden_sizes` | `256,256` | MLP hidden sizes. |
| `--gamma` | `0.99` | Discount factor. |
| `--gae_lambda` | `0.95` | GAE lambda. |
| `--clip_coef` | `0.2` | PPO clip ratio. |
| `--entropy_coef` | `0.01` | Entropy bonus coefficient. |
| `--value_coef` | `0.5` | Value loss coefficient. |
| `--max_grad_norm` | `0.5` | Gradient clipping norm. |
| `--update_epochs` | `4` | PPO epochs per rollout. |
| `--minibatch_size` | `256` | PPO minibatch size. |
| `--action_masking` | `true` | Mask invalid primitive actions. |
| `--benchmark_levels` | `level1,level2,level3,level4` | Levels mixed into curriculum. |
| `--filter_generated` | `true` | Try to filter generated puzzles with planner when available. |
| `--eval_interval` | `10` | Evaluate every N updates. Use `0` to disable. |
| `--eval_episodes` | `20` | Number of eval episodes when eval runs. |
| `--save_interval` | `10` | Save latest checkpoint every N updates. |
| `--comet_enabled` | `false` | Enable Comet logging. |
| `--comet_project_name` | `pushworld-ppo` | Comet project. |
| `--comet_workspace` | empty | Optional Comet workspace. |
| `--comet_experiment_name` | empty | Optional Comet experiment name. |
| `--comet_tags` | empty | Comma-separated Comet tags. |
| `--comet_log_artifacts` | `true` | Upload final artifacts to Comet. |

## PPO Evaluation Options

`scripts/evaluate_ppo.py` greedily evaluates a plain PPO checkpoint.

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `runs/ppo/latest.pt` | PPO checkpoint. |
| `--puzzle_path` | `../benchmark/puzzles` | Eval puzzle file or directory. |
| `--obs_config` | checkpoint dir `obs_config.json` | Optional explicit observation config. |
| `--output_csv` | checkpoint dir `evaluation.csv` | Per-episode output. |
| `--episodes_per_puzzle` | `1` | Eval repeats per puzzle. |
| `--max_steps` | `512` | Episode time limit. |
| `--action_masking` | `true` | Mask invalid actions. |
| `--device` | `auto` | Device. |
| `--seed` | `0` | Eval seed. |
| Comet options | same pattern | Optional metric/artifact logging. |

## DIAYN Skill Discovery

DIAYN pretraining trains a skill-conditioned actor-critic with PPO on intrinsic
reward. The low-level policy input is:

```text
vector_observation + one_hot(skill_id)
```

The discriminator predicts the skill id from compact dynamic object-state
features. The original DIAYN reward is:

```text
log q_phi(z | s) - log p(z)
```

In this codebase, the total pretraining reward is shaped for PushWorld:

```text
reward =
  diayn_reward_scale * DIAYN
  + object_change_reward
  + gated_object_novelty_reward
  + positive_goal_progress_reward
  - negative_goal_progress_penalty
```

Object-change reward encourages moving non-agent objects. Object novelty rewards
first-time object positions within the episode. Novelty can be gated so it pays
only when the move does not increase total object-to-goal distance. Negative goal
progress penalty punishes moves that increase object-to-goal distance.

Example pretraining run:

```powershell
$Run = "runs\diayn_v5_useful_novelty_base_300k"

python scripts/train_diayn_ppo.py `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=$Run `
  --total_timesteps=300000 `
  --num_skills=8 `
  --num_envs=8 `
  --num_steps=128 `
  --max_steps=256 `
  --diayn_reward_scale=0.2 `
  --object_change_reward_scale=0.75 `
  --object_change_reward_clip=1.0 `
  --object_novelty_reward_scale=1.0 `
  --object_novelty_reward_clip=1.0 `
  --object_novelty_requires_nonnegative_goal_progress=true `
  --goal_progress_reward_scale=2.0 `
  --negative_goal_progress_penalty_scale=1.0 `
  --entropy_coef=0.08 `
  --comet_enabled=true `
  --comet_project_name=pushworld-diayn `
  --comet_tags=diayn-v5,useful-novelty,level0-base
```

### DIAYN Pretraining Outputs

| File | Meaning |
| --- | --- |
| `config.json` | Full DIAYN training config. |
| `skill_config.json` | Skill count and intrinsic reward config. |
| `obs_config.json` | Policy observation config. |
| `discriminator_obs_config.json` | Discriminator observation config. |
| `train.csv` | PPO, discriminator, skill usage, and manipulation diagnostics. |
| `diayn_policy.pt` | Skill-conditioned low-level policy checkpoint. |
| `discriminator.pt` | Skill discriminator checkpoint. |

### Important DIAYN Metrics

| Metric | Meaning |
| --- | --- |
| `mean_intrinsic_reward` | Final reward used by PPO during DIAYN pretraining. |
| `mean_discriminator_reward` | DIAYN classifier reward term only. |
| `discriminator_accuracy` | How well discriminator predicts skill id. |
| `skill_usage_entropy` | How evenly sampled skills appear in rollouts. |
| `endpoint_diversity` | State-summary diversity across rollout endpoints. |
| `mean_object_displacement` | Average non-agent object cell movement per step. |
| `object_push_rate` | Fraction of primitive steps that move any non-agent object. |
| `mean_novel_object_positions` | New object positions reached per step. |
| `object_novelty_rate` | Fraction of steps reaching new object positions. |
| `mean_goal_progress` | Positive means objects got closer to goals on average. |
| `mean_goal_progress_penalty` | Penalty for moves that worsen goal distance. |
| `skill_*_...` | Per-skill manipulation metrics. |

### DIAYN Pretraining CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--puzzle_path` | `../benchmark/puzzles/level0/all/train` | Unsupervised pretraining puzzle set. |
| `--output_dir` | `runs/diayn_ppo` | Run directory. |
| `--total_timesteps` | `1000000` | Target primitive env steps. |
| `--seed` | `0` | RNG seed. |
| `--num_envs` | `8` | Synchronous vector env count. |
| `--num_steps` | `128` | Rollout length. |
| `--max_steps` | `512` | Episode time limit. |
| `--device` | `auto` | Device. |
| `--learning_rate` | `3e-4` | PPO Adam learning rate. |
| `--hidden_sizes` | `256,256` | Low-level policy hidden sizes. |
| `--discriminator_hidden_sizes` | `256,256` | Discriminator hidden sizes. |
| `--gamma` | `0.99` | Discount. |
| `--gae_lambda` | `0.95` | GAE lambda. |
| `--clip_coef` | `0.2` | PPO clip ratio. |
| `--value_coef` | `0.5` | Value loss coefficient. |
| `--max_grad_norm` | `0.5` | Gradient clipping. |
| `--update_epochs` | `4` | PPO epochs per rollout. |
| `--minibatch_size` | `256` | PPO minibatch size. |
| `--action_masking` | `true` | Mask invalid primitive actions. |
| `--save_interval` | `10` | Save checkpoints every N updates. |
| `--num_skills` | `8` | Number of discrete DIAYN skills. |
| `--diayn_reward_scale` | `1.0` | Scale on classifier reward. |
| `--object_change_reward_scale` | `0.5` | Scale on object displacement reward. |
| `--object_change_reward_clip` | `1.0` | Clip object displacement reward contribution. |
| `--object_novelty_reward_scale` | `1.0` | Scale on first-time object positions. |
| `--object_novelty_reward_clip` | `2.0` | Clip novelty reward contribution. |
| `--object_novelty_requires_nonnegative_goal_progress` | `true` | Suppress novelty reward when goal distance worsens. |
| `--goal_progress_reward_scale` | `0.25` | Positive goal-progress reward scale. |
| `--negative_goal_progress_penalty_scale` | `1.0` | Penalty scale for negative goal progress. |
| `--discriminator_lr` | `3e-4` | Discriminator Adam learning rate. |
| `--discriminator_update_epochs` | `4` | Discriminator epochs per rollout. |
| `--discriminator_minibatch_size` | `256` | Discriminator minibatch size. |
| `--entropy_coef` | `0.05` | PPO entropy coefficient during pretraining. |
| Comet options | same pattern | Optional tracking. |

## DIAYN Skill Evaluation and Selection

`scripts/evaluate_diayn_skills.py` evaluates every skill, or a configured subset,
on an evaluation puzzle set. It can run either deterministic or stochastic
evaluation.

Deterministic evaluation always takes the argmax action. Stochastic evaluation
samples from the policy distribution. In current experiments, most useful DIAYN
behavior is stochastic, so stochastic evaluation is usually more informative.

Single-rollout stochastic evaluation:

```powershell
$Run = "runs\diayn_v5_useful_novelty_base_300k"

python scripts/evaluate_diayn_skills.py `
  --pretrained_dir=$Run `
  --puzzle_path=..\benchmark\puzzles\level0\base\test `
  --output_csv="$Run\skill_eval_stochastic.csv" `
  --max_puzzles=50 `
  --max_steps=256 `
  --deterministic=false `
  --comet_enabled=true `
  --comet_project_name=pushworld-diayn
```

Best-of-N stochastic evaluation:

```powershell
python scripts/evaluate_diayn_skills.py `
  --pretrained_dir=$Run `
  --puzzle_path=..\benchmark\puzzles\level0\base\test `
  --output_csv="$Run\skill_eval_stochastic_n8.csv" `
  --max_puzzles=50 `
  --max_steps=256 `
  --deterministic=false `
  --rollouts_per_skill=8 `
  --comet_enabled=true `
  --comet_project_name=pushworld-diayn `
  --comet_tags=eval,best-of-8,v5
```

Best-of-N runs multiple rollouts per `(puzzle, skill)` pair. It reports raw
single-rollout metrics plus `best_of_n_*` metrics computed from the best rollout
per puzzle-skill pair. The script also computes puzzle coverage: whether any
skill/rollout solved a given puzzle.

### DIAYN Evaluation Outputs

| File | Meaning |
| --- | --- |
| `skill_eval_*.csv` | Raw rollout rows, including `rollout_id`. |
| `skill_eval_*_best.csv` | Best rollout per `(puzzle, skill)` pair. |
| `skill_eval_*_selection.json` | Best skill/checkpoint metadata and aggregate stats. |

### DIAYN Evaluation CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--pretrained_dir` | `runs/diayn_ppo` | DIAYN run dir or checkpoint path. |
| `--puzzle_path` | `../benchmark/puzzles/level0/all/test` | Eval puzzle file or directory. |
| `--output_csv` | run dir `skill_eval.csv` | Raw rollout output. |
| `--summary_csv` | `*_best.csv` | Optional best-of summary output. |
| `--selection_json` | `*_selection.json` | Optional selection metadata output. |
| `--max_steps` | `512` | Episode time limit. |
| `--max_puzzles` | `0` | Puzzle subsample size. `0` means all. |
| `--skill_ids` | empty | Comma-separated skill ids. Empty means all. |
| `--device` | `auto` | Device. |
| `--seed` | `0` | Eval seed. |
| `--rollouts_per_skill` | `1` | Number of rollouts per puzzle-skill pair. |
| `--action_masking` | `true` | Mask invalid primitive actions. |
| `--deterministic` | `true` | Use argmax actions instead of sampling. |
| Comet options | same pattern | Optional tracking. |

### Skill Ranking Examples

Rank raw stochastic rollouts by skill:

```powershell
Import-Csv "$Run\skill_eval_stochastic.csv" |
  Group-Object skill_id |
  ForEach-Object {
    $rows = $_.Group
    [pscustomobject]@{
      skill_id = $_.Name
      success = (($rows | Measure-Object success -Average).Average)
      avg_return = (($rows | Measure-Object return -Average).Average)
      push_rate = (($rows | Measure-Object object_push_count -Sum).Sum / [Math]::Max(1, ($rows | Measure-Object length -Sum).Sum))
      novelty_rate = (($rows | Measure-Object object_novelty_count -Sum).Sum / [Math]::Max(1, ($rows | Measure-Object length -Sum).Sum))
      avg_goal_progress = (($rows | Measure-Object goal_progress -Average).Average)
    }
  } |
  Sort-Object success, avg_return -Descending |
  Format-Table -Auto
```

List solved puzzles:

```powershell
Import-Csv "$Run\skill_eval_stochastic.csv" |
  Where-Object { [double]$_.success -gt 0 } |
  Select-Object puzzle_path, skill_id, rollout_id, return, length, object_displacement, goal_progress |
  Sort-Object puzzle_path, skill_id, rollout_id |
  Format-Table -Auto
```

## DIAYN Fine-Tuning

Fine-tuning initializes a skill-conditioned policy from `diayn_policy.pt` and
continues PPO on true PushWorld extrinsic rewards.

Use this only after skill evaluation shows useful behavior. Previous experiments
showed that fixed-skill fine-tuning can collapse stochastic manipulation behavior,
so compare against pretraining metrics carefully.

```powershell
$Pretrain = "runs\diayn_v5_useful_novelty_base_300k"

python scripts/finetune_diayn_ppo.py `
  --pretrained_dir=$Pretrain `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=runs\diayn_finetune_base `
  --skill_sampling=fixed `
  --fixed_skill_id=7 `
  --total_timesteps=300000 `
  --num_envs=8 `
  --num_steps=128 `
  --max_steps=256 `
  --entropy_coef=0.05 `
  --eval_interval=25 `
  --eval_episodes=50 `
  --comet_enabled=true `
  --comet_project_name=pushworld-diayn `
  --comet_tags=finetune,skill7,level0-base
```

### Fine-Tuning Outputs

| File | Meaning |
| --- | --- |
| `config.json` | Fine-tuning config. |
| `skill_config.json` | Number of skills loaded from pretraining. |
| `obs_config.json` | Observation config. |
| `train.csv` | PPO and evaluation metrics. |
| `latest.pt` | Latest fine-tuned skill-conditioned policy. |
| `best_success.pt` | Best eval-success fine-tuned checkpoint. |

### Fine-Tuning CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--pretrained_dir` | `runs/diayn_ppo` | DIAYN run dir or checkpoint path. |
| `--puzzle_path` | `../benchmark/puzzles/level0/base/train` | Downstream training puzzles. |
| `--output_dir` | `runs/diayn_finetune` | Output directory. |
| `--total_timesteps` | `1000000` | PPO training steps. |
| `--seed` | `0` | RNG seed. |
| `--num_envs` | `8` | Synchronous env count. |
| `--num_steps` | `128` | Rollout length. |
| `--max_steps` | `512` | Episode time limit. |
| `--device` | `auto` | Device. |
| `--learning_rate` | `3e-4` | Adam learning rate. |
| `--gamma` | `0.99` | Discount. |
| `--gae_lambda` | `0.95` | GAE lambda. |
| `--clip_coef` | `0.2` | PPO clip ratio. |
| `--entropy_coef` | `0.01` | Entropy bonus. Higher values preserve stochasticity longer. |
| `--value_coef` | `0.5` | Value loss coefficient. |
| `--max_grad_norm` | `0.5` | Gradient clipping. |
| `--update_epochs` | `4` | PPO epochs per rollout. |
| `--minibatch_size` | `256` | PPO minibatch size. |
| `--action_masking` | `true` | Mask invalid primitive actions. |
| `--eval_interval` | `10` | Evaluate every N updates. |
| `--eval_episodes` | `20` | Eval episodes when eval runs. |
| `--skill_sampling` | `fixed` | `fixed`, `uniform`, or `cycle`. |
| `--fixed_skill_id` | `0` | Skill id used when `skill_sampling=fixed`. |
| `--save_interval` | `10` | Save latest every N updates. |
| Comet options | same pattern | Optional tracking. |

Skill sampling modes:

- `fixed`: all environments use one skill id. Most sample-efficient for testing
  whether one discovered skill can adapt.
- `uniform`: each new episode resamples a skill. This trains all skills, but
  spreads sparse task reward across many policies.
- `cycle`: deterministic cycling through skills across envs/episodes.

## Hierarchical DIAYN PPO

Hierarchical DIAYN freezes the pretrained low-level skill-conditioned policy and
trains a PPO meta-controller. The meta-controller action space is the skill id.
Each meta-action executes the selected low-level skill for `skill_horizon`
primitive steps, unless the episode terminates or truncates earlier.

```powershell
$Pretrain = "runs\diayn_v5_useful_novelty_base_300k"

python scripts/train_hierarchical_diayn_ppo.py `
  --pretrained_dir=$Pretrain `
  --puzzle_path=..\benchmark\puzzles\level0\base\train `
  --output_dir=runs\diayn_hier_base_h8 `
  --total_timesteps=300000 `
  --num_envs=8 `
  --num_steps=64 `
  --max_steps=256 `
  --skill_horizon=8 `
  --deterministic_low_level=false `
  --entropy_coef=0.01 `
  --eval_interval=25 `
  --eval_episodes=50 `
  --comet_enabled=true `
  --comet_project_name=pushworld-diayn `
  --comet_tags=hierarchical,h8,level0-base
```

Use `--deterministic_low_level=false` when DIAYN skills only work stochastically.
Current experiments show deterministic low-level skill execution often collapses
object manipulation.

### Hierarchical Outputs

| File | Meaning |
| --- | --- |
| `config.json` | Hierarchical training config. |
| `skill_config.json` | Loaded low-level skill count. |
| `obs_config.json` | Meta-controller observation config. |
| `train.csv` | Meta-controller PPO metrics and eval success. |
| `latest.pt` | Latest meta-controller checkpoint. |
| `best_success.pt` | Best eval-success hierarchical checkpoint. |

### Hierarchical CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--pretrained_dir` | `runs/diayn_ppo` | Frozen DIAYN low-level policy. |
| `--puzzle_path` | `../benchmark/puzzles/level0/base/train` | Training puzzles. |
| `--output_dir` | `runs/diayn_hierarchical` | Output directory. |
| `--total_timesteps` | `1000000` | Primitive env step budget. |
| `--seed` | `0` | RNG seed. |
| `--num_envs` | `8` | Synchronous env count. |
| `--num_steps` | `64` | Meta-rollout length. |
| `--max_steps` | `512` | Primitive episode time limit. |
| `--skill_horizon` | `8` | Primitive steps per selected skill. |
| `--device` | `auto` | Device. |
| `--learning_rate` | `3e-4` | Meta-controller Adam learning rate. |
| `--hidden_sizes` | `256,256` | Meta-controller hidden sizes. |
| `--gamma` | `0.99` | Discount. |
| `--gae_lambda` | `0.95` | GAE lambda. |
| `--clip_coef` | `0.2` | PPO clip ratio. |
| `--entropy_coef` | `0.01` | Entropy bonus for meta-controller. |
| `--value_coef` | `0.5` | Value loss coefficient. |
| `--max_grad_norm` | `0.5` | Gradient clipping. |
| `--update_epochs` | `4` | PPO epochs per rollout. |
| `--minibatch_size` | `256` | PPO minibatch size. |
| `--deterministic_low_level` | `false` | Use argmax low-level actions instead of sampling. |
| `--eval_interval` | `10` | Evaluate every N updates. |
| `--eval_episodes` | `20` | Eval episodes. |
| `--save_interval` | `10` | Save latest every N updates. |
| Comet options | same pattern | Optional tracking. |

## Recommended End-to-End Protocols

### Protocol A: Baseline PPO

1. Train plain PPO on the target training split.
2. Evaluate on the matching test split.
3. Use success rate, return, and length as the scratch baseline.

This is the control condition. DIAYN methods should be compared against this.

### Protocol B: DIAYN Discovery Diagnostics

1. Pretrain DIAYN on an unsupervised puzzle set.
2. Evaluate skills stochastically with `--rollouts_per_skill=1`.
3. Evaluate deterministic behavior.
4. If deterministic collapses but stochastic has nonzero success, run
   best-of-N evaluation.
5. Inspect per-skill success, push rate, novelty rate, and goal progress.

Proceed only when skill evaluation shows object manipulation above random motion.

### Protocol C: Best-of-N Skill Selection

Use this to answer whether useful behavior exists inside the stochastic policy.

```powershell
python scripts/evaluate_diayn_skills.py `
  --pretrained_dir=$Run `
  --puzzle_path=..\benchmark\puzzles\level0\base\test `
  --output_csv="$Run\skill_eval_stochastic_n8.csv" `
  --max_puzzles=50 `
  --max_steps=256 `
  --deterministic=false `
  --rollouts_per_skill=8
```

Read `*_selection.json` for:

- selected checkpoint
- `best_skill_id`
- `best_success_rate`
- `puzzle_coverage_rate`
- solved puzzle count

If `best_of_n_puzzle_coverage_rate` is much higher than single-rollout success,
the skills contain useful behavior but are unreliable. That points to
entropy-preserving or stochastic fine-tuning rather than more DIAYN shaping.

### Protocol D: DIAYN Fine-Tuning

Only use this after skill selection finds a meaningful skill. Start with:

- `skill_sampling=fixed`
- selected best skill id
- higher entropy, for example `--entropy_coef=0.03` to `0.08`, if behavior is
  stochastic
- short runs first, for example `200k` to `300k` timesteps

Watch whether eval push rate, novelty rate, and success collapse. If they
collapse, plain PPO fine-tuning is selecting away the useful stochastic behavior.

### Protocol E: Hierarchical Skill Composition

Use this when best-of-N evaluation shows complementary skills or high puzzle
coverage across skills. Keep low-level stochastic if deterministic low-level
skills do not manipulate objects.

Good signs before hierarchical training:

- multiple skills solve different puzzles
- best-of-N puzzle coverage is high
- low-level skills have nontrivial push and novelty rates

Bad signs:

- all skills solve the same tiny easy subset
- deterministic and stochastic both have near-zero push rate
- mean goal progress is strongly negative

## Metric Definitions

| Metric | Meaning |
| --- | --- |
| `success_rate` | Fraction of episodes that solve the puzzle before timeout. |
| `best_success_rate` | Best per-skill success rate. |
| `best_skill_id` | Skill id with highest success rate. |
| `return` | Sum of PushWorld extrinsic rewards in one episode. |
| `length` | Primitive steps before success or timeout. |
| `object_push_rate` | Object-push steps divided by total primitive steps. |
| `object_displacement` | Total Manhattan movement of non-agent movable objects in an episode. |
| `novel_object_positions` | First-time object positions reached within an episode. |
| `object_novelty_rate` | Novel object-position steps divided by total primitive steps. |
| `goal_progress` | Reduction in object-to-goal Manhattan distance. Positive is good. |
| `best_of_n_*` | Metrics after selecting the best rollout among N attempts. |
| `puzzle_coverage_rate` | Fraction of puzzles solved by at least one skill/rollout. |

## Current Empirical Guidance

Recent experiments showed:

- Plain fixed-skill fine-tuning can collapse useful stochastic DIAYN behavior.
- Deterministic DIAYN evaluation often has near-zero object manipulation.
- Stochastic DIAYN can solve a nontrivial subset, but behavior is unreliable.
- Best-of-N evaluation is now the right diagnostic before another training change.

The current next recommended experiment is to compare v4 and v5 DIAYN runs with
`--rollouts_per_skill=8`. If puzzle coverage jumps substantially, implement
entropy-preserving/stochastic fine-tuning. If coverage stays low, improve skill
discovery before running more downstream training.

## Artifact Hygiene

Training and evaluation create large run directories under `python3/runs/`.
These should stay out of Git. Comet offline/local cache directories such as
`.cometml-runs/` should also be ignored.

