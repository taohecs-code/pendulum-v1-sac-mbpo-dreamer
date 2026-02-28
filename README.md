## Pendulum SAC vs MBPO vs Dreamer (Unified Runner)

This repo is set up to run **controlled comparisons** on `Pendulum-v1` with:
- **SAC**: model-free baseline
- **MBPO**: dynamics-model rollouts + SAC (key variable: `horizon`)
- **Dreamer (simplified MLP)**: lightweight latent world model + actor/critic (key variable: `horizon`)

The goal is to keep **the evaluation protocol, logging schema, and checkpoint rules identical** across algorithms.

### Install

This project assumes you have Python 3.10+ and PyTorch installed. A minimal set of deps:

```bash
pip install gymnasium numpy torch wandb matplotlib
```

If you use W&B, run `wandb login` once (or set `WANDB_API_KEY`).

### Quick start

Run everything from the project root (`pendulum-sac-mbpo-dreamer/`).

#### SAC (baseline, 3 seeds)

```bash
python experiment/run_pendulum.py --algo sac --seeds 1,2,3
```

#### MBPO (compare horizons)

```bash
python experiment/run_pendulum.py --algo mbpo --seeds 1,2,3 --horizon 5
python experiment/run_pendulum.py --algo mbpo --seeds 1,2,3 --horizon 15
```

#### Dreamer (simplified MLP)

```bash
python experiment/run_pendulum.py --algo dreamer --seeds 1,2,3 --horizon 15
```

### W&B naming rules (matches proposal)

- **Project**: `CS5180_Pendulum_MBRL` (default; override via `--wandb-project`)
- **Run name format**: `[Algorithm]_[KeyVars]_[Seed]`
  - SAC: `SAC_Baseline_Seed{seed}`
  - MBPO: `MBPO_Horizon{horizon}_Seed{seed}`
  - Dreamer: `DREAMER_Horizon{horizon}_MLP_Seed{seed}`

You can also add tags:

```bash
python experiment/run_pendulum.py --algo mbpo --horizon 15 --wandb-tags Compute_Constrained,Horizon_Sweep
```

Disable W&B entirely:

```bash
python experiment/run_pendulum.py --algo sac --seeds 1 --no-wandb
```

Upload checkpoints to W&B Artifacts (recommended for Colab runs):

```bash
python experiment/run_pendulum.py --algo sac --seeds 1,2,3,4,5 --wandb-artifacts
```

### Unified evaluation protocol (all algorithms)

All algorithms use the same logic in `experiment/run_pendulum.py`:
- **Deterministic evaluation**: `deterministic=True`
- **Evaluation frequency**: `--eval-freq` (default 2000 env steps)
- **Evaluation episodes**: `--eval-episodes` (default 5)
- **Convergence threshold**: `--threshold` (default `-150`)
- **Asymptotic performance**: mean of the last `--asymptotic-window` evaluation points (default 10)

### Checkpoints (same rules & names)

Checkpoints are saved under:
- `checkpoints/<run_name>/...`

Saved files:
- **Regular step checkpoints**: `step_10k.pt`, `step_20k.pt`, ... (controlled by `--checkpoint-every`)
- **Convergence checkpoint**: `converged_150.pt` (saved when eval return first exceeds `--threshold`)
- **Final checkpoint**: `final_stage.pt`

### Key experiment knobs

#### MBPO

- `--horizon`: model rollout horizon (primary variable for error accumulation)
- `--mbpo-model-steps`: model gradient steps per env step
- `--mbpo-synth-updates`: synthetic policy updates per env step

Example "fast debug" MBPO:

```bash
python experiment/run_pendulum.py --algo mbpo --seeds 1 --train-steps 5000 --prefill-steps 500 --horizon 5 --mbpo-model-steps 1 --mbpo-synth-updates 1 --no-wandb
```

#### Dreamer (simplified)

- `--horizon`: placeholder for imagination horizon (used in run naming/config; simplified agent uses a lightweight training loop)
- `--dreamer-seq-len`: sequence length used to train the world model
- `--batch-size`: sequence batch size

Example "fast debug" Dreamer:

```bash
python experiment/run_pendulum.py --algo dreamer --seeds 1 --train-steps 5000 --prefill-steps 500 --dreamer-seq-len 20 --batch-size 32 --no-wandb
```

### Notes / assumptions

- This code prioritizes a **unified experiment harness** and **consistent evaluation** over perfectly matching every detail of canonical MBPO/Dreamer papers.
- The Dreamer implementation is a **simplified MLP-only** variant tailored to low-dimensional `Pendulum-v1`.

### ReplayBuffer conventions (important for reproducibility)

The shared `ReplayBuffer` stores two termination signals:
- **done_for_learning** (`done`): `1` if **terminated** (used for bootstrapping targets; time-limit truncation does not cut off TD targets)
- **episode_end**: `1` if **terminated OR truncated** (used to prevent sequence sampling from crossing episode boundaries; used by Dreamer continuation training depending on `--dreamer-cont-target`)

Dreamer uses `ReplayBuffer.sample_sequences(...)` which enforces **no episode crossing** using `episode_end`.

