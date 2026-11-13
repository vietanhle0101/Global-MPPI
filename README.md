# Global-MPPI

Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation
via KernelSOS.

Contact-rich manipulation costs are highly non-convex, so purely local
sampling-based MPC (MPPI, predictive sampling, ...) easily gets stuck in bad
local minima. GlobalMPPI adds a global search step: candidate control-spline
knots are evaluated, smoothed with a log-sum-exp estimate, and handed to a
[KernelSOS](https://github.com/Simple-Robotics/ksos-tools) solver (vendored
here as the `ksos-tools` submodule) to fit a new, better-informed mean before
the usual MPPI local refinement runs on top.

## Algorithms

All algorithms share the same `SamplingBasedController` interface
(`global_mppi/alg_base.py`) and control-spline knot representation
(`global_mppi/utils/spline.py`), so they can be swapped in for the same task
with the same simulation loop.

| Algorithm | Flag | Description |
| --- | --- | --- |
| Predictive Sampling | `ps` | Samples control tapes around the current mean and keeps the lowest-cost one. |
| MPPI | `mppi` | Model-predictive path integral control: a softmax-weighted average over sampled rollouts. |
| DIAL-MPC | `dial` | MPPI with noise annealed across both optimization iterations and the planning horizon. |
| GlobalMPPI | `globalmppi` | This work: a KernelSOS global-search step seeds the mean, followed by an MPPI local-search refinement. |

## Tasks

- **PushT** (`global_mppi/tasks/pusht.py`) — push a T-shaped block to a target
  pose with a point pusher.
- **CubeRotation** (`global_mppi/tasks/cube.py`) — reorient a cube to a target
  orientation using a LEAP hand.

Both tasks define their dynamics via a MuJoCo model (`global_mppi/models/`)
and their running/terminal costs via the abstract `Task` interface in
`global_mppi/task_base.py`.

## Requirements

- Python **>= 3.12**
- An NVIDIA GPU with CUDA 12 (for `jax[cuda12]` / `mujoco-mjx`)

## Installation

The repository uses a git submodule (`ksos-tools`) for the KernelSOS solver, so
clone recursively (or initialize the submodule after cloning):

```bash
# Fresh clone
git clone --recurse-submodules <repo-url>
cd Global-MPPI

# Already cloned without --recurse-submodules?
git submodule update --init --recursive
```

Create an environment (Python 3.12) and install the package plus the submodule.
Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .            # main package + all Python dependencies
uv pip install -e ./ksos-tools # KernelSOS solver (git submodule, not on PyPI)
```

Or with plain `pip` inside a Python 3.12 environment:

```bash
pip install -e .
pip install -e ./ksos-tools
```

All Python dependencies (jax, mujoco, jaxlie, cvxpy, eigenpy, ipdb,
imageio-ffmpeg, warp, ...) are declared in `pyproject.toml` and installed
automatically. The `ksos-tools` submodule is the only piece that must be
installed separately, because it is a git submodule rather than a PyPI package.

## Project layout

```
global_mppi/
├── algs/          # Controllers: PS, MPPI, DIAL, GlobalMPPI
├── tasks/         # Task definitions (dynamics + costs): PushT, CubeRotation
├── models/        # MuJoCo XML models for each task
├── simulation/    # Interactive/async simulation loops, W&B + video logging
├── utils/         # Control-spline interpolation utilities
├── alg_base.py    # Abstract SamplingBasedController base class
├── task_base.py   # Abstract Task interface
└── risk.py        # Domain-randomization risk strategies (average, worst-case, CVaR, ...)
examples/          # Runnable scripts: pusht.py, pusht_nf.py, cube.py
ksos_l2o/          # Learned KSOS sampler (normalizing flow) for PushT
tests/             # pytest test suite
data/logs/         # Example best-cost-history logs + plotting script
ksos-tools/        # KernelSOS solver (git submodule)
```

## Running the examples

The example scripts live in `examples/`. Each takes an algorithm as a
positional argument (`ps`, `mppi`, `dial`, `globalmppi`).

```bash
source .venv/bin/activate

# Push-T task with Predictive Sampling
python examples/pusht.py ps

# Cube rotation task with GlobalMPPI
python examples/cube.py globalmppi
```

`examples/cube.py` always opens an interactive MuJoCo viewer; the
`--headless`/`--wandb` flags below apply only to `examples/pusht.py`.

On a **headless machine** (remote server / dev desktop with no display), add
`--headless` so MuJoCo uses the offscreen GL backend and records a video:

```bash
python examples/pusht.py --headless ps
```

Video recording (and W&B video upload) needs an `ffmpeg` binary on `PATH`. The
`imageio-ffmpeg` dependency bundles one; expose it with:

```bash
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" ~/.local/bin/ffmpeg
export PATH="$HOME/.local/bin:$PATH"
```

### Weights & Biases logging (optional)

```bash
python examples/pusht.py --headless --wandb --wandb-entity <your-entity> ps
```

Pass `--wandb-entity` (or set `export WANDB_ENTITY=<your-entity>`) if your W&B
account has no default entity, otherwise `wandb.init` errors with
`entity not specified`. The first run prompts for your API key (from
https://wandb.ai/authorize) and caches it in `~/.netrc`.

## Learned KSOS sampler (normalizing flow)

`ksos_l2o/` trains a state-conditioned normalizing flow (RealNVP) that
proposes GlobalMPPI's KSOS candidates for PushT, in place of uniform or Sobol
samples. `examples/pusht_nf.py` runs GlobalMPPI with it.

### Training

Train on near-goal states with the LSE-smoothed cost. These are the settings
of the current model, `ksos_l2o/data/sampler_lse_20ms`:

```bash
PYTHONPATH=. python -m ksos_l2o.train_state_sampler \
  --output ksos_l2o/data/sampler_lse_20ms \
  --cost lse --sigma-lse 0.1 --layers 8 --hidden 256 \
  --goal-fraction 1.0 --goal-max 0.15 --near-start-prob 0 --rollout-fraction 0 \
  --pusher-clearance 0.05 --epochs 50 --steps-per-epoch 100 --batch-size 64 --val-size 128
```

Each epoch logs the flow's best-of-32 candidate cost, divided by the same
figure for 32 uniform candidates (`ratio`; lower is better). The learning
rate halves after `--lr-patience` epochs without improvement of at least
0.01, and training stops after `--stop-patience` such epochs. The checkpoint
with the best ratio is saved to `--output`.

To fine-tune an existing model, pass `--init <dir>`. The `--layers` and
`--hidden` values must match that model. The starting model is scored and
saved first, so use a new `--output` to avoid overwriting it:

```bash
PYTHONPATH=. python -m ksos_l2o.train_state_sampler \
  --init ksos_l2o/data/sampler_lse_20ms --output ksos_l2o/data/sampler_plain_20ms \
  --cost plain --layers 8 --hidden 256 --lr 5e-4 \
  --goal-fraction 1.0 --goal-max 0.15 --near-start-prob 0 --rollout-fraction 0 \
  --pusher-clearance 0.05 --epochs 20 --steps-per-epoch 100 --batch-size 64 --val-size 128
```

A trained model is only valid for the physics, control range and cost it was
trained with. Retrain after changing `global_mppi/models/pusht/pusht.xml` or
the PushT cost.

### Testing

```bash
# NF sampler, default start
PYTHONPATH=. python examples/pusht_nf.py --seeds 0 \
  --ksos-sampling flow --ksos-flow-dir ksos_l2o/data/sampler_lse_20ms

# NF sampler, block started close to the goal
PYTHONPATH=. python examples/pusht_nf.py --seeds 0 \
  --ksos-sampling flow --ksos-flow-dir ksos_l2o/data/sampler_lse_20ms --near-goal-init

# Baselines: same commands with --ksos-sampling uniform or sobol (no --ksos-flow-dir)
PYTHONPATH=. python examples/pusht_nf.py --seeds 0 --ksos-sampling uniform --near-goal-init
```

Useful flags:

- `--seeds 0,1,2`: run several episodes (the default is seeds 0–5).
- `--duration 10`: episode length in seconds (the default, 500 steps at
  50 Hz). `--cycles N` sets the number of steps directly.
- `--goal-pos-tol` / `--goal-angle-tol`: the episode ends once the block is
  within 5 mm and 0.05 rad of the goal (by default). `--goal-pos-tol 0`
  turns this off.
- `--random-init` / `--init-qpos x,y,theta,px,py`: other start states.
- `--ksos-num-samples 128 --elite-num 32`: more flow candidates (default 32).
  This reaches the goal faster from close starts, at a higher cost per cycle.
- `--ksos-cost plain`: score KSOS candidates with one rollout each instead of
  the LSE-smoothed cost (100 perturbed rollouts each), which is faster.
- `--no-headless`: open the viewer instead of recording a video.
- `--wandb`: log to Weights & Biases (off by default).

The output for each run (cost history and video) goes to
`logs/pushT_globalmppi_globalmppi/<timestamp>/`. Each control cycle prints its
timing and the safeguard decisions:

- `used best candidate`: the best KSOS candidate beat the KSOS solution and
  seeded MPPI instead.
- `kept pre-MPPI plan`: MPPI made the plan worse, so it was discarded.

To run two JAX jobs on one GPU (for example training and testing), start them
with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Otherwise each job reserves 75% of
GPU memory at startup.

## Running the tests

Install the `dev` extra (adds `pytest`) and run the suite from the repo root:

```bash
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pytest tests/
```

The tests instantiate real MuJoCo/MJX models and JIT-compile rollouts, so
expect the first run of each test file to take a while.

## Citation

If you use this codebase, please cite:

```bibtex
@article{wei2026global,
  title={Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation via KernelSOS},
  author={Wei, Zhongqi and Duembgen, Frederike},
  journal={arXiv preprint arXiv:2604.27175},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

This codebase builds on [hydrax](https://github.com/vincekurtz/hydrax), a
library for sampling-based MPC in MuJoCo MJX.
