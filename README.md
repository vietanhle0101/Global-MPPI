# Global-MPPI

Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation
via KernelSOS.

Local sampling-based MPC (MPPI, predictive sampling) easily gets stuck in the
local minima of contact-rich costs. GlobalMPPI adds a global step: candidate
control-spline knots are rolled out, scored with a log-sum-exp smoothed cost
and passed to a [KernelSOS](https://github.com/Simple-Robotics/ksos-tools)
solver (the `ksos-tools` submodule), whose solution seeds MPPI.

| Algorithm | Flag | Description |
| --- | --- | --- |
| Predictive Sampling | `ps` | Keeps the lowest-cost sampled control tape. |
| MPPI | `mppi` | Softmax-weighted average of sampled rollouts. |
| DIAL-MPC | `dial` | MPPI with noise annealed over iterations and horizon. |
| GlobalMPPI | `globalmppi` | KernelSOS global step followed by MPPI refinement. |

Tasks: **PushT** (`global_mppi/tasks/pusht.py`), which pushes a T block to a goal
pose with a point pusher, and **CubeRotation** (`global_mppi/tasks/cube.py`),
which reorients a cube with a LEAP hand.

## Installation

Requires Python >= 3.12 and an NVIDIA GPU with CUDA 12.

```bash
git clone --recurse-submodules <repo-url> && cd Global-MPPI
# already cloned: git submodule update --init --recursive

uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .             # or: pip install -e .
uv pip install -e ./ksos-tools  # KernelSOS solver (submodule, not on PyPI)
```

Video recording needs `ffmpeg` on `PATH`. The bundled one from `imageio-ffmpeg` works:

```bash
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" ~/.local/bin/ffmpeg
```

## Layout

```
global_mppi/
├── algs/          # PS, MPPI, DIAL, GlobalMPPI
├── tasks/         # PushT, CubeRotation (dynamics + costs)
├── models/        # MuJoCo XML models
├── simulation/    # Closed-loop runner, video and W&B logging
└── utils/         # Splines, video helpers
examples/          # pusht.py, pusht_nf.py, cube.py, plot_results.py, plot_timing.py
ksos_l2o/          # Learned parts for PushT: flow sampler and KSOS dual solver
tests/             # pytest suite
ksos-tools/        # KernelSOS solver (submodule)
```

All commands below run from the repo root with the environment active. The
`ksos_l2o` modules also need `PYTHONPATH=.`.

## Running the examples

Each script takes the algorithm as a positional argument.

```bash
python examples/pusht.py globalmppi        # PushT, headless, records a video
python examples/pusht.py --no-headless ps  # PushT in the MuJoCo viewer
python examples/cube.py globalmppi         # CubeRotation (always opens the viewer)
```

`examples/pusht_nf.py` is the PushT runner with all GlobalMPPI options,
including the learned sampler and solver:

```bash
# Plain GlobalMPPI: uniform candidates, Newton solver with restarts
PYTHONPATH=. python examples/pusht_nf.py --seeds 0,1,2 --near-goal-init globalmppi

# Flow sampler + learned solver (NF + L2O), LSE cost
PYTHONPATH=. python examples/pusht_nf.py --seeds 0,1,2 --near-goal-init \
  --ksos-sampling flow --ksos-flow-dir ksos_l2o/data/sampler_lse_20ms \
  --ksos-solver learned --ksos-solver-dir ksos_l2o/data/sdp_solver_k2 globalmppi

# Same with the plain (single-rollout) cost: faster per cycle
PYTHONPATH=. python examples/pusht_nf.py --seeds 0,1,2 --near-goal-init --ksos-cost plain \
  --ksos-sampling flow --ksos-flow-dir ksos_l2o/data/sampler_plain_20ms \
  --ksos-solver learned --ksos-solver-dir ksos_l2o/data/sdp_solver_k2_plain globalmppi
```

Main flags (see `--help` for all of them):

| Flag | Meaning |
| --- | --- |
| `--seeds 0,1,2` | One episode per seed (default 0-5). |
| `--near-goal-init`, `--random-init`, `--init-qpos x,y,th,px,py` | Start state; default is a fixed start. |
| `--duration 10` / `--cycles N` | Episode length in seconds / steps (50 Hz). |
| `--goal-pos-tol 0.005 --goal-angle-tol 0.05` | Stop when the block is this close to the goal; `0` never stops. |
| `--ksos-sampling {uniform,sobol,flow}` | How KSOS candidates are drawn. |
| `--ksos-solver {newton,newton-rs,learned}` | KSOS dual solver (default `newton-rs`). |
| `--ksos-cost {lse,plain}` | LSE-smoothed cost (100 noisy rollouts per candidate) or one rollout. |
| `--ksos-num-samples 128 --elite-num 32` | More candidates per pass (default 32). |
| `--no-headless` | Open the viewer instead of recording video. |
| `--wandb --wandb-entity <entity>` | Log to Weights & Biases. |

Each run writes to `logs/pushT_globalmppi_globalmppi/<timestamp>/`: per-seed
`robot_pos_history` and `best_cost_history` (`.npy`) and a video. With
several seeds the videos are joined into `simulation_all_seeds.mp4`. Per-cycle timings
are appended to `test/log/globalmppi_timing.txt`.

To run two JAX jobs on one GPU, start both with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Learned KSOS components (`ksos_l2o/`)

Both models are tied to the physics, control range and cost they were trained
with. Retrain both after changing `global_mppi/models/pusht/pusht.xml` or the
PushT cost.

### 1. Flow sampler

A state-conditioned normalizing flow (RealNVP) that proposes KSOS candidates:

```bash
PYTHONPATH=. python -m ksos_l2o.train_state_sampler \
  --output ksos_l2o/data/sampler_lse_20ms \
  --cost lse --sigma-lse 0.1 --layers 8 --hidden 256 \
  --goal-fraction 1.0 --goal-max 0.15 --near-start-prob 0 --rollout-fraction 0 \
  --pusher-clearance 0.05 --epochs 50 --steps-per-epoch 100 --batch-size 64 --val-size 128
```

Each epoch logs `ratio`, the flow's best-of-32 cost divided by that of 32
uniform candidates (lower is better), and the best checkpoint is saved. To
fine-tune, add `--init <dir>` with matching `--layers`/`--hidden` and a new
`--output`. For the plain-cost sampler use `--cost plain`.

### 2. Learned dual solver

Generate KSOS problems from the flow's candidates, then train the unrolled
solver on them. Use `JAX_PLATFORMS=cpu` for `train_solver`: its CVXPY
reference solves run in a CPU process pool.

```bash
PYTHONPATH=. python -m ksos_l2o.generate_flow_dataset \
  --sampler-dir ksos_l2o/data/sampler_lse_20ms --out ksos_l2o/data/dual_dataset.npz

JAX_PLATFORMS=cpu PYTHONPATH=. python -m ksos_l2o.train_solver --newton-steps 2 \
  --data ksos_l2o/data/dual_dataset.npz --output-dir ksos_l2o/data/sdp_solver_k2
# plain cost: add --cost plain and use --output-dir ksos_l2o/data/sdp_solver_k2_plain
```

To check an existing solver on new data without retraining, add `--eval-only`
(keep `--newton-steps` equal to the checkpoint's). Note that this overwrites the
`report.json` in that directory.

## M20 box pushing (`m20push`)

A Deep Robotics M20, a wheeled-legged quadruped, pushes a 0.6 x 0.4 x 0.4 m,
5 kg box to a goal pose with a bumper post ahead of its front wheels. The
robot is planar and tracks body-frame velocity commands (forward, sideways, yaw
rate), as the real robot's stock gait takes them; only the bumper may touch
the box.

| File | What |
| --- | --- |
| `global_mppi/tasks/m20push.py` | `M20BoxPush` (task and cost), `PUSH_WEIGHTS`, `m20push_context`, `sample_push_state` |
| `global_mppi/algs/steered_global_mppi.py` | `SteeredGlobalMPPI`: GlobalMPPI with the task's `steer` and `drop_glitch` in every rollout |
| `global_mppi/models/m20push/build.py` | Planner model (MJX, 50 ms) and CPU plant (1 ms), sizes, `Plant`; MuJoCo and NumPy only |
| `global_mppi/models/m20push/*.xml` | Snapshots of both models (`python -m global_mppi.models.m20push.build`) |
| `global_mppi/simulation/m20push.py` | `Settings`, `make_controller`, `Loop`: closed loop against a separate plant |
| `examples/m20push.py` | Push-phase episodes from `sample_push_state` starts |
| `tests/test_m20push.py` | Models, states, steering, glitch guard |

```bash
PYTHONPATH=.:ksos-tools python examples/m20push.py --seeds 0,1,2
PYTHONPATH=.:ksos-tools python examples/m20push.py --seeds 0 --video   # mp4 in logs/m20push/
```

The full task (`M20BoxPush()`, default weights) is the whole job from anywhere
on the floor. GlobalMPPI alone did not solve it: getting round the box takes a
multi-second detour that a sampler re-seeded every cycle doesn't commit to. The
working setup splits the job. A motion planner puts the robot at a contact,
GlobalMPPI pushes from there (the push phase), and short rule-based pushes
finish within 40 cm of the goal. On the Mac that passed 9 of 10 scenarios,
every box within 3.5 cm and 2.5 degrees. The planner and the fine pushes live
in the deployment repo; this is the part to improve.

**Push phase.** `M20BoxPush(weights=PUSH_WEIGHTS)`, controlled by
`SteeredGlobalMPPI`:

- The cost holds the bumper at the chosen contact (mocap body `contact`, set
  with `M20BoxPush.with_contact`) and keeps the robot square to the face.
- The third control is the robot's heading relative to the face, not its yaw
  rate. `steer` turns that heading into a yaw rate by feedback, in the rollouts
  and in the plant. Pushing off centre turns the box, and the face swings into
  a front wheel unless the robot turns with it; sampling a yaw-rate profile
  that does this rarely worked.

`examples/m20push.py` runs push phases only. Each episode ends on reaching the
goal, a wheel touching the box, losing contact for 1 s, or a time limit. On a
Mac CPU (32 KSOS candidates, 1 pass, 512 MPPI samples, plain cost):

- seeds 0-2 each ended "lost";
- the boxes moved 3 to 36 cm closer (seed 2 got to 16 cm);
- no wheel touched;
- each 0.5 s replan took 0.5-1.1 s.

Results go to `logs/m20push/results.json`.

**Two departures from `run_interactive`.**

- *Replan period and spline shift.* The loop replans every 0.5 s, not every
  20 ms. Before planning it shifts the previous plan to now, holding the last
  knot. The cubic's own shift extrapolates past the last knot; at 0.5 s that
  tail ran to the command limits and into the box.
- *Commitment.* The previous plan stays unless the new one is 3% cheaper.
  Without this, the first command flipped between very different, equally
  cheap plans every cycle.

**MJX glitch.** At first contact MJX 3.3.2 sometimes throws the box 0.86 m in
one 50 ms step. MuJoCo's CPU engine doesn't, from the same state. Whether MJX
does depends on how XLA compiled the surrounding program: a 22-step rollout was
clean, a 60-step one threw the box. It does not depend on the solver, contact
shape, friction or margins. `M20BoxPush.drop_glitch` keeps the previous state
for any step that moves the box or robot more than 5 cm (real motion is under
1.5 cm a step). The planner model uses Euler steps, a sphere bumper and extra
hinge inertia on the box, for the same reason.

**For `ksos_l2o`:**

- `m20push_context(qpos, qvel, contact)` (18 values) plays the part of
  `pusht_context`.
- `sample_push_state(key, task)` plays the part of
  `problem.sample_initial_state`. It returns push-ready states: the box
  0.1-1.2 m from the goal, the robot at a contact on the face that points at
  the goal, offset to turn the box, plus jitter.
- `state_flow.make_flow_sampler` and the dataset scripts call
  `pusht_context` directly. They need the context function passed in before
  they work for this task.

## Plotting results

```bash
# Success rate, time to goal, final error, distance over time, pusher paths, summary.csv
python examples/plot_results.py \
  --run "NF+L2O LSE=logs/pushT_globalmppi_globalmppi/<run1>" \
  --run "NF+L2O plain=logs/pushT_globalmppi_globalmppi/<run2>" --out logs/results_cmp

# Box plot of KSOS rollout / KSOS solver / MPPI time per step (default: latest run)
python examples/plot_timing.py --run-dir logs/pushT_globalmppi_globalmppi/<run>
```

Compare methods on the same seeds and use 20 or more seeds; with only a few
seeds most differences are noise.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```

The first run of each test file JIT-compiles MJX rollouts and takes a while.

## Citation

```bibtex
@article{wei2026global,
  title={Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation via KernelSOS},
  author={Wei, Zhongqi and Duembgen, Frederike},
  journal={arXiv preprint arXiv:2604.27175},
  year={2026}
}
```

MIT license, see [LICENSE](LICENSE). Built on
[hydrax](https://github.com/vincekurtz/hydrax).
