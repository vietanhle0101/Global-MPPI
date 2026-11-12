import argparse
import os
import sys
import getpass

# On a headless machine (e.g. a remote server with no display) MuJoCo must use
# an offscreen GL backend. Select it BEFORE importing mujoco so the render
# context is created correctly. EGL uses the GPU; fall back to `osmesa` (CPU)
# by setting MUJOCO_GL yourself if EGL is unavailable.
# Headless is the default; only an explicit --no-headless opts out.
if "--no-headless" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from global_mppi.algs import PredictiveSampling, MPPI, DIAL, GlobalMPPI
from global_mppi.simulation.deterministic import run_interactive
from global_mppi.tasks.pusht import PushT
"""
Run an interactive simulation of the push-T task with predictive sampling.
"""


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
)
# Weights & Biases logging options (place before the algorithm, e.g.
# `python pusht.py --wandb ps`)
parser.add_argument(
    "--wandb",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Log best_cost_history to Weights & Biases "
    "(on by default; use --no-wandb to disable)",
)
parser.add_argument(
    "--ksos-solver",
    choices=["newton", "newton-rs"],
    default="newton-rs",
    help="KSOS solver used by GlobalMPPI (default: newton-rs)",
)
parser.add_argument(
    "--ksos-sampling",
    choices=["uniform", "sobol", "flow"],
    default="uniform",
    help="Candidate-knot sampling distribution used by GlobalMPPI's KSOS "
    "pass (default: uniform; unchanged behavior). 'flow' draws pass 0 from "
    "the trained sampler in --ksos-flow-dir; later passes sample uniformly "
    "in the shrinking trust region.",
)
parser.add_argument(
    "--ksos-flow-dir",
    default=None,
    help="Trained sampler directory (ksos_l2o/train_state_sampler.py output); "
    "required for --ksos-sampling flow",
)
parser.add_argument(
    "--ksos-num-samples",
    type=int,
    default=32,
    help="KSOS candidates per pass (default: 32)",
)
parser.add_argument(
    "--ksos-passes",
    type=int,
    default=1,
    help="KSOS passes (restarts) per control cycle, each followed by MPPI "
    "(default: 1; the original GlobalMPPI setting is 5)",
)
parser.add_argument(
    "--mppi-noise",
    type=float,
    default=0.1,
    help="With --ksos-sampling flow: MPPI noise std in pass 0 (default 0.1; "
    "the KSOS smoothing there uses the flow's sigma)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.001,
    help="GlobalMPPI softmax temperature for the MPPI update (default: 0.001)",
)
parser.add_argument(
    "--mppi-iterations",
    type=int,
    default=1,
    help="GlobalMPPI: MPPI iterations after each KSOS solve (default: 1; "
    "the original GlobalMPPI setting is 5)",
)
parser.add_argument(
    "--mppi-samples",
    type=int,
    default=1024,
    help="GlobalMPPI: noisy samples per MPPI iteration (default: 1024)",
)
parser.add_argument(
    "--ksos-cost",
    choices=["lse", "plain"],
    default="lse",
    help="How KSOS candidates are scored: 'lse' = soft-min over 100 perturbed "
    "rollouts (default), 'plain' = one rollout per candidate",
)
parser.add_argument(
    "--elite-num",
    type=int,
    default=32,
    help="KSOS elites kept per pass; at most --ksos-num-samples (default: 32)",
)
parser.add_argument(
    "--seeds",
    type=lambda s: [int(x) for x in s.split(",")],
    default=list(range(6)),
    help="Comma-separated seeds, one episode per seed, e.g. 2 or 0,1,2 "
    "(default: 0,1,2,3,4,5)",
)
parser.add_argument(
    "--init-qpos",
    type=lambda s: [float(x) for x in s.split(",")],
    default=[0.1, 0.1, 1.3, 0.0, 0.0],
    help="Initial qpos as block_x,block_y,block_angle,pusher_x,pusher_y "
    "(default: 0.1,0.1,1.3,0,0; the goal is block at 0,0,0)",
)
parser.add_argument(
    "--random-init",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Per-seed random start (block in [-0.2, 0.2]^2, any angle, pusher 5-15 cm away)",
)
parser.add_argument(
    "--near-goal-init",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Per-seed random start near the goal (block 1-12 cm away, |angle| < 0.3 rad, pusher within 5 cm)",
)
parser.add_argument(
    "--frequency",
    type=float,
    default=50.0,
    help="Replanning frequency in Hz (default: 50)",
)
parser.add_argument(
    "--duration",
    type=float,
    default=20.0,
    help="Episode length in simulated seconds (default: 20)",
)
parser.add_argument(
    "--goal-pos-tol",
    type=float,
    default=0.005,
    help="Stop when the block is within this distance (m) and --goal-angle-tol of the goal "
    "(default: 0.005); 0 = never stop early",
)
parser.add_argument(
    "--goal-angle-tol",
    type=float,
    default=0.05,
    help="Goal angle tolerance in rad (default: 0.05)",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=None,
    help="Control cycles per seed; overrides --duration",
)
parser.add_argument(
    "--log-l2s",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Log raw KSOS candidate data (knots, costs, box bounds, ...) per "
    "cycle to data/l2s/raw/ for offline flow-sampler training. Off by "
    "default; GlobalMPPI only.",
)
parser.add_argument(
    "--wandb-project",
    default="global-mppi",
    help="W&B project name (default: global-mppi)",
)
parser.add_argument(
    "--wandb-entity",
    default=None,
    help="W&B entity/team (defaults to your W&B account's default entity)",
)
parser.add_argument(
    "--wandb-key",
    default=None,
    help="W&B API key. Only needed the first time; afterwards it is cached.",
)
parser.add_argument(
    "--headless",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run without a viewer window and record the rollout to a video "
    "(uploaded to W&B when --wandb is set). Use on remote/headless machines. "
    "On by default; use --no-headless to open an interactive viewer.",
)

parser.add_argument(
    "--warm-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="GlobalMPPI: add the previous plan, shifted to the current time, as a KSOS candidate (default: on)",
)
parser.add_argument(
    "--exec-physics",
    choices=["planner", "fine"],
    default="planner",
    help="Execution solver settings: 'planner' = same as the MJX planner (XML: 10 ms, 1 iteration; "
    "default), 'fine' = 1 ms, 100 iterations",
)

subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one; default: globalmppi)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
subparsers.add_parser(
    "globalmppi", help="GlobalMPPI"
)
args = parser.parse_args()
if args.algorithm is None:
    args.algorithm = "globalmppi"

# Define the task (cost and dynamics)
task = PushT()

# Authenticate with Weights & Biases. The first time, prompt for an API key
# (grab yours from https://wandb.ai/authorize); it is cached in ~/.netrc so
# subsequent runs log in automatically.
if args.wandb:
    import wandb

    def _wandb_logged_in() -> bool:
        if os.environ.get("WANDB_API_KEY"):
            return True
        netrc = os.path.expanduser("~/.netrc")
        try:
            with open(netrc) as f:
                return "api.wandb.ai" in f.read()
        except OSError:
            return False

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    elif _wandb_logged_in():
        wandb.login()
    else:
        key = getpass.getpass(
            "Enter your W&B API key (from https://wandb.ai/authorize): "
        ).strip()
        wandb.login(key=key)

# Set up the controller
# Set the controller based on command-line arguments
if args.algorithm == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=256,
        noise_level=0.4,
        num_randomizations=1,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
    ) 
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=256,
        noise_level=1.0,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations = 5,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=256,
        noise_level=1.0,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
    )
elif args.algorithm == "globalmppi":
    print("Running GlobalMPPI")
    flow_sampler = None
    if args.ksos_sampling == "flow":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ksos_l2o
        from ksos_l2o.state_flow import make_flow_sampler

        flow_sampler, flow_sigma = make_flow_sampler(args.ksos_flow_dir, task, 6)
    ctrl = GlobalMPPI(
        task,
        num_samples=args.mppi_samples,
        noise_level=0.1,
        temperature=args.temperature,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations=args.mppi_iterations,
        ksos_solver=args.ksos_solver,
        ksos_sampling=args.ksos_sampling,
        ksos_num_samples=args.ksos_num_samples,
        elite_num=args.elite_num,
        ksos_flow_sampler=flow_sampler,
    )
    ctrl.ksos_num_restart = args.ksos_passes
    ctrl.ksos_warm_start = args.warm_start
    ctrl.is_lse_smoothing = args.ksos_cost == "lse"
    if flow_sampler is not None and flow_sigma is not None:
        # pass 0 smooths at the sigma the flow was trained for
        ctrl.lse_smoothing_sigma = ctrl.lse_smoothing_sigma.at[0].set(flow_sigma)
        ctrl.mppi_noise = ctrl.mppi_noise.at[0].set(args.mppi_noise)

def goal_reached(data):
    """Block (qpos x, y, angle) within the tolerances of the goal pose (0, 0, 0)."""
    angle = np.arctan2(np.sin(data.qpos[2]), np.cos(data.qpos[2]))
    return np.hypot(data.qpos[0], data.qpos[1]) < args.goal_pos_tol and abs(angle) < args.goal_angle_tol


# Run the interactive simulation
for trial_idx in args.seeds:
    # Define the model used for simulation
    mj_model = task.mj_model
    if args.exec_physics == "fine":
        mj_model.opt.timestep = 0.001
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    assert len(args.init_qpos) == 5, "--init-qpos needs 5 comma-separated values"
    if args.near_goal_init:
        import jax
        import jax.numpy as jnp
        from ksos_l2o.train_state_sampler import place_pusher

        rng = np.random.default_rng(trial_idx)
        phi = rng.uniform(-np.pi, np.pi)
        block_xy = jnp.array(rng.uniform(0.01, 0.12) * np.array([np.cos(phi), np.sin(phi)]))
        theta = rng.uniform(-0.3, 0.3)
        pusher = place_pusher(jax.random.PRNGKey(trial_idx), block_xy, theta, 0.05, 0.3)
        mj_data.qpos = [*np.asarray(block_xy), theta, *np.asarray(pusher)]
        print(f"near-goal start qpos: {np.round(mj_data.qpos, 3)}, block distance {100 * np.linalg.norm(block_xy):.1f} cm")
    elif args.random_init:
        rng = np.random.default_rng(trial_idx)
        block_xy = rng.uniform(-0.2, 0.2, 2)
        phi = rng.uniform(-np.pi, np.pi)
        pusher = block_xy + rng.uniform(0.05, 0.15) * np.array([np.cos(phi), np.sin(phi)]) - [0.0, 0.1]  # qpos is an offset from (0, 0.1)
        mj_data.qpos = [*block_xy, rng.uniform(-np.pi, np.pi), *pusher]
        print(f"random start qpos: {np.round(mj_data.qpos, 3)}")
    else:
        mj_data.qpos = args.init_qpos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    print(f"\n=== Seed {trial_idx} ===")
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=args.frequency,
        show_traces=False,
        record_video=False,
        max_traces=5,
        current_seed=trial_idx,
        max_cycles=args.cycles if args.cycles is not None else round(args.duration * args.frequency),
        goal_reached=goal_reached if args.goal_pos_tol > 0 else None,
        current_task="pushT_globalmppi",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        headless=args.headless,
        log_l2s=args.log_l2s,
    )
