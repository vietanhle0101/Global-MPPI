import time
from datetime import datetime
from typing import Sequence
import os

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from global_mppi.alg_base import SamplingBasedController
from global_mppi import ROOT
from global_mppi.utils.video import VideoRecorder

# Timestamp for this process, computed once at import so every seed of a single
# run shares the same log subfolder: logs/<task>_<algorithm>/<RUN_TIMESTAMP>/.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from scipy.optimize import minimize

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


class _HeadlessViewer:
    """Drop-in stand-in for the MuJoCo passive viewer for headless runs.

    Exposes just the attributes/methods the simulation loop touches
    (``is_running``, ``sync``, ``cam``, ``user_scn``) so the exact same loop can
    run on a machine with no display. Frames are rendered offscreen instead of
    being shown in a window.
    """

    def __init__(self, mj_model: mujoco.MjModel, max_cycles: int) -> None:
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(mj_model, self.cam)
        self.user_scn = mujoco.MjvScene(mj_model, maxgeom=10000)
        self._running = True

    def __enter__(self) -> "_HeadlessViewer":
        return self

    def __exit__(self, *exc) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def sync(self) -> None:  # no window to sync to
        pass

def rbf_kernel(X1, X2=None, sigma=1.0):
    X1 = np.atleast_2d(X1)
    if X2 is None:
        X2 = X1
    else:
        X2 = np.atleast_2d(X2)

    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    dists = X1_sq - 2 * X1 @ X2.T + X2_sq
    return np.exp(-0.5 * dists / sigma**2)

def neg_lml(log_sigma, X, y):
    lam = 1e-6
    sigma = np.exp(log_sigma)
    K = rbf_kernel(X, X, sigma)
    Ky = K + lam * np.eye(len(X))

    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        print("Cholesky failed")
        return 1e10

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    logdet = 2 * np.sum(np.log(np.diag(L)))
    n = len(X)
    return float(0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2*np.pi))

def apply_mppi_update(
    policy_params,
    rollouts,
    temperature: float = 0.1,
    eps: float = 1e-5,
):
    """Update the MPPI proposal distribution given rollout statistics."""
    costs = jnp.sum(rollouts.costs, axis=1)  # total cost per rollout
    centered_costs = costs - jnp.min(costs)
    weights = jnp.exp(-centered_costs / temperature)
    weights = weights / (jnp.sum(weights) + eps)
    mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
    return policy_params.replace(mean=jax.device_put(mean))


def apply_cem_update(
    policy_params,
    rollouts,
    num_elites: int = 3,
    min_std: float = 0.1,
):
    """Update the proposal distribution using a CEM-style elite fit."""
    costs = jnp.sum(rollouts.costs, axis=1)
    elite_ids = jnp.argsort(costs)[:num_elites]
    elite_knots = rollouts.knots[elite_ids]
    mean = jnp.mean(elite_knots, axis=0)
    cov = jnp.maximum(jnp.std(elite_knots, axis=0), min_std)
    return policy_params.replace(
        mean=jax.device_put(mean),
        cov=jax.device_put(cov),
    )


def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
    current_seed: int = 0,
    max_cycles: int = 100,
    current_task: str = "default",
    use_wandb: bool = False,
    wandb_project: str = "global-mppi",
    wandb_entity: str = None,
    headless: bool = False,
    log_l2s: bool = False,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        record_video: Whether to record a video of the simulation.
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots, seed=current_seed)
    jit_interp_func = jax.jit(controller.interp_func)
    
    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()

    if controller.ctrl_name == "globalmppi":
        jit_ksos_rollout = jax.jit(controller.ksos_rollout)
        policy_params = jit_ksos_rollout(mjx_data, policy_params)
        policy_params = jit_ksos_rollout(mjx_data, policy_params)

        jit_optimize = jax.jit(controller.optimize)
        policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)
        policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)
    else:

        jit_optimize = jax.jit(controller.optimize)

        policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)
        policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = max_traces

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # In headless mode there is no window: always record so there is something
    # to look at (and to upload to W&B), and skip on-screen-only traces.
    if headless:
        record_video = True
        show_traces = False

    # Per-run log directory at the repo root:
    # logs/<task>_<algorithm>/<timestamp>/. All outputs of this run (history
    # arrays and the rollout video) are collected here. RUN_TIMESTAMP is fixed
    # at import, so every seed of one invocation shares the same folder. ROOT is
    # the package dir, so its parent is the repo root.
    repo_root = os.path.dirname(ROOT)
    run_log_dir = os.path.join(
        repo_root,
        "logs",
        f"{current_task}_{controller.ctrl_name}",
        RUN_TIMESTAMP,
    )
    os.makedirs(run_log_dir, exist_ok=True)

    # Opt-in L2S (learned-sampler) raw data directory: one .npz per
    # (seed, cycle) with every KSOS pass of that cycle, used offline to train
    # the flow sampler. Off by default; does not affect the RNG stream.
    if log_l2s:
        l2s_log_dir = os.path.join(
            repo_root,
            "data",
            "l2s",
            "raw",
            f"{current_task}_{controller.ctrl_name}",
            RUN_TIMESTAMP,
        )
        os.makedirs(l2s_log_dir, exist_ok=True)

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=run_log_dir,
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Start the simulation. Use a headless stand-in (no window) when requested,
    # otherwise launch the interactive passive viewer.
    viewer_ctx = (
        _HeadlessViewer(mj_model, max_cycles)
        if headless
        else mujoco.viewer.launch_passive(mj_model, mj_data)
    )
    with viewer_ctx as viewer:
        # Tracking for min cost across cycles
        min_cost_history = []
        cycle_count = 0
        plotted_min_cost = False
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

            # for best trace
            for j in range(i+1, i+controller.ctrl_steps+1):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[j+1],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 0.0, 0.0, 0.3]),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        robot_pos_history = []
        best_cost_history = []
        best_sample_history = []
        error_history = []
        pre_best_cost = np.inf
        best_sample_cost = np.inf

        # Optionally start a Weights & Biases run to log best_cost_history
        wandb_run = None
        if use_wandb:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{current_task}_{controller.ctrl_name}_seed{current_seed}_{RUN_TIMESTAMP}",
                group=f"{current_task}_{controller.ctrl_name}",
                reinit=True,
                config={
                    "algorithm": controller.ctrl_name,
                    "task": current_task,
                    "seed": current_seed,
                    "num_samples": getattr(controller, "num_samples", None),
                    "plan_horizon": controller.plan_horizon,
                    "num_knots": controller.num_knots,
                    "frequency": actual_frequency,
                    "max_cycles": max_cycles,
                },
            )

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )
            robot_pos_history.append(np.array(mj_data.qpos))

            # Do a replanning step
            plan_start = time.time()

            # run ksos update to initilize the mean
            if controller.ctrl_name == "globalmppi":
                enable_auto_calib = True          # run ksos update
                
                log_dir = "test/log"
                os.makedirs(log_dir, exist_ok=True)
                timing_file = os.path.join(log_dir, "globalmppi_timing.txt")
                
                if log_l2s:
                    l2s_knots, l2s_lse_cost, l2s_cost_summed = [], [], []
                    l2s_lb, l2s_ub, l2s_elite_indices = [], [], []
                    l2s_sigma, l2s_ksos_mean, l2s_ksos_result_cost = [], [], []

                if cycle_count % 1 == 0:          # 5 for pushT with ksos
                    total_ksos_rollout_time = 0.0
                    total_calibration_time = 0.0
                    total_ksos_solver_time = 0.0
                    total_mppi_time = 0.0
                    total_all_time = 0.0
                    for iter in range(controller.ksos_num_restart):
                        iter_start_time = time.perf_counter()
                        # ============================================================
                        # 1. Time KSOS rollout
                        # ============================================================
                        ksos_rollout_start = time.perf_counter()

                        policy_params = policy_params.replace(iter=iter)

                        if log_l2s:
                            # Mirror sample_ksos_knots' center/radius exactly,
                            # for logging only (reads params, no RNG use).
                            center = (
                                np.zeros_like(np.asarray(policy_params.ksos_mean))
                                if iter == 0
                                else np.asarray(policy_params.ksos_mean)
                            )
                            radius = (controller.decay_rate**iter) * np.asarray(
                                controller.task.u_max
                            )
                            l2s_lb.append(center - radius)
                            l2s_ub.append(center + radius)

                        policy_params = jit_ksos_rollout(mjx_data, policy_params)
                        policy_params = jax.block_until_ready((policy_params))

                        ksos_rollout_time = time.perf_counter() - ksos_rollout_start

                        if log_l2s:
                            # Elite ranking must be captured here: solve_ksos
                            # below overwrites params.lse_cost with the
                            # filtered elite subset.
                            l2s_knots.append(np.asarray(policy_params.knots))
                            l2s_lse_cost.append(np.asarray(policy_params.lse_cost))
                            l2s_cost_summed.append(
                                np.asarray(policy_params.cost).sum(axis=-1)
                            )
                            l2s_elite_indices.append(
                                np.argsort(np.asarray(policy_params.lse_cost))[
                                    : controller.elite_num
                                ]
                            )

                        # ============================================================
                        # 2. Time GP sigma auto-calibration
                        # ============================================================
                        calibration_start = time.perf_counter()
                        if enable_auto_calib:
                            n_samples = policy_params.lse_cost.shape[0]
                            X = policy_params.knots.reshape(n_samples, -1)
                            y = policy_params.lse_cost

                            # ---- standardize X ----
                            X_mean = X.mean(axis=0, keepdims=True)
                            X_std  = X.std(axis=0, keepdims=True) + 1e-8   # avoid divide-by-zero
                            Xn = (X - X_mean) / X_std

                            # ---- demean y ----
                            y_mean = y.mean()
                            yn = y - y_mean
                            # ---- grid search for sigma init ----
                            sigma_grid = np.linspace(0.01, 30.0, 10)   # 10 numbers from 0.01 to 30
                            log_sigma_grid = np.log(sigma_grid)

                            vals = np.array([neg_lml(ls, Xn, yn) for ls in log_sigma_grid])
                            best_i = int(np.argmin(vals))
                            log_sigma0 = log_sigma_grid[best_i]
                            sigma0 = float(np.exp(log_sigma0))

                            # ---- refine with L-BFGS-B ----
                            res = minimize(
                                neg_lml,
                                x0=np.array([log_sigma0]),     # pass as 1D array is safer
                                args=(Xn, yn),
                                method="L-BFGS-B",
                                bounds=[(np.log(1e-2), np.log(30.0))],
                            )
                            controller.ksos_sigma = np.exp(res.x[0])
                            print(f"Auto-calibrated sigma: {controller.ksos_sigma}")
                        calibration_time = time.perf_counter() - calibration_start

                        if log_l2s:
                            l2s_sigma.append(float(controller.ksos_sigma))

                        # ============================================================
                        # 3. Time the KSOS solve
                        # ============================================================
                        ksos_solver_start = time.perf_counter()
                        policy_params = controller.solve_ksos(policy_params)
                        ksos_solver_time = time.perf_counter() - ksos_solver_start

                        if log_l2s:
                            l2s_ksos_mean.append(np.asarray(policy_params.ksos_mean))
                            l2s_ksos_result_cost.append(
                                float(np.asarray(policy_params.ksos_result_cost))
                            )

                        policy_params = policy_params.replace(mean=policy_params.ksos_mean)

                        # ============================================================
                        # 4. Time the MPPI local-search refinement
                        # ============================================================
                        mppi_start = time.perf_counter()
                        policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)
                        policy_params = jax.block_until_ready(policy_params)
                        mppi_time = time.perf_counter() - mppi_start

                        iter_total_time = time.perf_counter() - iter_start_time

                        total_ksos_rollout_time += ksos_rollout_time
                        total_calibration_time += calibration_time
                        total_ksos_solver_time += ksos_solver_time
                        total_mppi_time += mppi_time

                        total_all_time += iter_total_time

                    if log_l2s:
                        num_passes = controller.ksos_num_restart
                        l2s_path = os.path.join(
                            l2s_log_dir,
                            f"seed{current_seed}_cycle{int(cycle_count)}.npz",
                        )
                        np.savez(
                            l2s_path,
                            episode=current_seed,
                            cycle=int(cycle_count),
                            pass_index=np.arange(num_passes),
                            qpos=np.array(mj_data.qpos),
                            qvel=np.array(mj_data.qvel),
                            mocap_pos=np.array(mj_data.mocap_pos),
                            mocap_quat=np.array(mj_data.mocap_quat),
                            time=mj_data.time,
                            knots=np.stack(l2s_knots),
                            lse_cost=np.stack(l2s_lse_cost),
                            cost_summed=np.stack(l2s_cost_summed),
                            lb=np.stack(l2s_lb),
                            ub=np.stack(l2s_ub),
                            elite_indices=np.stack(l2s_elite_indices),
                            sigma=np.array(l2s_sigma),
                            ksos_mean=np.stack(l2s_ksos_mean),
                            ksos_result_cost=np.array(l2s_ksos_result_cost),
                        )
            else:
                policy_params, rollouts, rollouts_best = jit_optimize(mjx_data, policy_params)
            
            # ============================================================
            # Save total timing over all restarts / iterations
            # (only globalmppi populates these timing variables)
            # ============================================================
            if controller.ctrl_name == "globalmppi":
                with open(timing_file, "a") as f:
                    f.write(
                        f"cycle={cycle_count}, "
                        f"num_restart={controller.ksos_num_restart}, "
                        f"total_ksos_rollout_time={total_ksos_rollout_time:.6f}, "
                        f"total_calibration_time={total_calibration_time:.6f}, "
                        f"total_ksos_solver_time={total_ksos_solver_time:.6f}, "
                        f"total_mppi_time={total_mppi_time:.6f}, "
                        f"total_all_time={total_all_time:.6f}, "
                        f"avg_ksos_rollout_time={total_ksos_rollout_time / controller.ksos_num_restart:.6f}, "
                        f"avg_calibration_time={total_calibration_time / controller.ksos_num_restart:.6f}, "
                        f"avg_ksos_solver_time={total_ksos_solver_time / controller.ksos_num_restart:.6f}, "
                        f"avg_mppi_time={total_mppi_time / controller.ksos_num_restart:.6f}, "
                        f"avg_all_time={total_all_time / controller.ksos_num_restart:.6f}, "
                        f"sigma={getattr(controller, 'ksos_sigma', None)}\n"
                    )
                print(
                    f"[Timing total] cycle={cycle_count}, "
                    f"num_restart={controller.ksos_num_restart}, "
                    f"total_ksos_rollout={total_ksos_rollout_time:.3f}s, "
                    f"total_calibration={total_calibration_time:.3f}s, "
                    f"total_ksos_solver={total_ksos_solver_time:.3f}s, "
                    f"total_mppi={total_mppi_time:.3f}s, "
                    f"total_all={total_all_time:.3f}s"
                )
            # data logging: rollouts_best (from jit_optimize) is already the
            # rollout of params.mean, so its summed cost is the "best cost" -
            # no need to re-roll-out via get_cost_with_best_samples.
            best_cost = rollouts_best.costs.sum()
            if controller.ctrl_name == "globalmppi":
                print(f"\nCycle {cycle_count}: Best cost: {best_cost}")
                best_cost_history.append(best_cost)
                error_history.append(best_sample_cost - best_cost)
            else:
                costs = np.sum(rollouts.costs, axis=1)  # sum over time steps
                print(f"\nCycle {cycle_count}: Best cost: {best_cost}, Best sample: {np.min(costs)}")
                best_cost_history.append(best_cost)
                error_history.append(np.min(costs) - best_cost)

            # Stream the latest best cost to Weights & Biases. best_cost may be a
            # size-1 array (shape (1,)), so flatten to a scalar first.
            if wandb_run is not None:
                wandb.log(
                    {"best_cost": float(np.asarray(best_cost).ravel()[0])},
                    step=int(cycle_count),
                )

            # cost evaulation
            if cycle_count % 10 == 0 or cycle_count >= max_cycles:
                print(f"Cost for optimal action (cycle {cycle_count}): {best_cost_history}")
                    # If we collected any cost history during the interactive run, save a plot.
                if (
                    'best_cost_history' in locals()
                    and len(best_cost_history) > 0
                ):
                    logs_dir = run_log_dir
                    out_cost_npy = os.path.join(logs_dir, f"{current_task}_{controller.ctrl_name}_best_cost_history_seed{current_seed}.npy")
                    out_sample_npy = os.path.join(logs_dir, f"{current_task}_{controller.ctrl_name}_best_sample_history_seed{current_seed}.npy")
                    out_pos_npy = os.path.join(logs_dir, f"{current_task}_{controller.ctrl_name}_robot_pos_history_seed{current_seed}.npy")
                    # Save arrays separately for downstream analysis
                    np.save(
                        out_pos_npy,
                        np.asarray(robot_pos_history, dtype=float),
                    )
                    np.save(
                        out_cost_npy,
                        np.asarray(best_cost_history, dtype=float),
                    )
                if cycle_count >= max_cycles:
                    print(policy_params.mean[None, ...])
                    break
                
            plan_time = time.time() - plan_start
            print(f"\nCycle {cycle_count}: plan time: {plan_time:.4f}s"),
            cycle_count += 1.0
            
            # Visualize the rollouts
            if show_traces:
                ii = 0
                if controller.ctrl_name == "globalmppi":
                    for k in range(num_trace_sites):
                        for i in range(num_traces):
                            for j in range(controller.ctrl_steps):
                                mujoco.mjv_connector(
                                    viewer.user_scn.geoms[ii],
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    trace_width,
                                    policy_params.ksos_trace[i, j, k],
                                    policy_params.ksos_trace[i, j + 1, k],
                                )
                                ii += 1
                    
                    # trace the best  
                    for j in range(controller.ctrl_steps):      
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[ii],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            trace_width,
                            policy_params.best_trace[1, j, 1],
                            policy_params.best_trace[1, j + 1, 1],
                        )
                        ii += 1
                    
                else:
                    for k in range(num_trace_sites):
                        for i in range(num_traces):
                            for j in range(controller.ctrl_steps):
                                mujoco.mjv_connector(
                                    viewer.user_scn.geoms[ii],
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    trace_width,
                                    rollouts.trace_sites[i, j, k],
                                    rollouts.trace_sites[i, j + 1, k],
                                )
                                ii += 1
                                    
                    # trace the best  
                    for j in range(controller.ctrl_steps):      
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[ii],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            trace_width,
                            policy_params.best_trace[1, j, 1],
                            policy_params.best_trace[1, j + 1, 1],
                        )
                        ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)
            
            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    # Preserve the last printout
    print(f"finish the seed {current_seed} interactive simulation for controller {controller.ctrl_name}.")

    # Finalize the video first, so the finished .mp4 can be attached to the run.
    if record_video and recorder is not None:
        recorder.stop()

    # Finalize the Weights & Biases run: log best_cost_history and the video.
    if wandb_run is not None:
        if len(best_cost_history) > 0:
            # Entries may be size-1 arrays; flatten to a 1-D array of scalars.
            hist = np.asarray(best_cost_history, dtype=float).ravel()
            wandb_run.summary["final_best_cost"] = float(hist[-1])
            wandb_run.summary["min_best_cost"] = float(hist.min())
            table = wandb.Table(
                data=[[i, float(c)] for i, c in enumerate(hist)],
                columns=["cycle", "best_cost"],
            )
            wandb_run.log(
                {
                    "best_cost_history": wandb.plot.line(
                        table, "cycle", "best_cost", title="Best cost history"
                    )
                }
            )
        if (
            record_video
            and recorder is not None
            and recorder.video_path is not None
            and os.path.exists(recorder.video_path)
        ):
            wandb_run.log(
                {
                    "rollout_video": wandb.Video(
                        recorder.video_path,
                        fps=int(actual_frequency),
                        format="mp4",
                    )
                }
            )
        wandb_run.finish()
