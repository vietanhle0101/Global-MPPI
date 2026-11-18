"""GlobalMPPI in closed loop with a separate plant, for M20BoxPush (global_mppi/tasks/m20push.py).

`run_interactive` steps the planner's own model; here the plant is another simulator (the CPU MuJoCo plant in
global_mppi/models/m20push/build.py, later the real robot), so `Loop` takes its state from the plant's poses and
velocities, as an overhead tracker would give them, and hands back commands to play until the next replan.

Two changes from `run_interactive`'s cycle, for replanning every 0.5 s rather than every 20 ms:

- the previous plan is shifted to now holding its last knot. The spline's own shift extends the cubic past its
  last knot, which is harmless at 20 ms replanning against 0.2 s knot spacing; at 0.5 s the extrapolated tail
  ran to the command limits and drove a wheel into the box, so the previous plan scored 150,000 to 1,500,000
  when it had cost about 540, and was never kept;
- commitment (`Settings.commit`): the previous plan stays unless the new one is that fraction cheaper. Each
  cycle's KSOS candidates are fresh uniform samples over the whole command range, very different plans cost
  about the same, and a slow robot that can't move sideways goes nowhere if it follows a different one each cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI, SteeredGlobalMPPI
from global_mppi.tasks.m20push import M20BoxPush


@dataclass(frozen=True)
class Settings:
    """Controller settings: examples/pusht_nf.py's globalmppi defaults, stretched for a slow robot."""
    horizon: float = 4.0        # s; PushT uses 1 s for a pusher that crosses its workspace in under a second
    knots: int = 8
    mppi_samples: int = 512     # PushT uses 1024; halved for a laptop CPU
    temperature: float = 0.001
    ksos_samples: int = 32
    elite: int = 32
    passes: int = 1
    ksos_cost: str = "plain"    # "lse" scores each candidate by 100 noisy rollouts, as PushT's default
    ksos_sampling: str = "uniform"
    ksos_solver: str = "newton-rs"
    warm_start: bool = True
    keep_check: bool = True
    noise_scale: float = 1.0    # scales the per-pass LSE sigma and MPPI noise [0.3, 0.2, 0.1, 0.05, 0.001]
    commit: float = 0.03        # keep the previous plan unless the new one is this fraction cheaper
    replan: float = 0.5         # s between replans (PushT replans every 20 ms)


# The push phase (robot already at its contact): a shorter plan over forward speed and relative heading.
PUSH_SETTINGS = Settings(horizon=3.0, knots=4)


def make_controller(task: M20BoxPush, settings: Settings = Settings(), seed: int = 0, flow_sampler=None,
                    learned_solver=None) -> GlobalMPPI:
    """GlobalMPPI, or SteeredGlobalMPPI for the push phase (`task.weights.steer_gain`)."""
    kind = SteeredGlobalMPPI if task.weights.steer_gain else GlobalMPPI
    ctrl = kind(task, num_samples=settings.mppi_samples, noise_level=0.1, temperature=settings.temperature,
                spline_type="cubic", plan_horizon=settings.horizon, num_knots=settings.knots, iterations=1,
                ksos_solver=settings.ksos_solver, ksos_sampling=settings.ksos_sampling,
                ksos_num_samples=settings.ksos_samples, elite_num=settings.elite, seed=seed,
                ksos_flow_sampler=flow_sampler, ksos_learned_solver=learned_solver)
    ctrl.ksos_num_restart = settings.passes
    ctrl.ksos_warm_start = settings.warm_start
    ctrl.mppi_keep_check = settings.keep_check
    ctrl.is_lse_smoothing = settings.ksos_cost == "lse"
    ctrl.lse_smoothing_sigma = ctrl.lse_smoothing_sigma * settings.noise_scale
    ctrl.mppi_noise = ctrl.lse_smoothing_sigma
    return ctrl


def fit_kernel_width(ctrl, params) -> None:
    """The GP marginal-likelihood fit of the KSOS kernel width, as in simulation/deterministic.run_interactive."""
    from scipy.optimize import minimize

    from global_mppi.simulation.deterministic import neg_lml

    n = params.lse_cost.shape[0]
    x = np.asarray(params.knots).reshape(n, -1)
    y = np.asarray(params.lse_cost)
    xn = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    yn = y - y.mean()
    grid = np.log(np.linspace(0.01, 30.0, 10))
    start = grid[int(np.argmin([neg_lml(g, xn, yn) for g in grid]))]
    result = minimize(neg_lml, x0=np.array([start]), args=(xn, yn), method="L-BFGS-B",
                      bounds=[(np.log(1e-2), np.log(30.0))])
    ctrl.ksos_sigma = float(np.exp(result.x[0]))


class Loop:
    """GlobalMPPI's cycle (run_interactive's for 'globalmppi'), fed from a plant.

    The plant needs robot_pose(), box_pose() (x, y, yaw in the goal's frame), robot_velocity(), box_velocity()
    (world frame), `command` (the actuator command it is executing) and `data.time`.
    """

    def __init__(self, task: M20BoxPush, settings: Settings = Settings(), seed: int = 0, **controller):
        self.task = task
        self.ctrl = make_controller(task, settings, seed, **controller)
        self.settings, self.seed = settings, seed
        self.params = self.ctrl.init_params(seed=seed)
        model = self.task.mj_model
        self.state = mjx.put_data(model, mujoco.MjData(model))
        self.contact = np.zeros(3)
        self.ksos_rollout = jax.jit(self.ctrl.ksos_rollout)
        self.optimize = jax.jit(self.ctrl.safeguarded_optimize)
        self.interp = jax.jit(self.ctrl.interp_func)
        self.cost_of = jax.jit(lambda state, tk, knots: jnp.sum(self.ctrl.rollout(state, tk, knots[None]).costs))
        self.switches = 0

    def sync(self, plant) -> None:
        qpos = np.array([*plant.robot_pose(), *plant.box_pose()])
        qvel = np.array([*plant.robot_velocity(), *plant.box_velocity()])
        mocap = np.array([[0.0, 0.0, 0.001], self.contact])
        self.state = self.state.replace(qpos=jnp.array(qpos, jnp.float32), qvel=jnp.array(qvel, jnp.float32),
                                        ctrl=jnp.array(plant.command, jnp.float32),
                                        time=jnp.array(plant.data.time, jnp.float32),
                                        mocap_pos=jnp.array(mocap, jnp.float32))

    def restart(self, time: float, command=(0.0, 0.0, 0.0), contact=None) -> None:
        """Start a fresh plan at `time`, holding `command`, pushing at `contact` (box-frame x, y, normal angle)."""
        knots = jnp.tile(jnp.array(command, jnp.float32), (self.ctrl.num_knots, 1))
        params = self.ctrl.init_params(initial_knots=knots, seed=self.seed)
        tk = jnp.linspace(0.0, self.ctrl.plan_horizon, self.ctrl.num_knots) + time
        self.params = params.replace(tk=tk)
        if contact is not None:
            self.contact = np.asarray(contact, float)

    def plan(self) -> float:
        """One replanning cycle; returns the cost of the plan it keeps."""
        tk = jnp.linspace(0.0, self.ctrl.plan_horizon, self.ctrl.num_knots) + self.state.time
        held = self.interp(jnp.clip(tk, self.params.tk[0], self.params.tk[-1]), self.params.tk,
                           self.params.mean[None, ...])[0]
        held = jnp.clip(held, self.task.u_min, self.task.u_max)
        self.params = self.params.replace(tk=tk, mean=held)
        cost = np.nan
        for i in range(self.ctrl.ksos_num_restart):
            params = self.ksos_rollout(self.state, self.params.replace(iter=i))
            if self.ctrl.ksos_auto_calib:
                fit_kernel_width(self.ctrl, params)
            params = self.ctrl.solve_ksos(params, self.state)
            params = params.replace(mean=params.ksos_mean)
            params, _, cost, _ = self.optimize(self.state, params)
            self.params = params
        cost = float(cost)
        if self.settings.commit > 0:
            held_cost = float(self.cost_of(self.state, self.params.tk, held))
            if cost > (1 - self.settings.commit) * held_cost:
                self.params = self.params.replace(mean=held)
                return held_cost
            self.switches += 1
        return cost

    def commands(self, t0: float, dt: float, n: int) -> np.ndarray:
        """The plan's controls at t0, t0 + dt, ..., for the plant to play until the next replan."""
        tq = jnp.arange(n) * dt + t0
        return np.asarray(self.interp(tq, self.params.tk, self.params.mean[None, ...]))[0]
