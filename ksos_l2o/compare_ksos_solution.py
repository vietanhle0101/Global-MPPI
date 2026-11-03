"""One KernelSOS solve (no MPC) at the PushT benchmark start state with Sobol vs flow candidates.

    python -m ksos_l2o.compare_ksos_solution [--sampler-dir ksos_l2o/data/sampler_state_lse]
        [--num-samples 32] [--repeats 20] [--sigma-lse 0.3]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from global_mppi.simulation.deterministic import neg_lml
from global_mppi.tasks.pusht import PushT

from .compare_samples import lse_cost, make_controller, make_state
from .context import pusht_context
from .problem import rollout_cost
from .state_flow import load_state_sampler, sample as flow_sample
from .train_state_sampler import START_QPOS


def calibrate_sigma(ctrl, params):
    """The kernel-width auto-calibration `run_interactive` does before every solve."""
    X = np.asarray(params.knots.reshape(len(params.knots), -1))
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    y = np.asarray(params.lse_cost)
    y = y - y.mean()
    grid = np.log(np.linspace(0.01, 30.0, 10))
    x0 = grid[int(np.argmin([neg_lml(g, X, y) for g in grid]))]
    res = minimize(neg_lml, x0=np.array([x0]), args=(X, y), method="L-BFGS-B", bounds=[(np.log(1e-2), np.log(30.0))])
    ctrl.ksos_sigma = float(np.exp(res.x[0]))


def flow_rollout(ctrl, flow, flow_params, n, state, params):
    """Replaces `ksos_rollout`: n flow candidates, then the controller's LSE rollout."""
    rng, key = jax.random.split(params.rng)
    lb, ub = jnp.tile(ctrl.task.u_min, ctrl.num_knots), jnp.tile(ctrl.task.u_max, ctrl.num_knots)
    z = jax.random.normal(key, (n, lb.size))
    knots = flow_sample(flow, flow_params, z, pusht_context(state.qpos, state.qvel), lb, ub)
    knots = jnp.clip(knots.reshape(n, ctrl.num_knots, ctrl.task.model.nu), ctrl.task.u_min, ctrl.task.u_max)
    return ctrl.lse_smoothing_rollout(state, params.tk, knots, params.replace(rng=rng))


def solve(ctrl, rollout, state, key):
    """Sample, calibrate sigma, solve_ksos. -> solution knots, whether the solve failed."""
    params = rollout(state, ctrl.init_params().replace(rng=key))
    calibrate_sigma(ctrl, params)
    solved = ctrl.solve_ksos(params)
    return np.asarray(solved.ksos_mean), solved is params  # solve_ksos returns its input on failure


def main(sampler_dir, n, repeats, sigma_lse, seed):
    flow, flow_params, config = load_state_sampler(sampler_dir)
    sigma_lse = sigma_lse if sigma_lse is not None else (config["sigma_lse"] or 0.3)
    task = PushT()
    state = make_state(task, START_QPOS, np.zeros(5))
    ctrl = make_controller(task, n, sampling="sobol")
    ctrl.elite_num = n
    ctrl.lse_smoothing_sigma = jnp.full((5,), sigma_lse, dtype=jnp.float32)
    arms = {f"sobol{n}": jax.jit(ctrl.ksos_rollout),
            f"flow{n}": jax.jit(lambda s, p: flow_rollout(ctrl, flow, flow_params, n, s, p))}
    print(f"start state {START_QPOS}, {repeats} repeats, {n} samples, sampler {sampler_dir}, sigma_lse={sigma_lse}\n")

    def score(knots):  # (R, num_knots, nu) -> plain J, LSE cost
        knots = jnp.clip(jnp.asarray(knots, dtype=jnp.float32), task.u_min, task.u_max)
        return np.asarray(rollout_cost(ctrl, state, knots)), np.atleast_1d(np.asarray(lse_cost(ctrl, state, knots, jax.random.PRNGKey(seed))))

    print(f"{'arm':<14}{'plain J mean +/- std':>24}{'min':>9}{'LSE mean':>11}{'failed':>8}{'s/solve':>9}")
    for name, rollout in arms.items():
        start = time.perf_counter()
        runs = [solve(ctrl, rollout, state, k) for k in jax.random.split(jax.random.PRNGKey(seed), repeats)]
        elapsed = (time.perf_counter() - start) / repeats
        plain, lse = score(np.stack([r[0] for r in runs]))
        print(f"{name:<14}{plain.mean():>15.2f} +/- {plain.std(ddof=1) if repeats > 1 else 0:>6.2f}{plain.min():>9.2f}"
              f"{lse.mean():>11.2f}{sum(r[1] for r in runs):>8d}{elapsed:>9.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-dir", default="ksos_l2o/data/sampler_state_lse")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--sigma-lse", type=float, default=None, help="default: the flow's trained value")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.sampler_dir, a.num_samples, a.repeats, a.sigma_lse, a.seed)
