"""Dual-solver (Phase 2a) training data: near-goal PushT states, n candidates
drawn from the state flow, and their LSE-smoothed costs from the controller's
own KSOS rollout (same physics, cost and LSE sigma as pusht_nf.py). No elite
selection or filtering: each problem is the n flow samples as drawn.

    python -m ksos_l2o.generate_flow_dataset [--sampler-dir ksos_l2o/data/sampler_lse_20ms]
        [--states 8192] [--samples 32] [--out ksos_l2o/data/dual_dataset.npz]

Saved arrays: x (B, n, dim) knots, J (B, n) LSE costs (the KSOS f values),
J_plain (B, n) plain rollout costs, context (B, 11), qpos, qvel, lb, ub.
"""
import argparse

import jax
import numpy as np

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .context import pusht_context
from .state_flow import make_flow_sampler
from .train_state_sampler import sample_synthetic


def make_controller(sampler_dir, samples):
    """Same KSOS settings as examples/pusht_nf.py with --ksos-sampling flow,
    except that every candidate comes from the flow (no warm-start candidate)."""
    task = PushT()
    sampler, flow_sigma = make_flow_sampler(sampler_dir, task, 6)
    ctrl = GlobalMPPI(task, num_samples=1024, noise_level=0.1, temperature=0.001, spline_type="cubic",
                      plan_horizon=1.0, num_knots=6, iterations=1, ksos_sampling="flow",
                      ksos_num_samples=samples, elite_num=samples, ksos_flow_sampler=sampler)
    ctrl.ksos_warm_start = False
    ctrl.lse_smoothing_sigma = ctrl.lse_smoothing_sigma.at[0].set(flow_sigma)
    return ctrl


def main(sampler_dir, states, samples, out, chunk=16, seed=0):
    ctrl = make_controller(sampler_dir, samples)
    task = ctrl.task
    state_of = lambda key: sample_synthetic(key, task, 0.0, 0.0, 0.2, goal_fraction=1.0, pusher_clearance=0.05,
                                            goal_max=0.15)
    params = ctrl.init_params(seed=seed).replace(iter=0)
    params = ctrl.ksos_rollout(state_of(jax.random.PRNGKey(seed)), params).replace(iter=0)  # full-shape params

    def one(key):
        k_state, k_rng = jax.random.split(key)
        state = state_of(k_state)
        p = ctrl.ksos_rollout(state, params.replace(rng=k_rng))
        x = p.knots.reshape(samples, -1)
        return x, p.lse_cost, p.cand_cost, pusht_context(state.qpos, state.qvel), state.qpos, state.qvel

    batch = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.PRNGKey(seed), states).reshape(-1, chunk, 2)
    parts = []
    for i, k in enumerate(keys):
        parts.append([np.asarray(a) for a in batch(k)])
        if (i + 1) % 32 == 0 or i + 1 == len(keys):
            print(f"{(i + 1) * chunk}/{states} states", flush=True)
    x, J, J_plain, context, qpos, qvel = (np.concatenate(a) for a in zip(*parts))

    lb, ub = np.tile(task.u_min, ctrl.num_knots), np.tile(task.u_max, ctrl.num_knots)
    np.savez(out, x=x, J=J, J_plain=J_plain, context=context, qpos=qpos, qvel=qvel, lb=lb, ub=ub)
    print(f"saved {out}: x {x.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-dir", default="ksos_l2o/data/sampler_lse_20ms")
    p.add_argument("--states", type=int, default=8192)
    p.add_argument("--samples", type=int, default=32, help="flow samples per state")
    p.add_argument("--out", default="ksos_l2o/data/dual_dataset.npz")
    a = p.parse_args()
    main(a.sampler_dir, a.states, a.samples, a.out)
