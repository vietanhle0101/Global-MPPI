"""Dual-solver training data: random PushT states, n candidates drawn from the
trained flow (same context), scored with the plain rollout cost.

    python -m ksos_l2o.generate_flow_dataset [--sampler-dir ksos_l2o/data/sampler]
        [--states 8192] [--samples 32] [--out ksos_l2o/data/dual_dataset.npz]
"""
import argparse

import jax
import numpy as np

from .context import pusht_context
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state
from .visualize_sampler import load_sampler


def main(sampler_dir, states, samples, out, chunk=256, seed=0):
    flow, params, lb, ub, _ = load_sampler(sampler_dir)
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu

    @jax.jit
    def batch(key):
        k_state, k_z = jax.random.split(key)
        state_keys = jax.random.split(k_state, chunk)
        s = jax.vmap(sample_initial_state, in_axes=(0, None))(state_keys, task)
        ctx = jax.vmap(pusht_context)(s.qpos, s.qvel)
        z = jax.random.normal(k_z, (chunk, samples, dim))
        x, _ = flow.sample_batch(params, z, ctx, lb, ub)
        knots = x.reshape(chunk, samples, controller.num_knots, task.model.nu)
        costs = jax.vmap(lambda st, k: rollout_cost(controller, st, k))(s, knots)
        return ctx, x, costs

    ctxs, xs, js = [], [], []
    for i, key in enumerate(jax.random.split(jax.random.PRNGKey(seed), states // chunk)):
        c, x, j = batch(key)
        ctxs.append(np.asarray(c)); xs.append(np.asarray(x)); js.append(np.asarray(j))
        print(f"chunk {i + 1}/{states // chunk}", flush=True)
    np.savez(out, context=np.concatenate(ctxs), x=np.concatenate(xs), J=np.concatenate(js),
             lb=np.asarray(lb), ub=np.asarray(ub))
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-dir", default="ksos_l2o/data/sampler")
    p.add_argument("--states", type=int, default=8192)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--out", default="ksos_l2o/data/dual_dataset.npz")
    a = p.parse_args()
    main(a.sampler_dir, a.states, a.samples, a.out)
