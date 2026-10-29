"""Sanity-check the (untrained) flow: for a few random problem instances,
build the context, sample control knots from the flow, and compute the same
MPC cost as problem.py's inspect_problem.py -- but from flow samples instead
of uniform ones.

Note: the flow is freshly initialized (zero-init output layers => starts at
the identity, a plain sigmoid-squashed Gaussian), not trained -- this is a
plumbing check, not a quality check.

    python -m ksos_l2o.inspect_flow [--instances 3] [--samples 256] [--seed 42]
"""
import argparse

import jax
import jax.numpy as jnp

from .context import CONTEXT_DIM, pusht_context
from .flow import ConditionalRealNVP
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state


def main(num_instances=3, num_samples=256, seed=42):
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu
    lb = jnp.tile(task.u_min, controller.num_knots)
    ub = jnp.tile(task.u_max, controller.num_knots)

    flow = ConditionalRealNVP(dim, CONTEXT_DIM, num_layers=6, hidden_dim=64)
    params = flow.init(jax.random.PRNGKey(0))

    keys = jax.random.split(jax.random.PRNGKey(seed), num_instances)
    for i, key in enumerate(keys):
        state_key, z_key = jax.random.split(key)
        state = sample_initial_state(state_key, task)
        context = pusht_context(state.qpos, state.qvel)

        z = jax.random.normal(z_key, (num_samples, dim))
        sample_fn = jax.vmap(flow.sample, in_axes=(None, 0, None, None, None))
        x, log_q = sample_fn(params, z, context, lb, ub)
        knots = x.reshape(num_samples, controller.num_knots, task.model.nu)

        costs = rollout_cost(controller, state, knots)
        best = int(jnp.argmin(costs))

        print(f"\n=== instance {i} ===")
        print(f"qpos    = {jnp.round(state.qpos, 3)}")
        print(f"qvel    = {jnp.round(state.qvel, 3)}")
        print(f"context = {jnp.round(context, 3)}")
        print(
            f"cost over {num_samples} flow samples: "
            f"min={float(costs.min()):.3f}  mean={float(costs.mean()):.3f}  "
            f"max={float(costs.max()):.3f}  std={float(costs.std()):.3f}"
        )
        print(f"log q of each sample: mean={float(log_q.mean()):.3f}  std={float(log_q.std()):.3f}")
        print(f"best candidate's knots (#{best}):\n{jnp.round(knots[best], 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.instances, args.samples, args.seed)
