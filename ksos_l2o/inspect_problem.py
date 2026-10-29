"""Print sampled qpos/qvel and the resulting candidate-knot costs for a few
random PushT problem instances -- a quick eyeball check of problem.py.

    python -m ksos_l2o.inspect_problem [--instances 3] [--samples 256] [--seed 42]
"""
import argparse

import jax
import jax.numpy as jnp

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .problem import rollout_cost, sample_candidate_knots, sample_initial_state


def make_controller():
    task = PushT()
    return GlobalMPPI(
        task,
        num_samples=256,
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
    )


def main(num_instances=3, num_samples=256, seed=42):
    controller = make_controller()
    keys = jax.random.split(jax.random.PRNGKey(seed), num_instances)

    for i, key in enumerate(keys):
        state_key, knot_key = jax.random.split(key)
        state = sample_initial_state(state_key, controller.task)
        knots = sample_candidate_knots(
            knot_key, controller.task, controller.num_knots, num_samples
        )
        costs = rollout_cost(controller, state, knots)
        best = int(jnp.argmin(costs))

        qpos = jnp.asarray(state.qpos)
        qvel = jnp.asarray(state.qvel)
        print(f"\n=== instance {i} ===")
        print(f"qpos  (T_x, T_y, T_z, r_x, r_y) = {jnp.round(qpos, 3)}")
        print(f"qvel  (same order, derivatives) = {jnp.round(qvel, 3)}")
        print(
            f"cost over {num_samples} candidates: "
            f"min={float(costs.min()):.3f}  mean={float(costs.mean()):.3f}  "
            f"max={float(costs.max()):.3f}  std={float(costs.std()):.3f}"
        )
        print(f"best candidate's knots (#{best}):\n{jnp.round(knots[best], 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.instances, args.samples, args.seed)
