"""Does the trained flow let us use FEWER pass-0 samples than uniform's real
budget (256, global_mppi/algs/global_mppi.py:ksos_num_samples) for the same
or better quality? No retraining needed -- the flow approximates a
per-context density, so drawing fewer samples at eval time is a free knob.

Compares, on the SAME held-out states used in training's final report:
  uniform @ budget       (fair, same-N comparison)
  flow    @ budget
  uniform @ 256          (today's real controller budget, for reference)

    python -m ksos_l2o.eval_sample_budget [--sampler-dir ksos_l2o/data/sampler]
        [--budget 32] [--reference-samples 256] [--eval-states 32] [--seed 0]
"""
import argparse

import jax
import jax.numpy as jnp

from .context import pusht_context
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state
from .visualize_sampler import load_sampler


def costs_for(controller, states, knots, eval_states, n):
    task = controller.task
    knots = knots.reshape(eval_states, n, controller.num_knots, task.model.nu)
    return jax.vmap(lambda s, k: rollout_cost(controller, s, k))(states, knots)


def main(sampler_dir="ksos_l2o/data/sampler", budget=32, reference_samples=256, eval_states=32, seed=0):
    flow, params, lb, ub, config = load_sampler(sampler_dir)
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu

    # Same held-out states as train_sampler.py's final report.
    eval_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), eval_states)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    def flow_costs(n, key):
        z = jax.random.normal(key, (eval_states, n, dim))
        x, _ = flow.sample_batch(params, z, contexts, lb, ub)
        return costs_for(controller, states, x, eval_states, n)

    def uniform_costs(n, key):
        x = jax.random.uniform(key, (eval_states, n, dim), minval=lb, maxval=ub)
        return costs_for(controller, states, x, eval_states, n)

    results = {
        f"uniform @ {budget}": uniform_costs(budget, jax.random.PRNGKey(seed + 2000)),
        f"flow @ {budget}": flow_costs(budget, jax.random.PRNGKey(seed + 2001)),
        f"uniform @ {reference_samples} (today's real budget)": uniform_costs(
            reference_samples, jax.random.PRNGKey(seed + 2002)
        ),
    }

    print(f"{'method':<34}{'mean cost':>12}{'best-of-n':>12}")
    for name, costs in results.items():
        print(f"{name:<34}{float(jnp.mean(costs)):>12.2f}{float(jnp.mean(jnp.min(costs, axis=1))):>12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-dir", default="ksos_l2o/data/sampler")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--reference-samples", type=int, default=256)
    parser.add_argument("--eval-states", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.sampler_dir, args.budget, args.reference_samples, args.eval_states, args.seed)
