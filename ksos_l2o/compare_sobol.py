"""Compare the trained flow against GlobalMPPI's *real* Sobol sampling path
(not a reimplementation) at a reduced budget: 32 samples, elite_num=32 (i.e.
no elite filtering -- the full budget counts as "elite").

Reuses the real controller's `sample_ksos_knots` directly, at iter=0 (pass
0), where its trust-region box is center=0, radius=u_max -- identical to the
fixed [-u_max, u_max] box the flow/uniform comparisons have used throughout.
Scored with the same plain rollout_cost used everywhere else in ksos_l2o/,
for an apples-to-apples comparison with the flow/uniform numbers already
reported (not the LSE-smoothed cost pass 0 also computes downstream).

    python -m ksos_l2o.compare_sobol [--sampler-dir ksos_l2o/data/sampler]
        [--budget 32] [--eval-states 32] [--seed 0]
"""
import argparse

import jax
import jax.numpy as jnp

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .context import pusht_context
from .problem import rollout_cost, sample_initial_state
from .visualize_sampler import load_sampler


def make_sobol_controller(budget):
    task = PushT()
    return GlobalMPPI(
        task, num_samples=256, noise_level=0.1, temperature=0.1,
        spline_type="cubic", plan_horizon=1.0, num_knots=6,
        ksos_sampling="sobol", ksos_num_samples=budget, elite_num=budget,
    )


def sobol_knots_for_state(controller, state_key):
    params = controller.init_params().replace(rng=state_key)
    knots, _ = controller.sample_ksos_knots(params)
    return jnp.clip(knots, controller.task.u_min, controller.task.u_max)


def main(sampler_dir="ksos_l2o/data/sampler", budget=32, eval_states=32, seed=0):
    flow, params, lb, ub, config = load_sampler(sampler_dir)
    sobol_ctrl = make_sobol_controller(budget)
    task = sobol_ctrl.task
    dim = sobol_ctrl.num_knots * task.model.nu

    eval_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), eval_states)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    def cost_for(knots):
        knots = knots.reshape(eval_states, budget, sobol_ctrl.num_knots, task.model.nu)
        return jax.vmap(lambda s, k: rollout_cost(sobol_ctrl, s, k))(states, knots)

    # Real Sobol sampling, one scramble/shift per held-out state.
    sobol_keys = jax.random.split(jax.random.PRNGKey(seed + 3000), eval_states)
    sobol_knots = jax.vmap(sobol_knots_for_state, in_axes=(None, 0))(sobol_ctrl, sobol_keys)
    sobol_costs = cost_for(sobol_knots)

    z = jax.random.normal(jax.random.PRNGKey(seed + 1001), (eval_states, budget, dim))
    flow_x, _ = flow.sample_batch(params, z, contexts, lb, ub)
    flow_costs = cost_for(flow_x)

    print(f"{'method':<28}{'mean cost':>12}{'best-of-n':>12}")
    for name, costs in (("sobol (real path)", sobol_costs), ("flow", flow_costs)):
        print(
            f"{name + f' @ {budget}, elite_num={budget}':<28}"
            f"{float(jnp.mean(costs)):>12.2f}{float(jnp.mean(jnp.min(costs, axis=1))):>12.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-dir", default="ksos_l2o/data/sampler")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--eval-states", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.sampler_dir, args.budget, args.eval_states, args.seed)
