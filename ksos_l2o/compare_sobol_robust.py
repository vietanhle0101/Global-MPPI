"""Same flow-vs-real-Sobol comparison as compare_sobol.py, but with (1) many
more held-out states per trial and (2) several independent seeds pooled
together, to get a statistically trustworthy answer instead of the
marginal one from a single seed x 32 states.

    python -m ksos_l2o.compare_sobol_robust [--sampler-dir ksos_l2o/data/sampler]
        [--budget 32] [--eval-states 256] [--seeds 0 1 2 3 4]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from .compare_sobol import make_sobol_controller, sobol_knots_for_state
from .context import pusht_context
from .problem import rollout_cost, sample_initial_state
from .visualize_sampler import load_sampler


def one_seed(flow, params, lb, ub, sobol_ctrl, budget, eval_states, seed):
    task = sobol_ctrl.task
    dim = sobol_ctrl.num_knots * task.model.nu

    eval_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), eval_states)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    def cost_for(knots):
        knots = knots.reshape(eval_states, budget, sobol_ctrl.num_knots, task.model.nu)
        return jax.vmap(lambda s, k: rollout_cost(sobol_ctrl, s, k))(states, knots)

    sobol_keys = jax.random.split(jax.random.PRNGKey(seed + 3000), eval_states)
    sobol_knots = jax.vmap(sobol_knots_for_state, in_axes=(None, 0))(sobol_ctrl, sobol_keys)
    sobol_costs = cost_for(sobol_knots)

    z = jax.random.normal(jax.random.PRNGKey(seed + 1001), (eval_states, budget, dim))
    flow_x, _ = flow.sample_batch(params, z, contexts, lb, ub)
    flow_costs = cost_for(flow_x)

    return np.asarray(sobol_costs), np.asarray(flow_costs)


def paired_report(name, sobol_stat, flow_stat):
    diff = flow_stat - sobol_stat  # negative = flow better
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    t = diff.mean() / se if se > 0 else float("nan")
    win_rate = np.mean(diff < 0)
    # Normal-approximation 95% CI on the win rate (n is large here, so this is fine).
    win_se = np.sqrt(win_rate * (1 - win_rate) / n)
    print(
        f"{name:<14} n={n:5d}  mean_diff={diff.mean():>8.3f}  std={diff.std():>8.2f}  "
        f"median={np.median(diff):>8.2f}  win_rate={win_rate:>6.1%} (+/-{1.96*win_se:.1%})  "
        f"paired t={t:>6.2f}"
    )


def main(sampler_dir="ksos_l2o/data/sampler", budget=32, eval_states=256, seeds=(0, 1, 2, 3, 4)):
    flow, params, lb, ub, config = load_sampler(sampler_dir)
    sobol_ctrl = make_sobol_controller(budget)

    all_sobol_mean, all_flow_mean, all_sobol_best, all_flow_best = [], [], [], []
    for seed in seeds:
        sobol_costs, flow_costs = one_seed(flow, params, lb, ub, sobol_ctrl, budget, eval_states, seed)
        sobol_mean, flow_mean = sobol_costs.mean(axis=1), flow_costs.mean(axis=1)
        sobol_best, flow_best = sobol_costs.min(axis=1), flow_costs.min(axis=1)

        print(f"\n--- seed {seed} ({eval_states} states) ---")
        paired_report("mean cost", sobol_mean, flow_mean)
        paired_report("best-of-n", sobol_best, flow_best)

        all_sobol_mean.append(sobol_mean); all_flow_mean.append(flow_mean)
        all_sobol_best.append(sobol_best); all_flow_best.append(flow_best)

    print(f"\n=== pooled across {len(seeds)} seeds ({len(seeds) * eval_states} total state-comparisons) ===")
    paired_report("mean cost", np.concatenate(all_sobol_mean), np.concatenate(all_flow_mean))
    paired_report("best-of-n", np.concatenate(all_sobol_best), np.concatenate(all_flow_best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-dir", default="ksos_l2o/data/sampler")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--eval-states", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(args.sampler_dir, args.budget, args.eval_states, tuple(args.seeds))
