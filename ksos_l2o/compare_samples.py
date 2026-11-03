"""Compare flow candidates with uniform sampling over many fresh PushT states.

Arms (N = 32 KernelSOS samples): uniform32; flow32 (trained flow); uniform256e32 (256 uniform
samples, keep the 32 with the lowest LSE cost, 8x the rollouts). Each candidate is scored with the
plain cost J and the LSE cost at --sigma-lse.

    python -m ksos_l2o.compare_samples [--sampler-dir ksos_l2o/data/sampler_state_lse]
        [--states 200] [--repeats 1] [--sigma-lse 0.3]
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import comb
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .context import pusht_context
from .problem import rollout_cost
from .state_flow import load_state_sampler, sample as flow_sample
from .train_state_sampler import sample_state

N = 32  # KernelSOS samples
ARMS = (f"uniform{N}", f"flow{N}", f"uniform256e{N}")
METRICS = {  # name -> (cost, reduction over the N candidates)
    f"plain best-of-{N}": ("plain", np.min), "plain mean": ("plain", np.mean),
    f"LSE best-of-{N}": ("lse", np.min), "LSE mean": ("lse", np.mean),
}


def make_controller(task, budget, sampling="uniform"):
    return GlobalMPPI(
        task, num_samples=256, noise_level=0.1, temperature=0.1, spline_type="cubic",
        plan_horizon=1.0, num_knots=6, ksos_sampling=sampling, ksos_num_samples=budget, elite_num=N,
    )


def make_state(task, qpos, qvel):
    data = mjx.make_data(task.model).replace(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
    return mjx.forward(task.model, data)


def pass0_knots(ctrl, key):
    knots, _ = ctrl.sample_ksos_knots(ctrl.init_params().replace(rng=key))
    return jnp.clip(knots, ctrl.task.u_min, ctrl.task.u_max)


def lse_cost(ctrl, state, knots, key):
    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    return ctrl.lse_smoothing_rollout(state, tk, knots, ctrl.init_params().replace(rng=key)).lse_cost


def sample_arms(flow, flow_params, ctrls, state, key):
    c32, c256 = ctrls  # c32 holds the N-sample controller
    task, shape = c32.task, (c32.num_knots, c32.task.model.nu)
    k_sobol, k_flow, k_sobol256, k_lse = jax.random.split(key, 4)

    lb, ub = jnp.tile(task.u_min, shape[0]), jnp.tile(task.u_max, shape[0])
    z = jax.random.normal(k_flow, (N, lb.size))
    flow_knots = flow_sample(flow, flow_params, z, pusht_context(state.qpos, state.qvel), lb, ub).reshape(N, *shape)

    s256 = pass0_knots(c256, k_sobol256)
    elites = s256[jnp.argsort(lse_cost(c256, state, s256, k_lse))[:N]]
    arms = (pass0_knots(c32, k_sobol), jnp.clip(flow_knots, task.u_min, task.u_max), elites)
    return {name: {"plain": rollout_cost(c32, state, k), "lse": lse_cost(c32, state, k, k_lse)}
            for name, k in zip(ARMS, arms)}


def per_state_metrics(results):
    """{arm: {metric: (num_states,)}}, each averaged over that state's repeats."""
    return {arm: {m: np.array([np.mean([red(r[arm][key]) for r in reps]) for reps in results])
                  for m, (key, red) in METRICS.items()} for arm in ARMS}


def report(results):
    metrics, n = per_state_metrics(results), len(results)
    print(f"=== pooled over {n} states ===")
    for arm in ARMS:
        print(f"{arm:<13}" + " | ".join(f"{m} {metrics[arm][m].mean():7.2f}" for m in METRICS))


def expected_best_of_k(costs):
    """E[min of k candidates drawn without replacement], k = 1..N; costs (..., N) -> (..., N)."""
    n = costs.shape[-1]
    c, i, k = np.sort(costs, axis=-1), np.arange(1, n + 1), np.arange(1, n + 1)[:, None]
    weights = comb(n - i[None, :], k - 1) / comb(n, k)  # P(i-th smallest is the min of k)
    return c @ weights.T


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics, n = per_state_metrics(results), len(results)
    u, f, e = ARMS
    colors = {u: "tab:gray", f: "tab:red", e: "tab:blue"}
    ks = np.arange(1, N + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for row, (key, label) in enumerate((("plain", "plain cost J"), ("lse", "LSE cost"))):
        ax = axes[row, 0]
        curves = {arm: expected_best_of_k(np.array([[r[arm][key] for r in reps] for reps in results])).mean(axis=1)
                  for arm in (u, f)}  # (states, N)
        for arm, curve in curves.items():
            ax.plot(ks, curve.mean(0), color=colors[arm], label=arm)
        diff = curves[f] - curves[u]
        half = 1.96 * diff.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else 0 * diff[0]
        ax2 = axes[row, 2]
        ax2.plot(ks, diff.mean(0), color=colors[f])
        ax2.fill_between(ks, diff.mean(0) - half, diff.mean(0) + half, color=colors[f], alpha=0.25)
        ax2.axhline(0, color="k", lw=1)
        ax2.set(xscale="log", xlabel="candidates evaluated (best of k)", ylabel=f"{f} - {u}, {label}")
        ax2.set_title("paired difference (below 0 = flow better, 95% CI)")
        elite = metrics[e][f"{label.split()[0]} best-of-{N}"]
        ax.errorbar([256], [elite.mean()], yerr=[1.96 * elite.std(ddof=1) / np.sqrt(n)] if n > 1 else None,
                    fmt="o", color=colors[e], label=f"{e} (256 evaluated)")
        ax.set(xscale="log", xlabel="candidates evaluated (best of k)", ylabel=f"mean best {label}")
        ax.legend()
        ax.set_title("sample efficiency (mean over states)")

        ax = axes[row, 1]
        name = f"{label.split()[0]} best-of-{N}"
        for arm in (u, e):
            ax.scatter(metrics[arm][name], metrics[f][name], s=8, alpha=0.5, color=colors[arm], label=f"vs {arm}")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel=f"baseline best-of-{N} {label}", ylabel=f"{f} best-of-{N} {label}")
        ax.set_title("per state (below the line = flow better)")
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main(sampler_dir, states, repeats, seed, sigma_lse, figure):
    flow, flow_params, config = load_state_sampler(sampler_dir)
    task = PushT()
    ctrls = (make_controller(task, N), make_controller(task, 256))
    sigma_lse = sigma_lse if sigma_lse is not None else config["sigma_lse"]  # None: controller's pass-0 value
    if sigma_lse is not None:
        for c in ctrls:
            c.lse_smoothing_sigma = jnp.full((5,), sigma_lse, dtype=jnp.float32)

    batch = jax.vmap(sample_state, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed + 10_000), states), task)
    all_states = [jax.tree_util.tree_map(lambda x: x[i], batch) for i in range(states)]
    print(f"{states} states, {repeats} repeats, sampler {sampler_dir} (trained on {config['cost']} cost), scoring sigma_lse={sigma_lse or 0.3}\n")

    run = jax.jit(lambda s, k: sample_arms(flow, flow_params, ctrls, s, k))
    results = [[jax.tree_util.tree_map(np.asarray, run(s, k))
                for k in jax.random.split(jax.random.PRNGKey(seed + i), repeats)]
               for i, s in enumerate(all_states)]
    report(results)
    plot(results, figure or os.path.join(sampler_dir, "compare_samples.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-dir", default="ksos_l2o/data/sampler_state_lse")
    p.add_argument("--states", type=int, default=200, help="fresh random states")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--sigma-lse", type=float, default=None, help="default: the flow's trained value")
    p.add_argument("--figure", default=None, help="output PNG (default: <sampler-dir>/compare_samples.png)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.sampler_dir, a.states, a.repeats, a.seed, a.sigma_lse, a.figure)
