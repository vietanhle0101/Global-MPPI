"""Practical test + figures for the learned dual solver on FRESH states.

For each fresh PushT state: 32 flow candidates -> solve the dual three ways
(converged reference, GNN start + k Newton, uniform start + k Newton) -> the
solution point z = alpha @ x -> real rollout cost at z.

    python -m ksos_l2o.visualize_dual [--config ksos_l2o/configs/train_dual.yaml]
        [--states 256] [--seed 7]
Figures go to <output_dir>/figures/.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon

from . import dual
from .checkpoint import load_pytree
from .context import pusht_context
from .gnn import init_gnn, warm_start
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state
from .train_dual import LAMBD, prepare
from .visualize_sampler import _t_block_polygons, load_sampler


def solve_path(x, f, c, params, ub, sigma, iters, use_gnn):
    phi, fn, tn = prepare(x, f, sigma)
    n = x.shape[0]
    a0 = warm_start(params, x, fn, c, ub, sigma) if use_gnn else jnp.full((n,), 1.0 / n)
    _, path, _, _ = dual.solve(fn, phi, tn, LAMBD, alpha0=a0, iterations=iters)
    values = jax.vmap(lambda a: dual.barrier_value(a, fn, phi, tn, LAMBD))(path)
    return path, values


def main(config, states_n, seed):
    cfg = yaml.safe_load(open(config))
    directory = Path(cfg["output_dir"])
    m = json.loads((directory / "config.json").read_text())
    k, sigma, n = m["layers"], m["sigma"], 32
    fig_dir = directory / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    flow, flow_params, lb, ub, _ = load_sampler("ksos_l2o/data/sampler")
    gnn_params = load_pytree(init_gnn(jax.random.PRNGKey(0), m["dim"], m["context_dim"], m["hidden_dim"], m["gnn_layers"]),
                             directory / "checkpoint.npz")
    controller = make_controller()
    task = controller.task
    nk, nu = controller.num_knots, task.model.nu

    keys = jax.random.split(jax.random.PRNGKey(seed), states_n)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(keys, task)
    ctx = jax.vmap(pusht_context)(states.qpos, states.qvel)
    z0 = jax.random.normal(jax.random.PRNGKey(seed + 1), (states_n, n, m["dim"]))
    x, _ = flow.sample_batch(flow_params, z0, ctx, lb, ub)
    cost = lambda st, kn: rollout_cost(controller, st, kn.reshape(-1, nk, nu))
    f = jax.vmap(cost)(states, x)

    run = lambda iters, use, p: jax.jit(jax.vmap(
        lambda x_, f_, c_: solve_path(x_, f_, c_, p, ub, sigma, iters, use)))(x, f, ctx)
    ref_path, ref_val = run(200, False, None)
    gnn_path, gnn_val = run(k, True, gnn_params)
    uni_path, uni_val = run(k, False, None)
    a_ref, a_gnn, a_uni = ref_path[:, -1], gnn_path[:, -1], uni_path[:, -1]

    zs = {name: jnp.einsum("bn,bnd->bd", a, x) for name, a in
          (("converged solver", a_ref), (f"GNN + {k} Newton", a_gnn), (f"uniform + {k} Newton", a_uni))}
    zs["best sample"] = x[jnp.arange(states_n), jnp.argmin(f, axis=1)]
    costs = {name: np.asarray(jax.vmap(lambda st, z: rollout_cost(controller, st, z.reshape(1, nk, nu))[0])(states, z))
             for name, z in zs.items()}
    costs["mean of samples"] = np.asarray(f.mean(axis=1))
    f_np = np.asarray(f)
    tied = (f_np <= f_np.min(1, keepdims=True) * (1 + 1e-6) + 1e-6).sum(1) >= 5

    print(f"{states_n} fresh states (seed {seed}), k={k}, sigma={sigma}; tied-minimum problems: {tied.sum()}")
    print(f"{'point z':<22}{'mean cost':>11}{'median':>9}{'mean (tied)':>13}{'beats ref':>11}{'worse than ref by >5%':>24}")
    ref_c = costs["converged solver"]
    for name, c in costs.items():
        print(f"{name:<22}{c.mean():>11.2f}{np.median(c):>9.2f}{(c[tied].mean() if tied.any() else float('nan')):>13.2f}"
              f"{np.mean(c < ref_c - 1e-6):>11.1%}{np.mean(c > ref_c * 1.05 + 1e-6):>24.1%}")

    # 1. convergence
    gap = lambda v: np.asarray(v - ref_val[:, -1:])
    dz = lambda p: np.asarray(jnp.linalg.norm(jnp.einsum("bkn,bnd->bkd", p, x) - zs["converged solver"][:, None], axis=-1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (g, u, title) in zip(axes, ((gap(gnn_val), gap(uni_val), "barrier gap H-H*"),
                                         (dz(gnn_path), dz(uni_path), "|z - z*|"))):
        for data, label, color in ((u, "uniform start", "#d62728"), (g, "GNN start", "#1f77b4")):
            lo, med, hi = np.percentile(np.maximum(data, 1e-12), [10, 50, 90], axis=0)
            it = np.arange(data.shape[1])
            ax.plot(it, med, color=color, label=label); ax.fill_between(it, lo, hi, color=color, alpha=0.2)
        ax.set(yscale="log", xlabel="Newton iterations after the start", title=title + " (median, 10-90%)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "convergence.png", dpi=130); plt.close(fig)

    # 2. cost at z
    names = [nm for nm in costs if nm != "mean of samples"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(ref_c, costs[f"GNN + {k} Newton"], s=12, alpha=0.6, label="GNN", color="#1f77b4")
    ax.scatter(ref_c[tied], costs[f"GNN + {k} Newton"][tied], s=30, facecolors="none", edgecolors="k", label="tied-minimum problems")
    lim = [0, max(ref_c.max(), costs[f"GNN + {k} Newton"].max()) * 1.05]
    ax.plot(lim, lim, "k--", linewidth=1); ax.set(xlabel="rollout cost at z (converged solver)", ylabel=f"rollout cost at z (GNN + {k})", xlim=lim, ylim=lim)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(fig_dir / "cost_at_z.png", dpi=130); plt.close(fig)

    # 3. alpha bars + 4. pusher traces: one typical problem, one tied problem
    typical = int(np.argmax(~tied & (np.abs(costs[f"GNN + {k} Newton"] - ref_c) < 1e-3)))
    picks = [("typical", typical)] + ([("tied", int(np.argmax(tied)))] if tied.any() else [])
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 3.8), squeeze=False)
    tk = jnp.linspace(0.0, controller.plan_horizon, nk)
    fig2, axes2 = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6), squeeze=False)
    colors = {"converged solver": "#2ca02c", f"GNN + {k} Newton": "#1f77b4", f"uniform + {k} Newton": "#d62728", "best sample": "#ff7f0e"}
    for col, (label, i) in enumerate(picks):
        ax = axes[0, col]; idx = np.arange(n)
        for off, (a, nm) in zip((-0.27, 0, 0.27), ((a_ref, "converged solver"), (a_gnn, f"GNN + {k} Newton"), (a_uni, f"uniform + {k} Newton"))):
            ax.bar(idx + off, np.asarray(a[i]), width=0.27, color=colors[nm], label=nm)
        ax.set(xlabel="sample index", ylabel="alpha", title=f"{label} problem (state {i}): weights"); ax.legend(fontsize=7)
        ax2 = axes2[0, col]; st = jax.tree_util.tree_map(lambda a: a[i], states)
        for poly in _t_block_polygons(np.asarray(st.qpos[:2]), float(st.qpos[2])):
            ax2.add_patch(Polygon(poly, closed=True, facecolor="#888888", alpha=0.6, edgecolor="black", zorder=2))
        for poly in _t_block_polygons(np.zeros(2), 0.0):
            ax2.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor="#2a9d2a", linestyle="--", zorder=1))
        for nm, c in colors.items():
            traj = controller.rollout(st, tk, zs[nm][i].reshape(1, nk, nu))
            path = np.asarray(traj.trace_sites[0, :, 0, :2])
            ax2.plot(path[:, 0], path[:, 1], color=c, linewidth=1.8, label=f"{nm} (J={costs[nm][i]:.1f})", zorder=4)
        ax2.scatter(0, 0, color="#2a9d2a", marker="*", s=120, zorder=5)
        ax2.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal", title=f"{label} problem: pusher trace at z"); ax2.legend(fontsize=7, loc="upper right"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "alpha_weights.png", dpi=130); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(fig_dir / "pusher_traces_at_z.png", dpi=130); plt.close(fig2)
    print("figures saved to", fig_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ksos_l2o/configs/train_dual.yaml")
    p.add_argument("--states", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    main(a.config, a.states, a.seed)
