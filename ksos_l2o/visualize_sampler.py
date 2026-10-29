"""Visualize the trained forward-KL flow sampler: training curves, a
cost-distribution comparison against uniform sampling, and a spatial plot of
where the pusher actually goes under flow vs. uniform samples.

Loads the checkpoint saved by train_sampler.py (ksos_l2o/data/sampler/ by
default). Saves PNGs to <sampler-dir>/figures/.

    python -m ksos_l2o.visualize_sampler [--sampler-dir ksos_l2o/data/sampler]
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
from matplotlib.patches import Polygon

from .checkpoint import load_pytree
from .context import CONTEXT_DIM, pusht_context
from .flow import ConditionalRealNVP
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state


def load_sampler(sampler_dir):
    sampler_dir = Path(sampler_dir)
    config = json.loads((sampler_dir / "config.json").read_text())
    flow = ConditionalRealNVP(config["dim"], config["context_dim"], num_layers=6, hidden_dim=64)
    params = load_pytree(flow.init(jax.random.PRNGKey(0)), sampler_dir / "checkpoint.npz")
    with np.load(sampler_dir / "lb_ub.npz") as d:
        lb, ub = jnp.asarray(d["lb"]), jnp.asarray(d["ub"])
    return flow, params, lb, ub, config


def plot_training_curves(sampler_dir, out_dir):
    sampler_dir = Path(sampler_dir)
    epoch = np.load(sampler_dir / "epoch.npy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epoch, np.load(sampler_dir / "train_loss.npy"), label="train")
    axes[0].plot(epoch, np.load(sampler_dir / "val_loss.npy"), label="validation")
    axes[0].set(xlabel="epoch", ylabel="forward-KL loss", title="Loss")
    axes[1].plot(epoch, np.load(sampler_dir / "train_ess.npy"), label="train")
    axes[1].plot(epoch, np.load(sampler_dir / "val_ess.npy"), label="validation")
    axes[1].set(xlabel="epoch", ylabel="effective sample size (relative)", title="ESS")
    axes[2].plot(epoch, np.load(sampler_dir / "lr.npy"), color="#444444")
    axes[2].set(xlabel="epoch", ylabel="learning rate", yscale="log", title="Plateau LR schedule")
    for ax in axes[:2]:
        ax.legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


def _rotate(points, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def _t_block_polygons(center_xy, theta):
    """Two rectangles (world-frame corner arrays) matching the XML's "bottom"
    and "top" box geoms (half-extents 0.01x0.04 and 0.03x0.01 at local
    (0, 0.03))."""
    def corners(half_w, half_h, local_center):
        local = np.array([
            [-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h],
        ]) + local_center
        return _rotate(local, theta) + center_xy

    bottom = corners(0.01, 0.04, np.array([0.0, 0.0]))
    top = corners(0.03, 0.01, np.array([0.0, 0.03]))
    return bottom, top


def held_out_states_and_costs(flow, params, lb, ub, controller, eval_states, eval_samples, seed):
    """Exactly reproduces train_sampler.py's evaluate_against_uniform (same
    keys), but returns per-state costs instead of the averaged report."""
    task = controller.task
    dim = controller.num_knots * task.model.nu
    eval_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), eval_states)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    def cost_for(knots):
        knots = knots.reshape(eval_states, eval_samples, controller.num_knots, task.model.nu)
        return jax.vmap(lambda s, k: rollout_cost(controller, s, k))(states, knots)

    z = jax.random.normal(jax.random.PRNGKey(seed + 1001), (eval_states, eval_samples, dim))
    flow_x, _ = flow.sample_batch(params, z, contexts, lb, ub)
    flow_costs = cost_for(flow_x)

    uniform_x = jax.random.uniform(
        jax.random.PRNGKey(seed + 1002), (eval_states, eval_samples, dim), minval=lb, maxval=ub
    )
    uniform_costs = cost_for(uniform_x)
    return states, flow_costs, uniform_costs


def find_largest_gap_state(flow, params, lb, ub, controller, eval_states, eval_samples, seed):
    """Index (within the held-out set) where flow improves best-of-n the
    most over uniform, so the visualization actually matches the aggregate
    numbers instead of an arbitrary example."""
    states, flow_costs, uniform_costs = held_out_states_and_costs(
        flow, params, lb, ub, controller, eval_states, eval_samples, seed
    )
    flow_best = jnp.min(flow_costs, axis=1)
    uniform_best = jnp.min(uniform_costs, axis=1)
    gap = uniform_best - flow_best
    idx = int(jnp.argmax(gap))
    print(
        f"largest-gap held-out state: index {idx}  uniform_best={float(uniform_best[idx]):.2f}  "
        f"flow_best={float(flow_best[idx]):.2f}  (gap={float(gap[idx]):.2f}, "
        f"median gap over {eval_states} states={float(jnp.median(gap)):.2f})"
    )
    return jax.tree_util.tree_map(lambda a: a[idx], states)


def plot_samples(flow, params, lb, ub, controller, out_dir, state, seed=0, num_samples=256, top_k=8):
    task = controller.task
    dim = controller.num_knots * task.model.nu
    context = pusht_context(state.qpos, state.qvel)

    tk = jnp.linspace(0.0, controller.plan_horizon, controller.num_knots)

    def rollouts_for(knots):
        return controller.rollout(state, tk, knots)

    z = jax.random.normal(jax.random.PRNGKey(seed + 2001), (num_samples, dim))
    sample_fn = jax.vmap(flow.sample, in_axes=(None, 0, None, None, None))
    flow_x, _ = sample_fn(params, z, context, lb, ub)
    flow_knots = flow_x.reshape(num_samples, controller.num_knots, task.model.nu)
    flow_costs = np.asarray(rollout_cost(controller, state, flow_knots))
    flow_traj = rollouts_for(flow_knots)

    uniform_knots = jax.random.uniform(
        jax.random.PRNGKey(seed + 2002), (num_samples, controller.num_knots, task.model.nu), minval=lb.reshape(controller.num_knots, task.model.nu), maxval=ub.reshape(controller.num_knots, task.model.nu)
    )
    uniform_costs = np.asarray(rollout_cost(controller, state, uniform_knots))
    uniform_traj = rollouts_for(uniform_knots)

    # ---- cost histogram ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(
        min(flow_costs.min(), uniform_costs.min()), max(flow_costs.max(), uniform_costs.max()), 30
    )
    ax.hist(uniform_costs, bins=bins, alpha=0.6, label=f"uniform (mean={uniform_costs.mean():.1f})")
    ax.hist(flow_costs, bins=bins, alpha=0.6, label=f"flow (mean={flow_costs.mean():.1f})")
    ax.set(xlabel="rollout cost J", ylabel="count", title=f"{num_samples} samples, one example state")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_histogram.png", dpi=130)
    plt.close(fig)

    # ---- spatial plot: pusher traces for the top_k lowest-cost samples ----
    fig, ax = plt.subplots(figsize=(6, 6))
    block_xy = np.asarray(state.qpos[:2])
    block_theta = float(state.qpos[2])
    bottom, top = _t_block_polygons(block_xy, block_theta)
    for poly in (bottom, top):
        ax.add_patch(Polygon(poly, closed=True, facecolor="#888888", alpha=0.6, edgecolor="black", zorder=2))
    goal_bottom, goal_top = _t_block_polygons(np.array([0.0, 0.0]), 0.0)
    for poly in (goal_bottom, goal_top):
        ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor="#2a9d2a", linestyle="--", linewidth=1.5, zorder=1))

    uniform_idx = np.argsort(uniform_costs)[:top_k]
    flow_idx = np.argsort(flow_costs)[:top_k]
    for i in uniform_idx:
        path = np.asarray(uniform_traj.trace_sites[i, :, 0, :2])
        ax.plot(path[:, 0], path[:, 1], color="#d62728", alpha=0.5, linewidth=1.2, zorder=3,
                label="uniform (best few)" if i == uniform_idx[0] else None)
    for i in flow_idx:
        path = np.asarray(flow_traj.trace_sites[i, :, 0, :2])
        ax.plot(path[:, 0], path[:, 1], color="#1f77b4", alpha=0.7, linewidth=1.5, zorder=4,
                label="flow (best few)" if i == flow_idx[0] else None)

    ax.scatter(*block_xy, color="#444444", marker="s", s=40, zorder=5, label="block (start)")
    ax.scatter(0, 0, color="#2a9d2a", marker="*", s=120, zorder=5, label="goal")
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), aspect="equal",
           xlabel="x", ylabel="y", title=f"pusher traces, best {top_k} of {num_samples} samples")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=130)
    plt.close(fig)

    return {
        "uniform_mean": float(uniform_costs.mean()), "flow_mean": float(flow_costs.mean()),
        "uniform_best": float(uniform_costs.min()), "flow_best": float(flow_costs.min()),
    }


def main(sampler_dir="ksos_l2o/data/sampler", seed=0, eval_states=32, eval_samples=256):
    flow, params, lb, ub, config = load_sampler(sampler_dir)
    out_dir = Path(sampler_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(sampler_dir, out_dir)
    controller = make_controller()
    state = find_largest_gap_state(flow, params, lb, ub, controller, eval_states, eval_samples, seed)
    stats = plot_samples(flow, params, lb, ub, controller, out_dir, state, seed=seed, num_samples=eval_samples)

    print(f"example state: uniform mean={stats['uniform_mean']:.2f} best={stats['uniform_best']:.2f}  "
          f"flow mean={stats['flow_mean']:.2f} best={stats['flow_best']:.2f}")
    print(f"saved figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler-dir", default="ksos_l2o/data/sampler")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.sampler_dir, args.seed)
