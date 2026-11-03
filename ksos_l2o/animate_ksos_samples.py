"""GIF of the candidates Sobol and the trained flow give KernelSOS at the PushT start state:
each candidate's rollout is a faint T-block coloured by its plain cost J; the KSOS solution is black.

    python -m ksos_l2o.animate_ksos_samples [--sampler-dir ksos_l2o/data/sampler_state_lse]
        [--num-samples 32] [--sigma-lse 0.3] [--seed 0] [--gif out.gif]
"""
import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from global_mppi.tasks.pusht import PushT  # noqa: E402

from .compare_ksos_solution import calibrate_sigma, flow_rollout  # noqa: E402
from .compare_samples import make_controller, make_state  # noqa: E402
from .state_flow import load_state_sampler  # noqa: E402
from .train_state_sampler import START_QPOS  # noqa: E402

T_SHAPE = np.array([[-.03, .04], [.03, .04], [.03, .02], [.01, .02], [.01, -.04], [-.01, -.04], [-.01, .02], [-.03, .02]])
PUSHER_OFFSET = np.array([0.0, 0.1])  # pusher qpos is relative to its body origin


def block_polygon(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return T_SHAPE @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def candidates(ctrl, rollout, state, key):
    """The sampler's knots with their plain costs J, and the KSOS solution with its J."""
    params = rollout(state, ctrl.init_params().replace(rng=key))
    calibrate_sigma(ctrl, params)
    solution = jnp.clip(ctrl.solve_ksos(params).ksos_mean, ctrl.task.u_min, ctrl.task.u_max)
    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)

    def cost(knots):
        return np.asarray(ctrl.rollout(state, tk, jnp.asarray(knots)).costs.sum(axis=1))

    return {"knots": np.asarray(params.knots), "solution": np.asarray(solution), "j": cost(params.knots), "sol_j": cost(solution[None])[0]}


def play(ctrl, state, knots):
    """qpos (R, H+1, 5) for each knot sequence."""
    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    controls = ctrl.interp_func(jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps), tk, jnp.asarray(knots))
    states, _ = ctrl.eval_rollouts(ctrl.model, state, controls, jnp.asarray(knots))
    q = np.asarray(states.qpos)
    return np.concatenate([np.tile(np.asarray(state.qpos), (len(q), 1, 1)), q], axis=1)


def main(sampler_dir, n, sigma_lse, seed, gif, step):
    flow, flow_params, config = load_state_sampler(sampler_dir)
    sigma_lse = sigma_lse if sigma_lse is not None else (config["sigma_lse"] or 0.3)
    task = PushT()
    state = make_state(task, START_QPOS, np.zeros(5))
    ctrl = make_controller(task, n, sampling="sobol")
    ctrl.elite_num = n
    ctrl.lse_smoothing_sigma = jnp.full((5,), sigma_lse, dtype=jnp.float32)
    rollouts = {f"sobol{n}": jax.jit(ctrl.ksos_rollout),
                f"flow{n}": jax.jit(lambda s, p: flow_rollout(ctrl, flow, flow_params, n, s, p))}
    key = jax.random.PRNGKey(seed)
    runs = {}
    for name, fn in rollouts.items():
        d = candidates(ctrl, fn, state, key)
        q = play(ctrl, state, np.concatenate([d["knots"], d["solution"][None]]))
        runs[name] = dict(d, q=q)

    frames = list(range(0, q.shape[1], step)) + [q.shape[1] - 1] * 8
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(r["j"].max() for r in runs.values()))
    cmap = plt.get_cmap("viridis_r")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), layout="constrained")
    artists = {}
    for ax, (name, r) in zip(axes, runs.items()):
        ax.add_patch(Polygon(block_polygon(0, 0, 0), closed=True, fill=False, ec="tab:green", lw=2, label="goal"))
        ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), aspect="equal", xlabel="x", ylabel="y")
        ax.xaxis.label.set_size(16), ax.yaxis.label.set_size(16), ax.tick_params(labelsize=14)
        blocks = [ax.add_patch(Polygon(T_SHAPE, closed=True, fc=cmap(norm(j)), ec="none", alpha=0.18)) for j in r["j"]]
        sol_block = ax.add_patch(Polygon(T_SHAPE, closed=True, fill=False, ec="k", lw=2, label=f"KSOS solution (J={r['sol_j']:.1f})"))
        dots = ax.scatter(np.zeros(n), np.zeros(n), c=r["j"], cmap=cmap, norm=norm, s=22, edgecolors="none")
        sol_dot, = ax.plot([], [], "ko", ms=8)
        sol_trail, = ax.plot([], [], "k-", lw=1.5)
        ax.legend(loc="lower right", fontsize=14)
        artists[name] = (blocks, sol_block, dots, sol_dot, sol_trail)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="plain cost J of the candidate", shrink=0.7).set_label("plain cost J of the candidate", size=16)
    axes[0].figure.axes[-1].tick_params(labelsize=14)

    def draw(t):
        for ax, (name, r) in zip(axes, runs.items()):
            blocks, sol_block, dots, sol_dot, sol_trail = artists[name]
            q = r["q"][:, t]
            for poly, row in zip(blocks, q[:-1]):
                poly.set_xy(block_polygon(*row[:3]))
            sol_block.set_xy(block_polygon(*q[-1, :3]))
            dots.set_offsets(q[:-1, 3:] + PUSHER_OFFSET)
            sol_dot.set_data([q[-1, 3] + PUSHER_OFFSET[0]], [q[-1, 4] + PUSHER_OFFSET[1]])
            trail = r["q"][-1, : t + 1, 3:] + PUSHER_OFFSET
            sol_trail.set_data(trail[:, 0], trail[:, 1])
            ax.set_title(f"{name}   t = {t * ctrl.dt:.2f} s", fontsize=20)

    os.makedirs(os.path.dirname(gif) or ".", exist_ok=True)
    FuncAnimation(fig, draw, frames=frames).save(gif, writer=PillowWriter(fps=12), dpi=70)
    print(f"saved {gif}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-dir", default="ksos_l2o/data/sampler_state_lse")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--sigma-lse", type=float, default=None, help="default: the flow's trained value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=2, help="simulation steps per GIF frame")
    p.add_argument("--gif", default=None, help="output GIF (default: <sampler-dir>/ksos_samples.gif)")
    a = p.parse_args()
    main(a.sampler_dir, a.num_samples, a.sigma_lse, a.seed, a.gif or os.path.join(a.sampler_dir, "ksos_samples.gif"), a.step)
