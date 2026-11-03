"""MuJoCo GIF of what each dual solution does: for chosen fresh states, replay the
control sequence at (converged solver, GNN + k Newton, uniform + k Newton, best
sample) in plain MuJoCo, render a top-down view, and tile them into one GIF.

    python -m ksos_l2o.render_dual [--config ksos_l2o/configs/train_dual.yaml]
        [--states 0 26] [--seed 7] [--num-states 256]
Output: <output_dir>/videos/state_<i>.gif. The state indices refer to the same
fresh problems as visualize_dual.py (same seed and --num-states).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from .checkpoint import load_pytree
from .context import pusht_context
from .gnn import init_gnn
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state
from .visualize_dual import solve_path
from .visualize_sampler import load_sampler

TILE = 400


def replay_frames(mj_model, qpos, qvel, controls, renderer, camera, every):
    data = mujoco.MjData(mj_model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(mj_model, data)
    frames = []
    for step, u in enumerate(controls):
        if step % every == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render().copy())
        data.ctrl[:] = u
        mujoco.mj_step(mj_model, data)
    renderer.update_scene(data, camera)
    frames.append(renderer.render().copy())
    block_err = float(np.linalg.norm(data.qpos[:2]))
    return frames, block_err


def main(config, state_ids, seed, num_states, every=2):
    cfg = yaml.safe_load(open(config))
    directory = Path(cfg["output_dir"])
    m = json.loads((directory / "config.json").read_text())
    k, sigma, n = m["layers"], m["sigma"], 32
    out_dir = directory / "videos"; out_dir.mkdir(parents=True, exist_ok=True)

    flow, flow_params, lb, ub, _ = load_sampler("ksos_l2o/data/sampler")
    gnn_params = load_pytree(init_gnn(jax.random.PRNGKey(0), m["dim"], m["context_dim"], m["hidden_dim"], m["gnn_layers"]),
                             directory / "checkpoint.npz")
    controller = make_controller()
    task = controller.task
    nk, nu = controller.num_knots, task.model.nu
    tk = jnp.linspace(0.0, controller.plan_horizon, nk)

    keys = jax.random.split(jax.random.PRNGKey(seed), num_states)
    z0 = jax.random.normal(jax.random.PRNGKey(seed + 1), (num_states, n, m["dim"]))

    from global_mppi.tasks.pusht import ROOT
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/pusht/scene.xml")
    mj_model.vis.global_.offwidth = mj_model.vis.global_.offheight = TILE
    renderer = mujoco.Renderer(mj_model, height=TILE, width=TILE)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance, camera.azimuth, camera.elevation = 0.7, 90.0, -90.0

    for i in state_ids:
        state = sample_initial_state(keys[i], task)
        ctx = pusht_context(state.qpos, state.qvel)
        x, _ = flow.sample_batch(flow_params, z0[i][None], ctx[None], lb, ub)
        x = x[0]
        f = rollout_cost(controller, state, x.reshape(n, nk, nu))
        a = {}
        a["converged solver"] = solve_path(x, f, ctx, None, ub, sigma, 200, False)[0][-1]
        a[f"GNN + {k} Newton"] = solve_path(x, f, ctx, gnn_params, ub, sigma, k, True)[0][-1]
        a[f"uniform + {k} Newton"] = solve_path(x, f, ctx, None, ub, sigma, k, False)[0][-1]
        zs = {name: alpha @ x for name, alpha in a.items()}
        zs["best sample"] = x[jnp.argmin(f)]

        tiles = {}
        for name, z in zs.items():
            knots = z.reshape(1, nk, nu)
            cost = float(rollout_cost(controller, state, knots)[0])
            controls = np.asarray(controller.rollout(state, tk, knots).controls[0])
            frames, block_err = replay_frames(mj_model, np.asarray(state.qpos), np.asarray(state.qvel),
                                              controls, renderer, camera, every)
            tiles[name] = (frames, cost, block_err)
            print(f"state {i} {name:<22} J={cost:8.2f}  final block-to-goal distance {block_err:.3f} m")

        names = list(tiles)
        length = max(len(t[0]) for t in tiles.values())
        gif = []
        for t in range(length):
            canvas = Image.new("RGB", (2 * TILE, 2 * TILE))
            for j, name in enumerate(names):
                frames, cost, _ = tiles[name]
                tile = Image.fromarray(frames[min(t, len(frames) - 1)])
                draw = ImageDraw.Draw(tile)
                draw.rectangle([0, 0, TILE, 18], fill=(0, 0, 0))
                draw.text((4, 3), f"{name}   J={cost:.1f}", fill=(255, 255, 255))
                canvas.paste(tile, ((j % 2) * TILE, (j // 2) * TILE))
            gif.append(canvas)
        path = out_dir / f"state_{i}.gif"
        gif[0].save(path, save_all=True, append_images=gif[1:] + [gif[-1]] * 10, duration=40, loop=0)
        print("saved", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ksos_l2o/configs/train_dual.yaml")
    p.add_argument("--states", type=int, nargs="+", default=[0, 26])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-states", type=int, default=256)
    a = p.parse_args()
    main(a.config, a.states, a.seed, a.num_states)
