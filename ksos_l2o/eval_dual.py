"""Held-out tail evaluation of the GNN warm start vs the uniform start.

    python -m ksos_l2o.eval_dual [--config ksos_l2o/configs/train_dual.yaml] [--dir DIR] [--data DATA.npz]

By default the model directory and dataset are taken from the training config.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .checkpoint import load_pytree
from .gnn import init_gnn
from .train_dual import path_values, reference


def main(directory="ksos_l2o/data/dual", data="ksos_l2o/data/dual_dataset.npz", val_size=512):
    cfg = json.loads(open(f"{directory}/config.json").read())
    d = np.load(data)
    x, f, c = (jnp.asarray(d[k])[-val_size:] for k in ("x", "J", "context"))
    ub = jnp.asarray(d["ub"])
    sigma, layers = cfg["sigma"], cfg["layers"]
    template = init_gnn(jax.random.PRNGKey(0), cfg["dim"], cfg["context_dim"], cfg["hidden_dim"], cfg["gnn_layers"])
    params = load_pytree(template, f"{directory}/checkpoint.npz")

    h_star, z_star = jax.jit(jax.vmap(lambda x_, f_: reference(x_, f_, sigma)))(x, f)
    run = lambda p, use: jax.jit(jax.vmap(lambda x_, f_, c_: path_values(p, x_, f_, c_, ub, sigma, layers, use)))(x, f, c)
    hg, zg = run(params, True)
    hu, zu = run(None, False)
    gap_g, gap_u = np.asarray(hg - h_star[:, None]), np.asarray(hu - h_star[:, None])
    dz_g = np.asarray(jnp.linalg.norm(zg - z_star[:, None], axis=-1))
    dz_u = np.asarray(jnp.linalg.norm(zu - z_star[:, None], axis=-1))

    print(f"{val_size} held-out problems, sigma={sigma}; non-finite: GNN {int((~np.isfinite(gap_g)).any(axis=1).sum())}, "
          f"uniform {int((~np.isfinite(gap_u)).any(axis=1).sum())}")
    for name, g, u in (("barrier gap H-H*", gap_g, gap_u), ("|z - z*|", dz_g, dz_u)):
        print(f"\n{name}")
        print(f"{'k':>3} | {'GNN median':>11}{'p90':>10}{'p99':>10}{'max':>10} | {'uniform median':>14}{'p90':>10}{'max':>10} | {'GNN better':>10}")
        for k in (layers,):
            print(f"{k:>3} | {np.median(g[:, k]):>11.2e}{np.percentile(g[:, k], 90):>10.2e}{np.percentile(g[:, k], 99):>10.2e}{g[:, k].max():>10.2e}"
                  f" | {np.median(u[:, k]):>14.2e}{np.percentile(u[:, k], 90):>10.2e}{u[:, k].max():>10.2e} | {np.mean(g[:, k] < u[:, k]):>10.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ksos_l2o/configs/train_dual.yaml")
    p.add_argument("--dir", default=None)
    p.add_argument("--data", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    directory, data = a.dir or cfg["output_dir"], a.data or cfg["data"]
    print(f"evaluating model in {directory}")
    main(directory, data)
