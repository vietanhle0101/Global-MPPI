"""Why does z end up wrong on a few held-out problems? Compare the bad group
(|z - z*| > threshold) against the rest on properties of the problem.

    python -m ksos_l2o.inspect_dual_tail [--config ksos_l2o/configs/train_dual.yaml] [--threshold 1.0]
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from . import dual
from .checkpoint import load_pytree
from .gnn import init_gnn, warm_start
from .train_dual import LAMBD, EPSILON, prepare


def solve_alpha(x, f, c, params, ub, sigma, k, use_gnn):
    phi, fn, tn = prepare(x, f, sigma)
    n = x.shape[0]
    a0 = warm_start(params, x, fn, c, ub, sigma) if use_gnn else jnp.full((n,), 1.0 / n)
    a, _, _, _ = dual.solve(fn, phi, tn, LAMBD, alpha0=a0, iterations=k)
    return a, a0, fn


def main(config, threshold, val_size=512):
    cfg = yaml.safe_load(open(config))
    directory = cfg["output_dir"]
    m = json.loads(open(f"{directory}/config.json").read())
    k, sigma = m["layers"], m["sigma"]
    d = np.load(cfg["data"])
    x, f, c = (jnp.asarray(d[key])[-val_size:] for key in ("x", "J", "context"))
    ub = jnp.asarray(d["ub"])
    template = init_gnn(jax.random.PRNGKey(0), m["dim"], m["context_dim"], m["hidden_dim"], m["gnn_layers"])
    params = load_pytree(template, f"{directory}/checkpoint.npz")

    ref = jax.jit(jax.vmap(lambda x_, f_, c_: solve_alpha(x_, f_, c_, None, ub, sigma, 200, False)))
    gnn = jax.jit(jax.vmap(lambda x_, f_, c_: solve_alpha(x_, f_, c_, params, ub, sigma, k, True)))
    a_ref, _, fn = ref(x, f, c)
    a_gnn, a_init, _ = gnn(x, f, c)
    a_ref, a_gnn, a_init, fn, x = map(np.asarray, (a_ref, a_gnn, a_init, fn, x))

    z_ref, z_gnn = np.einsum("bn,bnd->bd", a_ref, x), np.einsum("bn,bnd->bd", a_gnn, x)
    dz = np.linalg.norm(z_gnn - z_ref, axis=1)
    bad = dz > threshold
    print(f"k={k}, sigma={sigma}: {bad.sum()}/{len(bad)} problems with |z-z*| > {threshold}")

    sorted_fn = np.sort(fn, axis=1)
    props = {
        "gap best-vs-2nd cost (normalised)": sorted_fn[:, 1] - sorted_fn[:, 0],
        "gap best-vs-5th cost (normalised)": sorted_fn[:, 4] - sorted_fn[:, 0],
        "ref alpha: max weight": a_ref.max(axis=1),
        "ref alpha: # samples with weight>0.05": (a_ref > 0.05).sum(axis=1),
        "ref alpha: entropy": -(np.clip(a_ref, 1e-12, None) * np.log(np.clip(a_ref, 1e-12, None))).sum(axis=1),
        "GNN-init alpha: max weight": a_init.max(axis=1),
        "|x_ref_argmax - x_gnn_argmax|": np.linalg.norm(
            x[np.arange(len(x)), a_ref.argmax(1)] - x[np.arange(len(x)), a_gnn.argmax(1)], axis=1),
    }
    same_top = a_ref.argmax(1) == a_gnn.argmax(1)
    print(f"{'property':<42}{'bad: median':>13}{'rest: median':>14}")
    for name, v in props.items():
        print(f"{name:<42}{np.median(v[bad]) if bad.any() else float('nan'):>13.3g}{np.median(v[~bad]):>14.3g}")
    print(f"{'GNN picks same top sample as reference':<42}{same_top[bad].mean() if bad.any() else float('nan'):>13.1%}{same_top[~bad].mean():>14.1%}")

    worst = np.argsort(-dz)[:8]
    print("\nworst 8: |dz|, ref top-sample (weight), GNN top-sample (weight), cost-gap best-vs-2nd, normalised cost of each pick")
    for i in worst:
        r, g = a_ref[i].argmax(), a_gnn[i].argmax()
        print(f"  |dz|={dz[i]:.2f} ref {r:2d} ({a_ref[i, r]:.2f}) gnn {g:2d} ({a_gnn[i, g]:.2f}) "
              f"gap12={props['gap best-vs-2nd cost (normalised)'][i]:.3f} fn[ref]={fn[i, r]:.3f} fn[gnn]={fn[i, g]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ksos_l2o/configs/train_dual.yaml")
    p.add_argument("--threshold", type=float, default=1.0)
    a = p.parse_args()
    main(a.config, a.threshold)
