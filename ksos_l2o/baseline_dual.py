"""Baseline for the learned warm start: convergence of the uniform-start
Newton solver at a given sigma, over many dataset states.

    python -m ksos_l2o.baseline_dual [--sigma 1.0] [--states 256]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from . import dual

LAMBD, EPSILON = 1e-5, 1e-2


def run_state(x, f, sigma, iterations):
    n = x.shape[0]
    phi = dual.features(x, sigma)
    fn, tn = dual.normalize(f, EPSILON / n)
    _, path, _, _ = dual.solve(fn, phi, tn, LAMBD, iterations=iterations)
    values = jax.vmap(lambda a: dual.barrier_value(a, fn, phi, tn, LAMBD))(path)
    return path @ x, values, path


def main(sigma=1.0, states=256, max_iters=100):
    d = np.load("ksos_l2o/data/dataset_32.npz")
    x, f = jnp.asarray(d["x"][:states]).reshape(states, 32, -1), jnp.asarray(d["J"][:states])
    z, values, path = jax.jit(jax.vmap(lambda x_, f_: run_state(x_, f_, sigma, max_iters)))(x, f)
    z, values = np.asarray(z), np.asarray(values)
    z_star, v_star = z[:, -1], values[:, -1]
    valid = np.isfinite(values).all(axis=1)
    print(f"sigma={sigma}, {valid.sum()}/{states} states finite")
    print(f"{'iters':>6}{'median |z-z*|':>16}{'p90 |z-z*|':>14}{'median H-H*':>14}{'p90 H-H*':>12}")
    for k in (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75):
        dz = np.linalg.norm(z[valid, k] - z_star[valid], axis=1)
        dh = values[valid, k] - v_star[valid]
        print(f"{k:>6}{np.median(dz):>16.3e}{np.percentile(dz, 90):>14.3e}{np.median(dh):>14.3e}{np.percentile(dh, 90):>12.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--states", type=int, default=256)
    a = p.parse_args()
    main(a.sigma, a.states)
