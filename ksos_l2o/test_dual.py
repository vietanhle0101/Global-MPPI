"""Check the JAX dual solver against the real ksos_tools newton-rs solver.

    python -m ksos_l2o.test_dual
"""
import jax
import jax.numpy as jnp
import numpy as np
from ksos_tools.solvers import ksos

from . import dual

LAMBD, EPSILON = 1e-5, 1e-2


def compare(x, f, sigma, iterations):
    n = x.shape[0]
    t = EPSILON / n
    z_ref, info = ksos.solve(
        f=None, samples=x.astype(np.float64), f_samples=f.astype(np.float64),
        lambd=LAMBD, sigma=float(sigma), epsilon=EPSILON, kernel="Laplace", solver="newton-rs",
    )
    a_ref = np.asarray(info["alpha"]).reshape(-1)
    z_ref = np.asarray(z_ref).reshape(-1)
    phi = dual.features(jnp.asarray(x), sigma)
    def run(f_, p_):
        fn, tn = dual.normalize(f_, t)
        return dual.solve(fn, p_, tn, LAMBD, iterations=iterations)

    alpha, _, c, decs = jax.jit(run)(jnp.asarray(f), phi)
    alpha = np.asarray(alpha)
    z = alpha @ x
    print(
        f"sigma={sigma:<5} iters={iterations:<3} |a-a_ref|_max={np.abs(alpha - a_ref).max():.2e} "
        f"|z-z_ref|={np.linalg.norm(z - z_ref):.2e} sum(a)={alpha.sum():.6f} "
        f"min(a)={alpha.min():.2e} "
        f"final decrement={float(decs[-1]):.2e} ref_status={info['status']}"
    )


def main():
    d = np.load("ksos_l2o/data/dataset_32.npz")
    for state in (0, 1):
        x, f = d["x"][state].reshape(32, -1), d["J"][state]
        print(f"--- state {state} ---")
        for sigma in (1.0, 0.01):
            for iters in (30, 50, 100, 200):
                compare(x, f, sigma, iters)


if __name__ == "__main__":
    main()
