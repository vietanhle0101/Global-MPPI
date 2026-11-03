"""Check the JAX dual solver against a direct CVXPY solve of the barrier SDP:

    max c - lambd tr(B) + t logdet(B)   s.t.  f_i - phi_i^T B phi_i >= c,  B >= 0

The dual variables of the n inequality constraints are the weights alpha.

    python -m ksos_l2o.test_dual_cvxpy
"""
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from . import dual

LAMBD, EPSILON = 1e-5, 1e-2


def cvxpy_solve(phi, f, t, lambd, solver):
    n = f.shape[0]
    B = cp.Variable((n, n), PSD=True)
    c = cp.Variable()
    cons = [f[i] - phi[:, i] @ B @ phi[:, i] >= c for i in range(n)]
    prob = cp.Problem(cp.Maximize(c - lambd * cp.trace(B) + t * cp.log_det(B)), cons)
    prob.solve(solver=solver)
    alpha = np.array([cons[i].dual_value for i in range(n)]).reshape(-1)
    return alpha, float(c.value), prob.value, prob.status


def main(solver="CLARABEL"):
    d = np.load("ksos_l2o/data/dataset_32.npz")
    n = 32
    t = EPSILON / n
    for state in (0, 1, 2, 3):
        x, f = d["x"][state].reshape(n, -1).astype(np.float64), d["J"][state].astype(np.float64)
        for sigma in (1.0, 0.01):
            phi = np.asarray(dual.features(jnp.asarray(x, dtype=jnp.float32), sigma), dtype=np.float64)
            fn, tn = dual.normalize(jnp.asarray(f, dtype=jnp.float32), t)
            fn, tn = np.asarray(fn, dtype=np.float64), float(tn)

            a_cvx, c_cvx, val, status = cvxpy_solve(phi, fn, tn, LAMBD, solver)
            alpha, _, c, _ = jax.jit(lambda f_, p_: dual.solve(f_, p_, tn, LAMBD, iterations=100))(
                jnp.asarray(fn, dtype=jnp.float32), jnp.asarray(phi, dtype=jnp.float32)
            )
            alpha = np.asarray(alpha)
            print(
                f"state {state} sigma={sigma:<5} [{status}] sum(a_cvx)={a_cvx.sum():.4f} "
                f"|a-a_cvx|_max={np.abs(alpha - a_cvx).max():.2e} |z-z_cvx|={np.linalg.norm(alpha @ x - a_cvx @ x):.2e} "
                f"c_jax={float(c):.5f} c_cvx={c_cvx:.5f}"
            )


if __name__ == "__main__":
    main()
