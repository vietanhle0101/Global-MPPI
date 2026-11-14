"""Checks of the learned KSOS dual solver on dataset problems.

    python -m pytest -q ksos_l2o/test_sdp_solver.py
"""
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from . import dual
from .sdp_solver import (SolverConfig, init_solver, kernel_and_features, moment_matrix, newton_step,
                         predict_logits, solve, solve_path, warm_start)

CFG = SolverConfig(hidden_dim=16, gnn_layers=2, newton_steps=3)
DATA = Path("ksos_l2o/data/dual_dataset.npz")


@pytest.fixture(scope="module")
def problem():
    if not DATA.exists():
        pytest.skip("run python -m ksos_l2o.generate_flow_dataset first")
    d = np.load(DATA)
    x, f, c = (jnp.asarray(d[k][:4], jnp.float64) for k in ("x", "J", "context"))
    return x, f, c, jnp.asarray(d["lb"], jnp.float64), jnp.asarray(d["ub"], jnp.float64)


def min_eig(x, alpha):
    phi = kernel_and_features(x, CFG.sigma, CFG.kernel_jitter)[1]
    return jnp.linalg.eigvalsh(moment_matrix(alpha, phi, CFG.lambd))[0]


def test_features_factor_the_laplace_kernel(problem):
    x = problem[0][0]
    kernel, phi = kernel_and_features(x, CFG.sigma, CFG.kernel_jitter)
    assert jnp.allclose(phi.T @ phi, kernel, atol=1e-10)
    assert jnp.allclose(kernel, dual.laplace_gram(x, CFG.sigma) + CFG.kernel_jitter * jnp.eye(x.shape[0]), atol=1e-5)


def test_logits_are_permutation_equivariant_and_warm_start_is_feasible(problem):
    x, f, c, lb, ub = (a[0] if a.ndim > 1 else a for a in problem)
    params = init_solver(jax.random.PRNGKey(0), x.shape[1], c.shape[0], CFG)
    kernel, phi = kernel_and_features(x, CFG.sigma, CFG.kernel_jitter)
    perm = jax.random.permutation(jax.random.PRNGKey(1), x.shape[0])
    logits = predict_logits(params["gnn"], x, f, kernel, c, lb, ub)
    permuted = predict_logits(params["gnn"], x[perm], f[perm], kernel[perm][:, perm], c, lb, ub)
    assert jnp.allclose(permuted, logits[perm])
    alpha = warm_start(params["gnn"], x, f, kernel, phi, c, lb, ub, CFG)
    assert jnp.allclose(jnp.sum(alpha), 1.0)
    assert min_eig(x, alpha) >= CFG.psd_floor - 1e-10


def test_untrained_solver_is_feasible_and_beats_uniform(problem):
    x, f, c, lb, ub = problem
    params = init_solver(jax.random.PRNGKey(0), x.shape[2], c.shape[1], CFG)
    for i in range(x.shape[0]):
        path = solve_path(params, x[i], f[i], c[i], lb, ub, CFG)
        assert path.shape == (CFG.newton_steps + 1, x.shape[1])
        assert jnp.allclose(path.sum(1), 1.0)
        assert min_eig(x[i], path[-1]) > 0
        assert path[-1] @ f[i] < jnp.mean(f[i])
        alpha, z = solve(params, x[i], f[i], c[i], lb, ub, CFG)
        assert jnp.allclose(z, alpha @ x[i])


def test_newton_layers_reach_the_newton_rs_barrier_optimum(problem):
    """Uniform start, unit steps and tau = n / epsilon: the layers converge to the
    minimiser of the barrier problem newton-rs solves (dual.py, checked against it)."""
    x, f = problem[0][0], problem[1][0]
    n = x.shape[0]
    phi = kernel_and_features(x, CFG.sigma, CFG.kernel_jitter)[1]
    alpha = jnp.full(n, 1.0 / n)
    for _ in range(60):
        alpha = newton_step(alpha, f, phi, 1.0, CFG.lambd, CFG.tau_final(n), 0.0, CFG.psd_floor)
    t = CFG.epsilon / n
    reference = dual.solve(f, dual.features(x, CFG.sigma, jitter=0.0), t, CFG.lambd, iterations=60)[0]
    assert jnp.abs(alpha - reference).max() < 1e-4
    assert jnp.abs(alpha @ f - reference @ f) < 1e-6
