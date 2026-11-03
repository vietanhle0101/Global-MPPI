"""Differentiable KernelSOS dual solver (damped Newton on the log-det barrier).

Mirrors ksos_tools' `damped_newton` (Rudi et al. 2020, Sec. 6) in JAX, float32:

    min_alpha  H(alpha) = alpha . f - t * logdet(M(alpha)),   sum(alpha) = 1
    M(alpha)   = Phi diag(alpha) Phi^T + lambd I,   K = Phi^T Phi,  K_ij = exp(-|x_i-x_j|/sigma)

grad H = f - t diag(Phi^T M^-1 Phi),   hess H = t (Phi^T M^-1 Phi)^2 (elementwise).
The primal point estimate is z = sum_i alpha_i x_i.
"""
import jax
import jax.numpy as jnp


def laplace_gram(x, sigma):
    diff = x[:, None, :] - x[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    gram = jnp.exp(-dist / sigma)
    return gram * (1 - jnp.eye(x.shape[0])) + jnp.eye(x.shape[0])


def features(x, sigma, jitter=1e-6):
    """Phi (n x n) with Phi^T Phi = K."""
    gram = laplace_gram(x, sigma) + jitter * jnp.eye(x.shape[0])
    return jnp.linalg.cholesky(gram).T


def moment_matrix(alpha, phi, lambd):
    return (phi * alpha[None, :]) @ phi.T + lambd * jnp.eye(phi.shape[0])


def grad_hess(alpha, f, phi, t, lambd):
    m = moment_matrix(alpha, phi, lambd)
    inner = phi.T @ jnp.linalg.solve(m, phi)
    return f - t * jnp.diag(inner), t * inner * inner


def barrier_value(alpha, f, phi, t, lambd):
    m = moment_matrix(alpha, phi, lambd)
    _, logdet = jnp.linalg.slogdet(m)
    return jnp.dot(alpha, f) - t * logdet


def newton_direction(alpha, f, phi, t, lambd):
    """Equality-constrained Newton direction (sum Delta = 0) and squared decrement."""
    grad, hess = grad_hess(alpha, f, phi, t, lambd)
    n = alpha.shape[0]
    hess = hess + 1e-9 * jnp.eye(n)
    g1 = jnp.linalg.solve(hess, grad)
    g2 = jnp.linalg.solve(hess, jnp.ones(n))
    c = jnp.sum(g1) / jnp.sum(g2)
    delta = g1 - c * g2
    return delta, c, delta @ hess @ delta


def newton_step(alpha, f, phi, t, lambd):
    delta, c, decrement = newton_direction(alpha, f, phi, t, lambd)
    step = 1.0 / (1.0 + jnp.sqrt(jnp.maximum(decrement, 0.0) / t))
    new = alpha - step * delta
    new = new + (1.0 - jnp.sum(new)) / new.shape[0]
    return new, c, decrement


def normalize(f, t):
    """Shift/scale f to [0, 1] and rescale t to match; leaves the optimal alpha unchanged
    (sum alpha = 1) but keeps float32 Newton well conditioned."""
    scale = jnp.maximum(jnp.max(f) - jnp.min(f), 1e-12)
    return (f - jnp.min(f)) / scale, t / scale


def solve(f, phi, t, lambd, alpha0=None, iterations=20):
    """Unrolled damped Newton. Returns (alpha, path (iterations+1, n), c, decrements)."""
    n = f.shape[0]
    alpha0 = jnp.full((n,), 1.0 / n) if alpha0 is None else alpha0

    def body(alpha, _):
        new, c, dec = newton_step(alpha, f, phi, t, lambd)
        return new, (new, c, dec)

    final, (path, cs, decs) = jax.lax.scan(body, alpha0, None, length=iterations)
    return final, jnp.concatenate([alpha0[None], path], axis=0), cs[-1], decs
