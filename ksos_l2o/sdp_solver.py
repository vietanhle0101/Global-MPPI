"""Learned KernelSOS dual solver (Phase 2a), ported from KSOS_L2O examples 3/4:
a theta-conditioned GNN warm start followed by K unrolled log-det Newton layers
with a learned step size and barrier weight per layer.

The SDP is the one GlobalMPPI.solve_ksos hands to ksos.solve:

    min_alpha  f^T alpha   s.t.  sum(alpha) = 1,  M(alpha) = Phi diag(alpha) Phi^T + lambd I >= 0
    K = Phi^T Phi,  K_ij = exp(-|x_i - x_j| / sigma)  (Laplace),  z = X^T alpha.

newton-rs solves its barrier version tau f^T alpha - logdet M(alpha) with
tau = n / epsilon, so tau_final is set to that value.
Train with jax_enable_x64; load_learned_solver runs it in float64 inside a
float32 controller.
"""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from .checkpoint import load_pytree


@dataclass(frozen=True)
class SolverConfig:
    # KSOS problem, as in GlobalMPPI (ksos_lambda, ksos_epsilon, ksos_kernel) with a fixed sigma
    sigma: float = 1.0
    lambd: float = 1e-5
    epsilon: float = 1e-2
    kernel_jitter: float = 1e-10
    # model
    hidden_dim: int = 96
    gnn_layers: int = 3
    newton_steps: int = 5
    tau: float = 20.0                   # lower bound of the learned barrier weights
    kkt_damping: float = 1e-6
    psd_floor: float = 1e-6
    warm_start_fraction: float = 0.95   # share of the way to the PSD boundary
    initial_step_size: float = 0.8
    warm_start_temperature: float = 0.2  # ray starts at softmax(-(f - f_min) / T), 0 = uniform start
    warm_start_mix: float = 0.1          # mixed with uniform

    def tau_final(self, n):
        """Upper bound of the barrier weights: newton-rs's tau = 1 / t = n / epsilon."""
        return n / self.epsilon

    @property
    def initial_step_logit(self):
        p = min(max(self.initial_step_size, 1e-3), 1.0 - 1e-3)
        return math.log(p / (1.0 - p))

    def to_dict(self):
        return asdict(self)


# ---- kernel ----

def laplace_kernel(x, sigma):
    sq = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    dist = jnp.sqrt(jnp.where(sq > 0, sq, 1.0)) * (sq > 0)  # safe gradient at 0
    return jnp.exp(-dist / sigma)


def kernel_and_features(x, sigma, jitter):
    """K and Phi with K + jitter I = Phi^T Phi."""
    kernel = laplace_kernel(x, sigma) + jitter * jnp.eye(x.shape[0], dtype=x.dtype)
    return kernel, jnp.linalg.cholesky(kernel).T


# ---- Newton layers (src/kernelsos_dual.py) ----

def moment_matrix(alpha, phi, lambd):
    m = lambd * jnp.eye(phi.shape[0], dtype=phi.dtype) + (phi * alpha[None, :]) @ phi.T
    return 0.5 * (m + m.T)


def _safe_matrix(m, psd_floor):
    # smallest shift that keeps the Cholesky reliable, without gradient
    shift = jnp.maximum(psd_floor - jnp.linalg.eigvalsh(m)[0], 0.0) + psd_floor * 1e-3
    return m + jax.lax.stop_gradient(shift) * jnp.eye(m.shape[0], dtype=m.dtype)


def newton_direction(alpha, f, phi, lambd, tau, kkt_damping, psd_floor):
    """Equality-constrained Newton direction of tau f^T alpha - logdet M(alpha)."""
    chol = jnp.linalg.cholesky(_safe_matrix(moment_matrix(alpha, phi, lambd), psd_floor))
    q = phi.T @ jax.scipy.linalg.cho_solve((chol, True), phi)
    gradient = tau * f - jnp.diag(q)
    n = alpha.shape[0]
    ones = jnp.ones((n, 1), dtype=alpha.dtype)
    kkt = jnp.block([[q * q + kkt_damping * jnp.eye(n, dtype=alpha.dtype), ones],
                     [ones.T, jnp.zeros((1, 1), dtype=alpha.dtype)]])
    rhs = jnp.concatenate((-gradient, jnp.zeros(1, dtype=alpha.dtype)))
    return jnp.linalg.solve(kkt, rhs)[:n]


def max_step_to_boundary(alpha, direction, phi, lambd, psd_floor):
    """Largest r with M(alpha + r direction) >= psd_floor I."""
    m = moment_matrix(alpha, phi, lambd) - psd_floor * jnp.eye(phi.shape[0], dtype=phi.dtype)
    chol = jnp.linalg.cholesky(m)
    normalized = jax.scipy.linalg.solve_triangular(chol, (phi * direction[None, :]) @ phi.T, lower=True)
    normalized = jax.scipy.linalg.solve_triangular(chol, normalized.T, lower=True).T
    return 1.0 / jnp.maximum(-jnp.linalg.eigvalsh(0.5 * (normalized + normalized.T))[0], 1e-12)


def newton_step(alpha, f, phi, step_size, lambd, tau, kkt_damping, psd_floor):
    direction = newton_direction(alpha, f, phi, lambd, tau, kkt_damping, psd_floor)
    cap = jnp.minimum(1.0, 0.9 * max_step_to_boundary(alpha, direction, phi, lambd, psd_floor))
    updated = alpha + step_size * cap * direction
    return updated + (1.0 - jnp.sum(updated)) / updated.size


def tau_schedule(gamma, tau_start, tau_final):
    """Increasing barrier weights in [tau_start, tau_final], one parameter per layer."""
    positions = jnp.sort(jax.nn.sigmoid(gamma))
    return jnp.exp(jnp.log(tau_start) + (jnp.log(tau_final) - jnp.log(tau_start)) * positions)


def init_tau_parameters(steps):
    positions = jnp.linspace(0.02, 0.98, steps)
    return jnp.log(positions) - jnp.log1p(-positions)


# ---- warm-start GNN (examples/example_3/model.py) ----

def _linear(key, fan_in, fan_out):
    return {"w": jax.random.normal(key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in), "b": jnp.zeros(fan_out)}


def _apply(layer, x):
    return x @ layer["w"] + layer["b"]


def init_gnn(key, dim, context_dim, hidden_dim, layers):
    keys = iter(jax.random.split(key, 4 + 2 * layers))
    params = {"theta_embed": [_linear(next(keys), context_dim, hidden_dim), _linear(next(keys), hidden_dim, hidden_dim)],
              "embed": _linear(next(keys), dim + 1 + hidden_dim, hidden_dim), "layers": []}
    for _ in range(layers):
        params["layers"].append({"self": _linear(next(keys), hidden_dim, hidden_dim),
                                 "message": _linear(next(keys), hidden_dim, hidden_dim)})
    params["head"] = _linear(next(keys), 2 * hidden_dim, 1)
    return params


def predict_logits(params, x, f, kernel, context, lb, ub):
    x = 2.0 * (x - lb) / (ub - lb) - 1.0
    f = (f - jnp.mean(f)) / (jnp.std(f) + 1e-6)
    h_theta = context
    for layer in params["theta_embed"]:
        h_theta = jax.nn.tanh(_apply(layer, h_theta))
    h_theta = jnp.broadcast_to(h_theta, (x.shape[0], h_theta.size))
    h = jax.nn.tanh(_apply(params["embed"], jnp.concatenate((x, f[:, None], h_theta), axis=-1)))
    adjacency = kernel / (jnp.sum(kernel, axis=-1, keepdims=True) + 1e-8)
    for layer in params["layers"]:
        h = jax.nn.tanh(_apply(layer["self"], h) + _apply(layer["message"], adjacency @ h))
    return _apply(params["head"], jnp.concatenate((h, h_theta), axis=-1))[:, 0]


def softmin_start(f, temperature, mix):
    """(1 - mix) softmax(-(f - f_min) / T) + mix uniform: feasible and close to the optimum."""
    return (1.0 - mix) * jax.nn.softmax(-(f - jnp.min(f)) / temperature) + mix / f.shape[0]


def warm_start(params, x, f, kernel, phi, context, lb, ub, cfg):
    """start + r d along the learned unit direction d (sum d = 0), r = warm_start_fraction
    of the distance to the PSD boundary."""
    n = x.shape[0]
    start = (softmin_start(f, cfg.warm_start_temperature, cfg.warm_start_mix) if cfg.warm_start_temperature > 0
             else jnp.full(n, 1.0 / n, dtype=x.dtype))
    logits = predict_logits(params, x, f, kernel, context, lb, ub)
    direction = logits - jnp.mean(logits)
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    alpha = start + cfg.warm_start_fraction * max_step_to_boundary(start, direction, phi, cfg.lambd, cfg.psd_floor) * direction
    return alpha + (1.0 - jnp.sum(alpha)) / n


# ---- schedule network: step size and tau per layer from the context ----

def init_schedule(key, context_dim, steps, initial_step_logit, hidden_dim=8):
    k1, k2 = jax.random.split(key)
    head = _linear(k2, hidden_dim, 2 * steps)
    return {"hidden": _linear(k1, context_dim, hidden_dim),
            "head": {"w": jnp.zeros_like(head["w"]),
                     "b": jnp.concatenate((jnp.full(steps, initial_step_logit), init_tau_parameters(steps)))}}


def predict_schedule(params, context):
    raw = _apply(params["head"], jax.nn.tanh(_apply(params["hidden"], context)))
    return jnp.split(raw, 2)


# ---- full solver ----

def init_solver(key, dim, context_dim, cfg):
    k1, k2 = jax.random.split(key)
    return {"gnn": init_gnn(k1, dim, context_dim, cfg.hidden_dim, cfg.gnn_layers),
            "schedule": init_schedule(k2, context_dim, cfg.newton_steps, cfg.initial_step_logit)}


def solve_path(params, x, f, context, lb, ub, cfg):
    """alpha after the warm start and after every Newton layer, shape (newton_steps + 1, n)."""
    kernel, phi = kernel_and_features(x, cfg.sigma, cfg.kernel_jitter)
    alpha = warm_start(params["gnn"], x, f, kernel, phi, context, lb, ub, cfg)
    step_logit, gamma = predict_schedule(params["schedule"], context)
    taus = tau_schedule(gamma, cfg.tau, cfg.tau_final(x.shape[0]))

    def layer(alpha, step_tau):
        alpha = newton_step(alpha, f, phi, step_tau[0], cfg.lambd, step_tau[1], cfg.kkt_damping, cfg.psd_floor)
        return alpha, alpha

    _, path = jax.lax.scan(layer, alpha, (jax.nn.sigmoid(step_logit), taus))
    return jnp.concatenate((alpha[None], path))


def solve(params, x, f, context, lb, ub, cfg):
    """(alpha, z = X^T alpha) of the last layer."""
    alpha = solve_path(params, x, f, context, lb, ub, cfg)[-1]
    return alpha, alpha @ x


# ---- closed loop ----

def load_learned_solver(solver_dir, lb, ub, context_fn):
    """Trained solver for GlobalMPPI.ksos_learned_solver: (x (n, dim), f (n,), state)
    -> (z, f^T alpha) as numpy. Runs jitted in float64 on the CPU, apart from the
    float32 controller graph. context_fn(qpos, qvel) gives the training context."""
    meta = json.loads((Path(solver_dir) / "config.json").read_text())
    cfg = SolverConfig(**{k: v for k, v in meta.items() if k in SolverConfig.__dataclass_fields__})
    cpu = jax.devices("cpu")[0]
    with enable_x64():
        template = init_solver(jax.random.PRNGKey(0), meta["dim"], meta["context_dim"], cfg)
        params = jax.device_put(load_pytree(template, Path(solver_dir) / "checkpoint.npz"), cpu)
        lb, ub = (jax.device_put(jnp.asarray(b, jnp.float64), cpu) for b in (lb, ub))

        @jax.jit
        def run(x, f, qpos, qvel):
            alpha, z = solve(params, x, f, context_fn(qpos, qvel), lb, ub, cfg)
            return z, alpha @ f

    def solver(x, f, state):
        with enable_x64():
            args = (np.asarray(a, np.float64) for a in (x, f, state.qpos, state.qvel))
            z, cost = run(*(jax.device_put(a, cpu) for a in args))
            return np.asarray(z), float(cost)

    return solver
