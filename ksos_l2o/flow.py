"""Conditional RealNVP over a box [lb, ub]:

    z ~ N(0, I),  y = F(z; c),  x = lb + (ub - lb) * sigmoid(y)

F is a stack of affine coupling layers conditioned on the context c.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

Params = dict


def coupling_masks(dim: int, num_layers: int) -> jax.Array:
    """0/1 masks, shape (num_layers, dim), 1 = passed through unchanged.
    Cycles through even/odd, odd/even, first half, second half."""
    assert dim >= 2
    index = jnp.arange(dim)
    patterns = jnp.stack(
        [index % 2 == 0, index % 2 == 1, index < dim // 2, index >= dim // 2]
    ).astype(jnp.float32)
    return jnp.stack([patterns[layer % 4] for layer in range(num_layers)])


def _dense(key, fan_in: int, fan_out: int, zero: bool = False) -> Params:
    w = (
        jnp.zeros((fan_in, fan_out))
        if zero
        else jax.random.normal(key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
    )
    return {"w": w, "b": jnp.zeros(fan_out)}


def _apply(layer: Params, x: jax.Array) -> jax.Array:
    return x @ layer["w"] + layer["b"]


def init_flow(
    key: jax.Array, dim: int, context_dim: int, num_layers: int, hidden_dim: int
) -> Params:
    """One MLP [z * mask, c] -> (scale, shift) per layer. The last layer is
    zero-initialized, so the flow starts as the identity."""
    layers = []
    for layer_key in jax.random.split(key, num_layers):
        k1, k2 = jax.random.split(layer_key)
        layers.append(
            {
                "hidden1": _dense(k1, dim + context_dim, hidden_dim),
                "hidden2": _dense(k2, hidden_dim, hidden_dim),
                "out": _dense(None, hidden_dim, 2 * dim, zero=True),
            }
        )
    return {"layers": layers}


def _scale_shift(
    layer: Params, kept: jax.Array, context: jax.Array, mask: jax.Array, scale_bound: float
) -> Tuple[jax.Array, jax.Array]:
    h = jnp.concatenate((kept, context))
    h = jax.nn.silu(_apply(layer["hidden1"], h))
    h = jax.nn.silu(_apply(layer["hidden2"], h))
    s_raw, shift = jnp.split(_apply(layer["out"], h), 2)
    scale = scale_bound * jnp.tanh(s_raw / scale_bound)
    return scale * (1.0 - mask), shift * (1.0 - mask)


def flow_forward(
    params: Params, z: jax.Array, context: jax.Array, scale_bound: float = 2.0
) -> Tuple[jax.Array, jax.Array]:
    """y = F(z; c) and log|det dF/dz| for a single z of shape (dim,)."""
    masks = coupling_masks(z.shape[-1], len(params["layers"]))
    logdet = jnp.zeros((), dtype=z.dtype)
    for layer, mask in zip(params["layers"], masks):
        scale, shift = _scale_shift(layer, z * mask, context, mask, scale_bound)
        z = z * mask + (1.0 - mask) * (z * jnp.exp(scale) + shift)
        logdet = logdet + jnp.sum(scale)
    return z, logdet


def flow_inverse(
    params: Params, y: jax.Array, context: jax.Array, scale_bound: float = 2.0
) -> Tuple[jax.Array, jax.Array]:
    """z = F^-1(y; c) and log|det dF^-1/dy|."""
    masks = coupling_masks(y.shape[-1], len(params["layers"]))
    logdet = jnp.zeros((), dtype=y.dtype)
    for layer, mask in reversed(list(zip(params["layers"], masks))):
        scale, shift = _scale_shift(layer, y * mask, context, mask, scale_bound)
        y = y * mask + (1.0 - mask) * (y - shift) * jnp.exp(-scale)
        logdet = logdet - jnp.sum(scale)
    return y, logdet


def _box_logdet(y: jax.Array, lb: jax.Array, ub: jax.Array) -> jax.Array:
    # log|det dx/dy| of x = lb + (ub - lb) * sigmoid(y)
    return jnp.sum(jnp.log(ub - lb) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))


def _log_normal(z: jax.Array) -> jax.Array:
    return -0.5 * jnp.sum(z**2) - z.shape[-1] * jnp.log(jnp.sqrt(2.0 * jnp.pi))


class ConditionalRealNVP:
    """Conditional RealNVP sampler. The box bounds lb, ub are passed per call."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_layers: int = 6,
        hidden_dim: int = 64,
        scale_bound: float = 2.0,
    ) -> None:
        self.dim = dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.scale_bound = scale_bound

    def init(self, key: jax.Array) -> Params:
        return init_flow(key, self.dim, self.context_dim, self.num_layers, self.hidden_dim)

    def forward(self, params: Params, z: jax.Array, context: jax.Array):
        return flow_forward(params, z, context, self.scale_bound)

    def inverse(self, params: Params, y: jax.Array, context: jax.Array):
        return flow_inverse(params, y, context, self.scale_bound)

    def sample(
        self, params: Params, z: jax.Array, context: jax.Array, lb: jax.Array, ub: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Map one latent z to (x, log q(x | c))."""
        y, logdet = self.forward(params, z, context)
        x = lb + (ub - lb) * jax.nn.sigmoid(y)
        return x, _log_normal(z) - logdet - _box_logdet(y, lb, ub)

    def log_prob(
        self, params: Params, x: jax.Array, context: jax.Array, lb: jax.Array, ub: jax.Array,
        eps: float = 1e-6,
    ) -> jax.Array:
        """log q(x | c) of a point x in the box."""
        u = jnp.clip((x - lb) / (ub - lb), eps, 1.0 - eps)
        y = jnp.log(u) - jnp.log1p(-u)
        z, logdet = self.inverse(params, y, context)
        return _log_normal(z) + logdet - _box_logdet(y, lb, ub)

    def sample_batch(
        self, params: Params, z: jax.Array, contexts: jax.Array, lb: jax.Array, ub: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """z: (B, N, dim), contexts: (B, context_dim) -> x (B, N, dim), log_q (B, N)."""
        per_point = jax.vmap(self.sample, in_axes=(None, 0, None, None, None))
        per_context = jax.vmap(per_point, in_axes=(None, 0, 0, None, None))
        return per_context(params, z, contexts, lb, ub)

    def log_prob_batch(
        self, params: Params, x: jax.Array, contexts: jax.Array, lb: jax.Array, ub: jax.Array
    ) -> jax.Array:
        """x: (B, N, dim), contexts: (B, context_dim) -> (B, N)."""
        per_point = jax.vmap(self.log_prob, in_axes=(None, 0, None, None, None))
        per_context = jax.vmap(per_point, in_axes=(None, 0, 0, None, None))
        return per_context(params, x, contexts, lb, ub)
