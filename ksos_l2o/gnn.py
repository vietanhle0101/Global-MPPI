"""Permutation-equivariant GNN warm start for the KernelSOS dual.

Nodes are the n candidate samples; edge weights are the (row-normalised)
Laplace Gram matrix, so message passing follows the same kernel the dual
uses. Output is a feasible alpha (positive, sums to 1) via softmax. The
logits include a learnable `-a * f_normalised` skip, so the untrained net
already behaves like a softmin average. Pure float32, problem-agnostic.
"""
import jax
import jax.numpy as jnp

from . import dual

Params = dict


def _dense(key, fan_in, fan_out, zero=False):
    w = jnp.zeros((fan_in, fan_out)) if zero else jax.random.normal(key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
    return {"w": w, "b": jnp.zeros(fan_out)}


def _apply(layer, x):
    return x @ layer["w"] + layer["b"]


def init_gnn(key, dim, context_dim, hidden_dim=64, num_layers=3, init_temperature_inv=5.0) -> Params:
    keys = jax.random.split(key, num_layers + 1)
    node_in = dim + 2 + context_dim
    return {
        "embed": _dense(keys[0], node_in, hidden_dim),
        "layers": [_dense(k, 3 * hidden_dim, hidden_dim) for k in keys[1:]],
        "out": _dense(jax.random.PRNGKey(0), hidden_dim, 1, zero=True),
        "log_a": jnp.log(jnp.asarray(init_temperature_inv)),
    }


def warm_start(params, x, fn, context, ub, sigma):
    """x: (n, dim), fn: (n,) normalised costs in [0,1], context: (context_dim,). Returns alpha (n,)."""
    n = x.shape[0]
    rank = jnp.argsort(jnp.argsort(fn)) / (n - 1)
    node = jnp.concatenate(
        [x / ub, fn[:, None], rank[:, None], jnp.broadcast_to(context, (n, context.shape[0]))], axis=-1
    )
    gram = dual.laplace_gram(x, sigma)
    gram = gram / jnp.sum(gram, axis=1, keepdims=True)
    h = jax.nn.relu(_apply(params["embed"], node))
    for layer in params["layers"]:
        msg = jnp.concatenate([h, gram @ h, jnp.broadcast_to(jnp.mean(h, axis=0), h.shape)], axis=-1)
        h = h + jax.nn.relu(_apply(layer, msg))
    logits = _apply(params["out"], h)[:, 0] - jnp.exp(params["log_a"]) * fn
    return jax.nn.softmax(logits)
