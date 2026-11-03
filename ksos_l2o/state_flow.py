"""Load and sample the state-conditioned flow trained by train_state_sampler.py.

The flow models u = (knots - lb) / (ub - lb) in [0, 1]^dim over the full control
box, conditioned on the PushT state context; samples are mapped back through the
runtime bounds.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from .checkpoint import load_pytree
from .flow import ConditionalRealNVP


def load_state_sampler(sampler_dir):
    sampler_dir = Path(sampler_dir)
    config = json.loads((sampler_dir / "config.json").read_text())
    flow = ConditionalRealNVP(config["dim"], config["context_dim"], num_layers=config["layers"], hidden_dim=config["hidden"])
    params = load_pytree(flow.init(jax.random.PRNGKey(0)), sampler_dir / "checkpoint.npz")
    return flow, params, config


def sample(flow, params, z, context, lb, ub):
    """`z` (N, dim), `context` (context_dim,), `lb`/`ub` (dim,) -> knots (N, dim) inside [lb, ub]."""
    unit, _ = flow.sample_batch(params, z[None], context[None], jnp.zeros_like(lb), jnp.ones_like(ub))
    return lb + (ub - lb) * unit[0]
