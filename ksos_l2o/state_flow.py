"""Load the flow trained by train_state_sampler.py and sample knots from it.
The flow works in the unit box, samples are scaled to the control bounds."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from .checkpoint import load_pytree
from .context import pusht_context
from .flow import ConditionalRealNVP


def load_state_sampler(sampler_dir):
    sampler_dir = Path(sampler_dir)
    config = json.loads((sampler_dir / "config.json").read_text())
    flow = ConditionalRealNVP(config["dim"], config["context_dim"], num_layers=config["layers"], hidden_dim=config["hidden"])
    params = load_pytree(flow.init(jax.random.PRNGKey(0)), sampler_dir / "checkpoint.npz")
    return flow, params, config


def sample(flow, params, z, context, lb, ub):
    """z: (N, dim), context: (context_dim,) -> knots (N, dim) in [lb, ub]."""
    unit, _ = flow.sample_batch(params, z[None], context[None], jnp.zeros_like(lb), jnp.ones_like(ub))
    return lb + (ub - lb) * unit[0]


def make_flow_sampler(sampler_dir, task, num_knots):
    """Returns sampler(state, key, n) -> knots (n, num_knots, nu), for
    GlobalMPPI's ksos_flow_sampler, and the sigma_lse the flow was trained with."""
    flow, params, config = load_state_sampler(sampler_dir)
    lb, ub = jnp.tile(task.u_min, num_knots), jnp.tile(task.u_max, num_knots)

    def sampler(state, key, n):
        z = jax.random.normal(key, (n, lb.size))
        knots = sample(flow, params, z, pusht_context(state.qpos, state.qvel), lb, ub)
        return knots.reshape(n, num_knots, task.model.nu)

    return sampler, config["sigma_lse"]
