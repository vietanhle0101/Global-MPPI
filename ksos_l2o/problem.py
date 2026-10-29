"""Random PushT problem instances and pass-0-style knot sampling/costing.

A "problem instance" is a random initial state x0 = (qpos, qvel). For each
instance we draw candidate control-spline knots the same way GlobalMPPI's
pass 0 does (uniform over [-u_max, u_max]) and score them with the plain,
single-rollout MPC cost J(U; x0) -- the same cost GlobalMPPI itself computes
via `SamplingBasedController.rollout`, not re-derived here.
"""
import jax
import jax.numpy as jnp
from mujoco import mjx

from global_mppi.alg_base import SamplingBasedController

# Randomization ranges, relative to the fixed goal at the origin. Provisional
# -- see the conversation that picked them; recalibrate once real mid-episode
# logged data is available.
BLOCK_XY_RANGE = 0.2       # block (T_x, T_y) ~ U(-0.2, 0.2)^2
PUSHER_RADIUS_RANGE = (0.05, 0.15)  # pusher offset from the block, polar radius
QVEL_RANGE = 0.3           # all 5 qvel components ~ U(-0.3, 0.3)


def sample_initial_state(key: jax.Array, task) -> mjx.Data:
    """One random PushT state x0 = (qpos, qvel), forwarded so derived
    quantities (sensordata, etc.) used by the cost function are populated."""
    block_key, angle_key, radius_key, phi_key, qvel_key = jax.random.split(key, 5)

    block_xy = jax.random.uniform(
        block_key, (2,), minval=-BLOCK_XY_RANGE, maxval=BLOCK_XY_RANGE
    )
    block_theta = jax.random.uniform(angle_key, (), minval=-jnp.pi, maxval=jnp.pi)

    radius = jax.random.uniform(
        radius_key, (), minval=PUSHER_RADIUS_RANGE[0], maxval=PUSHER_RADIUS_RANGE[1]
    )
    phi = jax.random.uniform(phi_key, (), minval=-jnp.pi, maxval=jnp.pi)
    pusher_world = block_xy + radius * jnp.array([jnp.cos(phi), jnp.sin(phi)])
    # qpos[3:5] is an offset added to the pusher body's reference pos (0, 0.1).
    pusher_qpos = pusher_world - jnp.array([0.0, 0.1])

    qpos = jnp.concatenate((block_xy, block_theta[None], pusher_qpos))
    qvel = jax.random.uniform(qvel_key, (5,), minval=-QVEL_RANGE, maxval=QVEL_RANGE)

    state = mjx.make_data(task.model).replace(qpos=qpos, qvel=qvel)
    return jax.jit(mjx.forward)(task.model, state)


def sample_candidate_knots(
    key: jax.Array, task, num_knots: int, num_samples: int
) -> jax.Array:
    """`(num_samples, num_knots, nu)` candidates, uniform over the full box
    -- the same box GlobalMPPI's pass 0 samples from (center 0, radius
    u_max)."""
    knots = jax.random.uniform(
        key,
        (num_samples, num_knots, task.model.nu),
        minval=task.u_min,
        maxval=task.u_max,
    )
    return jnp.clip(knots, task.u_min, task.u_max)


def rollout_cost(
    controller: SamplingBasedController, state: mjx.Data, knots: jax.Array
) -> jax.Array:
    """`(num_samples,)` plain single-rollout cost J(U; x0) for each
    candidate, via the controller's own rollout (same cost GlobalMPPI uses,
    not re-derived)."""
    tk = jnp.linspace(0.0, controller.plan_horizon, controller.num_knots)
    rollouts = controller.rollout(state, tk, knots)
    return jnp.sum(rollouts.costs, axis=1)
