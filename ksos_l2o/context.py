"""Context for the PushT flow: block pose relative to the goal, pusher
position in the block frame, and qvel.

qpos = (block x, y, theta, pusher x, y), where the pusher qpos is relative to
its body position (0, 0.1).
"""
import jax.numpy as jnp

CONTEXT_DIM = 11

PUSHER_BODY_POS = jnp.array([0.0, 0.1])


def pusht_context(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    goal_pos: jnp.ndarray = jnp.zeros(3),
    goal_quat: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0]),
) -> jnp.ndarray:
    """Context vector of shape (11,). The default goal is at the origin."""
    block_xy = qpos[:2]
    block_theta = qpos[2]
    pusher_world = qpos[3:5] + PUSHER_BODY_POS

    dx, dy = block_xy - goal_pos[:2]
    # goal_quat only rotates about z
    goal_theta = 2.0 * jnp.arctan2(goal_quat[3], goal_quat[0])
    dtheta = block_theta - goal_theta

    offset = pusher_world - block_xy
    cos_t, sin_t = jnp.cos(block_theta), jnp.sin(block_theta)
    pusher_in_block_frame = jnp.array([
        cos_t * offset[0] + sin_t * offset[1],
        -sin_t * offset[0] + cos_t * offset[1],
    ])

    return jnp.concatenate((
        jnp.array([dx, dy, jnp.sin(dtheta), jnp.cos(dtheta)]),
        pusher_in_block_frame,
        qvel,
    ))
