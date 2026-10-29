"""Context features for the PushT flow sampler: (dx, dy, sin dtheta, cos
dtheta, pusher position in the block frame, qvel). Pure JAX, float32, usable
both offline (on logged qpos/qvel/mocap arrays) and inside the jitted
controller.

qpos = (T_x, T_y, T_z, r_x, r_y): block (x, y, theta), pusher (x, y). The
pusher body's reference position is (0, 0.1); qpos[3:5] is an offset added
to it (see global_mppi/tasks/pusht.py's `_close_to_block_err`).
"""
import jax.numpy as jnp

CONTEXT_DIM = 11

_PUSHER_REFERENCE = jnp.array([0.0, 0.1])


def pusht_context(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    goal_pos: jnp.ndarray = jnp.zeros(3),
    goal_quat: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0]),
) -> jnp.ndarray:
    """``(11,)`` context from one state. ``goal_pos``/``goal_quat`` default
    to the fixed goal used today (origin, identity orientation) but are
    accepted explicitly so this still works if the goal is ever moved."""
    block_xy = qpos[:2]
    block_theta = qpos[2]
    pusher_world = qpos[3:5] + _PUSHER_REFERENCE

    dx, dy = block_xy - goal_pos[:2]
    # goal_quat is a pure z-rotation (w, 0, 0, z); its angle is 2*atan2(z, w).
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
