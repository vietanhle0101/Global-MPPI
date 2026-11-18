"""M20BoxPush: models, push-ready states, steering and the MJX glitch guard (CPU, a few seconds).

    PYTHONPATH=.:ksos-tools python -m pytest tests/test_m20push.py
"""

import math

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from global_mppi.models.m20push.build import M20, M20_BOX, Plant, build_planner_model, box_error
from global_mppi.tasks.m20push import (CONTEXT_DIM, PUSH_WEIGHTS, M20BoxPush, heading_error, m20push_context,
                                       sample_push_state)


@pytest.fixture(scope="module")
def task():
    return M20BoxPush(weights=PUSH_WEIGHTS)


@pytest.fixture(scope="module")
def states(task):
    sample = jax.jit(lambda key: sample_push_state(key, task))
    return [sample(jax.random.PRNGKey(i)) for i in range(50)]


def test_models_build():
    planner = build_planner_model()
    plant = Plant().model
    assert planner.nq == 6 and planner.nu == 3 and planner.opt.timestep == 0.05
    assert planner.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert plant.nu == 3 and plant.joint("box_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert planner.body_mass[planner.body("robot").id] == pytest.approx(M20.mass)


def test_push_states_are_ready_and_clear(task, states):
    """Bumper just off the pushed face, facing into it; no wheel or knee touching the box in the plant."""
    plant = Plant()
    for state in states:
        robot, box = np.asarray(state.qpos[:3]), np.asarray(state.qpos[3:6])
        assert 0.1 - 1e-3 <= box_error(box)[0] <= 1.2 + 1e-3
        assert float(jnp.min(task.clearance(state.qpos[:3], state.qpos[3:5], state.qpos[5]))) > 0.03
        contact = np.asarray(state.mocap_pos[1])
        assert abs(math.remainder(robot[2] - (box[2] + contact[2]), 2 * math.pi)) <= 0.15 + 1e-5
        assert float(task.contact_error(state, state.qpos[:3], state.qpos[3:5], state.qpos[5])) < 0.15 ** 2
        plant.reset(robot=robot, box=box, settle=0.0)
        assert not plant.touching() - {"bumper"}
        assert m20push_context(state.qpos, state.qvel, state.mocap_pos[1]).shape == (CONTEXT_DIM,)


def test_steer_turns_towards_the_relative_heading(task, states):
    state = states[0]
    robot, theta, normal = state.qpos[2], state.qpos[5], state.mocap_pos[1, 2]
    for offset in (-0.2, 0.0, 0.2):
        u = jnp.array([0.1, 0.0, offset])
        command = task.steer(state, u)
        error = math.remainder(float(theta + normal + offset - robot), 2 * math.pi)
        assert float(command[2]) == pytest.approx(np.clip(2.0 * error, -M20.max_yaw, M20.max_yaw), abs=1e-5)
    assert M20BoxPush().steer(state, jnp.array([0.1, 0.0, 0.3]))[2] == pytest.approx(0.3)


def test_drop_glitch_undoes_thrown_boxes(task, states):
    before = states[0]
    thrown = before.replace(qpos=before.qpos.at[3].add(0.5))
    nudged = before.replace(qpos=before.qpos.at[3].add(0.01))
    assert jnp.allclose(task.drop_glitch(before, thrown).qpos, before.qpos)
    assert jnp.allclose(task.drop_glitch(before, nudged).qpos, nudged.qpos)


def test_heading_error_is_modulo_a_half_turn():
    assert float(heading_error(math.pi)) == pytest.approx(0.0, abs=1e-6)
    assert float(heading_error(math.pi / 2 + 0.1)) == pytest.approx(-math.pi / 2 + 0.1, abs=1e-6)


def test_box_fits_under_the_body_and_not_between_the_wheels():
    assert M20.bumper_x - M20_BOX.pusher_radius > M20.wheel_x + M20.wheel_radius + 0.05
    assert 2 * min(M20_BOX.half_x, M20_BOX.half_y) > 2 * (M20.wheel_y - M20.wheel_radius)
