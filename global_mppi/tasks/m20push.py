"""M20BoxPush: a Deep Robotics M20 (wheeled-legged quadruped) pushes a 0.6 x 0.4 x 0.4 m, 5 kg box to a goal pose.

The model (global_mppi/models/m20push/build.py) is a planar M20 with a bumper post ahead of its front wheels and a
planar box, at 50 ms steps. Poses are in the goal's frame, so the goal is the origin with heading 0. The controls
are body-frame velocity commands (forward m/s, sideways m/s, yaw rad/s), as the real robot's stock gait takes them.

The cost follows PushT: 50 times the squared position error, 15 times the squared heading error and the
control-rate term. Some terms are new for a robot this size:

- keep-out: the wheels and knees must stay a few centimetres off the box, so only the bumper pushes;
- approach (whole task): the time a unicycle needs to get its bumper to the push point behind the box, around the
  box. A local sampler never finds the box from a metre away, and the KSOS step drops candidates whose cost never
  falls in the first 20 steps. Its weight fades as the box nears the goal;
- arena (whole task): the robot stays on the 6 m floor;
- contact and square (push phase, `PUSH_WEIGHTS`): with the robot already at a chosen contact on one face (the
  "contact" mocap body), hold the bumper there and the robot square to that face.

In the push phase the third control is not a yaw rate but the robot's heading relative to the pushed face, and
`steer` turns it into a yaw rate by feedback, so the robot turns with the box (see `steer`); use
`global_mppi.algs.SteeredGlobalMPPI`, which applies it in every rollout. Pushing off centre turns the box, and its
face swings into a front wheel unless the robot turns with it; finding that yaw-rate profile by sampling splines
rarely worked.

The box is a rectangle, symmetric under a half turn, so its heading error is taken modulo pi.

For the learned KSOS parts (ksos_l2o), `m20push_context` is the analogue of `pusht_context` and
`sample_push_state` the analogue of `problem.sample_initial_state`: push-phase states with the robot ready at a
contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from global_mppi.models.m20push.build import (FLOOR_HALF, M20, M20_BOX, BoxParams, M20Robot,
                                              build_planner_model)
from global_mppi.task_base import Task

SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class Weights:
    position: float = 50.0
    heading: float = 15.0
    approach: float = 10.0      # per second of travel time to the push point
    approach_fade: float = 0.3  # m: the approach weight scales with the box's distance from the goal up to this
    align_range: float = 0.15   # m: within about this distance of the push point, face the goal
    keep_out: float = 200000.0  # a wheel touching the box (3 cm into the margin) costs 180 per second
    arena: float = 1000.0
    smoothness: float = 2e-4    # PushT's value, divided by dt in control_rate_cost
    margin: float = 0.03        # wheels and knees keep this far from the box, m
    terminal: float = 5.0       # terminal cost = this x the running cost (PushT uses 1)
    contact: float = 0.0        # push phase: bumper to the chosen contact (replaces approach and arena)
    square: float = 0.0         # push phase: robot square to the pushed face, per (1 - cos) of the angle
    steer_gain: float = 0.0     # push phase: the third control is a heading relative to the face (see steer)


# The push phase: the robot is already at a chosen contact (a motion planner put it there), so the cost keeps the
# bumper near there while pushing, and the approach term goes. Pushing straight on for 3 s cost 1,575 against 347
# for standing still, almost all of it keep-out, while a push that stayed square to the face cost 137 to 268.
PUSH_WEIGHTS = Weights(approach=0.0, contact=20.0, square=60.0, steer_gain=2.0)


def heading_error(theta):
    """Heading error of a box that looks the same after a half turn, in (-pi/2, pi/2]."""
    return 0.5 * jnp.arctan2(jnp.sin(2 * theta), jnp.cos(2 * theta))


def path_around_rectangle(a, b, centre, theta, half):
    """Length of the shortest path from a to b (2D) that doesn't cut through the rectangle, and the world
    directions of its legs (three: a path round a rectangle passes at most two corners; legs past b repeat the
    last direction).

    The visibility graph of a, b and the four corners: a straight segment counts only if it misses the
    rectangle's inside (shrunk by 1 mm, so paths along its faces count), and the shortest path comes from five
    rounds of relaxation over the six nodes. A point inside the rectangle falls back to the straight line.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    to_local = lambda p: jnp.stack([c * (p[0] - centre[0]) + s * (p[1] - centre[1]),
                                    -s * (p[0] - centre[0]) + c * (p[1] - centre[1])])
    corners = jnp.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) * half
    nodes = jnp.concatenate([to_local(a)[None], to_local(b)[None], corners])
    inner = half - 1e-3

    def misses(p, q):
        """Liang-Barsky: does segment p-q miss the open rectangle |x| < inner?"""
        d = q - p
        safe = jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (-inner - p) / safe, (inner - p) / safe
        enter = jnp.maximum(jnp.max(jnp.minimum(t1, t2)), 0.0)
        leave = jnp.minimum(jnp.min(jnp.maximum(t1, t2)), 1.0)
        parallel_outside = jnp.any((jnp.abs(d) < 1e-9) & (jnp.abs(p) >= inner))
        return parallel_outside | (enter >= leave)

    lengths = jnp.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    clear = jax.vmap(lambda p: jax.vmap(lambda q: misses(p, q))(nodes))(nodes)
    edges = jnp.where(clear, lengths, jnp.inf)
    to_b = edges[1]                         # shortest distance from each node to b
    for _ in range(5):
        to_b = jnp.minimum(to_b, jnp.min(edges + to_b[None, :], axis=1))
    found = jnp.isfinite(to_b[0])

    def hop(i):
        """Next node from node i on a shortest path to b (b stays at b)."""
        through = (edges[i] + to_b).at[i].set(jnp.inf).at[0].set(jnp.inf)
        return jnp.where(i == 1, 1, jnp.argmin(through))

    stops = [jnp.array(0)]
    for _ in range(3):
        stops.append(jnp.where(found, hop(stops[-1]), 1))
    directions = []
    for start, end in zip(stops[:-1], stops[1:]):
        leg = nodes[end] - nodes[start]
        heading = jnp.arctan2(leg[1], leg[0]) + theta
        directions.append(heading if not directions else jnp.where(start == 1, directions[-1], heading))
    return jnp.where(found, to_b[0], lengths[0, 1]), jnp.stack(directions)


def rectangle_distance(points, centre, theta, half):
    """Signed distance from 2D points (n, 2) to a rectangle with half-sizes `half`, negative inside."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    d = points - centre
    local = jnp.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    q = jnp.abs(local) - half
    return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=1) + jnp.minimum(jnp.max(q, axis=1), 0.0)


def _rotate(theta, v):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


class M20BoxPush(Task):
    """Push the box to the origin, heading 0, with the M20's bumper.

    qpos = (robot x, y, yaw, box x, y, yaw); controls = (forward, sideways, yaw rate), or (forward, sideways,
    heading relative to the pushed face) with `steer_gain`. mocap body 1 ("contact") holds the push phase's chosen
    contact as (x, y, inward normal's angle), all in the box frame; `with_contact` sets it.
    """

    def __init__(self, p: BoxParams = M20_BOX, robot: M20Robot = M20, weights: Weights = Weights(),
                 model: mujoco.MjModel | None = None) -> None:
        mj_model = build_planner_model(p, robot) if model is None else model
        super().__init__(mj_model, trace_sites=["bumper"])
        self.p, self.robot, self.weights = p, robot, weights
        self.half = jnp.array([p.half_x, p.half_y])
        kx, ky, _ = robot.knee
        # Body-frame points that must stay clear of the box, with their radii: wheel centres, then knees.
        self.keep_out_points = jnp.array([[sx * robot.wheel_x, sy * robot.wheel_y] for sx, sy in SIGNS] +
                                         [[sx * kx, sy * ky] for sx, sy in SIGNS])
        self.keep_out_radii = jnp.array([robot.wheel_radius] * 4 + [robot.shank_radius] * 4)
        self.bumper = jnp.array([robot.bumper_x, 0.0])
        q = {name: mj_model.jnt_qposadr[mj_model.joint(name).id] for name in ("robot_x", "box_x", "box_yaw")}
        self.robot_q, self.box_q, self.box_yaw_q = q["robot_x"], q["box_x"], q["box_yaw"]

    def _poses(self, state: mjx.Data):
        robot = state.qpos[self.robot_q:self.robot_q + 3]
        box = state.qpos[self.box_q:self.box_q + 2]
        return robot, box, state.qpos[self.box_yaw_q]

    @staticmethod
    def _to_world(robot, points):
        return robot[:2] + _rotate(robot[2], points)

    @staticmethod
    def with_contact(state: mjx.Data, contact) -> mjx.Data:
        """`state` with the push phase's contact (x, y, inward normal's angle; box frame) in mocap body 1."""
        return state.replace(mocap_pos=state.mocap_pos.at[1].set(jnp.asarray(contact, state.mocap_pos.dtype)))

    def clearance(self, robot, box, theta):
        """Gap between each wheel and knee and the box, m (negative: overlapping)."""
        distance = rectangle_distance(self._to_world(robot, self.keep_out_points), box, theta, self.half)
        return distance - self.keep_out_radii

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        w = self.weights
        robot, box, theta = self._poses(state)
        position = jnp.sum(jnp.square(box))
        heading = jnp.square(heading_error(theta))
        keep_out = jnp.sum(jnp.square(jnp.maximum(w.margin - self.clearance(robot, box, theta), 0.0)))
        if w.contact:       # push phase: hold the bumper on the chosen contact, not the generic push point
            square = 1 - jnp.cos(robot[2] - (theta + state.mocap_pos[1, 2]))
            return (w.position * position + w.heading * heading
                    + w.contact * self.contact_error(state, robot, box, theta) + w.square * square
                    + w.keep_out * keep_out)
        bumper, target = self._to_world(robot, self.bumper[None])[0], self.push_point(box, theta)
        path, legs = path_around_rectangle(bumper, target, box, theta, self.half + self.p.pusher_radius)
        fade = jnp.minimum(jnp.sqrt(position) / w.approach_fade, 1.0)
        # Time a unicycle at full speed needs to get there: the path, plus every turn along it (onto the first
        # leg, at each corner, and into the push direction). The robot can't move sideways, so turning first is
        # progress, and this sees it at once, where the position terms only would after a detour longer than the
        # horizon. Counting only the first and last turns made rounding a corner look like a step backwards.
        facing = jnp.arctan2(-box[1], -box[0])          # the push direction: from the box towards the goal
        turn = lambda a, b: jnp.abs(jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b)))
        near = jnp.exp(-path / w.align_range)
        along = (turn(robot[2], legs[0]) + turn(legs[0], legs[1]) + turn(legs[1], legs[2])
                 + turn(legs[2], facing))
        turning = near * turn(robot[2], facing) + (1 - near) * along
        approach = fade * (path / self.robot.max_forward + turning / self.robot.max_yaw)
        arena = jnp.sum(jnp.square(jnp.maximum(jnp.abs(robot[:2]) - (FLOOR_HALF - 0.5), 0.0)))
        return (w.position * position + w.heading * heading + w.approach * approach + w.keep_out * keep_out
                + w.arena * arena)

    def drop_glitch(self, before: mjx.Data, after: mjx.Data) -> mjx.Data:
        """`after`, unless MJX threw the box or robot in that step; then `before`'s positions and velocities.

        At first contact MJX (3.3.2, float32 and float64 alike) sometimes throws the box 0.86 m and the robot
        0.12 m back in a single 50 ms step, at 17 m/s. MuJoCo's CPU engine never does from the same state, and
        whether MJX does depends on how XLA compiled the surrounding program (a 22-step rollout from that state was
        clean, a 60-step one threw the box), not on the step, solver, contact shape, friction or margins. Real
        motion here is under 1.5 cm a step. A rollout with a thrown box scored 100 to 1,000 times a good push, so
        the planner backed away from boxes it should have pushed.
        """
        robot0, box0, theta0 = self._poses(before)
        robot1, box1, theta1 = self._poses(after)
        turn = jnp.abs(jnp.arctan2(jnp.sin(theta1 - theta0), jnp.cos(theta1 - theta0)))
        glitch = ((jnp.linalg.norm(box1 - box0) > 0.05) | (turn > 0.3)
                  | (jnp.linalg.norm(robot1[:2] - robot0[:2]) > 0.05))
        return after.replace(qpos=jnp.where(glitch, before.qpos, after.qpos),
                             qvel=jnp.where(glitch, before.qvel, after.qvel))

    def steer(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The actuator command for a planned control.

        Without `steer_gain`, the identity. With it (push phase), control[2] is the robot's heading relative to the
        pushed face's inward normal (mocap body 1's angle), and the yaw rate turns the robot towards that heading,
        so following the box as it turns needs no planning. The plant must apply the same law.
        """
        if not self.weights.steer_gain:
            return control
        robot, _, theta = self._poses(state)
        error = theta + state.mocap_pos[1, 2] + control[2] - robot[2]
        error = jnp.arctan2(jnp.sin(error), jnp.cos(error))
        yaw_rate = jnp.clip(self.weights.steer_gain * error, -self.robot.max_yaw, self.robot.max_yaw)
        return control.at[2].set(yaw_rate)

    def contact_error(self, state: mjx.Data, robot, box, theta):
        """Squared distance from the bumper's axis to where it should be to push at the chosen contact."""
        local = state.mocap_pos[1]
        contact = box + _rotate(theta, local[:2])
        inward = theta + local[2]
        target = contact - jnp.array([jnp.cos(inward), jnp.sin(inward)]) * self.p.pusher_radius
        bumper = self._to_world(robot, self.bumper[None])[0]
        return jnp.sum(jnp.square(bumper - target))

    def push_point(self, box, theta):
        """Where the bumper's axis should be to push the box straight at the goal (the origin)."""
        away = box / jnp.maximum(jnp.linalg.norm(box), 1e-6)        # unit vector from the goal to the box
        c, s = jnp.cos(theta), jnp.sin(theta)
        reach = jnp.abs(c * away[0] + s * away[1]) * self.half[0] + jnp.abs(-s * away[0] + c * away[1]) * self.half[1]
        return box + away * (reach + self.p.pusher_radius + 0.02)

    def control_rate_cost(self, du: jax.Array) -> jax.Array:
        return self.weights.smoothness / self.dt * jnp.sum(jnp.square(du))

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.weights.terminal * self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        return {}


# --- learned KSOS parts ----------------------------------------------------------------------------------------

CONTEXT_DIM = 18


def m20push_context(qpos: jax.Array, qvel: jax.Array, contact: jax.Array) -> jax.Array:
    """Context vector (18,) for a push-phase state, as `ksos_l2o.context.pusht_context` for PushT.

    Box pose relative to the goal (x, y, sin 2 theta, cos 2 theta: the box looks the same after a half turn), robot
    pose in the box frame (x, y, sin, cos), the chosen contact in the box frame (x, y, sin and cos of the inward
    normal's angle; `state.mocap_pos[1]`), and qvel (robot then box, world frame).
    """
    robot, box, theta = qpos[0:3], qpos[3:5], qpos[5]
    local = _rotate(-theta, robot[:2] - box)
    relative = robot[2] - theta
    return jnp.concatenate((
        jnp.array([box[0], box[1], jnp.sin(2 * theta), jnp.cos(2 * theta)]),
        jnp.array([local[0], local[1], jnp.sin(relative), jnp.cos(relative)]),
        jnp.array([contact[0], contact[1], jnp.sin(contact[2]), jnp.cos(contact[2])]),
        qvel,
    ))


# Push-ready states for `sample_push_state`.
BOX_DISTANCE = (0.1, 1.2)       # m from the goal
GAP = (0.005, 0.12)             # bumper surface to the box face, m
HEADING_JITTER = 0.15           # rad, robot heading off the push direction
OFFSET_JITTER = 0.15            # of the face's half-length, on top of the turning offset
MAX_OFFSET = 0.4                # turning offset, of the face's half-length, at FULL_TURN_ERROR and beyond
FULL_TURN_ERROR = 0.52          # rad (30 degrees)
SPEED = (0.0, 0.15)             # robot forward speed, m/s
YAW_RATE = 0.1                  # rad/s


def sample_push_state(key: jax.Array, task: M20BoxPush) -> mjx.Data:
    """A random push-phase state, forwarded: the box 0.1-1.2 m from the goal, the robot ready to push it.

    The face and contact are chosen as a motion planner would put the robot there: the face whose inward normal
    points most nearly at the goal, pushed off centre so its torque turns the box towards the goal heading, plus
    jitter. The robot faces into the face (heading jitter up to 0.15 rad) with its bumper 0.5-12 cm off it, moving
    forward up to 0.15 m/s. Poses whose wheels or knees would be within 3 cm of the box fall back to a centred,
    square push, which always clears.
    """
    k_dist, k_dir, k_yaw, k_off, k_gap, k_head, k_speed, k_rate = jax.random.split(key, 8)
    p, robot_params = task.p, task.robot
    distance = jax.random.uniform(k_dist, (), minval=BOX_DISTANCE[0], maxval=BOX_DISTANCE[1])
    direction = jax.random.uniform(k_dir, (), minval=-jnp.pi, maxval=jnp.pi)
    box = distance * jnp.array([jnp.cos(direction), jnp.sin(direction)])
    theta = jax.random.uniform(k_yaw, (), minval=-jnp.pi / 2, maxval=jnp.pi / 2)

    # Faces in the box frame: +x, -x, +y, -y. Pick the one whose inward normal points most nearly at the goal.
    normals = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    inward_world = _rotate(theta, -normals)
    face = jnp.argmax(inward_world @ (-box / distance))
    normal = normals[face]
    tangent = jnp.array([-normal[1], normal[0]])
    half_length = jnp.abs(tangent) @ task.half
    # Torque of pushing at s along the tangent with force along the inward normal: s (t x F); turn against the error.
    error = heading_error(theta)
    t_cross_f = tangent[0] * -normal[1] - tangent[1] * -normal[0]
    size = MAX_OFFSET * half_length * jnp.minimum(1.0, jnp.abs(error) / FULL_TURN_ERROR)
    s = -jnp.sign(error * t_cross_f) * size
    s = s + OFFSET_JITTER * half_length * jax.random.uniform(k_off, (), minval=-1.0, maxval=1.0)
    s = jnp.clip(s, -0.8 * half_length, 0.8 * half_length)

    def ready_pose(s, gap, jitter):
        contact_local = normal * task.half + s * tangent
        inward = _rotate(theta, -normal)
        bumper = box + _rotate(theta, contact_local) - inward * (gap + p.pusher_radius)
        heading = jnp.arctan2(inward[1], inward[0]) + jitter
        base = bumper - robot_params.bumper_x * jnp.array([jnp.cos(heading), jnp.sin(heading)])
        return jnp.array([base[0], base[1], heading]), contact_local

    gap = jax.random.uniform(k_gap, (), minval=GAP[0], maxval=GAP[1])
    jitter = HEADING_JITTER * jax.random.uniform(k_head, (), minval=-1.0, maxval=1.0)
    robot, contact_local = ready_pose(s, gap, jitter)
    safe_robot, safe_contact = ready_pose(0.0, gap, 0.0)
    clear = jnp.min(task.clearance(robot, box, theta)) > task.weights.margin
    robot = jnp.where(clear, robot, safe_robot)
    contact_local = jnp.where(clear, contact_local, safe_contact)

    speed = jax.random.uniform(k_speed, (), minval=SPEED[0], maxval=SPEED[1])
    rate = YAW_RATE * jax.random.uniform(k_rate, (), minval=-1.0, maxval=1.0)
    qpos = jnp.concatenate((robot, box, theta[None]))
    qvel = jnp.array([speed * jnp.cos(robot[2]), speed * jnp.sin(robot[2]), rate, 0.0, 0.0, 0.0])
    normal_angle = jnp.arctan2(-normal[1], -normal[0])
    state = mjx.make_data(task.model)
    state = state.replace(qpos=qpos.astype(state.qpos.dtype), qvel=qvel.astype(state.qvel.dtype),
                          ctrl=jnp.array([speed, 0.0, 0.0], state.ctrl.dtype))
    state = task.with_contact(state, jnp.array([contact_local[0], contact_local[1], normal_angle]))
    return mjx.forward(task.model, state)
