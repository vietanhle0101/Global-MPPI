"""The M20 pushing a box: the planner's MJX model, the CPU plant it controls, and their sizes.

    python -m global_mppi.models.m20push.build      # writes m20push_planner.xml and m20push_plant.xml

MuJoCo and NumPy only, no JAX, so it also loads outside this package's environment (MuJoCo 3.3 to 3.17).

Both models share a planar Deep Robotics M20, a wheeled-legged quadruped. Its base slides and turns on the floor
(z, roll and pitch fixed) and tracks body-frame velocity commands (forward, sideways, yaw rate) through velocity
servos on a site at its centre, as the real robot's stock gait tracks joystick commands. Its collision geoms are
what can reach a 0.4 m box: a round bumper post ahead of the front wheels, the four wheels (spheres) and the lower
legs (capsules). The trunk is 0.50 m up, above the box, so it is drawn but never collides.

The boxes differ on purpose, so the planner is tested against a model it doesn't share:

- planner (`build_planner_model`): MJX-safe and coarse, like PushT's. 50 ms Euler steps; the box slides on x and y
  and turns on a hinge, with the floor's friction as friction loss on those joints. 1.1 mu m g on each slide and
  0.03 mu m g x 1 m on the hinge reproduce the plant's pushes at three offsets to 2 cm and 4 degrees. Two closer
  models failed in MJX at this step: a box standing on 3 x 3 contact points with a vertical slide grew without
  bound in its height within 12 steps from rest, and fixed-depth points gave about 2,750 N of floor load per mm.
- plant (`build_plant_model`, `Plant`): 1 ms steps; a free-jointed box on a 5 x 5 grid of sprung contact points,
  which matched the quasi-static limit-surface model to 0.9 cm.

Contact groups: 1 is the floor against the box's points, 2 is the box's sides against the robot. Poses are in the
goal's frame, so the goal is the origin with heading 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

G = 9.81
HERE = Path(__file__).parent


@dataclass(frozen=True)
class BoxParams:
    """The box and the bumper that pushes it.

    0.6 m wide, 0.4 m deep and 0.4 m tall, 5 kg. The box's top stays below the M20's body (0.50 m up), and it is
    wider than the 0.27 m gap between the wheels, so only the bumper, wheels and lower legs can touch it.
    """
    half_x: float = 0.3            # m
    half_y: float = 0.2
    half_z: float = 0.2
    mass: float = 5.0              # kg
    mu_floor: float = 0.3          # box-floor friction
    mu_pusher: float = 0.8         # bumper-box friction
    pusher_radius: float = 0.05    # the bumper post's radius, m
    pusher_height: float = 0.2     # height of the bumper's centre above the floor, m (mid-box)
    grid: int = 5                  # the plant's box rests on grid x grid contact points
    foot_radius: float = 0.005     # radius of those points, m
    foot_sag: float = 0.002        # how far each point's spring compresses under the box's weight, m
    foot_mass: float = 0.01        # mass of each point, kg (part of `mass`)

    @property
    def friction_force(self) -> float:
        """Force that keeps the box sliding: mu m g."""
        return self.mu_floor * self.mass * G


@dataclass(frozen=True)
class M20Robot:
    """The M20 as a planar body that tracks body-frame velocity commands, with a bumper ahead of its wheels.

    Sizes are from DR's M20.xml (github.com/DeepRoboticsLab/deep_robotics_model) in its RL training's standing
    pose. The robot frame is the base centre on the floor, x forward.
    """
    mass: float = 34.53                 # kg
    yaw_inertia: float = 2.705          # about the base's vertical axis, standing, kg m^2
    wheel_x: float = 0.3141             # wheel centres at (+-wheel_x, +-wheel_y, wheel_radius)
    wheel_y: float = 0.2266
    wheel_radius: float = 0.09
    knee: tuple = (0.2402, 0.1669, 0.329)   # front-left knee; the others mirror it
    shank_radius: float = 0.05
    bumper_x: float = 0.55              # bumper post's axis, ahead of the front wheels' 0.404 m
    bumper_half_height: float = 0.1     # the post spans pusher_height +- this
    max_forward: float = 0.3            # command limits: m/s, m/s, rad/s. DR says sideways motion is negligible.
    max_lateral: float = 0.05
    max_yaw: float = 0.5
    response_time: float = 0.1          # velocity servo time constant, s
    accel_limit: tuple = (1.0, 0.5, 2.0)    # the plant's slew limit on commands: m/s^2, m/s^2, rad/s^2


M20_BOX = BoxParams()
M20 = M20Robot()

PLANNER_DT = 0.05
SLIDE_FRICTION = 1.1        # planner box's slide friction loss, in mu m g
SPIN_FRICTION = 0.03        # its hinge friction loss, in mu m g x 1 m
YAW_ARMATURE = 0.5          # kg m^2 added on the hinge, for stability
FLOOR_HALF = 3.0
# DR's base sits 0.4776 m above its wheel centres in the standing pose.
STAND_BASE_ABOVE_WHEEL = 0.4776

ROBOT_JOINTS = ("robot_x", "robot_y", "robot_yaw")
BOX_JOINTS = ("box_x", "box_y", "box_yaw")      # planner only; the plant's box has a free joint

# Adds drawing-only geoms (e.g. the robot's meshes) to the robot body: (spec, body) -> None.
Decorate = Callable[[mujoco.MjSpec, mujoco.MjsBody], None]


def _scene(p: BoxParams, timestep: float) -> mujoco.MjSpec:
    """Floor, light, cameras and the goal outline."""
    spec = mujoco.MjSpec()
    spec.modelname = "m20_push"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0, 0, 0]

    spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.82, 0.84, 0.86],
                     rgb2=[0.72, 0.74, 0.76], width=256, height=256)
    floor_material = spec.add_material(name="floor", texrepeat=[1, 1], texuniform=True)
    floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checker"

    world = spec.worldbody
    light = world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.4, 0.4, 0.4])
    try:                                    # MuJoCo 3.3
        light.directional = True
    except AttributeError:                  # 3.17
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[FLOOR_HALF, FLOOR_HALF, 0.1],
                   material="floor", condim=3, contype=1, conaffinity=1, friction=[p.mu_floor, 0.005, 0.0001])
    # Frames a 4.4 x 3.3 m area around the goal, which is at the origin.
    world.add_camera(name="top", pos=[0, 0, 5.3], quat=[1, 0, 0, 0])
    world.add_camera(name="perspective", pos=[2.6, -3.4, 2.6], xyaxes=[0.8, 0.6, 0, -0.3, 0.4, 0.866])
    goal = world.add_body(name="goal", mocap=True, pos=[0, 0, 0.001])
    goal.add_geom(name="goal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.half_x, p.half_y, 0.001],
                  contype=0, conaffinity=0, rgba=[0.2, 0.7, 0.3, 0.35])
    return spec


def add_robot(spec: mujoco.MjSpec, robot: M20Robot, p: BoxParams, decorate: Decorate | None = None,
              sphere_bumper: bool = False) -> None:
    """The planar M20: three joints, three velocity servos, and the geoms that can touch the box."""
    body = spec.worldbody.add_body(name="robot")
    # The mass sits in an invisible block on the yaw axis, sized so its yaw inertia m (a^2 + b^2) / 3 is the
    # standing M20's; every other geom has no mass.
    a = 0.4
    b = math.sqrt(3 * robot.yaw_inertia / robot.mass - a * a)
    body.add_geom(name="mass", type=mujoco.mjtGeom.mjGEOM_BOX, size=[a, b, 0.05], pos=[0, 0, 0.5],
                  mass=robot.mass, contype=0, conaffinity=0, group=3, rgba=[0, 0, 0, 0])
    body.add_joint(name="robot_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    body.add_joint(name="robot_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    body.add_joint(name="robot_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    body.add_site(name="base", pos=[0, 0, 0])
    body.add_site(name="bumper", pos=[robot.bumper_x, 0, p.pusher_height])

    touch = dict(condim=3, contype=2, conaffinity=2, friction=[p.mu_pusher, 0.005, 0.0001], density=0)
    low, high = p.pusher_height - robot.bumper_half_height, p.pusher_height + robot.bumper_half_height
    if sphere_bumper:
        # The planner's bumper: a sphere at mid-box height, so it meets a face at one point. The post meets a face
        # along a line, as two contacts at the same depth, and at 50 ms that first contact sometimes threw the box
        # 0.86 m in one step in MJX.
        body.add_geom(name="bumper", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[p.pusher_radius, 0, 0],
                      pos=[robot.bumper_x, 0, p.pusher_height], rgba=[0.85, 0.3, 0.15, 1], **touch)
    else:
        body.add_geom(name="bumper", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[p.pusher_radius, 0, 0],
                      fromto=[robot.bumper_x, 0, low, robot.bumper_x, 0, high], rgba=[0.85, 0.3, 0.15, 1],
                      **touch)
    alpha = 0 if decorate else 1
    kx, ky, kz = robot.knee
    for leg, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("hl", -1, 1), ("hr", -1, -1)):
        wheel = [sx * robot.wheel_x, sy * robot.wheel_y, robot.wheel_radius]
        body.add_geom(name=f"{leg}_wheel", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[robot.wheel_radius, 0, 0],
                      pos=wheel, rgba=[0.15, 0.15, 0.15, alpha], **touch)
        body.add_geom(name=f"{leg}_shank", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[robot.shank_radius, 0, 0],
                      fromto=[sx * kx, sy * ky, kz, *wheel], rgba=[0.3, 0.3, 0.3, alpha], **touch)

    # The bracket from the nose down to the bumper post: drawn only, it stays above the box.
    base_z = robot.wheel_radius + STAND_BASE_ABOVE_WHEEL
    visual = dict(contype=0, conaffinity=0, density=0, group=2)
    body.add_geom(name="bracket", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.02, 0, 0],
                  fromto=[0.36, 0, base_z - 0.04, robot.bumper_x, 0, base_z - 0.04], rgba=[0.5, 0.5, 0.5, 1],
                  **visual)
    body.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.02, 0, 0],
                  fromto=[robot.bumper_x, 0, base_z - 0.04, robot.bumper_x, 0, high], rgba=[0.5, 0.5, 0.5, 1],
                  **visual)
    if decorate:
        decorate(spec, body)
    else:
        body.add_geom(name="trunk", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.375, 0.115, 0.07],
                      pos=[0, 0, base_z], rgba=[0.7, 0.7, 0.7, 1], **visual)

    gains = (robot.mass / robot.response_time, robot.mass / robot.response_time,
             robot.yaw_inertia / robot.response_time)
    limits = (robot.max_forward, robot.max_lateral, robot.max_yaw)
    for name, axis, kv, limit in zip(("forward", "lateral", "yaw"), (0, 1, 5), gains, limits):
        gear = [0.0] * 6
        gear[axis] = 1.0
        act = spec.add_actuator(name=name, target="base", trntype=mujoco.mjtTrn.mjTRN_SITE, gear=gear)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kv
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0, 0, -kv]
        act.ctrllimited = True
        act.ctrlrange = [-limit, limit]


def _set(element, field: str, value: float) -> None:
    """Set a joint's spring or damper: a number in MuJoCo 3.3, an array's first (linear) term in 3.17."""
    try:
        setattr(element, field, value)
    except (TypeError, ValueError):
        values = np.zeros_like(np.asarray(getattr(element, field), dtype=float))
        values.flat[0] = value
        setattr(element, field, values)


def _add_planner_box(spec: mujoco.MjSpec, p: BoxParams) -> None:
    """A box that slides on x and y and turns about z, held back by friction loss on those joints."""
    friction = p.friction_force
    box = spec.worldbody.add_body(name="box", pos=[0, 0, p.half_z + 0.001])
    for name, kind, axis, loss, damping in (
            ("box_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0], SLIDE_FRICTION * friction, 2.0),
            ("box_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0], SLIDE_FRICTION * friction, 2.0),
            ("box_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1], SPIN_FRICTION * friction, 0.2)):
        joint = box.add_joint(name=name, type=kind, axis=axis)
        joint.frictionloss = loss
        _set(joint, "damping", damping)
        if name == "box_yaw":
            # The box's own 0.22 kg m^2 and 0.04 N m of spin friction left a push exactly square to a face, in one
            # repeatable state, launching the box at 17 m/s in MJX. Extra inertia on the hinge stops it and
            # barely changes a slow push.
            joint.armature = YAW_ARMATURE
    # Group 2 only: nothing under the box touches the floor.
    box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.half_x, p.half_y, p.half_z], mass=p.mass,
                 condim=3, contype=2, conaffinity=2, friction=[p.mu_pusher, 0.005, 0.0001],
                 rgba=[0.25, 0.45, 0.7, 1])


def foot_positions(p: BoxParams) -> np.ndarray:
    """The grid x grid contact points under the box, at the centres of equal cells (box frame, xy)."""
    xs = (np.arange(p.grid) + 0.5) / p.grid * 2 * p.half_x - p.half_x
    ys = (np.arange(p.grid) + 0.5) / p.grid * 2 * p.half_y - p.half_y
    return np.array([(x, y) for x in xs for y in ys])


def add_sprung_box(spec: mujoco.MjSpec, p: BoxParams) -> None:
    """The plant's free-jointed box on its grid of sprung contact points (group 1 below, 2 for the sides).

    Each point's spring sags `foot_sag` under its share of the weight and is critically damped, so the floor load
    spreads over the whole footprint as the quasi-static pushing model assumes.
    """
    feet = foot_positions(p)
    share = p.mass / len(feet)
    stiffness = share * G / p.foot_sag
    damping = 2 * math.sqrt(stiffness * share)
    box = spec.worldbody.add_body(name="box", pos=[0, 0, p.half_z + p.foot_radius])
    box.add_freejoint(name="box_free")
    box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.half_x, p.half_y, p.half_z],
                 mass=p.mass - len(feet) * p.foot_mass, condim=3, contype=2, conaffinity=2,
                 friction=[p.mu_pusher, 0, 0], rgba=[0.25, 0.45, 0.7, 1])
    for i, (x, y) in enumerate(feet):
        foot = box.add_body(name=f"foot_{i}", pos=[x, y, -p.half_z])
        joint = foot.add_joint(name=f"foot_{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        _set(joint, "stiffness", stiffness)
        _set(joint, "damping", damping)
        foot.add_geom(name=f"foot_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[p.foot_radius, 0, 0],
                      mass=p.foot_mass, condim=3, contype=1, conaffinity=1, friction=[p.mu_floor, 0, 0],
                      rgba=[0.2, 0.2, 0.2, 1])


def planner_spec(p: BoxParams = M20_BOX, robot: M20Robot = M20, timestep: float = PLANNER_DT,
                 iterations: int = 4) -> mujoco.MjSpec:
    spec = _scene(p, timestep)
    # Euler, as PushT: at 50 ms MuJoCo's CPU engine with implicitfast launches the box at 15 m/s on first contact.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.iterations = iterations
    spec.option.ls_iterations = 4
    add_robot(spec, robot, p, sphere_bumper=True)
    _add_planner_box(spec, p)
    # Not a body in the scene: its position carries the push phase's chosen contact to the cost, as (x, y in the
    # box frame, inward normal's angle in the box frame), so it can change without recompiling.
    spec.worldbody.add_body(name="contact", mocap=True, pos=[0, 0, -1])
    return spec


def build_planner_model(p: BoxParams = M20_BOX, robot: M20Robot = M20, timestep: float = PLANNER_DT,
                        iterations: int = 4) -> mujoco.MjModel:
    return planner_spec(p, robot, timestep, iterations).compile()


def plant_spec(p: BoxParams = M20_BOX, robot: M20Robot = M20, decorate: Decorate | None = None) -> mujoco.MjSpec:
    spec = _scene(p, 0.001)
    # Elliptic cones and no-slip iterations stop the box creeping under a load below its friction limit.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100
    spec.option.noslip_iterations = 5
    add_robot(spec, robot, p, decorate)
    add_sprung_box(spec, p)
    return spec


def build_plant_model(p: BoxParams = M20_BOX, robot: M20Robot = M20,
                      decorate: Decorate | None = None) -> mujoco.MjModel:
    return plant_spec(p, robot, decorate).compile()


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def box_error(box, goal=np.zeros(3)):
    """Distance (m) and heading error (rad) of the box from the goal; a half turn counts as no error."""
    turn = wrap(box[2] - goal[2])
    turn = 0.5 * math.atan2(math.sin(2 * turn), math.cos(2 * turn))
    return float(np.hypot(*(np.asarray(box[:2]) - goal[:2]))), abs(turn)


class Plant:
    """The plant: put the robot and box somewhere, command the robot, read poses as an overhead tracker would."""

    def __init__(self, p: BoxParams = M20_BOX, robot: M20Robot = M20, decorate: Decorate | None = None):
        self.p, self.robot = p, robot
        self.model = build_plant_model(p, robot, decorate)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._robot_q = m.jnt_qposadr[m.joint("robot_x").id]
        self._box_q = m.jnt_qposadr[m.joint("box_free").id]
        self._box_v = m.jnt_dofadr[m.joint("box_free").id]
        self._foot_q = [m.jnt_qposadr[m.joint(f"foot_{i}").id] for i in range(p.grid ** 2)]
        self._box_geom = m.geom("box").id
        self._robot_body = m.body("robot").id
        self.command = np.zeros(3)
        self.limits = np.array([robot.max_forward, robot.max_lateral, robot.max_yaw])

    def reset(self, robot=(0.0, 0.0, 0.0), box=(0.0, 0.0, 0.0), goal=(0.0, 0.0, 0.0), settle: float = 0.3) -> None:
        p, d = self.p, self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self._robot_q:self._robot_q + 3] = robot
        d.qpos[self._box_q:self._box_q + 3] = (box[0], box[1], p.half_z + p.foot_radius - p.foot_sag)
        d.qpos[self._box_q + 3:self._box_q + 7] = (math.cos(box[2] / 2), 0, 0, math.sin(box[2] / 2))
        d.qpos[self._foot_q] = p.foot_sag
        d.mocap_pos[0] = (goal[0], goal[1], 0.001)
        d.mocap_quat[0] = (math.cos(goal[2] / 2), 0, 0, math.sin(goal[2] / 2))
        self.command[:] = 0
        mujoco.mj_forward(self.model, d)
        mujoco.mj_step(self.model, d, round(settle / self.model.opt.timestep))
        d.time = 0.0

    def step(self, u, accel_limit=None) -> None:
        """One physics step towards command u (forward, lateral, yaw rate), slew-limited per step."""
        dt = self.model.opt.timestep
        accel = np.array(self.robot.accel_limit if accel_limit is None else accel_limit)
        wanted = np.clip(u, -self.limits, self.limits)
        self.command += np.clip(wanted - self.command, -accel * dt, accel * dt)
        self.data.ctrl[:3] = self.command
        mujoco.mj_step(self.model, self.data)

    def robot_pose(self) -> np.ndarray:
        q = self.data.qpos[self._robot_q:self._robot_q + 3].copy()
        q[2] = wrap(q[2])
        return q

    def robot_velocity(self) -> np.ndarray:
        return self.data.qvel[self._robot_q:self._robot_q + 3].copy()

    def box_pose(self) -> np.ndarray:
        q = self.data.qpos
        w, _, _, z = q[self._box_q + 3:self._box_q + 7]
        return np.array([q[self._box_q], q[self._box_q + 1], 2 * math.atan2(z, w)])

    def box_velocity(self) -> np.ndarray:
        v = self.data.qvel
        return np.array([v[self._box_v], v[self._box_v + 1], v[self._box_v + 5]])

    def touching(self) -> set:
        """Names of the robot's geoms in contact with the box."""
        names = set()
        for c in self.data.contact[:self.data.ncon]:
            if self._box_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._box_geom else c.geom1
                if self.model.geom_bodyid[other] == self._robot_body:
                    names.add(self.model.geom(other).name)
        return names


def main() -> None:
    for name, spec in (("m20push_planner", planner_spec()), ("m20push_plant", plant_spec())):
        spec.compile()
        (HERE / f"{name}.xml").write_text(spec.to_xml())
        print(f"wrote {HERE / name}.xml")


if __name__ == "__main__":
    main()
