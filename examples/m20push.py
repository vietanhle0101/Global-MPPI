"""Push phase of M20BoxPush in closed loop: GlobalMPPI pushes the box from a ready pose, against a CPU plant.

    PYTHONPATH=. python examples/m20push.py --seeds 0,1,2
    PYTHONPATH=. python examples/m20push.py --seeds 0 --video          # mp4 in logs/m20push/

Each episode starts from `sample_push_state(PRNGKey(seed))`: the box 0.1-1.2 m from the goal and the robot at a
contact on the face that points most nearly at the goal, as a motion planner would leave it. The plant is the
CPU MuJoCo model in global_mppi/models/m20push/build.py (1 ms steps, a box on 25 sprung contact points), not the
planner's MJX model, and applies the same steering law as `M20BoxPush.steer`. GlobalMPPI replans every 0.5 s
(`--replan`) while simulated time waits for it.

An episode ends when the box holds 10 cm and 10 degrees of the goal for 1 s ("goal"), a wheel or knee touches
the box ("touch"), the bumper is more than 12 cm off the box for 1 s ("lost"), or after `--duration` s. One push
from one contact rarely finishes the job: in the full task a planner then picks a new contact; here the score is
how far the box got. Results go to logs/m20push/results.json.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import jax
import mujoco
import numpy as np

from global_mppi.models.m20push.build import M20, M20_BOX, Plant, box_error, wrap
from global_mppi.simulation.m20push import PUSH_SETTINGS, Loop
from global_mppi.tasks.m20push import PUSH_WEIGHTS, M20BoxPush, rectangle_distance, sample_push_state

GOAL = (0.10, math.radians(10))     # m, rad
HOLD = 1.0                          # s within GOAL before stopping
LOST_GAP, LOST_TIME = 0.12, 1.0     # m, s
START_SPEED = 0.15                  # m/s, the first plan's forward command
FPS = 25
OUT = Path("logs/m20push")
# Pushing, the planner leaves out sideways motion (DR: negligible), so its search is over speed and heading.
PUSH_ROBOT = replace(M20, max_lateral=1e-3)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", default="0", help="Comma-separated seeds, one episode each, e.g. 0,1,2")
parser.add_argument("--duration", type=float, default=30.0, help="Simulated seconds per episode")
parser.add_argument("--video", action="store_true", help="Record a top-view mp4 per episode")
parser.add_argument("--ksos-sampling", choices=["uniform", "sobol"], default=PUSH_SETTINGS.ksos_sampling,
                    help="KSOS candidates ('flow' needs a sampler trained on m20push_context)")
parser.add_argument("--ksos-solver", choices=["newton", "newton-rs"], default=PUSH_SETTINGS.ksos_solver)
parser.add_argument("--ksos-cost", choices=["plain", "lse"], default=PUSH_SETTINGS.ksos_cost)
parser.add_argument("--ksos-num-samples", type=int, default=PUSH_SETTINGS.ksos_samples)
parser.add_argument("--elite-num", type=int, default=PUSH_SETTINGS.elite)
parser.add_argument("--ksos-passes", type=int, default=PUSH_SETTINGS.passes)
parser.add_argument("--mppi-samples", type=int, default=PUSH_SETTINGS.mppi_samples)
parser.add_argument("--horizon", type=float, default=PUSH_SETTINGS.horizon, help="Plan horizon (s)")
parser.add_argument("--knots", type=int, default=PUSH_SETTINGS.knots)
parser.add_argument("--replan", type=float, default=PUSH_SETTINGS.replan, help="Seconds between replans")
parser.add_argument("--commit", type=float, default=PUSH_SETTINGS.commit,
                    help="Keep the previous plan unless the new one is this fraction cheaper")


def steer(plant: Plant, u: np.ndarray, normal_angle: float) -> np.ndarray:
    """The plant's side of M20BoxPush.steer: u[2] is the heading relative to the pushed face."""
    error = wrap(plant.box_pose()[2] + normal_angle + u[2] - plant.robot_pose()[2])
    yaw = np.clip(PUSH_WEIGHTS.steer_gain * error, -M20.max_yaw, M20.max_yaw)
    return np.array([u[0], u[1], yaw])


def episode(seed: int, loop: Loop, sample, plant: Plant, duration: float, video: bool) -> dict:
    start = sample(jax.random.PRNGKey(seed))
    robot, box = np.asarray(start.qpos[:3]), np.asarray(start.qpos[3:6])
    contact = np.asarray(start.mocap_pos[1])
    plant.reset(robot=robot, box=box, settle=0.05)
    loop.seed, loop.switches = seed, 0
    loop.restart(plant.data.time, (START_SPEED, 0.0, 0.0), contact)

    recorder = None
    if video:
        from global_mppi.utils.video import VideoRecorder

        renderer = mujoco.Renderer(plant.model, 480, 640)
        recorder = VideoRecorder(str(OUT), 640, 480, FPS, name=f"m20push_seed{seed}")
        recorder.start()
    dt = plant.model.opt.timestep
    steps, frame_every = round(loop.settings.replan / dt), round(1 / (FPS * dt))
    plan_s, best, held, lost, touches, reason = [], np.inf, 0.0, 0.0, set(), "time"
    half = np.array([M20_BOX.half_x, M20_BOX.half_y])
    while plant.data.time < duration:
        loop.sync(plant)
        t = time.time()
        loop.plan()
        plan_s.append(time.time() - t)
        for k, u in enumerate(loop.commands(plant.data.time, dt, steps)):
            plant.step(steer(plant, u, contact[2]))
            touches |= plant.touching() - {"bumper"}
            if recorder and k % frame_every == 0:
                renderer.update_scene(plant.data, camera="top")
                recorder.add_frame(renderer.render().tobytes())
        pose, box_now = plant.robot_pose(), plant.box_pose()
        distance, turn = box_error(box_now)
        best = min(best, distance)
        bumper = pose[:2] + M20.bumper_x * np.array([math.cos(pose[2]), math.sin(pose[2])])
        gap = float(np.asarray(rectangle_distance(bumper[None], box_now[:2], box_now[2], half))[0])
        lost = lost + loop.settings.replan if gap - M20_BOX.pusher_radius > LOST_GAP else 0.0
        held = held + loop.settings.replan if distance <= GOAL[0] and turn <= GOAL[1] else 0.0
        if touches:
            reason = "touch"
        elif held >= HOLD:
            reason = "goal"
        elif lost >= LOST_TIME:
            reason = "lost"
        if reason != "time":
            break
    if recorder:
        recorder.stop()
        renderer.close()
    distance, turn = box_error(plant.box_pose())
    start_cm, start_deg = box_error(box)
    return {"seed": seed, "end": reason, "start_cm": round(100 * start_cm, 1),
            "start_deg": round(math.degrees(start_deg), 1), "end_cm": round(100 * distance, 1),
            "end_deg": round(math.degrees(turn), 1), "best_cm": round(100 * best, 1),
            "sim_s": round(plant.data.time, 1), "touches": sorted(touches),
            "plan_s_mean": round(float(np.mean(plan_s)), 3), "plans": len(plan_s),
            "plan_switches": loop.switches}


def main() -> None:
    args = parser.parse_args()
    settings = replace(PUSH_SETTINGS, ksos_sampling=args.ksos_sampling, ksos_solver=args.ksos_solver,
                       ksos_cost=args.ksos_cost, ksos_samples=args.ksos_num_samples, elite=args.elite_num,
                       passes=args.ksos_passes, mppi_samples=args.mppi_samples, horizon=args.horizon,
                       knots=args.knots, replan=args.replan, commit=args.commit)
    task = M20BoxPush(robot=PUSH_ROBOT, weights=PUSH_WEIGHTS)
    plant = Plant()
    results = {"settings": asdict(settings), "goal": [GOAL[0], math.degrees(GOAL[1])], "episodes": []}
    OUT.mkdir(parents=True, exist_ok=True)
    loop = Loop(task, settings)          # compiled once, restarted per episode
    sample = jax.jit(lambda key: sample_push_state(key, task))
    for seed in [int(s) for s in args.seeds.split(",")]:
        result = episode(seed, loop, sample, plant, args.duration, args.video)
        results["episodes"].append(result)
        print(f"seed {seed}: {result['end']} after {result['sim_s']} s; box from {result['start_cm']} cm, "
              f"{result['start_deg']} deg to {result['end_cm']} cm, {result['end_deg']} deg "
              f"(best {result['best_cm']} cm); plan {result['plan_s_mean']} s; touches "
              f"{result['touches'] or 'bumper only'}")
        (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    episodes = results["episodes"]
    print(f"{sum(e['end'] == 'goal' for e in episodes)} of {len(episodes)} reached the goal; "
          f"box moved {np.mean([e['start_cm'] - e['end_cm'] for e in episodes]):.1f} cm closer on average")


if __name__ == "__main__":
    main()
