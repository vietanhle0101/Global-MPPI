"""GlobalMPPI for tasks whose planned controls pass through a feedback law before the simulator.

A task may define

- `steer(state, u) -> ctrl`: the actuator command for a planned control u in state x (default: u itself);
- `drop_glitch(before, after) -> state`: a guard on each step's result (default: `after`).

Every rollout (KSOS candidates, MPPI samples, the keep check) steps the model with `ctrl = steer(x, u)` and keeps
`drop_glitch(x, step(x))`. The control-rate cost is on the planned controls, not the steered commands, and the
first step isn't charged: `x.ctrl` holds the steered command, which can be in other units than u.

M20BoxPush (global_mppi/tasks/m20push.py) uses both: in its push phase u[2] is the robot's heading relative to
the pushed face and `steer` turns it into a yaw rate, and `drop_glitch` undoes MJX's occasional throw of the box
at first contact.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from global_mppi.alg_base import Trajectory
from global_mppi.algs.global_mppi import GlobalMPPI


class SteeredGlobalMPPI(GlobalMPPI):
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        steer = getattr(self.task, "steer", lambda x, u: u)
        guard = getattr(self.task, "drop_glitch", lambda before, after: after)

        def step(carry, u):
            x, previous = carry
            after = mjx.step(model, x.replace(ctrl=steer(x, u)))
            x = guard(x, after)
            cost = self.dt * self.task.running_cost(x, u) + self.task.control_rate_cost(u - previous)
            return (x, u), (x, cost, self.task.get_trace_sites(x))

        (final, _), (states, costs, sites) = jax.lax.scan(step, (state, controls[0]), controls)
        costs = jnp.append(costs, self.task.terminal_cost(final))
        sites = jnp.append(sites, self.task.get_trace_sites(final)[None], axis=0)
        return states, Trajectory(controls=controls, knots=knots, costs=costs, trace_sites=sites)
