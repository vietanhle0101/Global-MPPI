"""Does a softened-contact MJX model give a *useful* gradient for improving
the REAL (hard-contact) rollout cost? Two separate models, never mixed: the
"hard" model is the one GlobalMPPI actually uses (iterations=1, default XML
contact params) for real cost evaluation; the "soft" model has widened
solref/solimp/margin, used ONLY to compute a gradient. Both stay at
iterations=1 -- the while_loop reverse-mode block (check_solver_iterations.py)
applies regardless of contact softness, so more solver iterations is not an
option for either model.

For several (state, candidate-knots) pairs that genuinely involve contact:
  1. Cosine similarity of grad_soft against grad_hard_fd (the real cost's
     finite-difference gradient) -- does the soft gradient even point the
     right way, before we trust it for anything?
  2. The real test: take K normalized gradient-descent steps using
     grad_soft, vs grad_hard_auto (today's raw/unreliable gradient), vs a
     fixed random direction (zero-information baseline) -- then compare the
     REAL (hard) cost after those steps.

    python -m ksos_l2o.check_soft_contact [--states 5] [--candidates 5]
        [--steps 20] [--step-size 0.03] [--eps 1e-3]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .check_mjx_gradients import finite_diff_grad, single_cost
from .problem import rollout_cost, sample_candidate_knots, sample_initial_state


def make_hard_controller():
    return GlobalMPPI(
        PushT(), num_samples=256, noise_level=0.1, temperature=0.1,
        spline_type="cubic", plan_horizon=1.0, num_knots=6,
    )


def make_soft_controller(solref=(0.05, 1.0), solimp=(0.0, 0.9, 0.01, 0.5, 2.0), margin=0.001):
    """Same task/controller, but with widened contact solref/solimp/margin on
    every geom (harmless on the goal markers, which don't collide anyway).
    iterations is left untouched at the XML default (1) -- deliberately not
    varied here, since it's a separate, already-closed-off axis.

    margin=0.001, not the 0.005 first tried: PushT's geometry is tiny (block
    ~0.03-0.04, pusher radius 0.01), and a margin that large relative to the
    scene caused the backward pass to produce NaN (a lineax/equinox linear
    solver failure) even though the forward rollout stayed finite -- margin
    needs to be scaled to the task's own geometry, not used as an absolute
    constant.
    """
    task = PushT()
    task.mj_model.geom_solref[:] = solref
    task.mj_model.geom_solimp[:] = solimp
    task.mj_model.geom_margin[:] = margin
    task.model = mjx.put_model(task.mj_model)
    return GlobalMPPI(
        task, num_samples=256, noise_level=0.1, temperature=0.1,
        spline_type="cubic", plan_horizon=1.0, num_knots=6,
    )


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def descend(cost_hard_jit, grad_fn, knots_flat0, state, lb, ub, steps, step_size):
    """Normalized gradient descent (direction only, fixed step size), scored
    throughout by the REAL (hard) cost. Returns the cost trace, length
    steps+1."""
    x = np.asarray(knots_flat0, dtype=np.float64)
    costs = [float(cost_hard_jit(jnp.asarray(x), state))]
    for _ in range(steps):
        g = np.asarray(grad_fn(jnp.asarray(x), state))
        norm = np.linalg.norm(g)
        step = step_size * g / norm if norm > 1e-10 else np.zeros_like(x)
        x = np.clip(x - step, lb, ub)
        costs.append(float(cost_hard_jit(jnp.asarray(x), state)))
    return np.array(costs)


def main(num_states=5, num_candidates=5, steps=20, step_size=0.03, eps=1e-3, seed=0):
    hard = make_hard_controller()
    soft = make_soft_controller()
    dim = hard.num_knots * hard.task.model.nu
    lb = np.tile(np.asarray(hard.task.u_min), hard.num_knots)
    ub = np.tile(np.asarray(hard.task.u_max), hard.num_knots)

    cost_hard_jit = jax.jit(single_cost, static_argnums=2)
    grad_hard_jit = jax.jit(jax.grad(single_cost), static_argnums=2)
    grad_soft_jit = jax.jit(jax.grad(single_cost), static_argnums=2)
    rollout_jit = jax.jit(lambda state, tk, knots: hard.rollout(state, tk, knots).costs)
    tk = jnp.linspace(0.0, hard.plan_horizon, hard.num_knots)

    grad_fn_soft = lambda x, s: grad_soft_jit(x, s, soft)
    grad_fn_hard = lambda x, s: grad_hard_jit(x, s, hard)
    cost_fn_hard = lambda x, s: cost_hard_jit(x, s, hard)

    state_keys = jax.random.split(jax.random.PRNGKey(seed), num_states)
    cos_sims = []
    final_costs = {"start": [], "soft_grad": [], "hard_grad": [], "random": []}

    rng = np.random.default_rng(seed)
    for s_idx, state_key in enumerate(state_keys):
        state_key, knot_key = jax.random.split(state_key)
        state = sample_initial_state(state_key, hard.task)
        knots = sample_candidate_knots(knot_key, hard.task, hard.num_knots, num_candidates)

        for c in range(num_candidates):
            knots_flat = knots[c].reshape(-1)
            step_costs = rollout_jit(state, tk, knots_flat.reshape(1, hard.num_knots, hard.task.model.nu))
            if float(jnp.std(step_costs[0])) <= 1e-4:
                continue  # only contact-relevant candidates, as in check_mjx_gradients.py

            g_soft = np.asarray(grad_fn_soft(knots_flat, state))
            g_hard_fd = finite_diff_grad(cost_hard_jit, knots_flat, state, hard, eps=eps)
            cos_sims.append(cosine_similarity(g_soft, g_hard_fd))

            random_dir = rng.normal(size=dim)
            random_dir /= np.linalg.norm(random_dir)
            grad_fn_random = lambda x, s, d=random_dir: d  # fixed direction every step

            start_cost = float(cost_fn_hard(knots_flat, state))
            trace_soft = descend(cost_fn_hard, grad_fn_soft, knots_flat, state, lb, ub, steps, step_size)
            trace_hard = descend(cost_fn_hard, grad_fn_hard, knots_flat, state, lb, ub, steps, step_size)
            trace_rand = descend(cost_fn_hard, grad_fn_random, knots_flat, state, lb, ub, steps, step_size)

            final_costs["start"].append(start_cost)
            final_costs["soft_grad"].append(trace_soft[-1])
            final_costs["hard_grad"].append(trace_hard[-1])
            final_costs["random"].append(trace_rand[-1])

            print(
                f"state {s_idx} cand {c:2d}  cos(soft, hard_fd)={cos_sims[-1]:+.3f}  "
                f"start={start_cost:.2f}  after {steps} steps: "
                f"soft_grad={trace_soft[-1]:.2f}  hard_grad={trace_hard[-1]:.2f}  random={trace_rand[-1]:.2f}"
            )

    print("\n=== summary ===")
    print(f"n contact-relevant trials: {len(cos_sims)}")
    print(f"cosine(soft grad, hard finite-diff grad): mean={np.mean(cos_sims):.3f}  "
          f"median={np.median(cos_sims):.3f}  frac>0={np.mean(np.array(cos_sims) > 0):.1%}")
    print(f"\n{'method':<12}{'mean start':>12}{'mean final':>12}{'mean change':>13}{'frac improved':>15}")
    start = np.array(final_costs["start"])
    for name in ("soft_grad", "hard_grad", "random"):
        final = np.array(final_costs[name])
        print(
            f"{name:<12}{start.mean():>12.2f}{final.mean():>12.2f}"
            f"{(final-start).mean():>13.2f}{np.mean(final < start):>15.1%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.states, args.candidates, args.steps, args.step_size, args.eps, args.seed)
