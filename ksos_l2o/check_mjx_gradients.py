"""Gradient sanity check: does jax.grad through the PushT MJX rollout cost
agree with finite differences, and is it (as expected for contact-rich
dynamics) only usable where the rollout actually makes contact?

For several random (state, candidate-knots) pairs: classify the rollout as
"moved" or "didn't move" (position-cost variance across the horizon), then
compare the autodiff gradient against a central finite-difference estimate.
Reuses one jitted cost function throughout (not a fresh closure per call) --
repeated ad-hoc jitting exhausted the CPU LLVM JIT's memory in an earlier
version of this script.

    python -m ksos_l2o.check_mjx_gradients [--states 5] [--candidates 10]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from .inspect_problem import make_controller
from .problem import rollout_cost, sample_candidate_knots, sample_initial_state


def single_cost(knots_flat, state, controller):
    knots = knots_flat.reshape(1, controller.num_knots, controller.task.model.nu)
    return rollout_cost(controller, state, knots)[0]


def finite_diff_grad(cost_jit, knots_flat, state, controller, eps=1e-3):
    dim = knots_flat.shape[0]
    grad = np.zeros(dim)
    base = np.asarray(knots_flat)
    for i in range(dim):
        plus = base.copy(); plus[i] += eps
        minus = base.copy(); minus[i] -= eps
        grad[i] = (
            float(cost_jit(jnp.asarray(plus), state, controller))
            - float(cost_jit(jnp.asarray(minus), state, controller))
        ) / (2 * eps)
    return grad


def main(num_states=5, num_candidates=10, eps=1e-3, seed=0):
    controller = make_controller()
    dim = controller.num_knots * controller.task.model.nu

    cost_jit = jax.jit(single_cost, static_argnums=2)
    cost_grad = jax.jit(jax.grad(single_cost), static_argnums=2)
    rollout_jit = jax.jit(
        lambda state, tk, knots: controller.rollout(state, tk, knots).costs
    )
    tk = jnp.linspace(0.0, controller.plan_horizon, controller.num_knots)

    state_keys = jax.random.split(jax.random.PRNGKey(seed), num_states)
    moved_rel_errs, still_rel_errs = [], []
    moved_grad_norms, still_grad_norms = [], []

    for s, state_key in enumerate(state_keys):
        state_key, knot_key = jax.random.split(state_key)
        state = sample_initial_state(state_key, controller.task)
        knots = sample_candidate_knots(knot_key, controller.task, controller.num_knots, num_candidates)

        for c in range(num_candidates):
            knots_flat = knots[c].reshape(-1)
            step_costs = rollout_jit(state, tk, knots_flat.reshape(1, controller.num_knots, controller.task.model.nu))
            moved = float(jnp.std(step_costs[0]))

            g_auto = np.asarray(cost_grad(knots_flat, state, controller))
            g_fd = finite_diff_grad(cost_jit, knots_flat, state, controller, eps=eps)

            denom = np.maximum(np.linalg.norm(g_auto), np.linalg.norm(g_fd))
            rel_err = np.linalg.norm(g_auto - g_fd) / denom if denom > 1e-8 else 0.0

            tag = "MOVED" if moved > 1e-4 else "still"
            (moved_rel_errs if tag == "MOVED" else still_rel_errs).append(rel_err)
            (moved_grad_norms if tag == "MOVED" else still_grad_norms).append(np.linalg.norm(g_auto))

            print(
                f"state {s} cand {c:2d}  {tag}  movement_score={moved:.5f}  "
                f"|grad_auto|={np.linalg.norm(g_auto):.6f}  |grad_fd|={np.linalg.norm(g_fd):.6f}  "
                f"rel_err={rel_err:.4f}  any_nan={bool(np.any(~np.isfinite(g_auto)))}"
            )

    print("\n=== summary ===")
    print(f"MOVED candidates: {len(moved_rel_errs)}  still candidates: {len(still_rel_errs)}")
    if moved_rel_errs:
        print(f"MOVED  rel_err: mean={np.mean(moved_rel_errs):.4f}  max={np.max(moved_rel_errs):.4f}  "
              f"|grad| mean={np.mean(moved_grad_norms):.6f}")
    if still_rel_errs:
        print(f"still  rel_err: mean={np.mean(still_rel_errs):.4f}  max={np.max(still_rel_errs):.4f}  "
              f"|grad| mean={np.mean(still_grad_norms):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.states, args.candidates, args.eps, args.seed)
