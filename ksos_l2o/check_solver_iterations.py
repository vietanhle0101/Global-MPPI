"""Does giving the contact solver more iterations fix MJX's unreliable
gradients? Training-only, opt-in: builds a *separate* controller whose model
has more Newton/line-search iterations than the real controller's (which
stays at the XML default iterations=1, ls_iterations=4, untouched, for real
control-loop speed). global_mppi/ itself is never modified -- the higher-
fidelity model is built by mutating a cloned PushT task's mj_model.opt
*after* construction and rebuilding its MJX model from that.

Same (state, knots) pairs are reused across every solver setting, so the
comparison is apples-to-apples -- only the solver fidelity changes.

    python -m ksos_l2o.check_solver_iterations [--states 5] [--candidates 5]
        [--iterations 1 4 10 25 100] [--ls-iterations 50]
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


def make_controller_with_solver(iterations, ls_iterations):
    """A GlobalMPPI controller identical to inspect_problem.make_controller(),
    but with the contact solver's iteration counts overridden. Rebuilds the
    MJX model after mutating mj_model.opt, since PushT.__init__ already
    called mjx.put_model with the XML defaults."""
    task = PushT()
    if iterations is not None:
        task.mj_model.opt.iterations = iterations
    if ls_iterations is not None:
        task.mj_model.opt.ls_iterations = ls_iterations
    # tolerance=0 disables the solver's dynamic early-exit while_loop in favor
    # of a static-length (scan-based) loop that always runs the full
    # `iterations` count -- required for reverse-mode autodiff at all once
    # iterations > 1 (google-deepmind/mujoco#2259); harmless at iterations=1.
    task.mj_model.opt.tolerance = 0.0
    task.model = mjx.put_model(task.mj_model)
    return GlobalMPPI(
        task,
        num_samples=256,
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
    )


def evaluate_one_setting(controller, states, knots_list, eps):
    dim = controller.num_knots * controller.task.model.nu
    cost_jit = jax.jit(single_cost, static_argnums=2)
    cost_grad = jax.jit(jax.grad(single_cost), static_argnums=2)
    rollout_jit = jax.jit(lambda state, tk, knots: controller.rollout(state, tk, knots).costs)
    tk = jnp.linspace(0.0, controller.plan_horizon, controller.num_knots)

    moved_rel_errs, moved_grad_norms = [], []
    for state, knots in zip(states, knots_list):
        for c in range(knots.shape[0]):
            knots_flat = knots[c].reshape(-1)
            step_costs = rollout_jit(state, tk, knots_flat.reshape(1, controller.num_knots, controller.task.model.nu))
            moved = float(jnp.std(step_costs[0]))
            if moved <= 1e-4:
                continue  # only compare where the rollout actually moved, as before

            g_auto = np.asarray(cost_grad(knots_flat, state, controller))
            g_fd = finite_diff_grad(cost_jit, knots_flat, state, controller, eps=eps)
            denom = np.maximum(np.linalg.norm(g_auto), np.linalg.norm(g_fd))
            rel_err = np.linalg.norm(g_auto - g_fd) / denom if denom > 1e-8 else 0.0
            moved_rel_errs.append(rel_err)
            moved_grad_norms.append(np.linalg.norm(g_auto))
    return moved_rel_errs, moved_grad_norms


def main(num_states=5, num_candidates=5, eps=1e-3, seed=0, iterations=(1, 4, 10, 25, 100), ls_iterations=50):
    # Fix the (state, knots) draws once using the baseline controller, then
    # reuse the identical inputs for every solver setting.
    baseline = make_controller_with_solver(None, None)
    state_keys = jax.random.split(jax.random.PRNGKey(seed), num_states)
    states, knots_list = [], []
    for state_key in state_keys:
        state_key, knot_key = jax.random.split(state_key)
        states.append(sample_initial_state(state_key, baseline.task))
        knots_list.append(sample_candidate_knots(knot_key, baseline.task, baseline.num_knots, num_candidates))

    print(f"{'iterations':>10}  {'ls_iterations':>13}  {'n_moved':>8}  {'mean rel_err':>13}  {'max rel_err':>12}  {'mean |grad|':>12}")
    for it in iterations:
        controller = baseline if it == 1 and ls_iterations == 4 else make_controller_with_solver(it, ls_iterations)
        rel_errs, grad_norms = evaluate_one_setting(controller, states, knots_list, eps)
        if rel_errs:
            print(
                f"{it:>10}  {ls_iterations:>13}  {len(rel_errs):>8}  "
                f"{np.mean(rel_errs):>13.4f}  {np.max(rel_errs):>12.4f}  {np.mean(grad_norms):>12.4f}"
            )
        else:
            print(f"{it:>10}  {ls_iterations:>13}  {0:>8}  (no moved candidates)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 4, 10, 25, 100])
    parser.add_argument("--ls-iterations", type=int, default=50)
    args = parser.parse_args()
    main(args.states, args.candidates, args.eps, args.seed, tuple(args.iterations), args.ls_iterations)
