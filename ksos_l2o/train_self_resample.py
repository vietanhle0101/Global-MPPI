"""Iterative self-resampling (adaptive importance sampling / learned
Cross-Entropy Method): after bootstrapping on the fixed uniform dataset,
repeatedly draw fresh samples from the CURRENT flow for new random states,
score them with the real (forward-only) MPC cost, and continue forward-KL
training on those self-samples -- concentrating mass toward low-cost regions
over rounds, without ever differentiating through MJX.

Each round's weights correct for the flow's own (non-uniform) proposal
density via `log_r = stop_gradient(log q_phi(x|c))`, so this stays a
consistent importance-sampling estimate of the Boltzmann target rather than
a runaway positive-feedback loop onto whatever the flow already favors.

    python -m ksos_l2o.train_self_resample [--rounds 10] [--round-batch 128]
        [--samples 128] [--steps-per-round 10] [--bootstrap-steps 800]
        [--temperature-scale 1.0]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .context import CONTEXT_DIM, pusht_context
from .flow import ConditionalRealNVP
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state
from .train_sampler import forward_kl_loss

DATASET_PATH = "ksos_l2o/data/dataset.npz"
TEMPERATURE = 15.0


def bootstrap(flow, params, optimizer, opt_state, steps, batch_size, seed):
    d = np.load(DATASET_PATH)
    contexts = jnp.asarray(d["context"])
    x = jnp.asarray(d["x"])
    J = jnp.asarray(d["J"])
    lb = jnp.asarray(d["lb"])
    ub = jnp.asarray(d["ub"])
    n = contexts.shape[0]

    loss_fn = jax.jit(lambda p, c, xb, j: forward_kl_loss(flow, p, c, xb, j, lb, ub, TEMPERATURE))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    key = jax.random.PRNGKey(seed)
    for step in range(steps):
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        (loss, ess), grads = grad_fn(params, contexts[idx], x[idx], J[idx])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if step % 200 == 0 or step == steps - 1:
            print(f"bootstrap step {step:4d}  loss={float(loss):.3f}  ess={float(ess):.3f}")
    return params, opt_state, lb, ub


def _draw_round_batch(controller, flow, params, key, batch_size, num_samples, lb, ub):
    task = controller.task
    state_key, z_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, batch_size)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(state_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    z = jax.random.normal(z_key, (batch_size, num_samples, lb.shape[0]))
    x, _ = flow.sample_batch(params, z, contexts, lb, ub)
    log_r = jax.lax.stop_gradient(flow.log_prob_batch(params, x, contexts, lb, ub))

    knots = x.reshape(batch_size, num_samples, controller.num_knots, task.model.nu)
    costs = jax.vmap(lambda s, k: rollout_cost(controller, s, k))(states, knots)
    return contexts, x, costs, log_r


def make_draw_round_batch(controller, flow, batch_size, num_samples):
    """JIT-compiled as a whole (like generate_dataset.py's generate_batch),
    so XLA can dead-code-eliminate the unused per-step state trajectories
    `eval_rollouts` stacks internally -- calling the pieces eagerly instead
    ran out of GPU memory at this batch size."""
    return jax.jit(
        lambda params, key, lb, ub: _draw_round_batch(
            controller, flow, params, key, batch_size, num_samples, lb, ub
        )
    )


def main(
    rounds=10,
    round_batch=128,
    num_samples=128,
    steps_per_round=10,
    bootstrap_steps=800,
    temperature_scale=1.0,
    eval_states=16,
    eval_samples=256,
    seed=0,
):
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu

    flow = ConditionalRealNVP(dim, CONTEXT_DIM, num_layers=6, hidden_dim=64)
    params = flow.init(jax.random.PRNGKey(0))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-3))
    opt_state = optimizer.init(params)

    print("=== bootstrap on fixed uniform dataset ===")
    params, opt_state, lb, ub = bootstrap(
        flow, params, optimizer, opt_state, bootstrap_steps, batch_size=64, seed=seed
    )

    # Fixed held-out eval states, reused every round for a comparable metric.
    eval_keys = jax.random.split(jax.random.PRNGKey(999), eval_states)
    held_out_states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    held_out_contexts = jax.vmap(pusht_context)(held_out_states.qpos, held_out_states.qvel)

    @jax.jit
    def _evaluate_jit(params, z):
        x, _ = flow.sample_batch(params, z, held_out_contexts, lb, ub)
        knots = x.reshape(eval_states, eval_samples, controller.num_knots, task.model.nu)
        costs = jax.vmap(lambda s, k: rollout_cost(controller, s, k))(held_out_states, knots)
        return jnp.mean(costs), jnp.mean(jnp.min(costs, axis=1))

    def evaluate(params):
        z = jax.random.normal(jax.random.PRNGKey(1), (eval_states, eval_samples, dim))
        mean_cost, best_cost = _evaluate_jit(params, z)
        return float(mean_cost), float(best_cost)

    mean0, best0 = evaluate(params)
    print(f"round  0 (post-bootstrap): eval mean cost={mean0:.2f}  mean best-of-{eval_samples}={best0:.2f}")

    # Fresh optimizer state for the resampling phase: bootstrap's Adam momentum
    # was accumulated on a very different (large, fixed-dataset) gradient
    # distribution and shouldn't carry over into the small, round-by-round
    # updates below.
    opt_state = optimizer.init(params)

    loss_fn = jax.jit(lambda p, c, xb, j, lr, t: forward_kl_loss(flow, p, c, xb, j, lb, ub, t, lr))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    draw_round_batch_jit = make_draw_round_batch(controller, flow, round_batch, num_samples)

    key = jax.random.PRNGKey(seed + 1)
    for r in range(1, rounds + 1):
        key, round_key = jax.random.split(key)
        contexts, x, costs, log_r = draw_round_batch_jit(params, round_key, lb, ub)
        contexts, x, costs, log_r = jax.block_until_ready((contexts, x, costs, log_r))

        # Adaptive temperature: scaled to *this round's own* within-context
        # cost spread, rather than the fixed value calibrated for the much
        # wider-spread uniform dataset. Otherwise, as the flow concentrates
        # and each round's candidates get less spread out, the same absolute
        # temperature acts increasingly sharp, which is what collapsed ESS
        # last time.
        round_temperature = temperature_scale * float(jnp.mean(jnp.std(costs, axis=-1)))
        round_temperature = max(round_temperature, 1e-3)

        for _ in range(steps_per_round):
            (loss, ess), grads = grad_fn(params, contexts, x, costs, log_r, round_temperature)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        mean_cost, best_cost = evaluate(params)
        print(
            f"round {r:2d}  T={round_temperature:.2f}  train_loss={float(loss):.3f}  ess={float(ess):.3f}  "
            f"self-sample cost mean={float(costs.mean()):.2f}  |  "
            f"eval mean cost={mean_cost:.2f}  mean best-of-{eval_samples}={best_cost:.2f}"
        )

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--round-batch", type=int, default=128)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps-per-round", type=int, default=10)
    parser.add_argument("--bootstrap-steps", type=int, default=800)
    parser.add_argument("--temperature-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        rounds=args.rounds,
        round_batch=args.round_batch,
        num_samples=args.samples,
        steps_per_round=args.steps_per_round,
        bootstrap_steps=args.bootstrap_steps,
        temperature_scale=args.temperature_scale,
        seed=args.seed,
    )
