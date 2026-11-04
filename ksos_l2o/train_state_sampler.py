"""Train the state-conditioned flow with an importance-weighted forward-KL loss, generating data online.

    L = -E_state sum_j w_j log q(u_j | state),   w = softmax(-cost / T - log p(u_j | state))

Candidates come from a proposal p: uniform over the control box with probability --uniform-fraction,
else a draw from the flow as of the epoch start (uniform only for the first --warmup-epochs). T is set
per state so the relative effective sample size equals --target-ess. Validation uses the same recipe on
held-out states; the best checkpoint is the one with the lowest cost of flow-only candidates.
--init starts from an existing checkpoint. --cost plain is one rollout per candidate; --cost lse is the controller's smoothed cost at --sigma-lse.
Training states: synthetic (log-scale distance from the goal) plus --rollout-fraction mid-rollout states.
No gradients through MJX.

    python -m ksos_l2o.train_state_sampler --cost lse --sigma-lse 0.3 --output ksos_l2o/data/sampler_state_lse_ais
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

from .checkpoint import load_pytree, save_pytree
from .context import pusht_context
from .flow import ConditionalRealNVP
from .problem import BLOCK_XY_RANGE, PUSHER_RADIUS_RANGE, rollout_cost, sample_initial_state
from .train_sampler import plateau_step

START_QPOS = jnp.array([0.1, 0.1, 1.3, 0.0, 0.0])  # examples/pusht.py
JITTER = jnp.array([0.03, 0.03, 0.3, 0.03, 0.03])
QVEL_RANGE = jnp.array([0.3, 0.3, 0.3, 0.8, 0.8])


def sample_state(key, task):
    k_mix, k_jit, k_rand, k_vel = jax.random.split(key, 4)
    random_state = sample_initial_state(k_rand, task)
    near_start = jax.random.bernoulli(k_mix, 0.5)
    qpos_start = START_QPOS + JITTER * jax.random.uniform(k_jit, (5,), minval=-1.0, maxval=1.0)
    qvel_random = QVEL_RANGE * jax.random.uniform(k_vel, (5,), minval=-1.0, maxval=1.0)
    qpos = jnp.where(near_start, qpos_start, random_state.qpos)
    qvel = jnp.where(near_start, jnp.zeros(5), qvel_random)
    return forward_state(task, qpos, qvel)


def forward_state(task, qpos, qvel):
    return mjx.forward(task.model, mjx.make_data(task.model).replace(qpos=qpos, qvel=qvel))


def sample_synthetic(key, task, near_start, log_fraction, min_scale):
    """Random training state. With probability `log_fraction` the block's offset and angle from the goal are
    scaled by a log-uniform factor in [min_scale, 1] (near-goal states), else drawn over the full range; with
    probability `near_start` it is a jittered benchmark start."""
    k_near, k_jit, k_vel, k_log, k_scale, k_xy, k_th, k_rad, k_phi = jax.random.split(key, 9)
    log_scale = jnp.exp(jax.random.uniform(k_scale, minval=jnp.log(min_scale), maxval=0.0))
    scale = jnp.where(jax.random.bernoulli(k_log, log_fraction), log_scale, 1.0)
    block_xy = BLOCK_XY_RANGE * scale * jax.random.uniform(k_xy, (2,), minval=-1.0, maxval=1.0)
    theta = jnp.pi * scale * jax.random.uniform(k_th, (), minval=-1.0, maxval=1.0)
    radius = jax.random.uniform(k_rad, (), minval=PUSHER_RADIUS_RANGE[0], maxval=PUSHER_RADIUS_RANGE[1])
    phi = jax.random.uniform(k_phi, (), minval=-jnp.pi, maxval=jnp.pi)
    pusher = block_xy + radius * jnp.array([jnp.cos(phi), jnp.sin(phi)]) - jnp.array([0.0, 0.1])
    qpos_random = jnp.concatenate((block_xy, theta[None], pusher))
    qvel_random = QVEL_RANGE * jax.random.uniform(k_vel, (5,), minval=-1.0, maxval=1.0)
    qpos_start = START_QPOS + JITTER * jax.random.uniform(k_jit, (5,), minval=-1.0, maxval=1.0)
    near = jax.random.bernoulli(k_near, near_start)
    return forward_state(task, jnp.where(near, qpos_start, qpos_random), jnp.where(near, jnp.zeros(5), qvel_random))


def ess_weights(costs, target_ess, log_prop=0.0, iters=40):
    """softmax(-cost / T - log_prop) per row, T chosen so the relative ESS 1/(N sum w^2) = target_ess."""
    n = costs.shape[-1]
    shifted = costs - jnp.min(costs, axis=-1, keepdims=True)
    weights = lambda log_t: jax.nn.softmax(-shifted / jnp.exp(log_t)[..., None] - log_prop, axis=-1)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_flat = 1.0 / (n * jnp.sum(weights(mid) ** 2, axis=-1)) > target_ess  # ESS grows with T
        return jnp.where(too_flat, lo, mid), jnp.where(too_flat, mid, hi)

    lo = jnp.full(costs.shape[:-1], jnp.log(1e-3))
    hi = jnp.full(costs.shape[:-1], jnp.log(1e3))
    lo, hi = jax.lax.fori_loop(0, iters, bisect, (lo, hi))
    log_t = 0.5 * (lo + hi)
    return weights(log_t), jnp.exp(log_t)


def make_batch_fn(ctrl, flow, cost, num_samples, near_start, log_fraction, min_scale, rollout_fraction, derive_samples):
    """-> (batch, idle). jitted batch(key, batch_size, sampler_params, uniform_fraction) -> contexts, unit
    candidates, costs, log proposal; idle(key, batch_size) -> cost of zero controls on batch's states.
    A `rollout_fraction` of the states are mid-rollout states: the state some random time into the best of
    `derive_samples` uniform candidates rolled out from a synthetic state (realistic mid-push states)."""
    task = ctrl.task
    shape = (ctrl.num_knots, task.model.nu)
    dim = shape[0] * shape[1]
    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    lb, ub = jnp.tile(task.u_min, (shape[0], 1)), jnp.tile(task.u_max, (shape[0], 1))
    zeros, ones = jnp.zeros(dim), jnp.ones(dim)

    def candidate_cost(state, unit, key):
        knots = lb + (ub - lb) * unit.reshape(num_samples, *shape)
        if cost == "plain":
            return rollout_cost(ctrl, state, knots)
        return ctrl.lse_smoothing_rollout(state, tk, knots, ctrl.init_params().replace(rng=key, iter=0)).lse_cost

    def rollout_state(state, key):
        k_cand, k_time = jax.random.split(key)
        knots = lb + (ub - lb) * jax.random.uniform(k_cand, (derive_samples, *shape))
        best = knots[jnp.argmin(rollout_cost(ctrl, state, knots))][None]
        controls = ctrl.interp_func(jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps), tk, best)
        states, _ = ctrl.eval_rollouts(ctrl.model, state, controls, best)
        t = jax.random.randint(k_time, (), 0, controls.shape[1])
        return states.qpos[0, t], states.qvel[0, t]

    def draw_states(key, batch_size):
        k_state, k_roll, k_use = jax.random.split(key, 3)
        states = jax.vmap(lambda k: sample_synthetic(k, task, near_start, log_fraction, min_scale))(jax.random.split(k_state, batch_size))
        if rollout_fraction > 0:
            qpos, qvel = jax.vmap(rollout_state)(states, jax.random.split(k_roll, batch_size))
            use = jax.random.bernoulli(k_use, rollout_fraction, (batch_size, 1))
            states = jax.vmap(lambda qp, qv: forward_state(task, qp, qv))(
                jnp.where(use, qpos, states.qpos), jnp.where(use, qvel, states.qvel))
        return states

    def idle(key, batch_size):
        states = draw_states(jax.random.split(key, 5)[0], batch_size)
        knots = jnp.zeros((1, *shape))

        def one(state, k):
            if cost == "plain":
                return rollout_cost(ctrl, state, knots)[0]
            return ctrl.lse_smoothing_rollout(state, tk, knots, ctrl.init_params().replace(rng=k, iter=0)).lse_cost.reshape(())

        return jax.vmap(one)(states, jax.random.split(key, batch_size))

    def batch(key, batch_size, sampler_params, uniform_fraction):
        k_state, k_u, k_z, k_mix, k_cost = jax.random.split(key, 5)
        states = draw_states(k_state, batch_size)
        contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)
        unit = jax.random.uniform(k_u, (batch_size, num_samples, dim))
        log_prop = jnp.zeros((batch_size, num_samples))  # log density of the uniform proposal
        if uniform_fraction < 1.0:
            from_flow, _ = flow.sample_batch(sampler_params, jax.random.normal(k_z, unit.shape), contexts, zeros, ones)
            from_flow = jnp.clip(from_flow, 1e-6, 1.0 - 1e-6)  # where log_prob is evaluated
            use_uniform = jax.random.bernoulli(k_mix, uniform_fraction, (batch_size, num_samples, 1))
            unit = jnp.where(use_uniform, unit, from_flow)
            log_flow = flow.log_prob_batch(sampler_params, unit, contexts, zeros, ones)
            log_prop = jnp.logaddexp(jnp.log(uniform_fraction), jnp.log1p(-uniform_fraction) + log_flow)
        costs = jax.vmap(candidate_cost)(states, unit, jax.random.split(k_cost, batch_size))
        return contexts, unit, costs, log_prop

    return jax.jit(batch, static_argnums=(1, 3)), jax.jit(idle, static_argnums=1)


def main(a):
    ctrl = GlobalMPPI(
        PushT(), num_samples=256, noise_level=0.1, temperature=0.1, spline_type="cubic",
        plan_horizon=1.0, num_knots=6, ksos_sampling="uniform", ksos_num_samples=a.num_samples, elite_num=a.num_samples,
    )
    if a.cost == "lse":
        ctrl.lse_smoothing_sigma = jnp.full((5,), a.sigma_lse, dtype=jnp.float32)
    dim = ctrl.num_knots * ctrl.task.model.nu
    context_dim = pusht_context(jnp.zeros(5), jnp.zeros(5)).shape[0]

    flow = ConditionalRealNVP(dim, context_dim, num_layers=a.layers, hidden_dim=a.hidden)
    batch_fn, idle_fn = make_batch_fn(ctrl, flow, a.cost, a.num_samples, a.near_start_prob, a.log_scale_fraction, a.min_scale,
                                      a.rollout_fraction, a.derive_samples)
    params = flow.init(jax.random.PRNGKey(a.seed))
    if a.init:  # continue from a trained flow (same --layers/--hidden)
        params = jax.tree_util.tree_map(jnp.asarray, load_pytree(params, Path(a.init) / "checkpoint.npz"))
    zeros, ones = jnp.zeros(dim), jnp.ones(dim)
    optimizer = optax.inject_hyperparams(
        lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    )(learning_rate=a.lr)
    opt_state = optimizer.init(params)

    def loss_fn(prm, c, u, w):
        return -jnp.mean(jnp.sum(w * flow.log_prob_batch(prm, u, c, zeros, ones), axis=-1))

    weigh = jax.jit(lambda c, u, cost, log_prop: (c, u, ess_weights(cost, a.target_ess, log_prop)[0]))  # constants for the loss
    grad_fn, val_fn = jax.jit(jax.value_and_grad(loss_fn)), jax.jit(loss_fn)
    val_key = jax.random.PRNGKey(a.seed + 500)
    val_raw = batch_fn(val_key, a.val_size, params, 1.0)  # held-out states; uniform candidates (epoch 0 and the uniform-cost reference)
    val, uniform_cost = jax.block_until_ready(weigh(*val_raw)), float(val_raw[2].mean())
    idle_cost = float(idle_fn(val_key, a.val_size).mean())  # zero controls on the same validation states

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**vars(a), "sigma_lse": a.sigma_lse if a.cost == "lse" else None, "dim": dim, "context_dim": context_dim}, indent=2))
    print(f"cost={a.cost}" + (f" sigma_lse={a.sigma_lse}" if a.cost == "lse" else "")
          + f"  {a.num_samples} samples/state, batch {a.batch_size}, target ESS {a.target_ess}, "
          f"uniform fraction {a.uniform_fraction} after {a.warmup_epochs} warm-up epoch(s)")
    print(f"epoch 0 (untrained) val loss {float(val_fn(params, *val)):.3f}")

    best_cost, plateau_best, stale, early, lr = np.inf, np.inf, 0, 0, a.lr
    key = jax.random.PRNGKey(a.seed + 1)
    for epoch in range(1, a.epochs + 1):
        losses = []
        sampler = params  # this epoch's proposal: the flow as trained so far
        mix = 1.0 if epoch <= a.warmup_epochs else a.uniform_fraction
        for _ in range(a.steps_per_epoch):
            key, sub = jax.random.split(key)
            loss, grads = grad_fn(params, *weigh(*batch_fn(sub, a.batch_size, sampler, mix)))
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
        v = float(val_fn(params, *weigh(*batch_fn(val_key, a.val_size, params, mix))))  # same recipe as train, held-out states
        flow_cost = float(batch_fn(val_key, a.val_size, params, 0.0)[2].mean())  # flow-only candidates, same states
        plateau_best, stale, lr, reduced = plateau_step(flow_cost, plateau_best, stale, lr, patience=3, factor=0.5,
                                                         minimum=a.lr / 50, min_delta=0.05)
        opt_state.hyperparams["learning_rate"] = lr
        improved = flow_cost < best_cost - 0.05
        if improved:
            best_cost, early = flow_cost, 0
            save_pytree(params, out / "checkpoint.npz")
        else:
            early += 1
        print(f"epoch {epoch:3d} train {np.mean(losses):.3f} val {v:.3f} flow cost {flow_cost:.1f} (uniform {uniform_cost:.1f}, idle {idle_cost:.1f}) lr {lr:.1e}"
              f"{' (reduced)' if reduced else ''}{' (best)' if improved else ''}", flush=True)
        if early >= 10:
            print("early stopping"); break
    print(f"best flow cost {best_cost:.1f}; saved {out / 'checkpoint.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cost", choices=["plain", "lse"], default="plain")
    p.add_argument("--sigma-lse", type=float, default=0.1, help="LSE smoothing sigma (only with --cost lse)")
    p.add_argument("--output", default="ksos_l2o/data/sampler_state")
    p.add_argument("--target-ess", type=float, default=0.15)
    p.add_argument("--uniform-fraction", type=float, default=0.25, help="share of uniform candidates; 1 = uniform only (earlier behavior)")
    p.add_argument("--warmup-epochs", type=int, default=1, help="initial epochs with uniform candidates only")
    p.add_argument("--near-start-prob", type=float, default=0.1, help="share of synthetic states that are jittered benchmark starts")
    p.add_argument("--log-scale-fraction", type=float, default=1 / 3, help="share of synthetic states with log-scaled (near-goal) block offset")
    p.add_argument("--min-scale", type=float, default=0.2, help="smallest block-offset scale for the log-scaled states")
    p.add_argument("--rollout-fraction", type=float, default=0.4, help="share of states taken mid-rollout of a good candidate")
    p.add_argument("--derive-samples", type=int, default=32, help="uniform candidates tried to find the rollout that gives a state")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--init", default=None, help="sampler dir whose checkpoint to start from (same --layers/--hidden)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
