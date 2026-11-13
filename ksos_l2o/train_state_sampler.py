"""Train the state-conditioned flow sampler for PushT.

Loss: importance-weighted forward KL, with data generated on the fly,

    L = -E_state sum_j w_j log q(u_j | state),   w = softmax(-cost / T - log p(u_j | state))

where p mixes uniform samples and samples from the current flow, and T is
picked per state to get the effective sample size --target-ess. The saved
checkpoint is the one with the best best-of-N cost relative to uniform samples.

    python -m ksos_l2o.train_state_sampler --cost lse --sigma-lse 0.1 --output ksos_l2o/data/my_sampler
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
from .problem import BLOCK_XY_RANGE, rollout_cost, sample_initial_state
from .train_sampler import plateau_step

START_QPOS = jnp.array([0.1, 0.1, 1.3, 0.0, 0.0])  # default start in examples/pusht.py
JITTER = jnp.array([0.03, 0.03, 0.3, 0.03, 0.03])
QVEL_RANGE = jnp.array([0.3, 0.3, 0.3, 0.5, 0.5])


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


# near-goal states: block distance range and max |angle|
GOAL_RADIUS = (0.01, 0.06)
GOAL_ANGLE = 0.3
PUSHER_R = 0.01
# the two boxes of the T as (center, half size), in the block frame
T_BOXES = ((jnp.array([0.0, 0.0]), jnp.array([0.01, 0.04])), (jnp.array([0.0, 0.03]), jnp.array([0.03, 0.01])))


def block_sdf(p):
    """Signed distance from p (block frame) to the T."""
    def box(centre, half):
        q = jnp.abs(p - centre) - half
        return jnp.linalg.norm(jnp.maximum(q, 0.0) + 1e-12) + jnp.minimum(jnp.max(q), 0.0)
    return jnp.minimum(box(*T_BOXES[0]), box(*T_BOXES[1]))


def place_pusher(key, block_xy, theta, clearance, contact_prob):
    """Put the pusher in a random direction from the block, at a random gap in
    [0, clearance] from its surface (gap 0 with probability contact_prob)."""
    k_dir, k_gap, k_contact = jax.random.split(key, 3)
    phi = jax.random.uniform(k_dir, (), minval=-jnp.pi, maxval=jnp.pi)
    gap = jnp.where(jax.random.bernoulli(k_contact, contact_prob), 0.0, clearance * jax.random.uniform(k_gap))
    direction = jnp.array([jnp.cos(phi), jnp.sin(phi)])
    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    # last point of the ray that is still inside the T
    ts = jnp.linspace(0.0, 0.08, 81)
    inside = jax.vmap(lambda t: block_sdf(t * direction) <= 0.0)(ts)
    t_out = ts[jnp.max(jnp.where(inside, jnp.arange(ts.shape[0]), 0))]
    point = (t_out + 0.005) * direction
    # a few Newton steps to sdf(point) = PUSHER_R + gap
    target = PUSHER_R + gap
    for _ in range(4):
        point = point + (target - block_sdf(point)) * jax.grad(block_sdf)(point)
    return block_xy + rot @ point - jnp.array([0.0, 0.1])  # pusher qpos is relative to (0, 0.1)


def sample_synthetic(key, task, near_start, log_fraction, min_scale, goal_fraction=0.0, block_range=BLOCK_XY_RANGE,
                     angle_range=float(np.pi), pusher_clearance=0.10, contact_prob=0.3, goal_max=GOAL_RADIUS[1]):
    """Random training state. The block is placed within block_range / angle_range
    of the goal, or (with prob. goal_fraction) between GOAL_RADIUS[0] and goal_max
    from it. With prob. log_fraction the offset is scaled down log-uniformly, and
    with prob. near_start we use a jittered version of the default start."""
    k_near, k_jit, k_vel, k_log, k_scale, k_xy, k_th, k_rad, k_phi, k_goal, k_gr, k_gd, k_ga = jax.random.split(key, 13)
    log_scale = jnp.exp(jax.random.uniform(k_scale, minval=jnp.log(min_scale), maxval=0.0))
    scale = jnp.where(jax.random.bernoulli(k_log, log_fraction), log_scale, 1.0)
    block_xy = block_range * scale * jax.random.uniform(k_xy, (2,), minval=-1.0, maxval=1.0)
    theta = angle_range * scale * jax.random.uniform(k_th, (), minval=-1.0, maxval=1.0)
    goal = jax.random.bernoulli(k_goal, goal_fraction)
    goal_dist = jax.random.uniform(k_gr, (), minval=GOAL_RADIUS[0], maxval=goal_max)
    goal_dir = jax.random.uniform(k_gd, (), minval=-jnp.pi, maxval=jnp.pi)
    block_xy = jnp.where(goal, goal_dist * jnp.array([jnp.cos(goal_dir), jnp.sin(goal_dir)]), block_xy)
    theta = jnp.where(goal, GOAL_ANGLE * jax.random.uniform(k_ga, (), minval=-1.0, maxval=1.0), theta)
    pusher = place_pusher(k_rad, block_xy, theta, pusher_clearance, contact_prob)
    qpos_random = jnp.concatenate((block_xy, theta[None], pusher))
    qvel_random = QVEL_RANGE * jax.random.uniform(k_vel, (5,), minval=-1.0, maxval=1.0)
    qpos_start = START_QPOS + JITTER * jax.random.uniform(k_jit, (5,), minval=-1.0, maxval=1.0)
    near = jax.random.bernoulli(k_near, near_start)
    return forward_state(task, jnp.where(near, qpos_start, qpos_random), jnp.where(near, jnp.zeros(5), qvel_random))


def ess_weights(costs, target_ess, log_prop=0.0, iters=40):
    """w = softmax(-cost / T - log_prop) per row, with T found by bisection so
    that the relative ESS 1 / (N sum w^2) equals target_ess."""
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


def make_batch_fn(ctrl, flow, cost, num_samples, near_start, log_fraction, min_scale, rollout_fraction, derive_samples, goal_fraction=0.0,
                  state_kw=None):
    """Build two jitted functions:

    batch(key, batch_size, sampler_params, uniform_fraction) -> contexts,
        candidates in the unit box, their costs, and log proposal density.
    idle(key, batch_size) -> cost of doing nothing in the same states.

    A rollout_fraction of the states are taken from the middle of a rollout
    (the best of derive_samples uniform plans), to get states during a push.
    """
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
        states = jax.vmap(lambda k: sample_synthetic(k, task, near_start, log_fraction, min_scale, goal_fraction, **(state_kw or {})))(jax.random.split(k_state, batch_size))
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
        log_prop = jnp.zeros((batch_size, num_samples))  # uniform density is 1
        if uniform_fraction < 1.0:
            from_flow, _ = flow.sample_batch(sampler_params, jax.random.normal(k_z, unit.shape), contexts, zeros, ones)
            from_flow = jnp.clip(from_flow, 1e-6, 1.0 - 1e-6)  # same clip as in log_prob
            use_uniform = jax.random.bernoulli(k_mix, uniform_fraction, (batch_size, num_samples, 1))
            unit = jnp.where(use_uniform, unit, from_flow)
            log_flow = flow.log_prob_batch(sampler_params, unit, contexts, zeros, ones)
            log_prop = jnp.logaddexp(jnp.log(uniform_fraction), jnp.log1p(-uniform_fraction) + log_flow)
        costs = jax.vmap(candidate_cost)(states, unit, jax.random.split(k_cost, batch_size))
        return contexts, unit, costs, log_prop

    return jax.jit(batch, static_argnums=(1, 3)), jax.jit(idle, static_argnums=1)


def main(args):
    ctrl = GlobalMPPI(
        PushT(), num_samples=256, noise_level=0.1, temperature=0.1, spline_type="cubic",
        plan_horizon=1.0, num_knots=6, ksos_sampling="uniform",
        ksos_num_samples=args.num_samples, elite_num=args.num_samples,
    )
    if args.cost == "lse":
        ctrl.lse_smoothing_sigma = jnp.full((5,), args.sigma_lse, dtype=jnp.float32)
    dim = ctrl.num_knots * ctrl.task.model.nu
    context_dim = pusht_context(jnp.zeros(5), jnp.zeros(5)).shape[0]

    flow = ConditionalRealNVP(dim, context_dim, num_layers=args.layers, hidden_dim=args.hidden)
    state_kw = dict(block_range=args.block_range, angle_range=args.angle_range,
                    pusher_clearance=args.pusher_clearance, contact_prob=args.contact_prob, goal_max=args.goal_max)
    batch_fn, idle_fn = make_batch_fn(ctrl, flow, args.cost, args.num_samples, args.near_start_prob,
                                      args.log_scale_fraction, args.min_scale, args.rollout_fraction,
                                      args.derive_samples, args.goal_fraction, state_kw)
    params = flow.init(jax.random.PRNGKey(args.seed))
    if args.init:
        params = jax.tree_util.tree_map(jnp.asarray, load_pytree(params, Path(args.init) / "checkpoint.npz"))
    zeros, ones = jnp.zeros(dim), jnp.ones(dim)
    optimizer = optax.inject_hyperparams(
        lambda learning_rate: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    )(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    def loss_fn(prm, c, u, w):
        return -jnp.mean(jnp.sum(w * flow.log_prob_batch(prm, u, c, zeros, ones), axis=-1))

    # weights are computed outside the loss, so no gradient goes through them
    weigh = jax.jit(lambda c, u, cost, log_prop: (c, u, ess_weights(cost, args.target_ess, log_prop)[0]))
    grad_fn, val_fn = jax.jit(jax.value_and_grad(loss_fn)), jax.jit(loss_fn)

    # validation states, with uniform candidates as the reference
    val_key = jax.random.PRNGKey(args.seed + 500)
    val_raw = batch_fn(val_key, args.val_size, params, 1.0)
    val = jax.block_until_ready(weigh(*val_raw))
    uniform_cost = float(val_raw[2].mean())
    uniform_best = val_raw[2].min(axis=-1)
    idle_cost = float(idle_fn(val_key, args.val_size).mean())

    def flow_scores(prm):
        # mean cost and best-of-N of flow samples, and the geometric mean of
        # best-of-N relative to uniform (the number we select on)
        costs = batch_fn(val_key, args.val_size, prm, 0.0)[2]
        best = costs.min(axis=-1)
        ratio = jnp.exp(jnp.mean(jnp.log((best + 1e-4) / (uniform_best + 1e-4))))
        return float(costs.mean()), float(best.mean()), float(ratio)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "sigma_lse": args.sigma_lse if args.cost == "lse" else None,
              "dim": dim, "context_dim": context_dim}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"cost={args.cost}" + (f" sigma_lse={args.sigma_lse}" if args.cost == "lse" else "")
          + f"  {args.num_samples} samples/state, batch {args.batch_size}, target ESS {args.target_ess}, "
          f"uniform fraction {args.uniform_fraction} after {args.warmup_epochs} warm-up epoch(s)")
    print(f"epoch 0 (untrained) val loss {float(val_fn(params, *val)):.3f}")

    best_ratio, plateau_best, stale, early, lr = np.inf, np.inf, 0, 0, args.lr
    if args.init:
        # the model we start from is the one to beat
        best_ratio = plateau_best = flow_scores(params)[2]
        save_pytree(params, out / "checkpoint.npz")
        print(f"epoch 0 baseline best-of-{args.num_samples} ratio vs uniform {best_ratio:.3f}")
    key = jax.random.PRNGKey(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        losses = []
        sampler = params  # proposal for this epoch
        mix = 1.0 if epoch <= args.warmup_epochs else args.uniform_fraction
        for _ in range(args.steps_per_epoch):
            key, sub = jax.random.split(key)
            loss, grads = grad_fn(params, *weigh(*batch_fn(sub, args.batch_size, sampler, mix)))
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
        v = float(val_fn(params, *weigh(*batch_fn(val_key, args.val_size, params, mix))))
        flow_cost, flow_best, ratio = flow_scores(params)
        plateau_best, stale, lr, reduced = plateau_step(ratio, plateau_best, stale, lr, patience=args.lr_patience,
                                                         factor=0.5, minimum=args.lr / 50, min_delta=0.01)
        opt_state.hyperparams["learning_rate"] = lr
        improved = ratio < best_ratio - 0.01
        if improved:
            best_ratio, early = ratio, 0
            save_pytree(params, out / "checkpoint.npz")
        else:
            early += 1
        print(f"epoch {epoch:3d} train {np.mean(losses):.3f} val {v:.3f} "
              f"best-of-{args.num_samples} {flow_best:.3f} (uniform {float(uniform_best.mean()):.3f}) ratio {ratio:.3f} | "
              f"mean cost {flow_cost:.1f} (uniform {uniform_cost:.1f}, idle {idle_cost:.1f}) lr {lr:.1e}"
              f"{' (reduced)' if reduced else ''}{' (best)' if improved else ''}", flush=True)
        if early >= args.stop_patience:
            print("early stopping")
            break
    print(f"best ratio {best_ratio:.3f}; saved {out / 'checkpoint.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cost", choices=["plain", "lse"], default="plain")
    p.add_argument("--sigma-lse", type=float, default=0.1, help="LSE smoothing sigma (only with --cost lse)")
    p.add_argument("--output", default="ksos_l2o/data/sampler_state")
    p.add_argument("--target-ess", type=float, default=0.15)
    p.add_argument("--uniform-fraction", type=float, default=0.25, help="share of uniform candidates, the rest come from the flow")
    p.add_argument("--warmup-epochs", type=int, default=1, help="initial epochs with uniform candidates only")
    p.add_argument("--near-start-prob", type=float, default=0.1, help="share of synthetic states that are jittered benchmark starts")
    p.add_argument("--log-scale-fraction", type=float, default=1 / 3, help="share of synthetic states with log-scaled (near-goal) block offset")
    p.add_argument("--min-scale", type=float, default=0.2, help="smallest block-offset scale for the log-scaled states")
    p.add_argument("--goal-fraction", type=float, default=0.0, help="share of synthetic states with the block 1 cm to --goal-max from the goal at |angle| < 0.3")
    p.add_argument("--block-range", type=float, default=BLOCK_XY_RANGE, help="block x,y drawn within this distance of the goal (m)")
    p.add_argument("--angle-range", type=float, default=float(np.pi), help="block angle drawn within +-this (rad)")
    p.add_argument("--pusher-clearance", type=float, default=0.10, help="pusher placed up to this far from the block surface (m)")
    p.add_argument("--contact-prob", type=float, default=0.3, help="share of states with the pusher exactly touching the block")
    p.add_argument("--rollout-fraction", type=float, default=0.4, help="share of states taken mid-rollout of a good candidate")
    p.add_argument("--derive-samples", type=int, default=32, help="uniform candidates tried to find the rollout that gives a state")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-patience", type=int, default=5, help="epochs without ratio improvement before halving the lr")
    p.add_argument("--stop-patience", type=int, default=15, help="epochs without ratio improvement before stopping")
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--goal-max", type=float, default=GOAL_RADIUS[1], help="largest block distance (m) of the --goal-fraction states")
    p.add_argument("--init", default=None, help="start from this sampler dir (needs the same --layers/--hidden)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
