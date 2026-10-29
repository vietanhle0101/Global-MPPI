"""Train a context-conditioned normalizing-flow sampler for GlobalMPPI's
PushT pass-0 KSOS candidate sampling.

Online, label-free, forward-KL training: every training step draws a fresh
batch of random problem instances and pass-0-style uniform candidate knots
(reusing generate_dataset.py's generate_batch), rolls them out through the
real MJX cost, and fits the flow to the resulting importance-weighted
target. No fixed dataset file is needed for training -- unlike the earlier
version of this script, which trained on a pre-generated, finite pool
(ksos_l2o/generate_dataset.py's dataset.npz). Online generation gives
unlimited instance diversity instead of a fixed 8192-context ceiling, and at
small num_samples (e.g. 32, instead of the real controller's 256) it's cheap
enough per step that this is now the simpler design, not just a feasible
one. A single *fixed* validation pool is still generated once up front, so
the plateau/early-stopping logic has a stable target to check against.

Forward KL to the Boltzmann target p*(x|c) ~ exp(-J(x,c)/T), fit by
self-normalized importance sampling against the sampled (uniform-proposal)
candidates:

    L(phi) = -mean_c sum_i w_i * log q_phi(x_i | c),   w = softmax(-J/T)

`log r` (the uniform proposal's log-density) is the same constant for every
candidate within one context, so it cancels exactly in the softmax and is
omitted.

This is the only loss used: a soft-min term (pulling samples toward low-cost
regions via a gradient through the real MJX rollout) was tried several ways
-- iterative self-resampling, randomized/bundled-gradient smoothing, more
solver iterations, softened-contact gradients -- and none improved on
forward-KL alone, so it was dropped rather than kept as unused complexity.

Training loop (epochs x steps-per-epoch, plateau LR, early stopping) mirrors
KSOS_L2O's src/recovery.py:train_step_sizes.

    python -m ksos_l2o.train_sampler [--epochs 30] [--steps-per-epoch 100]
        [--batch-size 64] [--num-samples 32] [--temperature 15.0]
        [--val-size 512] [--output ksos_l2o/data/sampler]
"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoint import save_pytree
from .context import CONTEXT_DIM, pusht_context
from .flow import ConditionalRealNVP
from .generate_dataset import generate_batch
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_initial_state


def importance_weights(
    costs: jax.Array, temperature: float, log_r: jax.Array = None
) -> Tuple[jax.Array, jax.Array]:
    """Self-normalized weights over the last axis and their relative
    effective sample size `1 / (N * sum w^2)` in (0, 1]. `log_r` is the
    proposal's log-density at each candidate; omit it when the proposal is
    uniform (its log-density is the same constant for every candidate within
    a context, so it cancels exactly in the softmax). It must be supplied
    when candidates come from the flow itself (e.g. self-resampling), since
    the flow's density varies per candidate and skipping it would bias the
    weights toward whatever the flow already favors."""
    logits = -costs / temperature
    if log_r is not None:
        logits = logits - log_r
    weights = jax.nn.softmax(logits, axis=-1)
    ess = 1.0 / (weights.shape[-1] * jnp.sum(weights**2, axis=-1))
    return weights, ess


def forward_kl_loss(
    flow: ConditionalRealNVP,
    params,
    contexts: jax.Array,
    x: jax.Array,
    costs: jax.Array,
    lb: jax.Array,
    ub: jax.Array,
    temperature: float,
    log_r: jax.Array = None,
) -> Tuple[jax.Array, jax.Array]:
    """`contexts`: (B, context_dim), `x`: (B, N, dim), `costs`: (B, N) ->
    (mean loss, mean ESS)."""
    log_q = flow.log_prob_batch(params, x, contexts, lb, ub)  # (B, N)
    weights, ess = importance_weights(costs, temperature, log_r)
    weights = jax.lax.stop_gradient(weights)
    loss = -jnp.mean(jnp.sum(weights * log_q, axis=-1))
    return loss, jnp.mean(ess)


def plateau_step(value, best, stale_epochs, learning_rate, *, patience, factor, minimum, min_delta):
    """Reduce the learning rate when `value` (validation loss) stalls for
    `patience` epochs. Mirrors KSOS_L2O's src/training.py:plateau_step."""
    reduced = False
    if value < best - min_delta:
        best = value
        stale_epochs = 0
    else:
        stale_epochs += 1
    if stale_epochs >= patience and learning_rate > minimum:
        new_learning_rate = max(minimum, learning_rate * factor)
        reduced = new_learning_rate < learning_rate
        learning_rate = new_learning_rate
        stale_epochs = 0
    return best, stale_epochs, learning_rate, reduced


def _generate_batch_flat(controller, key, batch_size, num_samples):
    """generate_dataset.generate_batch, with knots flattened to (B, N, dim)
    -- the shape forward_kl_loss/flow.log_prob_batch expect."""
    contexts, knots, costs = generate_batch(controller, key, batch_size, num_samples)
    dim = controller.num_knots * controller.task.model.nu
    return contexts, knots.reshape(batch_size, num_samples, dim), costs


def train(
    epochs=30,
    steps_per_epoch=100,
    batch_size=64,
    num_samples=32,
    temperature=15.0,
    learning_rate=3e-3,
    grad_clip=1.0,
    val_size=512,
    plateau_patience=3,
    plateau_factor=0.5,
    plateau_min_lr=None,
    plateau_min_delta=1e-4,
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4,
    eval_states=32,
    eval_samples=256,
    seed=0,
    output_dir="ksos_l2o/data/sampler",
):
    if plateau_min_lr is None:
        plateau_min_lr = learning_rate / 50

    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu
    lb = jnp.tile(task.u_min, controller.num_knots)
    ub = jnp.tile(task.u_max, controller.num_knots)
    context_dim = CONTEXT_DIM

    generate_batch_jit = jax.jit(_generate_batch_flat, static_argnums=(0, 2, 3))

    # One fixed validation pool, generated once -- so the plateau/early-
    # stopping logic has a stable target, even though training data itself
    # is regenerated fresh every step.
    val_contexts, val_x, val_J = generate_batch_jit(controller, jax.random.PRNGKey(seed + 500), val_size, num_samples)
    val_contexts, val_x, val_J = jax.block_until_ready((val_contexts, val_x, val_J))

    flow = ConditionalRealNVP(dim, context_dim, num_layers=6, hidden_dim=64)
    params = flow.init(jax.random.PRNGKey(seed))

    # inject_hyperparams lets the plateau schedule change the learning rate
    # in place (opt_state.hyperparams['learning_rate'] = ...) without
    # resetting Adam's momentum, unlike rebuilding the optimizer would.
    optimizer = optax.inject_hyperparams(
        lambda learning_rate: optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(learning_rate))
    )(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    loss_fn = jax.jit(lambda p, c, xb, j: forward_kl_loss(flow, p, c, xb, j, lb, ub, temperature))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    val_fn = jax.jit(loss_fn)

    print(
        "=== training config ===\n"
        f"epochs={epochs}  steps_per_epoch={steps_per_epoch}  batch_size={batch_size}  "
        f"num_samples={num_samples}  temperature={temperature}\n"
        f"learning_rate={learning_rate}  grad_clip={grad_clip}  val_size={val_size}\n"
        f"plateau: patience={plateau_patience}  factor={plateau_factor}  min_lr={plateau_min_lr:.2g}  "
        f"early_stopping_patience={early_stopping_patience}\n"
        f"flow: dim={dim}  context_dim={context_dim}  seed={seed}\n"
        f"output={output_dir}\n"
    )

    # Epoch 0: the untrained flow's validation loss, evaluated before any
    # gradient step -- a real baseline to compare against (and the correct
    # starting point for best/plateau/early-stopping tracking, rather than
    # implicitly assuming epoch 1 is already an improvement).
    init_val_loss, init_val_ess = val_fn(params, val_contexts, val_x, val_J)
    init_val_loss, init_val_ess = float(init_val_loss), float(init_val_ess)
    print(f"epoch    0/{epochs}  (untrained)                        val_loss={init_val_loss:.3f}  val_ess={init_val_ess:.3f}\n")

    curves = {"epoch": [], "train_loss": [], "train_ess": [], "val_loss": [], "val_ess": [], "lr": []}
    best = {"val_loss": init_val_loss, "epoch": 0, "params": params}
    plateau_best, plateau_stale = init_val_loss, 0
    early_best, early_stale = init_val_loss, 0
    lr = learning_rate

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "lb_ub.npz", lb=np.asarray(lb), ub=np.asarray(ub))

    def autosave():
        """Overwrite the checkpoint/config/curves with the current state --
        called after every epoch, so an interruption (crash, OOM, a
        contending process on a shared GPU) never loses more than one
        epoch's progress."""
        save_pytree(best["params"], output / "checkpoint.npz")
        (output / "config.json").write_text(
            json.dumps(
                {
                    "epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size,
                    "num_samples": num_samples, "temperature": temperature, "learning_rate": learning_rate,
                    "grad_clip": grad_clip, "val_size": val_size, "plateau_patience": plateau_patience,
                    "plateau_factor": plateau_factor, "plateau_min_lr": plateau_min_lr,
                    "early_stopping_patience": early_stopping_patience, "seed": seed,
                    "dim": dim, "context_dim": context_dim,
                    "best_val_loss": best["val_loss"], "best_epoch": best["epoch"],
                },
                indent=2,
            )
        )
        for name, values in curves.items():
            np.save(output / f"{name}.npy", np.asarray(values))

    key = jax.random.PRNGKey(seed + 1)
    for epoch in range(1, epochs + 1):
        losses, esses = [], []
        for _ in range(steps_per_epoch):
            key, batch_key = jax.random.split(key)
            contexts, x, J = generate_batch_jit(controller, batch_key, batch_size, num_samples)
            (loss, ess), grads = grad_fn(params, contexts, x, J)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
            esses.append(float(ess))
        train_loss, train_ess = float(np.mean(losses)), float(np.mean(esses))

        val_loss, val_ess = val_fn(params, val_contexts, val_x, val_J)
        val_loss, val_ess = float(val_loss), float(val_ess)

        curves["epoch"].append(epoch)
        curves["train_loss"].append(train_loss)
        curves["train_ess"].append(train_ess)
        curves["val_loss"].append(val_loss)
        curves["val_ess"].append(val_ess)
        curves["lr"].append(lr)

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "params": params}

        plateau_best, plateau_stale, lr, reduced = plateau_step(
            val_loss, plateau_best, plateau_stale, lr,
            patience=plateau_patience, factor=plateau_factor, minimum=plateau_min_lr, min_delta=plateau_min_delta,
        )
        opt_state.hyperparams["learning_rate"] = lr

        if val_loss < early_best - early_stopping_min_delta:
            early_best, early_stale = val_loss, 0
        else:
            early_stale += 1

        print(
            f"epoch {epoch:4d}/{epochs}  train_loss={train_loss:.3f}  train_ess={train_ess:.3f}  "
            f"val_loss={val_loss:.3f}  val_ess={val_ess:.3f}  lr={lr:.2g}"
            + ("  (best)" if best["epoch"] == epoch else "")
        )
        if reduced:
            print(f"  -> validation plateaued, reducing lr to {lr:.2g}")

        autosave()

        if early_stopping_patience > 0 and early_stale >= early_stopping_patience:
            print(f"early stopping at epoch {epoch}")
            break

    print(f"\nbest validation loss={best['val_loss']:.3f} at epoch {best['epoch']}")
    params = best["params"]

    # The deployment-relevant check: fresh (unseen, not from training or the
    # validation pool) held-out states, flow samples vs. uniform, scored by
    # the real cost.
    report = evaluate_against_uniform(flow, params, lb, ub, eval_states, eval_samples, seed)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"\nheld-out ({eval_states} fresh states, {eval_samples} samples each): "
        f"uniform mean={report['uniform_mean']:.2f}  flow mean={report['flow_mean']:.2f}  "
        f"uniform best-of-n={report['uniform_best']:.2f}  flow best-of-n={report['flow_best']:.2f}"
    )
    print(f"saved to {output}")
    return flow, params, lb, ub


def evaluate_against_uniform(flow, params, lb, ub, eval_states, eval_samples, seed):
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu

    eval_keys = jax.random.split(jax.random.PRNGKey(seed + 1000), eval_states)
    states = jax.vmap(sample_initial_state, in_axes=(0, None))(eval_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)

    def cost_for(knots):
        knots = knots.reshape(eval_states, eval_samples, controller.num_knots, task.model.nu)
        return jax.vmap(lambda s, k: rollout_cost(controller, s, k))(states, knots)

    z = jax.random.normal(jax.random.PRNGKey(seed + 1001), (eval_states, eval_samples, dim))
    flow_x, _ = flow.sample_batch(params, z, contexts, lb, ub)
    flow_costs = cost_for(flow_x)

    uniform_x = jax.random.uniform(
        jax.random.PRNGKey(seed + 1002), (eval_states, eval_samples, dim), minval=lb, maxval=ub
    )
    uniform_costs = cost_for(uniform_x)

    return {
        "uniform_mean": float(jnp.mean(uniform_costs)),
        "flow_mean": float(jnp.mean(flow_costs)),
        "uniform_best": float(jnp.mean(jnp.min(uniform_costs, axis=1))),
        "flow_best": float(jnp.mean(jnp.min(flow_costs, axis=1))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-min-lr", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--eval-states", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="ksos_l2o/data/sampler")
    args = parser.parse_args()
    train(
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
        num_samples=args.num_samples, temperature=args.temperature, learning_rate=args.learning_rate,
        grad_clip=args.grad_clip, val_size=args.val_size, plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor, plateau_min_lr=args.plateau_min_lr,
        early_stopping_patience=args.early_stopping_patience, eval_states=args.eval_states,
        eval_samples=args.eval_samples, seed=args.seed, output_dir=args.output,
    )
