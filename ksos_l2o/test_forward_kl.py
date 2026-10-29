"""Verify the forward-KL loss: unit checks on importance_weights, gradient
sanity on real data, and a short mini-training run showing the loss actually
concentrates the flow's samples on lower-cost regions.

    python -m ksos_l2o.test_forward_kl
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .flow import ConditionalRealNVP
from .train_sampler import forward_kl_loss, importance_weights

DATASET_PATH = "ksos_l2o/data/dataset.npz"
TEMPERATURE = 15.0


def test_uniform_costs_give_uniform_weights_and_full_ess():
    costs = jnp.full((4, 256), 7.0)  # same cost everywhere -> no preference
    weights, ess = importance_weights(costs, TEMPERATURE)
    assert jnp.allclose(weights, 1.0 / 256, atol=1e-6)
    assert jnp.allclose(ess, 1.0, atol=1e-5)


def test_one_dominant_low_cost_concentrates_weight():
    # gap large relative to TEMPERATURE=15, so the softmax is effectively one-hot
    costs = jnp.full((256,), 1000.0).at[0].set(0.0)
    weights, ess = importance_weights(costs[None], TEMPERATURE)
    assert weights[0, 0] > 0.99  # essentially all mass on the one low-cost sample
    assert ess[0] < 0.01  # badly degenerate, as expected


def _load_dataset():
    d = np.load(DATASET_PATH)
    return (
        jnp.asarray(d["context"]),
        jnp.asarray(d["x"]),
        jnp.asarray(d["J"]),
        jnp.asarray(d["lb"]),
        jnp.asarray(d["ub"]),
    )


def test_loss_and_grad_are_finite_on_real_data():
    contexts, x, J, lb, ub = _load_dataset()
    dim, context_dim = x.shape[-1], contexts.shape[-1]
    flow = ConditionalRealNVP(dim, context_dim, num_layers=6, hidden_dim=64)
    params = flow.init(jax.random.PRNGKey(0))

    batch = slice(0, 32)
    loss_fn = lambda p: forward_kl_loss(flow, p, contexts[batch], x[batch], J[batch], lb, ub, TEMPERATURE)
    (loss, ess), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(ess) and 0.0 < ess <= 1.0
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert any(jnp.any(g != 0.0) for g in leaves)  # not a dead/zero gradient


def test_training_concentrates_samples_on_lower_cost_knots():
    contexts, x, J, lb, ub = _load_dataset()
    dim, context_dim = x.shape[-1], contexts.shape[-1]
    flow = ConditionalRealNVP(dim, context_dim, num_layers=6, hidden_dim=64)
    params = flow.init(jax.random.PRNGKey(0))

    train_size = 512
    train_idx = slice(0, train_size)
    batch_size = 64
    steps = 400

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-3))
    opt_state = optimizer.init(params)

    loss_fn = jax.jit(
        lambda p, c, xb, j: forward_kl_loss(flow, p, c, xb, j, lb, ub, TEMPERATURE)
    )
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    key = jax.random.PRNGKey(1)
    losses = []
    for step in range(steps):
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(batch_key, (batch_size,), 0, train_size)
        (loss, ess), grads = grad_fn(params, contexts[train_idx][idx], x[train_idx][idx], J[train_idx][idx])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))

    print(f"loss: first={losses[0]:.3f} last10_mean={np.mean(losses[-10:]):.3f}")
    assert np.mean(losses[-10:]) < 0.7 * losses[0], "training loss did not meaningfully decrease"

    # The real check: for a held-out instance, do flow samples (post-
    # training) beat the logged i.i.d. uniform samples on average cost,
    # scored by the *same* real PushT rollout cost? The dataset's generation
    # scheme (seed=0, batch_size=64, 128 batches) is deterministic, so the
    # exact state for any instance index can be reconstructed and used to
    # roll out fresh flow samples for a true apples-to-apples comparison.
    from .inspect_problem import make_controller
    from .problem import rollout_cost, sample_initial_state

    controller = make_controller()
    held_out = train_size  # first instance after the training slice
    batch_size_gen = 64
    batch_idx, within_idx = divmod(held_out, batch_size_gen)
    batch_key = jax.random.split(jax.random.PRNGKey(0), 128)[batch_idx]
    state_key, _ = jax.random.split(batch_key)
    this_state_key = jax.random.split(state_key, batch_size_gen)[within_idx]
    state = sample_initial_state(this_state_key, controller.task)

    from .context import pusht_context
    reconstructed_context = pusht_context(state.qpos, state.qvel)
    assert jnp.allclose(reconstructed_context, contexts[held_out], atol=1e-4), (
        "reconstructed state's context does not match the logged one -- key derivation is wrong"
    )

    uniform_costs = np.asarray(J[held_out])

    z = jax.random.normal(jax.random.PRNGKey(2), (256, dim))
    sample_fn = jax.vmap(flow.sample, in_axes=(None, 0, None, None, None))
    flow_x, _ = sample_fn(params, z, contexts[held_out], lb, ub)
    flow_knots = flow_x.reshape(256, controller.num_knots, controller.task.model.nu)
    flow_costs = np.asarray(rollout_cost(controller, state, flow_knots))

    print(
        f"held-out instance {held_out}: uniform mean cost={uniform_costs.mean():.2f}  "
        f"flow mean cost={flow_costs.mean():.2f}"
    )
    assert flow_costs.mean() < uniform_costs.mean()


if __name__ == "__main__":
    test_uniform_costs_give_uniform_weights_and_full_ess()
    test_one_dominant_low_cost_concentrates_weight()
    test_loss_and_grad_are_finite_on_real_data()
    test_training_concentrates_samples_on_lower_cost_knots()
    print("all checks passed")
