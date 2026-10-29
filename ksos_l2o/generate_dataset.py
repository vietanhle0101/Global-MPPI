"""Generate a fixed pool of random (context, x, J) training instances for the
PushT flow sampler, via problem.py's random-instance + pass-0-style knot
sampling + rollout cost machinery. Run once; train_sampler.py trains on the
resulting .npz (no MJX calls during training itself).

    python -m ksos_l2o.generate_dataset [--instances 4000] [--samples 256]
        [--batch-size 16] [--seed 0] [--output ksos_l2o/data/dataset.npz]
"""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from .context import CONTEXT_DIM, pusht_context
from .inspect_problem import make_controller
from .problem import rollout_cost, sample_candidate_knots, sample_initial_state


def generate_batch(controller, key, batch_size, num_samples):
    """One batch of `batch_size` random instances, each with `num_samples`
    candidate knots and their costs."""
    task = controller.task
    state_key, knot_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, batch_size)
    knot_keys = jax.random.split(knot_key, batch_size)

    states = jax.vmap(sample_initial_state, in_axes=(0, None))(state_keys, task)
    contexts = jax.vmap(pusht_context)(states.qpos, states.qvel)
    knots = jax.vmap(sample_candidate_knots, in_axes=(0, None, None, None))(
        knot_keys, task, controller.num_knots, num_samples
    )
    costs = jax.vmap(lambda s, k: rollout_cost(controller, s, k))(states, knots)
    return contexts, knots, costs


def main(
    num_instances=4000,
    num_samples=256,
    batch_size=16,
    seed=0,
    output="ksos_l2o/data/dataset.npz",
):
    controller = make_controller()
    task = controller.task
    dim = controller.num_knots * task.model.nu
    lb = np.tile(np.asarray(task.u_min), controller.num_knots)
    ub = np.tile(np.asarray(task.u_max), controller.num_knots)

    generate_batch_jit = jax.jit(generate_batch, static_argnums=(0, 2, 3))

    num_batches = (num_instances + batch_size - 1) // batch_size
    keys = jax.random.split(jax.random.PRNGKey(seed), num_batches)

    all_contexts, all_x, all_J = [], [], []
    start = time.perf_counter()
    done = 0
    for i, key in enumerate(keys):
        this_batch = min(batch_size, num_instances - done)
        contexts, knots, costs = generate_batch_jit(controller, key, this_batch, num_samples)
        contexts, knots, costs = jax.block_until_ready((contexts, knots, costs))
        all_contexts.append(np.asarray(contexts))
        all_x.append(np.asarray(knots).reshape(this_batch, num_samples, dim))
        all_J.append(np.asarray(costs))
        done += this_batch
        elapsed = time.perf_counter() - start
        print(
            f"batch {i + 1}/{num_batches}  ({done}/{num_instances} instances)  "
            f"{elapsed:.1f}s elapsed  ({elapsed / done:.3f}s/instance)",
            flush=True,
        )

    contexts = np.concatenate(all_contexts, axis=0)
    x = np.concatenate(all_x, axis=0)
    J = np.concatenate(all_J, axis=0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, context=contexts, x=x, J=J, lb=lb, ub=ub)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "num_instances": int(contexts.shape[0]),
                "num_samples": num_samples,
                "context_dim": CONTEXT_DIM,
                "dim": dim,
                "num_knots": controller.num_knots,
                "nu": int(task.model.nu),
                "seed": seed,
            },
            indent=2,
        )
    )
    print(f"saved {contexts.shape[0]} instances x {num_samples} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="ksos_l2o/data/dataset.npz")
    args = parser.parse_args()
    main(args.instances, args.samples, args.batch_size, args.seed, args.output)
