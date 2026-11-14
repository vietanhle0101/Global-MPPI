"""Train the learned KernelSOS dual solver (Phase 2a) on dual_dataset.npz and
compare it on held-out problems with what the controller uses (newton-rs),
the converged barrier optimum and the exact SDP (CVXPY).

    python -m ksos_l2o.train_solver [--data ksos_l2o/data/dual_dataset.npz]
        [--output-dir ksos_l2o/data/sdp_solver] [--newton-steps 5] [--epochs 50] [--eval-only]

Label-free, as in KSOS_L2O: the loss is the dual objective f^T alpha_k of every
layer, weighted decay^(K - k) (deep supervision). The references are only used
for the test report. Saves checkpoint.npz, config.json, curves and report.json.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoint import load_pytree, save_pytree
from .sdp_solver import SolverConfig, init_solver, kernel_and_features, moment_matrix, solve, solve_path
from .train_sampler import plateau_step


def load_data(path, val_size, test_size):
    d = np.load(path)
    arrays = {k: d[k] for k in ("x", "J", "J_plain", "context", "qpos", "qvel")}
    arrays = {k: v.astype(np.float64) if k in ("x", "J", "context") else v for k, v in arrays.items()}
    n_train = len(arrays["x"]) - val_size - test_size
    split = lambda lo, hi: {k: v[lo:hi] for k, v in arrays.items()}
    bounds = d["lb"].astype(np.float64), d["ub"].astype(np.float64)
    return split(0, n_train), split(n_train, n_train + val_size), split(n_train + val_size, None), bounds


# ---- references ----

def newton_rs(x, f, cfg):
    """What GlobalMPPI.solve_ksos gets from ksos.solve(solver="newton-rs"): alpha, z, ms per solve."""
    import newton_sos
    t = cfg.epsilon / x.shape[1]
    alphas, start = [], time.perf_counter()
    for xi, fi in zip(x, f):
        problem = newton_sos.Problem(cfg.lambd, t, xi, fi)
        problem.initialize_native_kernel("Laplace", cfg.sigma)
        alphas.append(np.ravel(newton_sos.solve(problem, max_iter=100, verbose=False, method="partial_piv_lu").alpha))
    ms = 1e3 * (time.perf_counter() - start) / len(x)
    alphas = np.stack(alphas)
    return alphas, np.einsum("bn,bnd->bd", alphas, x), ms


def barrier_optimum(x, f, cfg, iterations=60):
    """Damped Newton (dual.py, same as newton-rs) run to convergence instead of stopping at decrement < epsilon."""
    from . import dual
    t = cfg.epsilon / x.shape[1]
    one = lambda xi, fi: dual.solve(fi, dual.features(xi, cfg.sigma, jitter=0.0), t, cfg.lambd, iterations=iterations)[0]
    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(x), jnp.asarray(f)))


def _cvxpy_alpha(args):
    import cvxpy as cp
    f, phi, lambd = args
    n = f.size
    alpha = cp.Variable(n)
    m = lambd * np.eye(n) + sum(alpha[i] * np.outer(phi[:, i], phi[:, i]) for i in range(n))
    cp.Problem(cp.Minimize(f @ alpha), [cp.sum(alpha) == 1, m >> 0]).solve(solver="CLARABEL")
    return np.asarray(alpha.value)


def exact_sdp(x, f, cfg):
    """alpha of the exact SDP (no barrier), CLARABEL in worker processes."""
    import multiprocessing
    phis = [np.asarray(kernel_and_features(jnp.asarray(xi), cfg.sigma, cfg.kernel_jitter)[1]) for xi in x]
    with multiprocessing.get_context("spawn").Pool() as pool:
        return np.stack(pool.map(_cvxpy_alpha, [(fi, p, cfg.lambd) for fi, p in zip(f, phis)]))


def references(test, cfg, cache):
    if cache.exists():
        with np.load(cache) as saved:
            if saved["x"].shape == test["x"].shape and np.array_equal(saved["x"], test["x"]):
                return {k: saved[k] for k in saved.files if k != "x"}
    print(f"computing references for {len(test['x'])} test problems (newton-rs, barrier optimum, CVXPY)")
    alpha_rs, _, rs_ms = newton_rs(test["x"], test["J"], cfg)
    refs = {"alpha_rs": alpha_rs, "rs_ms": np.asarray(rs_ms),
            "alpha_opt": barrier_optimum(test["x"], test["J"], cfg), "alpha_sdp": exact_sdp(test["x"], test["J"], cfg)}
    np.savez(cache, x=test["x"], **refs)
    return refs


# ---- training ----

def make_losses(cfg, lb, ub, decay):
    k = jnp.arange(cfg.newton_steps + 1)
    weights = jnp.where(k == cfg.newton_steps, 1.0, decay ** (cfg.newton_steps - k))

    def per_layer(params, x, f, c):
        return solve_path(params, x, f, c, lb, ub, cfg) @ f   # f^T alpha_k, k = 0..K

    batched = jax.vmap(per_layer, in_axes=(None, 0, 0, 0))

    def train_loss(params, x, f, c):
        values = batched(params, x, f, c)
        return jnp.mean(values @ weights), jnp.mean(values[:, -1])

    def val_loss(params, x, f, c):
        return jnp.mean(batched(params, x, f, c)[:, -1])

    return train_loss, val_loss


def train(cfg, train_set, val_set, bounds, out, epochs, steps_per_epoch, batch_size, learning_rate, decay, seed):
    lb, ub = map(jnp.asarray, bounds)
    x_tr, f_tr, c_tr = (jnp.asarray(train_set[k]) for k in ("x", "J", "context"))
    val = tuple(jnp.asarray(val_set[k]) for k in ("x", "J", "context"))
    params = init_solver(jax.random.PRNGKey(seed), x_tr.shape[-1], c_tr.shape[-1], cfg)
    optimizer = optax.inject_hyperparams(lambda learning_rate: optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)))(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    train_loss, val_loss = make_losses(cfg, lb, ub, decay)
    val_loss = jax.jit(val_loss)

    @jax.jit
    def step(params, opt_state, x, f, c):
        (total, final), grads = jax.value_and_grad(train_loss, has_aux=True)(params, x, f, c)
        finite = jnp.isfinite(total) & jnp.all(jnp.array([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # a near-singular Newton solve: drop the batch, keep Adam's state clean
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), new, old)
        return keep(new_params, params), keep(new_state, opt_state), final, finite

    print(f"train {len(x_tr)} / val {len(val[0])} problems, n = {x_tr.shape[1]}, dim = {x_tr.shape[2]}, "
          f"{cfg.newton_steps} Newton layers; untrained val f^T alpha {float(val_loss(params, *val)):.5f}")
    key = jax.random.PRNGKey(seed + 1)
    best, best_epoch, lr = np.inf, 0, learning_rate
    plateau_best, plateau_stale, stale = np.inf, 0, 0
    curves = {"train": [], "val": [], "lr": []}
    for epoch in range(1, epochs + 1):
        losses, skipped = [], 0
        for _ in range(steps_per_epoch):
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (batch_size,), 0, len(x_tr))
            params, opt_state, loss, finite = step(params, opt_state, x_tr[idx], f_tr[idx], c_tr[idx])
            if bool(finite):
                losses.append(float(loss))
            else:
                skipped += 1
        v = float(val_loss(params, *val))
        curves["train"].append(float(np.mean(losses)) if losses else np.nan)
        curves["val"].append(v)
        curves["lr"].append(lr)
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            save_pytree(params, out / "checkpoint.npz")
        else:
            stale += 1
        plateau_best, plateau_stale, lr, reduced = plateau_step(v, plateau_best, plateau_stale, lr, patience=3, factor=0.5,
                                                                minimum=2e-5, min_delta=1e-6)
        opt_state.hyperparams["learning_rate"] = lr
        print(f"epoch {epoch:3d} | train {curves['train'][-1]:.5f} val {v:.5f} | lr {lr:.1e}"
              + (" (reduced)" if reduced else "") + (f" | skipped {skipped}" if skipped else ""), flush=True)
        if stale >= 10:
            print("early stopping")
            break
    for name, values in curves.items():
        np.save(out / f"{name}_curve.npy", np.asarray(values))
    print(f"best val f^T alpha {best:.5f} at epoch {best_epoch}")


# ---- test report ----

def rollout_costs(qpos, qvel, plans):
    """Plain rollout cost of one plan (num_knots * nu,) per test state, with the controller's physics and cost.
    Runs in float32 like the controller: float64 MJX contacts diverge from it."""
    from jax.experimental import disable_x64
    from .generate_flow_dataset import make_controller
    from .train_state_sampler import forward_state
    with disable_x64():
        ctrl = make_controller("ksos_l2o/data/sampler_lse_20ms", 1)
        tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
        knots = jnp.asarray(plans, jnp.float32).reshape(len(plans), 1, ctrl.num_knots, -1)
        one = lambda q, v, k: ctrl.rollout(forward_state(ctrl.task, q, v), tk, k).costs.sum()
        return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(qpos, jnp.float32), jnp.asarray(qvel, jnp.float32), knots))


def latency_ms(params, x, f, c, lb, ub, cfg, repeats=100):
    """Median time of one jitted single-problem solve, as it would run once per control cycle."""
    fn = jax.jit(lambda x, f, c: solve(params, x, f, c, lb, ub, cfg)[1])
    jax.block_until_ready(fn(x[0], f[0], c[0]))
    times = []
    for i in range(repeats):
        j = i % len(x)
        start = time.perf_counter()
        jax.block_until_ready(fn(x[j], f[j], c[j]))
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def report(params, cfg, test, bounds, refs):
    lb, ub = map(jnp.asarray, bounds)
    x, f, c = (jnp.asarray(test[k]) for k in ("x", "J", "context"))
    alpha_learned = np.asarray(jax.jit(jax.vmap(lambda x, f, c: solve(params, x, f, c, lb, ub, cfg)[0]))(x, f, c))
    x, f = np.asarray(x), np.asarray(f)
    alphas = {"learned": alpha_learned, "newton-rs (controller)": refs["alpha_rs"],
              "barrier optimum": refs["alpha_opt"], "exact SDP (CVXPY)": refs["alpha_sdp"]}
    a_sdp, a_rs = refs["alpha_sdp"], refs["alpha_rs"]
    z_sdp = np.einsum("bn,bnd->bd", a_sdp, x)

    def min_eig(xi, a):
        phi = kernel_and_features(jnp.asarray(xi), cfg.sigma, cfg.kernel_jitter)[1]
        return jnp.linalg.eigvalsh(moment_matrix(jnp.asarray(a), phi, cfg.lambd))[0]

    # plain rollout cost of each solver's z, of the best sample, and of the best sample again as a check
    best = np.argmin(test["J_plain"], axis=1)
    plans = [np.einsum("bn,bnd->bd", a, x) for a in alphas.values()] + [x[np.arange(len(x)), best]]
    reps = len(plans)
    costs = rollout_costs(np.tile(test["qpos"], (reps, 1)), np.tile(test["qvel"], (reps, 1)), np.concatenate(plans))
    costs = costs.reshape(reps, len(x))
    best_cost = costs[-1]
    stored = test["J_plain"][np.arange(len(x)), best]
    print(f"rollout check: best sample recomputed vs stored J_plain, max rel. diff {np.max(np.abs(best_cost / stored - 1)):.1e}")

    rows = {}
    print(f"\n{len(x)} held-out problems, n = {x.shape[1]}. Against the exact SDP (CVXPY): dual gap = f^T (alpha - alpha_cvx), "
          f"|alpha - alpha_cvx|_1, |z - z_cvx| (median / max).\nrs gap = f^T (alpha - alpha_rs). "
          f"rollout = plain cost of z = X^T alpha (median, and median ratio to the best sample).")
    print(f"{'solver':<24}{'f^T alpha':>10}{'dual gap':>10}{'|da|_1':>8}{'|dz| med':>9}{'|dz| max':>9}{'rs gap':>10}"
          f"{'min eig':>9}{'|sum-1|':>9}{'rollout':>9}{'/best':>7}")
    for (name, a), cost in zip(alphas.items(), costs):
        dz = np.linalg.norm(np.einsum("bn,bnd->bd", a, x) - z_sdp, axis=1)
        row = {"dual_value": float(np.mean(np.sum(f * a, 1))),
               "dual_gap_vs_cvxpy": float(np.mean(np.sum(f * (a - a_sdp), 1))),
               "alpha_l1_to_cvxpy": float(np.mean(np.abs(a - a_sdp).sum(1))),
               "z_dist_to_cvxpy_median": float(np.median(dz)),
               "z_dist_to_cvxpy_max": float(np.max(dz)),
               "dual_gap_vs_rs": float(np.mean(np.sum(f * (a - a_rs), 1))),
               "min_eig": float(np.min(jax.vmap(min_eig)(x, a))),
               "sum_residual": float(np.max(np.abs(a.sum(1) - 1))),
               "rollout_median": float(np.median(cost)),
               "rollout_ratio_median": float(np.median(cost / best_cost))}
        rows[name] = row
        print(f"{name:<24}{row['dual_value']:>10.5f}{row['dual_gap_vs_cvxpy']:>10.1e}{row['alpha_l1_to_cvxpy']:>8.3f}"
              f"{row['z_dist_to_cvxpy_median']:>9.2e}{row['z_dist_to_cvxpy_max']:>9.2e}{row['dual_gap_vs_rs']:>10.1e}"
              f"{row['min_eig']:>9.1e}{row['sum_residual']:>9.1e}{row['rollout_median']:>9.4f}{row['rollout_ratio_median']:>7.3f}")
    print(f"{'best sample':<24}{'':>82}{np.median(best_cost):>9.4f}{1.0:>7.3f}")
    rows["learned"]["latency_ms"] = latency_ms(params, jnp.asarray(x), jnp.asarray(f), c, lb, ub, cfg)
    rows["newton-rs (controller)"]["latency_ms"] = float(refs["rs_ms"])
    print(f"time per solve: learned {rows['learned']['latency_ms']:.2f} ms ({jax.default_backend()}, single problem, jitted), "
          f"newton-rs {float(refs['rs_ms']):.2f} ms")
    return rows


def main(args):
    cfg = replace(SolverConfig(), newton_steps=args.newton_steps)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_set, val_set, test_set, bounds = load_data(args.data, args.val_size, args.test_size)
    if not args.eval_only:
        (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "data": args.data, "context_dim": train_set["context"].shape[1],
                                                     "dim": train_set["x"].shape[2]}, indent=2))
        train(cfg, train_set, val_set, bounds, out, args.epochs, args.steps_per_epoch, args.batch_size, args.lr,
              args.deep_supervision_decay, args.seed)
    template = init_solver(jax.random.PRNGKey(0), train_set["x"].shape[2], train_set["context"].shape[1], cfg)
    params = load_pytree(template, out / "checkpoint.npz")
    refs = references(test_set, cfg, Path(args.data).with_name("dual_test_references.npz"))
    rows = report(params, cfg, test_set, bounds, refs)
    (out / "report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ksos_l2o/data/dual_dataset.npz")
    p.add_argument("--output-dir", default="ksos_l2o/data/sdp_solver")
    p.add_argument("--newton-steps", type=int, default=SolverConfig.newton_steps)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--deep-supervision-decay", type=float, default=0.5, help="0 = last layer only")
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true", help="skip training, report the saved checkpoint")
    main(p.parse_args())
