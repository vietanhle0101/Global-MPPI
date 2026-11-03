"""Train the GNN warm start for the KernelSOS dual (Phase 2a), label-free.

alpha0 = GNN(x, f, context), then K unrolled damped-Newton layers. Loss is the
barrier objective H(alpha_k) averaged over layers (deep supervision, last layer
weighted most); H* (200-iteration uniform-start solve) is only used to report
the optimality gap.

    python -m ksos_l2o.train_dual [--config ksos_l2o/configs/train_dual.yaml] [--set key=value ...]

Everything is defined in the YAML config; `--set` overrides single entries.
Exactly one k (the number of unrolled Newton iterations) is trained and reported.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from . import dual
from .checkpoint import save_pytree
from .gnn import init_gnn, warm_start
from .train_sampler import plateau_step

LAMBD, EPSILON = 1e-5, 1e-2


def prepare(x, f, sigma):
    phi = dual.features(x, sigma)
    fn, tn = dual.normalize(f, EPSILON / x.shape[0])
    return phi, fn, tn


def path_values(params, x, f, ctx, ub, sigma, layers, use_gnn=True):
    """H(alpha_k) and z_k = alpha_k @ x for k = 0..layers."""
    phi, fn, tn = prepare(x, f, sigma)
    n = x.shape[0]
    alpha0 = warm_start(params, x, fn, ctx, ub, sigma) if use_gnn else jnp.full((n,), 1.0 / n)
    _, path, _, _ = dual.solve(fn, phi, tn, LAMBD, alpha0=alpha0, iterations=layers)
    values = jax.vmap(lambda a: dual.barrier_value(a, fn, phi, tn, LAMBD))(path)
    return values, path @ x


def reference(x, f, sigma):
    phi, fn, tn = prepare(x, f, sigma)
    _, path, _, _ = dual.solve(fn, phi, tn, LAMBD, iterations=200)
    return dual.barrier_value(path[-1], fn, phi, tn, LAMBD), path[-1] @ x


def train(cfg):
    sigma, k = cfg["sigma"], int(cfg["k"])
    d = np.load(cfg["data"])
    x_all, f_all, c_all = jnp.asarray(d["x"]), jnp.asarray(d["J"]), jnp.asarray(d["context"])
    ub = jnp.asarray(d["ub"])
    n_total, n, dim = x_all.shape
    print(f"data {cfg['data']}: {n_total} problems x {n} samples, dim {dim}; sigma={sigma}, k={k} Newton iterations")

    ref = jax.jit(jax.vmap(lambda x, f: reference(x, f, sigma)))
    h_star, z_star = [], []
    for i in range(0, n_total, 512):
        h, z = ref(x_all[i:i + 512], f_all[i:i + 512])
        h_star.append(h); z_star.append(z)
    h_star, z_star = jnp.concatenate(h_star), jnp.concatenate(z_star)

    n_train = n_total - cfg["val_size"]
    sl = lambda a: (a[:n_train], a[n_train:])
    (x_tr, x_va), (f_tr, f_va), (c_tr, c_va), (h_tr, h_va), (z_tr, z_va) = map(sl, (x_all, f_all, c_all, h_star, z_star))

    params = init_gnn(jax.random.PRNGKey(cfg["seed"]), dim, c_all.shape[1], cfg["hidden_dim"], cfg["gnn_layers"])
    learning_rate = cfg["learning_rate"]
    optimizer = optax.inject_hyperparams(lambda learning_rate: optax.apply_if_finite(optax.chain(
        optax.clip_by_global_norm(cfg["grad_clip"]), optax.adam(learning_rate)), max_consecutive_errors=20))(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    def final_values(p, x, f, c, use_gnn=True):
        values, zs = jax.vmap(lambda x_, f_, c_: path_values(p, x_, f_, c_, ub, sigma, k, use_gnn))(x, f, c)
        return values[:, -1], zs[:, -1]

    def loss_fn(p, x, f, c, h):
        gap = final_values(p, x, f, c)[0] - h
        if cfg["loss"] == "log":
            gap = jnp.log1p(jnp.maximum(gap, 0.0) / cfg["log_floor"])
        return jnp.mean(jnp.where(jnp.isfinite(gap), gap, 0.0))

    @jax.jit
    def step(p, s, x, f, c, h):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, f, c, h)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def evaluate(p, x, f, c, h, z):
        v, zs = final_values(p, x, f, c)
        return jnp.median(v - h), jnp.median(jnp.linalg.norm(zs - z, axis=-1))

    @jax.jit
    def evaluate_uniform(x, f, c, h, z):
        v, zs = final_values(None, x, f, c, use_gnn=False)
        return jnp.median(v - h), jnp.median(jnp.linalg.norm(zs - z, axis=-1))

    gap_u, dz_u = evaluate_uniform(x_va, f_va, c_va, h_va, z_va)
    gap0, dz0 = evaluate(params, x_va, f_va, c_va, h_va, z_va)
    print(f"val (median) at k={k} | uniform start: gap {float(gap_u):.2e}  |z-z*| {float(dz_u):.2e}")
    print(f"val (median) at k={k} | GNN epoch 0  : gap {float(gap0):.2e}  |z-z*| {float(dz0):.2e}")

    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "layers": k, "dim": dim, "context_dim": int(c_all.shape[1])}))
    key = jax.random.PRNGKey(cfg["seed"] + 1)
    best, stale, lr, best_val, early_stale = np.inf, 0, learning_rate, np.inf, 0
    for epoch in range(1, cfg["epochs"] + 1):
        losses = []
        for _ in range(cfg["steps_per_epoch"]):
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (cfg["batch_size"],), 0, n_train)
            params, opt_state, loss = step(params, opt_state, x_tr[idx], f_tr[idx], c_tr[idx], h_tr[idx])
            losses.append(float(loss))
        gap, dz = evaluate(params, x_va, f_va, c_va, h_va, z_va)
        val = float(gap)
        best, stale, lr, reduced = plateau_step(val, best, stale, lr, patience=cfg["plateau_patience"],
                                                factor=cfg["plateau_factor"], minimum=learning_rate * 1e-3, min_delta=1e-9)
        opt_state.hyperparams["learning_rate"] = lr
        if val < best_val:
            best_val, early_stale = val, 0
            save_pytree(params, out / "checkpoint.npz")
        else:
            early_stale += 1
        print(f"epoch {epoch:3d} train {np.mean(losses):.3e} | val gap(k={k}) {val:.2e} |z-z*| {float(dz):.2e} | lr {lr:.1e}"
              f"{' (reduced)' if reduced else ''}", flush=True)
        if early_stale >= cfg["early_stopping_patience"]:
            print("early stopping"); break
    print(f"best val gap at k={k}: {best_val:.3e}; saved {out / 'checkpoint.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="ksos_l2o/configs/train_dual.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for item in a.set:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"unknown config key {key!r}")
        cfg[key] = yaml.safe_load(value)
    # PyYAML reads "1e-3" (no decimal point) as a string, so convert explicitly.
    for key in ("sigma", "log_floor", "learning_rate", "grad_clip", "plateau_factor"):
        cfg[key] = float(cfg[key])
    for key in ("k", "hidden_dim", "gnn_layers", "epochs", "steps_per_epoch", "batch_size", "val_size",
                "plateau_patience", "early_stopping_patience", "seed"):
        cfg[key] = int(cfg[key])
    train(cfg)
