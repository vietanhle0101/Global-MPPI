"""Save/load any pytree of arrays (e.g. the flow's params) to/from a .npz."""
import json
from pathlib import Path

import jax
import numpy as np


def save_pytree(tree, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, structure = jax.tree_util.tree_flatten(tree)
    np.savez(path, **{f"arr_{i}": np.asarray(v) for i, v in enumerate(leaves)})
    path.with_suffix(".tree.json").write_text(json.dumps(str(structure)))


def load_pytree(template, path):
    """Loaded against `template`'s structure; only shapes are checked."""
    leaves, structure = jax.tree_util.tree_flatten(template)
    with np.load(path) as saved:
        restored = [np.asarray(saved[f"arr_{i}"]) for i in range(len(leaves))]
    for expected, actual in zip(leaves, restored):
        if expected.shape != actual.shape:
            raise ValueError(f"checkpoint at {path} is incompatible (shape mismatch) -- regenerate it")
    return jax.tree_util.tree_unflatten(structure, restored)
