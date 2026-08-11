"""Train/val/test splits for the catalysis set, and why the obvious one is wrong.

The sample is 10 sites on each of 40 surfaces. That structure makes the default
choice actively misleading.

**A random split leaks.** Put the 400 rows in a hat and draw 20% for testing, and
almost every test row comes from a surface whose other nine sites are in training.
A model can then score well by learning each surface's mean binding energy and
applying it to the held-out sites -- which is exactly the composition baseline of
1.189 eV, reproduced by a graph network and reported as if the network had learned
site chemistry. Nothing in the RMSE distinguishes the two.

**A surface-disjoint split cannot be gamed that way.** Whole (surface, facet)
groups go to one partition only. A test surface has never been seen, so its mean
is not available to be memorised, and the only way to predict is from structure.

Both are built, deliberately. The random split is not a result -- it is the
control that shows how much of the apparent skill is leakage. The gap between the
two is the finding, in the same way the element split was in Phase 5.

**What this split still cannot test.** The PBE subset is one publication, so
publication-disjoint is impossible here (LIMITATIONS 17). Surface-disjoint tests
generalisation to new surfaces within one group's methodology. Two labs
disagreeing about the same surface is a failure mode this sample cannot see.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)


def surface_key(row: dict) -> str:
    return f'{row.get("surface")}|{row.get("facet")}'


def random_split(rows, fractions=DEFAULT_FRACTIONS, seed: int = 42) -> dict:
    """Rows shuffled and cut. Leaks by construction; kept as the control."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_tr = int(round(fractions[0] * len(rows)))
    n_va = int(round(fractions[1] * len(rows)))
    return {"train": sorted(idx[:n_tr].tolist()),
            "val": sorted(idx[n_tr:n_tr + n_va].tolist()),
            "test": sorted(idx[n_tr + n_va:].tolist())}


def surface_split(rows, fractions=DEFAULT_FRACTIONS, seed: int = 42) -> dict:
    """Whole surfaces to one partition only.

    Groups are shuffled, then filled largest-first into whichever partition is
    furthest below quota. Size-ordered filling matters: with unequal groups a
    sequential fill overshoots and can leave the test set half again as large as
    asked for, which then gets reported as a 15% test set.

    Sorted before shuffling so the split depends on the seed alone and not on the
    order rows happened to be built in.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[surface_key(r)].append(i)

    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))

    quota = [f * len(rows) for f in fractions]
    buckets: list[list[int]] = [[], [], []]
    filled = [0.0, 0.0, 0.0]
    for k in keys:
        deficit = [(filled[i] / quota[i] if quota[i] else 1.0) for i in range(3)]
        t = int(np.argmin(deficit))
        buckets[t].extend(groups[k])
        filled[t] += len(groups[k])
    return {"train": sorted(buckets[0]), "val": sorted(buckets[1]),
            "test": sorted(buckets[2])}


def leakage_report(rows, split: dict) -> dict:
    """How much of the test set sits on a surface the model trained on.

    The number that distinguishes the two splits. For a random split it is near
    100%: the model has seen nine other sites on almost every test surface and
    can predict the tenth from their mean. For a surface-disjoint split it must
    be exactly 0, and that is asserted rather than hoped for.
    """
    tr = {surface_key(rows[i]) for i in split["train"]}
    te = [surface_key(rows[i]) for i in split["test"]]
    seen = sum(1 for k in te if k in tr)
    return {
        "test_rows": len(te),
        "test_surfaces": len(set(te)),
        "test_rows_on_a_training_surface": seen,
        "leak_fraction": (seen / len(te)) if te else 0.0,
        "train_surfaces": len(tr),
    }


def group_mean_baseline(y: np.ndarray, rows, split: dict) -> float:
    """RMSE of predicting each test row from its surface's TRAINING mean.

    This is the number a graph model has to beat, computed the honest way --
    means fitted on training rows only and applied to unseen ones, rather than
    the optimistic within-sample floor.

    On a surface-disjoint split most test surfaces have no training mean at all,
    so it falls back to the global training mean. That is not a flaw in the
    baseline; it is the point. Composition-level knowledge genuinely runs out
    when the surface is new, and a graph model that cannot beat it there has not
    learned structure.
    """
    tr_idx = np.array(split["train"], dtype=int)
    te_idx = np.array(split["test"], dtype=int)
    if te_idx.size == 0:
        return float("nan")

    sums: dict[str, list] = defaultdict(list)
    for i in tr_idx:
        sums[surface_key(rows[i])].append(y[i])
    global_mean = float(y[tr_idx].mean())
    pred = np.array([np.mean(sums[surface_key(rows[i])]) if surface_key(rows[i]) in sums
                     else global_mean for i in te_idx])
    return float(np.sqrt(((pred - y[te_idx]) ** 2).mean()))
