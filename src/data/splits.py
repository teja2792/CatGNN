"""Train / validation / test splits that do not lie to you.

The default in most materials ML is a random split, and on a database like
Materials Project it is optimistic to the point of being misleading. The reason
is concrete rather than theoretical: this download contains 221 entries of
Li7Mn2(CoO4)3, all at identical cell size -- the same lattice with different
cation orderings. A random split scatters those across train and test, so the
model memorises one and is graded on its near-twin. The score that comes back
measures recall, not generalisation.

So this module builds four splits of increasing strictness, and every result in
the repo is reported against all four. Each answers a different, honest question:

    random      "How well does it do on more of the same?"
                The optimistic baseline. Reported so the inflation is visible.

    formula     "How well does it do on a formula it has never seen?"
                All polymorphs of a formula go to the same side.

    chemsys     "How well does it do on a chemical system it has never seen?"
                Everything containing exactly {Li, Mn, Co, O} moves together.

    element     "How well does it do on an element it has never seen?"
                The extrapolation test, and the one that usually hurts.

The gap between random and element is not a nuisance -- it is one of the most
informative numbers this repository can produce, because it tells a practitioner
how much to discount a published leaderboard score when applying a model to
chemistry that was not in the training set.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import CACHE, RANDOM_SEED

SPLITS = CACHE / "splits"

SCHEMES = ("random", "formula", "chemsys", "element")

DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)  # train, validation, test


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_key(row: dict, scheme: str) -> str:
    """The unit that must not be split across train and test."""
    if scheme == "random":
        return row["material_id"]          # every material is its own group
    if scheme == "formula":
        return row["formula_pretty"]
    if scheme == "chemsys":
        return row.get("chemsys") or "-".join(sorted(row.get("elements") or []))
    raise ValueError(f"'{scheme}' is not a group-based scheme")


def split_by_groups(rows, scheme, fractions=DEFAULT_FRACTIONS, seed=RANDOM_SEED):
    """Assign whole groups to train / val / test, targeting the given fractions.

    Groups are shuffled and then filled largest-first into whichever partition is
    furthest below its quota. Filling in size order matters: with a few enormous
    groups (Li7Mn2(CoO4)3 has 221 members) a naive sequential fill overshoots
    badly and can leave the test set 50% larger than requested.
    """
    rng = np.random.default_rng(seed)

    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[group_key(r, scheme)].append(r["material_id"])

    # Sort before shuffling. `list(groups)` follows dict insertion order, which
    # follows the order the rows arrived in, so shuffling that directly would
    # make the split depend on how the input happened to be ordered -- and a
    # split you cannot reproduce from a seed alone is not reproducible.
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # largest first, ties broken by the shuffle

    total = len(rows)
    quota = [f * total for f in fractions]
    buckets: list[list[str]] = [[], [], []]
    filled = [0.0, 0.0, 0.0]

    for k in keys:
        # Whichever partition is proportionally emptiest gets the next group.
        deficit = [(filled[i] / quota[i] if quota[i] else 1.0) for i in range(3)]
        target = int(np.argmin(deficit))
        buckets[target].extend(groups[k])
        filled[target] += len(groups[k])

    return {"train": sorted(buckets[0]), "val": sorted(buckets[1]), "test": sorted(buckets[2])}


def split_by_element(rows, fractions=DEFAULT_FRACTIONS, seed=RANDOM_SEED):
    """Hold out whole elements: the test set contains chemistry never trained on.

    Elements are picked at random from those rare enough not to gut the training
    set, accumulating until the held-out materials reach the target fraction. A
    material joins the held-out side if it contains *any* held-out element, so
    the test set is genuinely disjoint in chemistry rather than merely enriched.

    This is the hardest split here and the one whose result transfers best to
    practice: a lab applying a published model to a new element is in exactly
    this position.
    """
    rng = np.random.default_rng(seed)

    counts = Counter()
    for r in rows:
        counts.update(r.get("elements") or [])

    total = len(rows)
    want = (fractions[1] + fractions[2]) * total

    by_element = defaultdict(set)
    for r in rows:
        for e in r.get("elements") or []:
            by_element[e].add(r["material_id"])

    # Excluding a very common element (O is in 41% of Materials Project) would
    # destroy the training set, so candidates are capped by share. The cap has to
    # adapt, though: on a dataset with few distinct elements, a fixed 6% cap
    # leaves no candidates at all and an earlier version of this function then
    # quietly held out nothing and returned an EMPTY TEST SET. A split function
    # that silently declines to split is the worst possible failure here, because
    # every downstream metric still computes and just means nothing.
    held: set[str] = set()
    covered: set[str] = set()
    for cap in (0.06, 0.10, 0.15, 0.25, 0.40):
        candidates = sorted(e for e, c in counts.items() if c < cap * total and e not in held)
        rng.shuffle(candidates)
        for e in candidates:
            if len(covered) >= want:
                break
            held.add(e)
            covered |= by_element[e]
        if len(covered) >= want * 0.5:
            break

    if not held:
        raise ValueError(
            "Could not hold out any element: every element appears in more than "
            f"40% of the {total} materials. An element-disjoint split is not "
            "meaningful for this dataset."
        )
    if len(covered) < want * 0.5:
        print(f"    ! element split reached only {100 * len(covered) / total:.1f}% "
              f"held out, target was {100 * want / total:.0f}%")

    holdout = [r for r in rows if set(r.get("elements") or []) & held]
    train = [r for r in rows if not (set(r.get("elements") or []) & held)]

    # Divide the held-out elements between validation and test.
    held_list = sorted(held)
    rng.shuffle(held_list)
    cut = max(1, len(held_list) // 2)
    val_elements = set(held_list[:cut])
    test_elements = held - val_elements

    # A material can contain a validation element AND a test element at once --
    # say one holding both Mg and S when Mg is a validation element and S a test
    # element. Sending it to either side puts the other side's chemistry into
    # model selection, which is a real if mild leak: hyperparameters would be
    # tuned having seen an element the final score is supposed to be blind to.
    #
    # Such materials cannot go to training either, since they contain held-out
    # elements. So they are dropped, and the count is reported rather than
    # buried -- a split that silently discards data is as misleading as one that
    # silently duplicates it.
    val, test, spanning = [], [], []
    for r in holdout:
        els = set(r.get("elements") or []) & held
        if els <= val_elements:
            val.append(r)
        elif els <= test_elements:
            test.append(r)
        else:
            spanning.append(r)

    return (
        {"train": sorted(r["material_id"] for r in train),
         "val": sorted(r["material_id"] for r in val),
         "test": sorted(r["material_id"] for r in test)},
        {"held_out_elements": sorted(held),
         "val_elements": sorted(val_elements),
         "test_elements": sorted(test_elements),
         "dropped_spanning_val_and_test": len(spanning),
         "dropped_ids": sorted(r["material_id"] for r in spanning)[:20]},
    )


# ---------------------------------------------------------------------------
# Leakage measurement
# ---------------------------------------------------------------------------

def leakage_report(rows, split: dict) -> dict:
    """How much of the test set is 'already known' from training?

    Three progressively weaker forms of overlap. A random split leaks heavily on
    all three; that is the point of measuring it rather than asserting it.
    """
    by_id = {r["material_id"]: r for r in rows}
    tr = [by_id[i] for i in split["train"] if i in by_id]
    te = [by_id[i] for i in split["test"] if i in by_id]
    if not te:
        return {}

    tr_formula = {r["formula_pretty"] for r in tr}
    tr_chemsys = {r.get("chemsys") for r in tr}
    tr_elements = set().union(*[set(r.get("elements") or []) for r in tr]) if tr else set()

    same_formula = sum(1 for r in te if r["formula_pretty"] in tr_formula)
    same_chemsys = sum(1 for r in te if r.get("chemsys") in tr_chemsys)
    all_elements_seen = sum(
        1 for r in te if set(r.get("elements") or []) <= tr_elements
    )

    return {
        "n_train": len(tr),
        "n_test": len(te),
        "test_with_formula_seen_in_train": same_formula,
        "test_with_formula_seen_pct": round(100 * same_formula / len(te), 1),
        "test_with_chemsys_seen_in_train": same_chemsys,
        "test_with_chemsys_seen_pct": round(100 * same_chemsys / len(te), 1),
        "test_with_all_elements_seen": all_elements_seen,
        "test_with_all_elements_seen_pct": round(100 * all_elements_seen / len(te), 1),
    }


# ---------------------------------------------------------------------------
# Build and persist
# ---------------------------------------------------------------------------

def build_all_splits(rows, fractions=DEFAULT_FRACTIONS, seed=RANDOM_SEED) -> dict:
    SPLITS.mkdir(parents=True, exist_ok=True)
    out = {}

    for scheme in SCHEMES:
        if scheme == "element":
            split, extra = split_by_element(rows, fractions, seed)
        else:
            split, extra = split_by_groups(rows, scheme, fractions, seed), {}

        report = leakage_report(rows, split)
        out[scheme] = {
            "split": split,
            "leakage": report,
            "sizes": {k: len(v) for k, v in split.items()},
            **extra,
        }
        (SPLITS / f"{scheme}.json").write_text(json.dumps(split), encoding="utf-8")

    summary = {
        "seed": seed,
        "fractions": list(fractions),
        "n_materials": len(rows),
        "schemes": {k: {"sizes": v["sizes"], "leakage": v["leakage"],
                        **({"held_out_elements": v["held_out_elements"]}
                           if "held_out_elements" in v else {})}
                    for k, v in out.items()},
    }
    (SPLITS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def load_split(scheme: str) -> dict:
    path = SPLITS / f"{scheme}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/make_splits.py")
    return json.loads(path.read_text(encoding="utf-8"))
