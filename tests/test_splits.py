"""Split tests.

A leaky split is the most dangerous bug in this repository, because it makes
everything look *better*. Nothing errors, no figure looks odd -- the numbers are
just quietly too good, and stay too good all the way into a talk. So these tests
assert the disjointness properties directly rather than checking that the code
runs.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.splits import (  # noqa: E402
    leakage_report,
    split_by_element,
    split_by_groups,
)

ELEMENTS = ["Li", "Na", "K", "Mg", "Ca", "Ti", "Fe", "Co", "Ni", "Cu",
            "O", "S", "Se", "N", "P", "F", "Cl", "Si", "Al", "Zn"]


def fake_rows(n=3000, seed=0):
    """Synthetic materials with realistic structure: repeated formulas, shared systems."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        k = int(rng.integers(2, 4))
        els = sorted(rng.choice(ELEMENTS, size=k, replace=False).tolist())
        # Deliberately few distinct formulas per chemical system, so polymorph
        # families exist and a random split has something to leak through.
        formula = "".join(els) + str(int(rng.integers(1, 4)))
        rows.append({
            "material_id": f"mp-{i}",
            "formula_pretty": formula,
            "chemsys": "-".join(els),
            "elements": els,
        })
    return rows


def by_id(rows):
    return {r["material_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# Every split must partition the data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["random", "formula", "chemsys"])
def test_split_is_a_partition(scheme):
    rows = fake_rows()
    s = split_by_groups(rows, scheme)

    ids = s["train"] + s["val"] + s["test"]
    assert len(ids) == len(rows), "materials lost or duplicated"
    assert len(set(ids)) == len(ids), "a material appears in more than one partition"
    assert set(ids) == {r["material_id"] for r in rows}


def test_element_split_has_no_duplicates():
    rows = fake_rows()
    s, _ = split_by_element(rows)
    ids = s["train"] + s["val"] + s["test"]
    assert len(set(ids)) == len(ids), "a material appears in more than one partition"


@pytest.mark.parametrize("scheme", ["random", "formula", "chemsys"])
def test_split_sizes_are_close_to_requested(scheme):
    rows = fake_rows(4000)
    s = split_by_groups(rows, scheme, fractions=(0.8, 0.1, 0.1))

    for part, want in zip(("train", "val", "test"), (0.8, 0.1, 0.1)):
        got = len(s[part]) / len(rows)
        assert abs(got - want) < 0.05, f"{scheme}/{part}: wanted {want}, got {got:.3f}"


# ---------------------------------------------------------------------------
# The disjointness each scheme promises
# ---------------------------------------------------------------------------

def test_formula_split_shares_no_formula():
    rows = fake_rows()
    s = split_by_groups(rows, "formula")
    m = by_id(rows)

    tr = {m[i]["formula_pretty"] for i in s["train"]}
    te = {m[i]["formula_pretty"] for i in s["test"]}
    va = {m[i]["formula_pretty"] for i in s["val"]}

    assert not (tr & te), f"{len(tr & te)} formulas in both train and test"
    assert not (tr & va) and not (va & te)


def test_chemsys_split_shares_no_chemical_system():
    rows = fake_rows()
    s = split_by_groups(rows, "chemsys")
    m = by_id(rows)

    tr = {m[i]["chemsys"] for i in s["train"]}
    te = {m[i]["chemsys"] for i in s["test"]}
    assert not (tr & te)


def test_chemsys_split_also_implies_formula_disjoint():
    """A stricter split must not be weaker on any axis than a looser one."""
    rows = fake_rows()
    s = split_by_groups(rows, "chemsys")
    m = by_id(rows)

    tr = {m[i]["formula_pretty"] for i in s["train"]}
    te = {m[i]["formula_pretty"] for i in s["test"]}
    assert not (tr & te)


def test_element_split_holds_out_whole_elements():
    rows = fake_rows()
    s, info = split_by_element(rows)
    m = by_id(rows)

    held = set(info["held_out_elements"])
    assert held, "no elements were held out"

    train_elements = set()
    for i in s["train"]:
        train_elements |= set(m[i]["elements"])
    assert not (train_elements & held), "a held-out element appears in training"

    for i in s["test"]:
        assert set(m[i]["elements"]) & held, "test material contains no held-out element"


def test_element_split_keeps_val_and_test_chemistry_apart():
    """No held-out element may appear in both validation and test.

    Otherwise hyperparameters get tuned on chemistry the final score is supposed
    to be blind to. Materials containing a validation element AND a test element
    cannot satisfy this on either side, so they are dropped -- and the count is
    reported, not buried.
    """
    rows = fake_rows()
    s, info = split_by_element(rows)
    m = by_id(rows)

    held = set(info["held_out_elements"])
    va = set().union(*[set(m[i]["elements"]) for i in s["val"]]) if s["val"] else set()
    te = set().union(*[set(m[i]["elements"]) for i in s["test"]]) if s["test"] else set()

    assert not ((va & held) & (te & held)), "an element appears in both val and test"
    assert "dropped_spanning_val_and_test" in info, "dropped count must be reported"


def test_element_split_is_a_partition_minus_reported_drops():
    """Nothing may vanish without being counted."""
    rows = fake_rows()
    s, info = split_by_element(rows)

    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert total + info["dropped_spanning_val_and_test"] == len(rows)


def test_element_split_leaves_a_usable_training_set():
    """Holding out a very common element would gut training. It must not happen."""
    rows = fake_rows(4000)
    s, _ = split_by_element(rows)
    assert len(s["train"]) > 0.5 * len(rows), "training set destroyed by the holdout"


# ---------------------------------------------------------------------------
# Leakage measurement itself
# ---------------------------------------------------------------------------

def test_random_split_leaks_more_than_formula_split():
    """The point of the whole module, asserted rather than assumed."""
    rows = fake_rows(4000)

    rand = leakage_report(rows, split_by_groups(rows, "random"))
    form = leakage_report(rows, split_by_groups(rows, "formula"))

    assert rand["test_with_formula_seen_pct"] > 10, "fixture has no polymorphs to leak"
    assert form["test_with_formula_seen_pct"] == 0.0
    assert rand["test_with_formula_seen_pct"] > form["test_with_formula_seen_pct"]


def test_leakage_is_monotonic_across_schemes():
    """Each stricter scheme must leak no more than the looser ones, on every axis."""
    rows = fake_rows(4000)
    reports = {s: leakage_report(rows, split_by_groups(rows, s))
               for s in ("random", "formula", "chemsys")}

    for key in ("test_with_formula_seen_pct", "test_with_chemsys_seen_pct"):
        seq = [reports[s][key] for s in ("random", "formula", "chemsys")]
        assert seq == sorted(seq, reverse=True), f"{key} not monotonic: {seq}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["random", "formula", "chemsys"])
def test_same_seed_gives_same_split(scheme):
    rows = fake_rows()
    assert split_by_groups(rows, scheme, seed=42) == split_by_groups(rows, scheme, seed=42)


@pytest.mark.parametrize("scheme", ["random", "formula", "chemsys"])
def test_different_seed_gives_a_different_split(scheme):
    rows = fake_rows()
    a = split_by_groups(rows, scheme, seed=1)
    b = split_by_groups(rows, scheme, seed=2)
    assert a["test"] != b["test"], "seed has no effect -- splits are not randomised"


def test_split_does_not_depend_on_input_order():
    """Shuffling the input rows must not change which materials land in test."""
    rows = fake_rows()
    shuffled = list(reversed(rows))
    assert split_by_groups(rows, "formula", seed=7) == \
        split_by_groups(shuffled, "formula", seed=7)


def test_large_groups_do_not_blow_the_quota():
    """One enormous polymorph family must not overshoot the test set.

    Modelled on the real Li7Mn2(CoO4)3, which has 221 entries. A naive
    sequential fill can hand that whole block to a partition that only needed 30
    more materials.
    """
    rows = [{"material_id": f"mp-{i}", "formula_pretty": "Big",
             "chemsys": "Li-Mn-Co-O", "elements": ["Li", "Mn", "Co", "O"]}
            for i in range(221)]
    rows += [{"material_id": f"mp-x{i}", "formula_pretty": f"F{i}",
              "chemsys": f"A{i}-B", "elements": ["Li", "O"]} for i in range(1779)]

    s = split_by_groups(rows, "formula", fractions=(0.8, 0.1, 0.1))
    assert len(s["test"]) / len(rows) < 0.25, f"test set blew out to {len(s['test'])}"


def test_element_split_refuses_rather_than_returning_an_empty_test_set():
    """If no element can be held out, say so loudly.

    An earlier version capped candidate elements at 6% of the dataset with no
    fallback. On a dataset with few distinct elements that left no candidates, so
    it held out nothing and returned an EMPTY TEST SET -- and every downstream
    metric still computed, on nothing. Failing loudly is the only safe option.
    """
    rows = [{"material_id": f"mp-{i}", "formula_pretty": "AB",
             "chemsys": "A-B", "elements": ["A", "B"]} for i in range(500)]

    with pytest.raises(ValueError, match="Could not hold out any element"):
        split_by_element(rows)


def test_element_split_adapts_its_cap_to_the_dataset():
    """With few distinct elements the cap must relax rather than give up."""
    rows = fake_rows(3000)          # only 20 distinct elements
    s, info = split_by_element(rows)

    assert info["held_out_elements"], "no elements held out"
    assert len(s["test"]) > 0, "empty test set"
    assert len(s["train"]) > 0.4 * len(rows), "training set destroyed"
