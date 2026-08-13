"""Tests for choosing which rows to spend geometry requests on.

Geometry costs one request per row against a 450/day budget, so a bug here is not
a wrong number in a table — it is a day of budget spent on a sample that cannot
answer the question. These tests check the properties the experiment depends on
rather than the exact rows, because the exact rows are allowed to change when the
table does.

Run:  pytest -q
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.geometry_sample import (  # noqa: E402
    decode_reaction_id, describe, eligible_rows, select)


def row(i, surf="Pt", facet="111", e=0.0, func="PBE", ads="CO", pub="P1"):
    return {"id": base64.b64encode(f"Reaction:{i}".encode()).decode(),
            "surfaceComposition": surf, "facet": facet, "reactionEnergy": e,
            "dftFunctional": func, "adsorbate": ads, "pubId": pub}


def table(n_surfaces=6, sites=12, **kw):
    return [row(s * 100 + j, surf=f"M{s}", e=0.1 * j, **kw)
            for s in range(n_surfaces) for j in range(sites)]


# ---------------------------------------------------------------------------
# The id, which is what makes a one-row-at-a-time fetch possible at all
# ---------------------------------------------------------------------------

def test_the_base64_id_decodes_to_the_integer_the_filter_wants():
    assert decode_reaction_id("UmVhY3Rpb246MTY4MQ==") == 1681


def test_a_malformed_id_is_skipped_rather_than_raising():
    """One bad id must not abandon a download that is already partly paid for."""
    for bad in ("not base64", "", None, 17,
                base64.b64encode(b"Publication:5").decode(),
                base64.b64encode(b"Reaction:abc").decode()):
        assert decode_reaction_id(bad) is None


# ---------------------------------------------------------------------------
# The three filters, each of which exists to keep one quantity in the target
# ---------------------------------------------------------------------------

def test_impossible_energies_are_dropped():
    """+32 eV for CO adsorption is a failed calculation, not a weak bond."""
    rows = [row(1, e=-1.5), row(2, e=32.02), row(3, e=-40.0)]
    assert [r["reactionEnergy"] for r in eligible_rows(rows)] == [-1.5]


def test_a_missing_or_nonnumeric_energy_is_dropped():
    assert eligible_rows([row(1, e=None), row(2, e="-1.2")]) == []


def test_other_adsorbates_are_dropped():
    """H binding and CO binding are different quantities."""
    rows = [row(1, ads="CO"), row(2, ads="H")]
    assert len(eligible_rows(rows, adsorbate="CO")) == 1


def test_other_functionals_are_dropped():
    """The electrochemical ones carry a potential reference PBE does not."""
    rows = [row(1, func="PBE"), row(2, func="BEEF-vdW_-0.73VSHE"),
            row(3, func="RPBE")]
    kept = eligible_rows(rows)
    assert [r["dftFunctional"] for r in kept] == ["PBE"]


# ---------------------------------------------------------------------------
# The properties the experiment actually needs from the sample
# ---------------------------------------------------------------------------

def test_the_sample_has_many_sites_on_each_surface():
    """Without within-surface variation the graph model is handed exactly what
    composition already had, and the 0.806 eV ceiling cannot be beaten."""
    s = select(table(), n_surfaces=4, sites_per_surface=10)
    assert describe(s)["sites_per_surface_min"] == 10


def test_the_sample_has_many_surfaces():
    """Held-out surfaces are how generalisation gets tested; 2 is not enough."""
    s = select(table(n_surfaces=40, sites=12), n_surfaces=40, sites_per_surface=10)
    assert describe(s)["surfaces"] == 40


def test_the_sample_size_is_the_budget_that_was_agreed():
    s = select(table(n_surfaces=60, sites=12), n_surfaces=40, sites_per_surface=10)
    assert len(s) == 400


def test_surfaces_with_too_few_sites_are_excluded_entirely():
    """A surface contributing one row adds no site signal and costs a request."""
    rows = table(n_surfaces=3, sites=12) + [row(999, surf="Thin")]
    s = select(rows, n_surfaces=10, sites_per_surface=10)
    assert "Thin" not in {r["surfaceComposition"] for r in s}


def test_asking_for_more_surfaces_than_exist_returns_what_there_is():
    s = select(table(n_surfaces=3, sites=12), n_surfaces=40, sites_per_surface=10)
    assert describe(s)["surfaces"] == 3


def test_no_row_appears_twice():
    """Duplicates would be paid for twice and would inflate the row count."""
    s = select(table(n_surfaces=20, sites=15), n_surfaces=8, sites_per_surface=10)
    assert len({r["id"] for r in s}) == len(s)


# ---------------------------------------------------------------------------
# Determinism, which is what makes a partial download resumable
# ---------------------------------------------------------------------------

def test_the_same_table_gives_the_same_sample():
    rows = table(n_surfaces=30, sites=14)
    assert [r["id"] for r in select(rows, 10, 10)] == \
           [r["id"] for r in select(rows, 10, 10)]


def test_the_sample_does_not_depend_on_the_order_rows_arrived_in():
    """JSONL order reflects download order, which is not a scientific choice."""
    rows = table(n_surfaces=30, sites=14)
    a = {r["id"] for r in select(rows, 10, 10)}
    b = {r["id"] for r in select(list(reversed(rows)), 10, 10)}
    assert a == b


def test_surfaces_are_taken_from_across_the_size_range_not_only_the_largest():
    """Taking the biggest groups samples the most-studied surfaces, which are
    studied because they are interesting. That bias would go into training."""
    rows = []
    for s in range(40):
        for j in range(10 + s):          # sizes 10 .. 49
            rows.append(row(s * 1000 + j, surf=f"M{s}", e=0.1 * j))
    picked = {r["surfaceComposition"] for r in select(rows, 5, 10)}
    sizes = sorted(int(p[1:]) for p in picked)
    assert sizes[0] != sizes[-1] - 4, "picked 5 consecutive (largest) groups"
    assert max(sizes) - min(sizes) > 20, "sample does not span the size range"


# ---------------------------------------------------------------------------
# The description, which is what the limitations section is written from
# ---------------------------------------------------------------------------

def test_the_description_reports_publication_concentration():
    """The PBE subset is one publication. That has to be visible, not implied."""
    d = describe(select(table(n_surfaces=6, sites=12), 4, 10))
    assert d["publications"] == {"P1": 40}


def test_the_description_reports_within_surface_spread():
    """This is the signal being bought. A sample with no spread bought nothing."""
    d = describe(select(table(n_surfaces=6, sites=12), 4, 10))
    assert d["median_within_surface_spread_eV"] > 0.5


def test_describe_survives_an_empty_sample():
    assert describe([])["rows"] == 0


# ---------------------------------------------------------------------------
# Widening the sample must never abandon rows already paid for
# ---------------------------------------------------------------------------

def test_raising_sites_alone_silently_drops_surfaces():
    """The trap. Documented as a test so it cannot be reintroduced by someone
    who assumes a bigger sample is a bigger sample."""
    rows = []
    for s in range(10):
        for j in range(10 + s):              # groups of size 10 .. 19
            rows.append(row(s * 100 + j, surf=f"M{s}", e=0.1 * j))
    at10 = {r["id"] for r in select(rows, 10, 10)}
    at12 = {r["id"] for r in select(rows, 10, 12)}
    assert at10 - at12, "expected the qualifying bar to drop the smaller groups"


def test_holding_min_sites_fixed_makes_widening_purely_additive():
    """The fix, and the property a multi-day download depends on."""
    rows = []
    for s in range(10):
        for j in range(10 + s):
            rows.append(row(s * 100 + j, surf=f"M{s}", e=0.1 * j))
    base = {r["id"] for r in select(rows, 10, 10, min_sites=10)}
    for take in (12, 15, 20, 40):
        wider = {r["id"] for r in select(rows, 10, take, min_sites=10)}
        assert base <= wider, f"--sites {take} stranded {len(base - wider)} rows"


def test_min_sites_defaults_to_sites_so_old_calls_are_unchanged():
    rows = [row(i, surf=f"M{i // 12}", e=0.1 * i) for i in range(60)]
    assert [r["id"] for r in select(rows, 5, 10)] == \
           [r["id"] for r in select(rows, 5, 10, min_sites=10)]


def test_a_surface_never_contributes_more_rows_than_it_has():
    rows = [row(i, surf=f"M{i // 11}", e=0.1 * i) for i in range(55)]
    s = select(rows, 5, 40, min_sites=10)
    from collections import Counter
    assert max(Counter(r["surfaceComposition"] for r in s).values()) <= 11
