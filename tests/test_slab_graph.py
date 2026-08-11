"""Tests for building graphs from Catalysis-Hub slab geometries.

The failure this file mostly guards against is not a crash. It is a dataset that
builds cleanly, trains, converges, and reports an RMSE while describing surfaces
whose atoms have the wrong number of neighbours. Coordination is the strongest
simple structural determinant of binding energy, so getting it wrong changes the
answer without changing the appearance of the answer.

Run:  pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.graph_build import image_range  # noqa: E402
from src.data.slab_graph import (  # noqa: E402
    adsorbate_mask, build_slab_graph, classify_systems, coordination, is_molecule,
    parse_ase_json, pbc_for, vacuum_gap)


def ase_blob(numbers, positions, cell):
    return json.dumps({
        "1": {"numbers": {"__ndarray__": [[len(numbers)], "int64",
                                          [int(v) for v in numbers]]},
              "positions": {"__ndarray__": [[len(numbers), 3], "float64",
                                            list(np.ravel(positions))]},
              "cell": {"array": {"__ndarray__": [[3, 3], "float64",
                                                 list(np.ravel(cell))]}},
              "pbc": {"__ndarray__": [[3], "bool", [False, False, False]]}},
        "ids": [1], "nextid": 2})


def slab_atoms(n_layers=4, a=2.8, vacuum=15.0):
    """A small square-lattice slab: 2x2 in plane, n_layers deep, vacuum above."""
    pos, num = [], []
    for k in range(n_layers):
        for i in range(2):
            for j in range(2):
                pos.append([i * a, j * a, k * a])
                num.append(78)                      # Pt
    height = (n_layers - 1) * a
    cell = np.diag([2 * a, 2 * a, height + vacuum])
    return {"numbers": np.array(num, dtype=np.int16),
            "positions": np.array(pos, dtype=float), "cell": cell}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_an_ase_payload_round_trips():
    s = slab_atoms()
    got = parse_ase_json(ase_blob(s["numbers"], s["positions"], s["cell"]))
    assert got is not None
    assert np.array_equal(got["numbers"], s["numbers"])
    assert np.allclose(got["positions"], s["positions"])


def test_unreadable_geometry_returns_none_rather_than_raising():
    """One bad row must not discard a download that cost a day of budget."""
    for bad in ("{not json", "", None, "[]", json.dumps({"ids": [1]}), 42,
                json.dumps({"1": {"numbers": {"__ndarray__": [[2], "int64", [1, 2]]}}})):
        assert parse_ase_json(bad) is None


def test_a_length_mismatch_is_refused():
    """Three atoms with two positions is corruption, not a small structure."""
    blob = ase_blob([1, 2, 3], np.zeros((2, 3)), np.eye(3))
    assert parse_ase_json(blob) is None


# ---------------------------------------------------------------------------
# The stored pbc flag is wrong; periodicity comes from what the system IS
# ---------------------------------------------------------------------------

def test_a_slab_is_periodic_in_plane_and_finite_along_the_normal():
    assert pbc_for(slab_atoms()) == (True, True, False)


def test_a_gas_molecule_is_periodic_in_nothing():
    co = {"numbers": np.array([6, 8]), "positions": np.array([[0., 0., 0.],
                                                              [0., 0., 1.13]]),
          "cell": np.diag([15., 15., 15.])}
    assert is_molecule(co)
    assert pbc_for(co) == (False, False, False)


def test_periodicity_ignores_the_stored_flag_entirely():
    """Every downloaded file says pbc=False; 584 of 794 slabs are mislabelled.

    The builder must not consult the field at all, so a file claiming otherwise
    changes nothing.
    """
    s = slab_atoms()
    blob = ase_blob(s["numbers"], s["positions"], s["cell"])   # flag says False
    parsed = parse_ase_json(blob)
    assert pbc_for(parsed) == (True, True, False)


def test_dropping_in_plane_periodicity_loses_bonds():
    """The measured cost of trusting the file: 34% of all bonds on real data."""
    s = slab_atoms()
    right = build_slab_graph(s)
    frac = np.linalg.solve(s["cell"].T, s["positions"].T).T
    from src.data.graph_build import neighbour_list
    _, _, dist = neighbour_list(s["cell"], frac, 8.0, 12, pbc=(False, False, False))
    assert (right["dist"] <= 3.0).sum() > (dist <= 3.0).sum()


def test_image_range_zeroes_only_the_non_periodic_axes():
    lattice = np.diag([3.0, 3.0, 30.0])
    assert list(image_range(lattice, 8.0, (True, True, False))) == \
           [*image_range(lattice, 8.0)[:2], 0]


def test_image_range_default_is_still_fully_periodic():
    """Bulk crystals must be unaffected by the new argument."""
    lattice = np.diag([3.0, 3.0, 3.0])
    assert np.array_equal(image_range(lattice, 8.0), image_range(lattice, 8.0, None))


# ---------------------------------------------------------------------------
# The physical statement the graphs have to satisfy
# ---------------------------------------------------------------------------

def test_surface_atoms_are_less_coordinated_than_interior_atoms():
    """Undercoordination is why a surface binds anything. Measured 0.71x on the
    real sample. A graph that fails this is describing a different solid."""
    s = slab_atoms(n_layers=6, a=2.8)
    g = build_slab_graph(s)
    cn = coordination(g, len(s["numbers"]))
    z = s["positions"][:, 2]
    layers = np.unique(z)
    top = z >= layers[-1] - 0.1                 # outermost layer
    mid = np.abs(z - layers[len(layers) // 2]) < 0.1    # an interior layer
    assert top.any() and mid.any()
    assert cn[top].mean() < cn[mid].mean()


def test_the_slab_does_not_bond_through_its_own_vacuum():
    """Repeating along the normal would stack the slab on its vacuum and invent
    a second surface. With 15 A of vacuum nothing should reach across."""
    s = slab_atoms(vacuum=15.0)
    g = build_slab_graph(s)
    z = s["positions"][:, 2]
    spans = np.abs(z[g["src"]] - z[g["dst"]])
    assert spans.max() < vacuum_gap(s)


def test_vacuum_gap_is_measured_not_assumed():
    assert vacuum_gap(slab_atoms(n_layers=4, a=2.8, vacuum=15.0)) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Which system is which, and which atoms are the adsorbate
# ---------------------------------------------------------------------------

def _systems():
    clean = slab_atoms()
    ads = {**clean,
           "numbers": np.append(clean["numbers"], [6, 8]).astype(np.int16),
           "positions": np.vstack([clean["positions"],
                                   [[0., 0., 10.0], [0., 0., 11.13]]])}
    gas = {"numbers": np.array([6, 8]),
           "positions": np.array([[0., 0., 0.], [0., 0., 1.13]]),
           "cell": np.diag([15., 15., 15.])}
    return [{"InputFile": ase_blob(a["numbers"], a["positions"], a["cell"]),
             "energy": e, "Formula": f}
            for a, e, f in ((ads, -239.4, "Pt16CO"), (clean, -224.4, "Pt16"),
                            (gas, -14.8, "CO"))]


def test_the_three_systems_are_identified_by_composition_not_list_order():
    """List order is not a documented guarantee, and mislabelling the clean slab
    as the adsorbed one inverts the quantity being learned."""
    for perm in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        d = classify_systems([_systems()[i] for i in perm])
        assert len(d["adsorbed"]["numbers"]) == 18
        assert len(d["clean"]["numbers"]) == 16
        assert len(d["gas"]["numbers"]) == 2


def test_two_slabs_of_equal_size_are_refused():
    """Adsorption adds atoms. Equal sizes mean the pair is not what it claims."""
    s = _systems()
    assert classify_systems([s[1], s[1], s[2]]) == {}


def test_the_adsorbate_atoms_are_found_by_composition_difference():
    d = classify_systems(_systems())
    mask = adsorbate_mask(d["clean"], d["adsorbed"])
    assert mask.sum() == 2
    assert set(d["adsorbed"]["numbers"][mask].tolist()) == {6, 8}


def test_the_adsorbate_is_found_even_when_the_surface_shares_its_elements():
    """A carbide surface contains carbon too. Position in the list will not
    distinguish them; being the outermost atom of that element will."""
    clean = slab_atoms()
    clean["numbers"][:4] = 6                        # carbon in the surface
    ads = {**clean,
           "numbers": np.append(clean["numbers"], [6, 8]).astype(np.int16),
           "positions": np.vstack([clean["positions"],
                                   [[0., 0., 10.0], [0., 0., 11.13]]])}
    mask = adsorbate_mask(clean, ads)
    assert mask.sum() == 2
    assert mask[-2] and mask[-1], "picked buried carbon over the adsorbate"


def test_a_reaction_with_no_slab_yields_nothing():
    assert classify_systems([_systems()[2]]) == {}


def test_missing_geometry_does_not_raise():
    assert classify_systems([{"InputFile": None}]) == {}
    assert classify_systems([]) == {}
