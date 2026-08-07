"""Graph construction tests.

Crystal graphs are where a silent bug does the most damage: a wrong neighbour
list raises no error, it just trains the model on chemistry that does not exist.
So these tests check physical properties with known answers -- coordination
numbers, bond lengths, and the symmetries the representation is supposed to have
-- rather than that the code merely runs.

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

from src.data.graph_build import (  # noqa: E402
    Z_OF,
    build_graph,
    gaussian_expand,
    image_range,
    neighbour_list,
)

RUTILE_A, RUTILE_C, RUTILE_U = 4.5937, 2.9587, 0.30478


def rutile():
    u = RUTILE_U
    lattice = np.diag([RUTILE_A, RUTILE_A, RUTILE_C])
    frac = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        [u, u, 0.0], [1 - u, 1 - u, 0.0],
        [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5],
    ])
    return lattice, frac, ["Ti", "Ti", "O", "O", "O", "O"]


def rocksalt(a=4.2):
    """NaCl: both ions octahedrally coordinated. A second known-answer case."""
    lattice = np.eye(3) * a
    frac = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
    ], dtype=float)
    return lattice, frac, ["Na"] * 4 + ["Cl"] * 4


# ---------------------------------------------------------------------------
# Known-answer chemistry
# ---------------------------------------------------------------------------

def test_rutile_coordination_and_bond_lengths():
    lattice, frac, _ = rutile()
    src, dst, dist = neighbour_list(lattice, frac, cutoff=2.4, max_neighbours=12)

    cn = np.bincount(src, minlength=6)
    assert cn.tolist() == [6, 6, 3, 3, 3, 3], f"rutile CN wrong: {cn.tolist()}"

    lengths = sorted(set(np.round(dist, 3).tolist()))
    assert len(lengths) == 2
    assert lengths[0] == pytest.approx(1.946, abs=0.01)   # x4 equatorial
    assert lengths[1] == pytest.approx(1.983, abs=0.01)   # x2 apical


def test_rocksalt_is_six_coordinate():
    lattice, frac, _ = rocksalt(a=4.2)
    src, _, _ = neighbour_list(lattice, frac, cutoff=2.3, max_neighbours=12)
    assert np.bincount(src, minlength=8).tolist() == [6] * 8


# ---------------------------------------------------------------------------
# Periodic image range -- the bug a hard-coded +/-1 would cause
# ---------------------------------------------------------------------------

def test_image_range_grows_as_the_cell_shrinks():
    assert (image_range(np.eye(3) * 3.0, 8.0) == 3).all()
    assert (image_range(np.eye(3) * 20.0, 8.0) == 1).all()


def test_image_range_uses_perpendicular_width_not_vector_length():
    """A skewed cell can be far thinner than its vectors are long.

    Vector b here has length 5.0, which would suggest 2 images at an 8 A cutoff.
    The perpendicular width is only 2.2 A, so 4 are needed. Using |b| would
    silently truncate the neighbour list of every triclinic structure.
    """
    lattice = np.array([[5.0, 0, 0], [4.5, 2.2, 0], [0, 0, 6.0]])
    n = image_range(lattice, 8.0)

    assert n[0] >= 4 and n[1] >= 4, f"perpendicular width ignored: {n}"
    assert np.ceil(8.0 / np.linalg.norm(lattice[1])) == 2  # what the naive rule gives


def test_small_cell_large_cutoff_finds_every_neighbour():
    """One atom in a 3 A cell at 8 A cutoff: answer checkable by brute force."""
    a = 3.0
    src, dst, dist = neighbour_list(np.eye(3) * a, np.array([[0.0, 0.0, 0.0]]),
                                    cutoff=8.0, max_neighbours=500)

    expected = sum(
        1
        for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)
        if 0 < np.linalg.norm(np.array([i, j, k]) * a) <= 8.0
    )
    assert len(dist) == expected, f"found {len(dist)}, brute force says {expected}"

    naive = sum(
        1
        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
        if 0 < np.linalg.norm(np.array([i, j, k]) * a) <= 8.0
    )
    assert naive < expected * 0.5, "this case must expose the +/-1 shortcut"


def test_atom_is_its_own_neighbour_through_periodic_images():
    """In a small cell an atom really does neighbour its own periodic copy."""
    src, dst, dist = neighbour_list(np.eye(3) * 3.0, np.array([[0.0, 0.0, 0.0]]),
                                    cutoff=4.0, max_neighbours=50)
    assert len(dist) > 0
    assert (src == dst).all()          # only one atom exists
    assert dist.min() == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Symmetries the representation must have
# ---------------------------------------------------------------------------

def test_translation_invariance():
    lattice, frac, _ = rutile()
    base = np.sort(neighbour_list(lattice, frac, 8.0, 12)[2])

    for shift in ([0.31, 0.17, 0.44], [0.5, 0.5, 0.5], [0.99, 0.01, 0.73]):
        moved = (frac + np.array(shift)) % 1.0
        got = np.sort(neighbour_list(lattice, moved, 8.0, 12)[2])
        assert np.allclose(base, got, atol=1e-6), f"changed under translation {shift}"


def test_permutation_invariance():
    """Relabelling the atoms must not change the multiset of bond lengths."""
    lattice, frac, _ = rutile()
    base = np.sort(neighbour_list(lattice, frac, 8.0, 12)[2])

    perm = np.random.default_rng(0).permutation(len(frac))
    got = np.sort(neighbour_list(lattice, frac[perm], 8.0, 12)[2])
    assert np.allclose(base, got, atol=1e-6)


def test_rotation_invariance():
    """Rotating the whole crystal must leave every distance unchanged."""
    lattice, frac, _ = rutile()
    base = np.sort(neighbour_list(lattice, frac, 8.0, 12)[2])

    t = 0.7
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    got = np.sort(neighbour_list(lattice @ R.T, frac, 8.0, 12)[2])
    assert np.allclose(base, got, atol=1e-5)


def test_supercell_gives_the_same_local_environments():
    """A 2x2x2 supercell is the same crystal, so every atom's environment repeats.

    This is the strongest single check on periodic handling: get the images wrong
    and the unit cell and the supercell disagree.
    """
    lattice, frac, _ = rutile()
    _, _, d1 = neighbour_list(lattice, frac, 5.0, 12)

    big = lattice * 2
    shifts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    frac2 = np.concatenate([(frac + s) / 2.0 for s in shifts])
    _, _, d2 = neighbour_list(big, frac2, 5.0, 12)

    assert len(d2) == 8 * len(d1)
    assert np.allclose(np.sort(d1), np.sort(d2)[: len(d1)], atol=1e-6) or \
        np.allclose(np.sort(np.tile(d1, 8)), np.sort(d2), atol=1e-6)


# ---------------------------------------------------------------------------
# Contract of build_graph
# ---------------------------------------------------------------------------

def test_build_graph_returns_expected_arrays():
    lattice, frac, species = rutile()
    g = build_graph({"lattice": lattice.tolist(),
                     "frac_coords": frac.tolist(),
                     "species": species})

    assert g is not None
    assert g["z"].tolist() == [Z_OF["Ti"], Z_OF["Ti"]] + [Z_OF["O"]] * 4
    assert g["src"].size == g["dst"].size == g["dist"].size
    assert g["dist"].min() > 0 and g["dist"].max() <= 8.0 + 1e-6


def test_unknown_element_is_refused_not_guessed():
    g = build_graph({"lattice": (np.eye(3) * 4).tolist(),
                     "frac_coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
                     "species": ["Ti", "Unobtainium"]})
    assert g is None


def test_malformed_structures_return_none():
    for bad in (
        {},
        {"lattice": (np.eye(3) * 4).tolist(), "frac_coords": [], "species": []},
        {"lattice": (np.eye(3) * 4).tolist(), "frac_coords": [[0, 0, 0]], "species": []},
        {"lattice": np.zeros((3, 3)).tolist(), "frac_coords": [[0, 0, 0]], "species": ["Ti"]},
    ):
        assert build_graph(bad) is None


def test_max_neighbours_is_respected():
    lattice, frac, species = rutile()
    for cap in (4, 8, 12):
        src, _, _ = neighbour_list(lattice, frac, 8.0, cap)
        assert np.bincount(src, minlength=6).max() <= cap


def test_neighbours_are_returned_nearest_first():
    lattice, frac, _ = rutile()
    src, _, dist = neighbour_list(lattice, frac, 8.0, 12)
    for atom in np.unique(src):
        d = dist[src == atom]
        assert np.all(np.diff(d) >= -1e-6), "neighbours not sorted by distance"


# ---------------------------------------------------------------------------
# Edge featurisation
# ---------------------------------------------------------------------------

def test_gaussian_expansion_shape_and_peak():
    d = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32)
    feat = gaussian_expand(d, dmin=0.0, dmax=8.0, step=0.2)

    assert feat.shape == (4, 41)
    assert np.allclose(feat.max(axis=1), 1.0, atol=1e-6)
    # the peak must sit at the centre nearest the true distance
    assert np.argmax(feat[1]) == pytest.approx(10, abs=1)   # 2.0 A -> centre 10
    assert np.argmax(feat[2]) == pytest.approx(20, abs=1)   # 4.0 A -> centre 20


def test_gaussian_expansion_is_smooth():
    """Feature distance must grow with physical distance, without saturating early.

    The property that matters is monotonicity: two bonds that are similar in
    length must look more alike to the model than two that are not. Testing a
    fixed ratio instead would just be asserting the value of sigma.
    """
    ref = gaussian_expand(np.array([3.00], dtype=np.float32))
    deltas = [
        float(np.linalg.norm(ref - gaussian_expand(np.array([d], dtype=np.float32))))
        for d in (3.02, 3.05, 3.2, 3.5, 6.0)
    ]

    assert deltas == sorted(deltas), f"not monotonic in distance: {deltas}"
    assert deltas[0] < deltas[-1] * 0.2, "a 0.02 A change should barely register"


def test_gaussian_expansion_separates_typical_bond_lengths():
    """Two bonds 0.2 A apart -- rutile's equatorial vs apical Ti-O -- must differ.

    If the basis cannot resolve that, it cannot represent the distortion that
    distinguishes many polymorphs, and the whole premise of the repo suffers.
    """
    eq = gaussian_expand(np.array([1.946], dtype=np.float32))
    ap = gaussian_expand(np.array([1.983], dtype=np.float32))
    far = gaussian_expand(np.array([4.0], dtype=np.float32))

    assert np.linalg.norm(eq - ap) > 1e-3, "cannot resolve a real bond-length difference"
    assert np.linalg.norm(eq - ap) < np.linalg.norm(eq - far)
