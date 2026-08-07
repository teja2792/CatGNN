"""Phase 0 tests.

These are correctness tests, not smoke tests. Each one checks a claim the README
makes, so that if the claim stops being true, CI says so rather than the figure
quietly going wrong.

Run:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "data" / "reference" / "mp_summary_snapshot.csv"


# ---------------------------------------------------------------------------
# Crystal structure and graph construction
# ---------------------------------------------------------------------------

def test_rutile_bond_lengths_match_neutron_diffraction():
    """The rutile cell we hard-code must reproduce the published bond lengths.

    If someone fat-fingers a lattice constant, every downstream figure is wrong.
    Literature: Ti-O 1.946 A (x4 equatorial), 1.983 A (x2 apical).
    """
    from src.figures.fig_crystal_to_graph import rutile_cell, periodic_neighbours

    frac, species, lattice = rutile_cell()
    pairs = periodic_neighbours(frac, lattice, 2.4)
    lengths = sorted({round(d, 3) for _, _, d, _ in pairs})

    assert len(lengths) == 2, f"rutile should show two distinct Ti-O lengths, got {lengths}"
    assert lengths[0] == pytest.approx(1.946, abs=0.01)
    assert lengths[1] == pytest.approx(1.983, abs=0.01)


def test_rutile_coordination_numbers():
    """Ti must come out 6-coordinate and O 3-coordinate.

    This is the test that catches the periodic-image bug: a neighbour search that
    keeps only the closest image per atom pair returns Ti CN = 4, silently
    destroying the octahedron, and nothing else in the pipeline would notice.
    """
    from src.figures.fig_crystal_to_graph import rutile_cell, periodic_neighbours

    frac, species, lattice = rutile_cell()
    pairs = periodic_neighbours(frac, lattice, 2.4)

    cn = {i: sum(1 for a, _, _, _ in pairs if a == i) for i in range(len(species))}
    ti_cn = [cn[i] for i, s in enumerate(species) if s == "Ti"]
    o_cn = [cn[i] for i, s in enumerate(species) if s == "O"]

    assert ti_cn == [6, 6], f"rutile Ti must be octahedral, got {ti_cn}"
    assert o_cn == [3, 3, 3, 3], f"rutile O must be 3-coordinate, got {o_cn}"


def test_neighbour_list_is_symmetric():
    """If i is a neighbour of j, then j is a neighbour of i. Undirected graph."""
    from src.figures.fig_crystal_to_graph import rutile_cell, periodic_neighbours

    frac, _, lattice = rutile_cell()
    pairs = periodic_neighbours(frac, lattice, 2.4)

    fwd = sorted((i, j, round(d, 4)) for i, j, d, _ in pairs)
    rev = sorted((j, i, round(d, 4)) for i, j, d, _ in pairs)
    assert fwd == rev


def test_graph_is_translation_invariant():
    """Rigidly shifting every atom must not change the neighbour list.

    A graph built from a crystal should encode relative geometry only. If this
    fails, the model could learn something about arbitrary cell origin choice,
    which is physically meaningless.
    """
    from src.figures.fig_crystal_to_graph import rutile_cell, periodic_neighbours

    frac, _, lattice = rutile_cell()
    base = sorted(round(d, 6) for _, _, d, _ in periodic_neighbours(frac, lattice, 2.4))

    for shift in ([0.31, 0.17, 0.44], [0.5, 0.5, 0.5], [0.99, 0.01, 0.73]):
        moved = (frac + np.array(shift)) % 1.0
        got = sorted(round(d, 6) for _, _, d, _ in periodic_neighbours(moved, lattice, 2.4))
        assert got == base, f"neighbour list changed under translation {shift}"


def test_cutoff_is_monotonic():
    """A larger cutoff can only add edges, never remove them."""
    from src.figures.fig_crystal_to_graph import rutile_cell, periodic_neighbours

    frac, _, lattice = rutile_cell()
    counts = [len(periodic_neighbours(frac, lattice, c)) for c in (2.0, 2.4, 2.6, 3.0)]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# The claim Figure 1 makes
# ---------------------------------------------------------------------------

def test_snapshot_present_and_shaped():
    assert SNAPSHOT.exists(), "Materials Project snapshot missing -- see SOURCES.md"
    df = pd.read_csv(SNAPSHOT)
    for col in ("formula_pretty", "band_gap_eV", "formation_energy_per_atom_eV", "dft_run_type"):
        assert col in df.columns, f"snapshot missing column {col}"
    assert len(df) > 50


def test_composition_only_floor_is_the_mean_absolute_deviation():
    """The error floor must be minimal at the median, by construction."""
    from src.figures.fig_polymorph_problem import composition_only_floor

    rng = np.random.default_rng(0)
    v = rng.normal(2.0, 0.7, 400)
    floor = composition_only_floor(v)

    for guess in np.linspace(v.min(), v.max(), 60):
        assert np.abs(v - guess).mean() >= floor - 1e-9, "median is not optimal for MAE"


def test_tio2_polymorph_spread_is_real_and_large():
    """The headline motivating claim, checked against the data it came from.

    A formula-only model cannot separate TiO2 polymorphs, so its unavoidable
    band-gap error is the spread between them. The README states this is larger
    than CGCNN's published total error of 0.388 eV.
    """
    from src.figures.fig_polymorph_problem import composition_only_floor, OUTLIER_FE_CUTOFF

    df = pd.read_csv(SNAPSHOT)
    tio2 = df[(df.formula_pretty == "TiO2") & (df.dft_run_type == "GGA")]
    assert len(tio2) >= 40, f"expected 40+ TiO2 GGA entries, got {len(tio2)}"

    clean = tio2[tio2.formation_energy_per_atom_eV < OUTLIER_FE_CUTOFF]
    floor = composition_only_floor(clean.band_gap_eV.values)

    assert 0.35 < floor < 0.55, f"TiO2 band-gap floor moved to {floor:.3f} eV -- update the README"
    assert floor > 0.388 * 0.9, "claim that the floor rivals CGCNN's total error no longer holds"


def test_functionals_are_never_mixed():
    """GGA, GGA+U and r2SCAN gaps are different quantities.

    Guards against a genuine methodological error: averaging band gaps across
    functionals would produce a meaningless number that still looks fine.
    """
    df = pd.read_csv(SNAPSHOT)
    for formula, sub in df.groupby("formula_pretty"):
        if sub.dft_run_type.nunique() > 1:
            assert formula == "TiO2", (
                f"{formula} unexpectedly mixes functionals; check the analysis groups by dft_run_type"
            )


# ---------------------------------------------------------------------------
# Figures actually build
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module",
    [
        "src.figures.fig_polymorph_problem",
        "src.figures.fig_crystal_to_graph",
        "src.figures.fig_data_provenance",
        "src.figures.fig_roadmap",
    ],
)
def test_figure_builds(module):
    import importlib
    import matplotlib
    matplotlib.use("Agg")

    mod = importlib.import_module(module)
    mod.main()


def test_readme_figures_exist():
    """Every figure the README embeds must be on disk."""
    figs = [
        "fig1_polymorph_problem.png",
        "fig2_crystal_to_graph.png",
        "fig3_data_provenance.png",
        "fig4_roadmap.png",
    ]
    missing = [f for f in figs if not (REPO / "results" / "figures" / f).exists()]
    assert not missing, f"missing figures: {missing} -- run python scripts/make_figures.py"
