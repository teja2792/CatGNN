"""Descriptor and baseline tests.

The single most important property here is negative: the composition block must
be *blind to structure*. If it can distinguish two polymorphs of the same formula
then the whole comparison this repository is built on is meaningless, because
"composition-only" would silently be "composition plus a bit of structure".

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

from src.features.descriptors import (  # noqa: E402
    BLOCKS,
    composition_features,
    element_table,
    feature_names,
    featurise,
    property_matrix,
    structure_features,
)
from src.models.baselines import MODELS, build_model, metrics  # noqa: E402

RUTILE = dict(nsites=6, volume=62.43, density=4.236, nelements=2,
              spacegroup_number=136, crystal_system="Tetragonal")
ANATASE = dict(nsites=12, volume=136.25, density=3.895, nelements=2,
               spacegroup_number=141, crystal_system="Tetragonal")
TIO2 = ["Ti", "Ti", "O", "O", "O", "O"]


# ---------------------------------------------------------------------------
# The element table
# ---------------------------------------------------------------------------

def test_element_table_covers_the_periodic_table():
    elements, props = element_table()
    assert len(elements) >= 100, f"only {len(elements)} elements"
    for s in ("H", "O", "Ti", "Fe", "Cu", "U"):
        assert s in elements


def test_known_element_properties_are_right():
    """Spot-check against values any chemist can verify."""
    elements, _ = element_table()

    assert elements["O"]["Z"] == 8
    assert elements["O"]["X"] == pytest.approx(3.44, abs=0.02)   # electronegativity
    assert elements["Ti"]["Z"] == 22
    assert elements["Ti"]["is_transition_metal"] == 1.0
    assert elements["Ne"]["is_noble_gas"] == 1.0
    assert elements["Na"]["is_alkali"] == 1.0
    assert elements["Fe"]["X"] < elements["O"]["X"], "O must be more electronegative than Fe"


def test_property_matrix_has_no_gaps_left():
    """Missing values must be imputed in one place, not propagated as NaN."""
    _, mat, _ = property_matrix()
    assert np.isfinite(mat).all(), "NaN survived into the property matrix"


# ---------------------------------------------------------------------------
# Composition must be structurally blind -- the load-bearing property
# ---------------------------------------------------------------------------

def test_composition_cannot_tell_rutile_from_anatase():
    """The premise of the whole experiment, asserted rather than assumed."""
    a = composition_features(TIO2)
    b = composition_features(TIO2 * 2)   # anatase cell, same formula
    assert np.allclose(a, b), "composition features leaked structural information"


def test_composition_depends_only_on_the_ratio_not_the_cell_size():
    single = composition_features(["Ti", "O", "O"])
    triple = composition_features(["Ti", "O", "O"] * 3)
    assert np.allclose(single, triple)


def test_composition_is_order_independent():
    a = composition_features(["Ti", "O", "O"])
    b = composition_features(["O", "Ti", "O"])
    assert np.allclose(a, b)


def test_composition_distinguishes_different_stoichiometries():
    """TiO2 and TiO must not collide, or the block is useless."""
    assert not np.allclose(composition_features(["Ti", "O", "O"]),
                           composition_features(["Ti", "O"]))


def test_composition_distinguishes_different_elements():
    assert not np.allclose(composition_features(["Ti", "O", "O"]),
                           composition_features(["Zr", "O", "O"]))


def test_composition_features_are_finite():
    for species in (["Ti"], ["Ti", "O", "O"], ["H"] * 30, ["Fe", "Co", "Ni", "O", "O", "O"]):
        v = composition_features(species)
        assert np.isfinite(v).all(), f"non-finite features for {species}"


def test_unknown_element_symbols_are_ignored_not_fatal():
    v = composition_features(["Ti", "O", "O", "Unobtainium"])
    assert np.isfinite(v).all()
    assert np.allclose(v, composition_features(["Ti", "O", "O"]))


def test_empty_composition_returns_zeros_of_the_right_length():
    v = composition_features([])
    assert v.shape == (len(feature_names("composition")),)
    assert np.isfinite(v).all()


# ---------------------------------------------------------------------------
# Structure block
# ---------------------------------------------------------------------------

def test_structure_block_does_separate_the_polymorphs():
    """The mirror image of the composition test: this block must see structure."""
    a = structure_features(RUTILE, TIO2)
    b = structure_features(ANATASE, TIO2 * 2)
    assert not np.allclose(a, b)


def test_packing_fraction_is_physically_plausible():
    """Rutile is denser than anatase, so it must pack more tightly."""
    names = feature_names("structure_lite")
    j = names.index("struct_packing_fraction")

    rut = structure_features(RUTILE, TIO2)[j]
    ana = structure_features(ANATASE, TIO2 * 2)[j]

    assert 0.0 < rut < 1.5 and 0.0 < ana < 1.5, f"implausible packing: {rut}, {ana}"
    assert rut > ana, "rutile is the denser polymorph and must pack tighter"


def test_crystal_system_one_hot_is_exclusive():
    names = feature_names("structure_lite")
    idx = [i for i, n in enumerate(names) if n.startswith("struct_system_")]
    v = structure_features(RUTILE, TIO2)
    assert sum(v[i] for i in idx) == 1.0


def test_unknown_crystal_system_gives_all_zeros_not_a_crash():
    v = structure_features(dict(RUTILE, crystal_system="Quasicrystal"), TIO2)
    names = feature_names("structure_lite")
    idx = [i for i, n in enumerate(names) if n.startswith("struct_system_")]
    assert sum(v[i] for i in idx) == 0.0


def test_missing_structure_fields_do_not_crash():
    v = structure_features({}, TIO2)
    assert np.isfinite(v).all()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block", BLOCKS)
def test_vector_length_matches_declared_names(block):
    v = featurise(RUTILE, TIO2, block)
    assert v.shape == (len(feature_names(block)),)


def test_both_is_exactly_the_concatenation():
    a = featurise(RUTILE, TIO2, "composition")
    b = featurise(RUTILE, TIO2, "structure_lite")
    both = featurise(RUTILE, TIO2, "both")
    assert np.allclose(both, np.concatenate([a, b]))


def test_feature_names_are_unique():
    for block in BLOCKS:
        names = feature_names(block)
        assert len(set(names)) == len(names), f"duplicate names in {block}"


def test_unknown_block_raises():
    with pytest.raises(ValueError, match="unknown block"):
        featurise(RUTILE, TIO2, "telepathy")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_every_model_fits_and_predicts(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=300)

    model = build_model(name)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (300,)
    assert np.isfinite(pred).all()


def test_the_dummy_baseline_predicts_the_median():
    """It must be the MAE-optimal constant, or the floor we quote is beatable."""
    y = np.array([0.0, 0.0, 0.0, 1.0, 5.0])
    model = build_model("mean")
    model.fit(np.zeros((5, 2)), y)
    assert model.predict(np.zeros((1, 2)))[0] == pytest.approx(np.median(y))


def test_real_models_beat_the_dummy_on_learnable_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 6))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.05, size=500)

    dummy = build_model("mean").fit(X, y)
    floor = np.abs(y - dummy.predict(X)).mean()

    for name in ("ridge", "rf", "gbm"):
        model = build_model(name).fit(X, y)
        mae = np.abs(y - model.predict(X)).mean()
        assert mae < floor * 0.5, f"{name} failed to beat the constant baseline"


def test_metrics_are_correct_on_a_hand_worked_example():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.0, 2.0, 5.0])

    m = metrics(y, p)
    assert m["mae"] == pytest.approx((0.5 + 0.0 + 1.0 + 1.0) / 4)
    assert m["medae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 1) / 4))
    assert m["n"] == 4


def test_r2_is_zero_for_a_constant_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)


def test_ridge_is_scaled():
    """Unscaled, melting point (thousands of K) would drown electronegativity (~2)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    X[:, 2] *= 5000.0                       # one feature on a wildly different scale
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=400)

    model = build_model("ridge").fit(X, y)
    assert np.abs(y - model.predict(X)).mean() < 0.2, "ridge is not scaling its inputs"
