"""Tests for Phase 6 -- attributing a prediction to named element properties.

The taxonomy tests run everywhere. They exist because the property list and the
family map live in different files, and the failure mode when they drift apart is
silent: a new property simply appears as "other" on the figure, or worse, an
existing one is quietly renamed and its bar disappears.

The integrated-gradients tests check the one property that makes IG worth using
over a plain gradient: COMPLETENESS. The attributions must sum to the change in
the model's output between the baseline and the real input. That is an arithmetic
identity, so it is either satisfied or the implementation is wrong -- there is no
"looks about right" for it.

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

from src.features.element_features import element_feature_table  # noqa: E402
from src.features.property_groups import (  # noqa: E402
    FAMILY_ORDER, PROPERTY_FAMILY, family_of, label_of)


def _has_torch() -> bool:
    try:
        __import__("torch")
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# The taxonomy, which must track the element table
# ---------------------------------------------------------------------------

def test_every_property_has_a_family():
    """Catches the drift when a property is added to the reference table."""
    _, _, names = element_feature_table()
    missing = [n for n in names if n not in PROPERTY_FAMILY]
    assert not missing, f"no family assigned to {missing}"


def test_no_family_entries_for_properties_that_no_longer_exist():
    """The other direction: a renamed property leaves a dangling entry."""
    _, _, names = element_feature_table()
    stale = [n for n in PROPERTY_FAMILY if n not in names]
    assert not stale, f"family map mentions properties not in the table: {stale}"


def test_families_are_all_known():
    assert set(PROPERTY_FAMILY.values()) <= set(FAMILY_ORDER)


def test_labels_are_readable():
    """No pymatgen shorthand should reach a figure. 'X' means electronegativity."""
    _, _, names = element_feature_table()
    assert label_of("X") == "electronegativity"
    for n in names:
        assert "_" not in label_of(n), f"{n} still renders with an underscore"


def test_family_of_is_total():
    assert family_of("something_invented") == "other"


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------

def toy_batch(torch):
    from src.models import cgcnn_reference as ref

    rng = np.random.default_rng(0)
    return {
        "z": torch.from_numpy(np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)),
        "src": torch.from_numpy(np.array([0, 0, 1, 2, 3, 1, 4, 5])),
        "dst": torch.from_numpy(np.array([1, 2, 0, 0, 1, 3, 5, 4])),
        "u": torch.from_numpy(ref.gaussian_expand(
            rng.uniform(1.5, 7.5, size=8)).astype(np.float32)),
        "batch_index": torch.from_numpy(np.array([0, 0, 0, 0, 1, 1])),
        "n_graphs": 2,
    }


def toy_model(torch, mode="properties"):
    from src.models.fusion import FusedGNN, FusionConfig

    return FusedGNN(FusionConfig(atom_features=mode, atom_fea_len=16, n_conv=2,
                                 h_fea_len=32, use_batch_norm=False)).eval()


@needs_torch
def test_completeness():
    """Attributions must sum to F(input) - F(baseline). Not a heuristic."""
    import torch as T

    from src.models.attribution import completeness_error, integrated_gradients

    model, batch = toy_model(T), toy_batch(T)
    a = integrated_gradients(model, batch, steps=256)
    err = completeness_error(model, batch, a)
    assert err < 1e-3, f"completeness violated by {err:.2e}"


@needs_torch
def test_completeness_improves_with_more_steps():
    """Distinguishes a discretisation error from a bug.

    A coarse integral should be inaccurate and a fine one accurate. If refining
    the integration does NOT help, the error is not coming from the integration.
    """
    import torch as T

    from src.models.attribution import completeness_error, integrated_gradients

    model, batch = toy_model(T), toy_batch(T)
    coarse = completeness_error(model, batch, integrated_gradients(model, batch, steps=2))
    fine = completeness_error(model, batch, integrated_gradients(model, batch, steps=128))
    assert fine < coarse


@needs_torch
def test_shape_matches_atoms_by_properties():
    import torch as T

    from src.models.attribution import integrated_gradients

    model, batch = toy_model(T), toy_batch(T)
    a = integrated_gradients(model, batch, steps=8)
    assert a.shape == (6, model.featuriser.table.shape[1])


@needs_torch
def test_a_property_the_model_cannot_use_gets_no_attribution():
    """Zeroing a column of the table makes it carry no information.

    A method that still assigns it importance is reporting something other than
    the model's use of the input.
    """
    import torch as T

    from src.models.attribution import integrated_gradients

    model, batch = toy_model(T), toy_batch(T)
    with T.no_grad():
        model.featuriser.table[:, 3] = 0.0

    a = integrated_gradients(model, batch, steps=32)
    assert np.allclose(a[:, 3], 0.0, atol=1e-9)


@needs_torch
def test_supplying_the_properties_reproduces_the_lookup():
    """The IG path must be the same computation as the ordinary forward pass.

    If passing the property rows in explicitly gave a different answer than
    letting the model look them up, the attribution would describe a model that
    is not the one being evaluated.
    """
    import torch as T

    model, batch = toy_model(T), toy_batch(T)
    x = model.featuriser.property_rows(batch["z"])
    with T.no_grad():
        looked_up = model(batch)
        passed_in = model({**batch, "atom_props": x})
    assert T.allclose(looked_up, passed_in, atol=1e-7)


@needs_torch
def test_learned_mode_ignores_supplied_properties():
    """'learned' has no property pathway, so there is nothing to attribute.

    Worth asserting: silently accepting atom_props and ignoring it would let
    someone run the Phase 6 analysis on a Phase 3 model and get a table of
    zeros that looks like a finding.
    """
    import torch as T

    model, batch = toy_model(T, mode="learned"), toy_batch(T)
    with T.no_grad():
        a = model(batch)
        b = model({**batch, "atom_props": T.randn(6, 31)})
    assert T.allclose(a, b)


def test_aggregate_sums_within_a_crystal_before_averaging():
    """Two identical crystals must not be counted differently from one."""
    from src.models.attribution import aggregate

    a = np.array([[1.0, 2.0], [3.0, -2.0], [1.0, 2.0], [3.0, -2.0]])
    one = aggregate(a[:2], np.array([0, 0]), 1)
    two = aggregate(a, np.array([0, 0, 1, 1]), 2)
    assert np.allclose(one["mean_abs"], two["mean_abs"])
    assert np.allclose(one["mean_signed"], two["mean_signed"])


def test_aggregate_lets_opposing_atoms_cancel():
    """Absolute value is taken per crystal, not per atom, on purpose."""
    from src.models.attribution import aggregate

    a = np.array([[1.0], [-1.0]])
    got = aggregate(a, np.array([0, 0]), 1)
    assert np.allclose(got["mean_abs"], 0.0)


def test_profile_similarity_is_a_cosine():
    from src.models.attribution import profile_similarity

    v = np.array([1.0, 2.0, 3.0])
    assert profile_similarity(v, v) == pytest.approx(1.0)
    assert profile_similarity(v, 5 * v) == pytest.approx(1.0)
    assert profile_similarity(np.array([1.0, 0.0]),
                              np.array([0.0, 1.0])) == pytest.approx(0.0)
