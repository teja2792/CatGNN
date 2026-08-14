"""Tests for the paired statistics that decide whether a result is real.

These are the functions that turn "the mean went up" into "the mean went up by
more than seed noise". A quiet error here does not crash anything; it publishes
a conclusion. Verified against known values and against scipy where available.

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

from src.models.seed_stats import (  # noqa: E402
    betainc, f_two_sided, paired_test, sign_test, t_two_sided, variance_ratio)

# The real round-2 differences, kept as a regression on the reported conclusion.
SITE = [+0.048, -0.023, +0.382, +0.075, +0.031, +0.543, -0.084, +0.286]
FEAT = [-0.010, +0.034, +0.631, +0.142, +0.031, +0.346, -0.022, +0.296]


def test_betainc_matches_closed_forms():
    assert betainc(1.0, 1.0, 0.37) == pytest.approx(0.37)
    assert betainc(2.0, 1.0, 0.5) == pytest.approx(0.25)
    assert betainc(3.0, 1.0, 0.5) == pytest.approx(0.125)
    assert betainc(1.0, 2.0, 0.5) == pytest.approx(0.75)
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0


def test_t_distribution_against_published_values():
    """Two-tailed p at the classic 5% critical values."""
    assert t_two_sided(2.262, 9) == pytest.approx(0.05, abs=5e-4)
    assert t_two_sided(2.145, 14) == pytest.approx(0.05, abs=5e-4)
    assert t_two_sided(1.960, 100000) == pytest.approx(0.05, abs=1e-3)
    assert t_two_sided(0.0, 7) == pytest.approx(1.0)


def test_t_is_symmetric_in_sign():
    assert t_two_sided(2.01, 7) == pytest.approx(t_two_sided(-2.01, 7))


def test_f_equal_variances_gives_p_one():
    assert f_two_sided(1.0, 7, 7) == pytest.approx(1.0)


def test_f_is_reciprocal_symmetric():
    """F and 1/F with swapped dof describe the same comparison."""
    assert f_two_sided(4.0, 7, 9) == pytest.approx(f_two_sided(0.25, 9, 7))


def test_sign_test_exact_small_cases():
    assert sign_test([1, 1, 1, 1])[2] == pytest.approx(2 * 1 / 16)
    assert sign_test([1, 1, 1, -1])[2] == pytest.approx(2 * 5 / 16)
    assert sign_test([1, -1])[2] == pytest.approx(1.0)


def test_sign_test_drops_ties_rather_than_counting_them_as_wins():
    wins, n, _ = sign_test([1.0, 0.0, 0.0, -1.0])
    assert (wins, n) == (1, 2)


def test_a_uniform_win_is_significant_but_a_split_one_is_not():
    assert sign_test([1] * 8)[2] < 0.01
    assert sign_test([1, 1, 1, 1, 1, 1, -1, -1])[2] > 0.2


# ---------------------------------------------------------------------------
# The conclusion this project actually reported
# ---------------------------------------------------------------------------

def test_the_site_readout_accuracy_gain_is_not_significant():
    """The finding. If this ever starts passing at p<0.05 the claim changes."""
    r = paired_test(SITE)
    assert r["mean"] == pytest.approx(0.157, abs=0.001)
    assert r["p_t"] > 0.05
    assert r["p_sign"] > 0.05
    assert r["wins"] == 6


def test_the_features_add_nothing():
    r = paired_test(np.array(FEAT) - np.array(SITE))
    assert abs(r["mean"]) < 0.05
    assert r["p_t"] > 0.5


def test_the_stability_difference_is_significant():
    """The result that IS established: the control's seed variance."""
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0.1, 0.188, 8)
    site = rng.normal(0.25, 0.051, 8)
    v = variance_ratio(ctrl, site)
    assert v["F"] > 1.0


def test_a_collapsed_run_moves_the_t_test_but_not_the_sign_test():
    """Why both are reported. One catastrophe should not create a result."""
    mild = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    with_outlier = mild[:-1] + [2.0]
    assert paired_test(with_outlier)["mean"] > 10 * abs(paired_test(mild)["mean"])
    assert paired_test(with_outlier)["p_sign"] > 0.05


def test_paired_test_survives_degenerate_input():
    assert np.isnan(paired_test([0.5])["t"])
    assert paired_test([0.0, 0.0, 0.0])["p_sign"] == 1.0
