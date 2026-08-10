"""Tests for deciding which Catalysis-Hub rows are adsorption energies.

Most of these cases are not invented. They are rows the API actually returned
when asked for a given adsorbate, and every one of them would have been silently
accepted into the training set by a filter that trusted the server's substring
match. They are kept as regression tests because the failure is invisible: the
resulting dataset is large, well-formed, and measuring the wrong thing.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.adsorption import (  # noqa: E402
    ADSORBATES, is_metal_atom_adsorption, is_single_adsorbate, which_adsorbate)


# ---------------------------------------------------------------------------
# The rows that should be kept
# ---------------------------------------------------------------------------

def test_a_clean_co_adsorption_is_kept():
    assert is_single_adsorbate('{"star": 1, "COgas": 1}', '{"COstar": 1}', "CO")


def test_a_clean_hydrogen_adsorption_is_kept():
    assert is_single_adsorbate('{"star": 1, "Hgas": 1}', '{"Hstar": 1}', "H")


def test_dicts_are_accepted_as_well_as_json_strings():
    assert is_single_adsorbate({"star": 1, "COgas": 1}, {"COstar": 1}, "CO")


# ---------------------------------------------------------------------------
# Real rows the substring pre-filter let through. Each is a different quantity.
# ---------------------------------------------------------------------------

def test_rhodium_deposition_is_not_a_hydrogen_adsorption():
    """~Hstar matched Rhstar. The API returned this when asked for H."""
    assert not is_single_adsorbate('{"star": 1, "Rhgas": 1}', '{"Rhstar": 1}', "H")


def test_zinc_deposition_is_not_a_nitrogen_adsorption():
    """~Nstar matched Znstar."""
    assert not is_single_adsorbate('{"star": 1, "Zngas": 1}', '{"Znstar": 1}', "N")


def test_ethanol_is_not_a_hydroxyl_adsorption():
    """~OHstar matched CH3CH2OHstar."""
    assert not is_single_adsorbate('{"star": 1, "CH3CH2OHgas": 1}',
                                   '{"CH3CH2OHstar": 1}', "OH")


def test_carboxyl_is_not_a_hydroperoxyl_adsorption():
    """~OOHstar matched COOHstar, and the row is a dissociation besides."""
    assert not is_single_adsorbate('{"HCOOHgas": 1}',
                                   '{"COOHstar": 1, "Hstar": 1}', "OOH")


def test_a_surface_step_is_not_an_adsorption():
    """CHO* -> hfH2(g) + CO*  produces CO* without adsorbing CO."""
    assert not is_single_adsorbate('{"CHOstar": 1}',
                                   '{"hfH2gas": 1, "COstar": 1}', "CO")


def test_a_multi_species_reaction_is_rejected():
    """3CH4 + H2O - 2H2 + * -> 3CH3* + HO*, including a negative coefficient."""
    assert not is_single_adsorbate(
        '{"star": 1, "CH4gas": 3.0, "H2Ogas": 1.0, "H2gas": -2.0}',
        '{"CH3star": 3.0, "HOstar": 1}', "O")


# ---------------------------------------------------------------------------
# The specific ways a row can fail to be one clean adsorption
# ---------------------------------------------------------------------------

def test_co_adsorption_of_two_molecules_is_rejected():
    """Coverage matters: 2CO* is not the same quantity as CO*."""
    assert not is_single_adsorbate('{"star": 1, "COgas": 2}', '{"COstar": 2}', "CO")


def test_co_adsorption_missing_the_clean_surface_is_rejected():
    assert not is_single_adsorbate('{"COgas": 1}', '{"COstar": 1}', "CO")


def test_two_adsorbates_on_the_surface_are_rejected():
    assert not is_single_adsorbate('{"star": 1, "COgas": 1, "Hgas": 1}',
                                   '{"COstar": 1, "Hstar": 1}', "CO")


def test_malformed_json_is_rejected_rather_than_raising():
    assert not is_single_adsorbate("{not json", '{"COstar": 1}', "CO")
    assert not is_single_adsorbate(None, None, "CO")
    assert not is_single_adsorbate('["a", "list"]', '{"COstar": 1}', "CO")


# ---------------------------------------------------------------------------
# Labelling by what the row IS, not by which query fetched it
# ---------------------------------------------------------------------------

def test_the_adsorbate_is_read_from_the_row():
    assert which_adsorbate('{"star": 1, "COgas": 1}', '{"COstar": 1}') == "CO"
    assert which_adsorbate('{"star": 1, "Rhgas": 1}', '{"Rhstar": 1}') == "Rh"


def test_a_row_that_is_not_an_adsorption_has_no_adsorbate():
    assert which_adsorbate('{"CHOstar": 1}', '{"hfH2gas": 1, "COstar": 1}') is None
    assert which_adsorbate('{"star": 1, "COgas": 1}',
                           '{"COstar": 1, "Hstar": 1}') is None


def test_labelling_does_not_trust_the_query_that_fetched_the_row():
    """The Rh row arrived from a query asking for H. It must not be labelled H."""
    r, p = '{"star": 1, "Rhgas": 1}', '{"Rhstar": 1}'
    assert which_adsorbate(r, p) != "H"


# ---------------------------------------------------------------------------
# Metal deposition versus molecular chemisorption
# ---------------------------------------------------------------------------

def test_metal_atom_adsorption_is_flagged():
    """Rh* passes every structural test and is still a different process."""
    assert is_metal_atom_adsorption("Rh")
    assert is_metal_atom_adsorption("Zn")
    assert is_metal_atom_adsorption("Au")


def test_molecular_adsorbates_are_not_flagged_as_metal_deposition():
    for ads in ("CO", "OH", "CH3", "NO", "OOH"):
        assert not is_metal_atom_adsorption(ads)


def test_nonmetal_atoms_are_not_metal_deposition():
    for ads in ("H", "O", "N", "C", "S"):
        assert not is_metal_atom_adsorption(ads)


def test_the_adsorbate_list_is_the_standard_intermediates():
    assert ADSORBATES[0] == "CO"
    assert {"CO", "H", "O", "OH", "N", "C", "CH3", "NO", "S", "OOH"} == set(ADSORBATES)
