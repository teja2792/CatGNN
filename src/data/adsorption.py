"""Deciding which Catalysis-Hub rows are actually adsorption energies.

Catalysis-Hub's `reactionEnergy` column is not one quantity. It holds whatever
each row's equation says, which ranges from a single atom landing on a surface to
four species forming at once with a negative stoichiometric coefficient. Training
on it unfiltered gives a model a target whose physical meaning changes row to row.

The API can filter, which helps, and then stops helping in a way worth
documenting because it is invisible from the counts alone. Filters like
`products: "~COstar"` are CASE-INSENSITIVE SUBSTRING matches. Measured
consequences, all from real rows the probe returned:

    ~Hstar    matched  Rhstar          rhodium, not hydrogen
    ~Nstar    matched  Znstar          zinc, not nitrogen
    ~OHstar   matched  CH3CH2OHstar    ethanol, not hydroxyl
    ~OOHstar  matched  COOHstar        carboxyl, not hydroperoxyl
    ~Ostar    matched  HOstar, COstar, CH3Ostar, ...

Those counts looked plausible -- 13,364 for "H", 34,709 for "O" -- and were mostly
pollution. Anyone who scoped a dataset from them would have got a large, clean-
looking table of the wrong thing.

So the server filter is a PRE-FILTER whose only job is to reduce how much has to
be downloaded. Membership is decided here, exactly, on the parsed JSON:

    products  == {"<A>star": 1}
    reactants == {"star": 1, "<A>gas": 1}

Nothing else qualifies. A row with two products is a co-adsorption, a row with a
stoichiometric coefficient is a multi-species reaction, a row whose reactant is
another adsorbate is a surface step -- all real chemistry, none of it the same
quantity as a chemisorption energy.
"""

from __future__ import annotations

import json

# The standard intermediates of heterogeneous catalysis, fixed on chemistry
# before any counting. Choosing adsorbates after seeing which have the most rows
# would let data availability define the scientific question.
ADSORBATES = ["CO", "H", "O", "OH", "N", "C", "CH3", "NO", "S", "OOH"]


def _as_dict(blob) -> dict | None:
    if isinstance(blob, dict):
        return blob
    if not isinstance(blob, str):
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_single_adsorbate(reactants, products, adsorbate: str) -> bool:
    """True only for  A(g) + * -> A*  with unit coefficients.

    Deliberately strict. Every relaxation of this admits a different physical
    quantity into the same target column, which is the failure this whole module
    exists to prevent.
    """
    r, p = _as_dict(reactants), _as_dict(products)
    if r is None or p is None:
        return False

    gas, star = f"{adsorbate}gas", f"{adsorbate}star"

    # Exactly one product, the adsorbed species, with coefficient 1.
    if set(p) != {star} or abs(p.get(star, 0.0) - 1.0) > 1e-9:
        return False

    # Exactly the clean surface plus the gas-phase species, both coefficient 1.
    if set(r) != {"star", gas}:
        return False
    return (abs(r.get("star", 0.0) - 1.0) < 1e-9
            and abs(r.get(gas, 0.0) - 1.0) < 1e-9)


def which_adsorbate(reactants, products) -> str | None:
    """Which adsorbate, if this row is a clean single-adsorbate adsorption.

    Derived from the row rather than assumed from the query that fetched it. A
    row pulled by the `~Hgas` pre-filter may well be a rhodium deposition, and
    labelling it "H" because of how it was requested is exactly the error the
    pre-filter invites.
    """
    p = _as_dict(products)
    if p is None or len(p) != 1:
        return None
    (species,) = p
    if not species.endswith("star"):
        return None
    candidate = species[:-4]
    return candidate if is_single_adsorbate(reactants, products, candidate) else None


def is_metal_atom_adsorption(adsorbate: str) -> bool:
    """Is this a single metal atom depositing, rather than a molecule binding?

    Rh(g) + * -> Rh* passes every structural test above and is still a different
    process from CO(g) + * -> CO*: one is metal deposition, the other
    chemisorption of a molecule. They belong in the same table only if that is a
    decision someone made, so this flags them rather than silently mixing them.

    Chemistry, not a string test: an adsorbate is molecular if it contains any
    non-metal that catalysis cares about, otherwise it is an element being
    deposited.
    """
    from ..features.descriptors import element_table

    elements, _ = element_table()
    if adsorbate not in elements:
        return False                      # CO, OH, CH3 ... are not elements
    props = elements[adsorbate]
    return bool(props.get("is_metal") or props.get("is_transition_metal"))
