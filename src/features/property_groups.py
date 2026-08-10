"""Grouping and naming the 31 element properties, for Phase 6.

Kept apart from the attribution code, and free of PyTorch, for two reasons. It is
the piece most likely to fall out of step with the element table -- add a property
to the reference table and this file must gain a row, or a bar on figure 14 ends
up labelled "other" -- and a test that catches that should run on any machine.

The family assignment is by what each quantity IS, and was fixed before any
attribution was computed. Deciding the groupings after seeing which properties
scored highly would let the grouping be chosen to make the story tidier.
"""

from __future__ import annotations


# Families, so 31 bars become a chemical story rather than a list. Assignment is
# by what the quantity IS, decided before any attribution was computed.
PROPERTY_FAMILY = {
    "X": "electronic", "electron_affinity": "electronic",
    "n_valence": "electronic", "n_s": "electronic", "n_p": "electronic",
    "n_d": "electronic", "n_f": "electronic",
    "max_oxidation_state": "electronic", "min_oxidation_state": "electronic",

    "atomic_radius": "size", "atomic_radius_calculated": "size",
    "average_ionic_radius": "size", "molar_volume": "size",
    "atomic_mass": "size",

    "melting_point": "thermal", "boiling_point": "thermal",
    "thermal_conductivity": "thermal",

    "Z": "position", "row": "position", "group": "position",
    "mendeleev_no": "position",

    "is_metal": "block", "is_transition_metal": "block",
    "is_metalloid": "block", "is_alkali": "block", "is_alkaline": "block",
    "is_chalcogen": "block", "is_halogen": "block", "is_lanthanoid": "block",
    "is_actinoid": "block", "is_noble_gas": "block",
}

FAMILY_ORDER = ["electronic", "size", "position", "block", "thermal", "other"]

# Plain-language names. "X" is pymatgen's symbol for electronegativity and means
# nothing to a reader who has not used pymatgen.
PROPERTY_LABEL = {
    "X": "electronegativity",
    "n_valence": "valence electrons",
    "n_s": "s electrons", "n_p": "p electrons",
    "n_d": "d electrons", "n_f": "f electrons",
    "Z": "atomic number",
    "mendeleev_no": "Mendeleev number",
    "average_ionic_radius": "ionic radius",
    "atomic_radius_calculated": "atomic radius (calc.)",
    "max_oxidation_state": "max oxidation state",
    "min_oxidation_state": "min oxidation state",
    "electron_affinity": "electron affinity",
    "thermal_conductivity": "thermal conductivity",
    "molar_volume": "molar volume",
    "atomic_mass": "atomic mass",
    "is_transition_metal": "is a transition metal",
    "is_noble_gas": "is a noble gas",
    "is_alkaline": "is alkaline earth",
}


def family_of(name: str) -> str:
    return PROPERTY_FAMILY.get(name, "other")


def label_of(name: str) -> str:
    return PROPERTY_LABEL.get(name, name.replace("_", " "))
