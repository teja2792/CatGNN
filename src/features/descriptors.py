"""Chemical descriptors: what a chemist already knows, written as numbers.

This is the bar every neural network in this repository has to clear. It exists
because "our GNN gets 0.3 eV" means nothing on its own -- the question is whether
it beats what you get from looking up electronegativities in a table, which takes
seconds rather than hours and needs no crystal structure at all.

Three blocks, deliberately separated so the comparison stays honest:

**composition** -- Magpie-style statistics over element properties. Knows the
formula and nothing else: it literally cannot distinguish rutile from anatase,
which is the point. This is the ceiling on what chemistry alone can do.

**structure_lite** -- cheap structural facts that need the crystal but not a
graph: density, volume per atom, space group, crystal system, packing fraction.
Cheap to compute and often surprisingly strong, so it sits between composition
and a GNN and stops "structure helps" from being a claim about message passing
when it is really a claim about density.

**both** -- the two concatenated.

Element properties come from ``data/reference/element_properties.json``, which is
committed rather than read from pymatgen at runtime, so a library upgrade cannot
silently change the features underneath a comparison.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache

import numpy as np

from ..config import REFERENCE

TABLE_PATH = REFERENCE / "element_properties.json"

# Statistics computed over each element property, weighted by how much of the
# formula each element accounts for. These are Magpie's, and they are chosen so
# that a property's spread matters as well as its average: an alloy of two very
# different metals is not the same as one metal with the average property.
STATS = ("mean", "avg_dev", "min", "max", "range", "mode")

CRYSTAL_SYSTEMS = (
    "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal",
    "Trigonal", "Hexagonal", "Cubic",
)


@lru_cache(maxsize=1)
def element_table() -> tuple[dict[str, dict], tuple[str, ...]]:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"{TABLE_PATH} missing. Regenerate with:\n"
            "    python scripts/make_element_table.py"
        )
    blob = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    return blob["elements"], tuple(blob["properties"])


@lru_cache(maxsize=1)
def property_matrix() -> tuple[dict[str, int], np.ndarray, tuple[str, ...]]:
    """Element properties as a dense array, with gaps filled by the column median.

    Some properties genuinely do not exist for some elements -- boiling point is
    missing for 10, atomic radius for 15. Filling with the median of the other
    elements is a modelling choice, not a fact, so it happens here in one visible
    place rather than being scattered through the feature code. The alternative,
    propagating NaN, would silently poison every statistic that touches it.
    """
    elements, props = element_table()
    symbols = sorted(elements)
    index = {s: i for i, s in enumerate(symbols)}

    mat = np.full((len(symbols), len(props)), np.nan, dtype=np.float64)
    for s, i in index.items():
        for j, p in enumerate(props):
            v = elements[s].get(p)
            if v is not None:
                mat[i, j] = float(v)

    medians = np.nanmedian(mat, axis=0)
    gaps = np.isnan(mat)
    mat[gaps] = np.take(medians, np.where(gaps)[1])
    return index, mat, props


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def composition_feature_names() -> list[str]:
    _, _, props = property_matrix()
    names = [f"comp_{p}_{s}" for p in props for s in STATS]
    names += ["comp_n_elements", "comp_L2_norm", "comp_L3_norm",
              "comp_L5_norm", "comp_L7_norm", "comp_max_fraction"]
    return names


def composition_features(species: list[str]) -> np.ndarray:
    """Magpie-style descriptors from the list of atoms in the cell.

    Note what is *not* used: positions. Two polymorphs of the same formula give
    byte-identical vectors here. That is the whole experiment -- this block is
    structurally blind on purpose, so the difference between it and a
    structure-aware model measures exactly what structure is worth.
    """
    index, mat, props = property_matrix()

    counts = Counter(s for s in species if s in index)
    if not counts:
        return np.zeros(len(composition_feature_names()), dtype=np.float32)

    rows = np.array([index[s] for s in counts])
    frac = np.array([counts[s] for s in counts], dtype=np.float64)
    frac /= frac.sum()

    vals = mat[rows]                                  # (n_distinct, n_props)
    mean = frac @ vals                                # weighted mean
    avg_dev = frac @ np.abs(vals - mean)              # weighted mean abs deviation
    vmin, vmax = vals.min(axis=0), vals.max(axis=0)
    mode = vals[int(np.argmax(frac))]                 # the most abundant element's value

    stats = np.stack([mean, avg_dev, vmin, vmax, vmax - vmin, mode], axis=1)  # (P, 6)

    # Stoichiometry attributes: Lp norms of the fraction vector describe how
    # evenly the formula is shared out. L2 separates AB from AB3 even when the
    # elements involved are identical.
    extra = np.array([
        len(counts),
        np.linalg.norm(frac, 2),
        np.sum(frac ** 3) ** (1 / 3),
        np.sum(frac ** 5) ** (1 / 5),
        np.sum(frac ** 7) ** (1 / 7),
        frac.max(),
    ])

    return np.concatenate([stats.ravel(), extra]).astype(np.float32)


# ---------------------------------------------------------------------------
# Cheap structure
# ---------------------------------------------------------------------------

def structure_feature_names() -> list[str]:
    return (
        ["struct_density", "struct_volume_per_atom", "struct_nsites",
         "struct_nelements", "struct_spacegroup_number", "struct_packing_fraction"]
        + [f"struct_system_{c}" for c in CRYSTAL_SYSTEMS]
    )


def structure_features(row: dict, species: list[str] | None = None) -> np.ndarray:
    """Structural facts that need the crystal but not a graph.

    Included so that "structure beats composition" cannot quietly mean "density
    beats composition". If these cheap numbers close most of the gap, then the
    message passing is not doing the work, and that is worth knowing before
    anyone builds four architectures on top of it.
    """
    index, mat, props = property_matrix()

    nsites = float(row.get("nsites") or 0)
    volume = float(row.get("volume") or 0)
    vpa = volume / nsites if nsites else 0.0

    # Packing fraction: how much of the cell the atoms would occupy as hard
    # spheres. A crude but genuinely structural quantity -- it separates a dense
    # close-packed phase from an open framework of the same composition.
    packing = 0.0
    if species and volume > 0 and "atomic_radius" in props:
        j = props.index("atomic_radius")
        radii = np.array([mat[index[s], j] for s in species if s in index])
        if radii.size:
            packing = float((4.0 / 3.0) * np.pi * np.sum(radii ** 3) / volume)

    system = str(row.get("crystal_system") or "")
    one_hot = [1.0 if system == c else 0.0 for c in CRYSTAL_SYSTEMS]

    return np.array(
        [
            float(row.get("density") or 0.0),
            vpa,
            nsites,
            float(row.get("nelements") or 0),
            float(row.get("spacegroup_number") or 0),
            packing,
        ] + one_hot,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

BLOCKS = ("composition", "structure_lite", "both")


def feature_names(block: str) -> list[str]:
    if block == "composition":
        return composition_feature_names()
    if block == "structure_lite":
        return structure_feature_names()
    if block == "both":
        return composition_feature_names() + structure_feature_names()
    raise ValueError(f"unknown block '{block}', expected one of {BLOCKS}")


def featurise(row: dict, species: list[str], block: str = "both") -> np.ndarray:
    if block == "composition":
        return composition_features(species)
    if block == "structure_lite":
        return structure_features(row, species)
    if block == "both":
        return np.concatenate([composition_features(species),
                               structure_features(row, species)])
    raise ValueError(f"unknown block '{block}', expected one of {BLOCKS}")
