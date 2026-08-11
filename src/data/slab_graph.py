"""Turning Catalysis-Hub slab geometries into graphs.

Three things make this different from the bulk-crystal builder, and each one is a
place where trusting the file would have produced a plausible-looking dataset that
was quietly wrong.

**1. The `pbc` field in the downloaded files is not usable.**

Catalysis-Hub returns geometries as ASE database JSON, which carries a periodic-
boundary flag. Measured across all 1,191 systems in the sample:

    gas-phase CO molecules    397 of 397  marked periodic in all three directions
    slabs                     584 of 794  marked periodic in NONE

Both are wrong, and the second is dangerous. A lone CO molecule in a 15 A box is
not periodic in anything. A slab is periodic in its two surface directions -- that
is what makes it a surface rather than a cluster. 26 of the 36 surfaces in the
sample contain systems that disagree with each other about the flag, so it does
not even track the material; it records however each entry happened to be written.

Trusting it would have built 584 slabs with no in-plane periodicity at all. The
in-plane cell vectors here are 2.8 to 8.1 A against an 8 A cutoff, so every atom
would lose most of its real neighbours, and the atoms that lose the most are the
ones at the cell boundary. Coordination number is the single strongest structural
determinant of binding energy -- it is the basis of the generalised-coordination-
number model of Calle-Vallejo et al. (Science 350, 185, 2015). A dataset with
systematically wrong coordination would still train, still converge, and still
report an RMSE.

So periodicity is decided from the geometry, which was measured, rather than the
flag, which contradicts itself. The two populations separate with no overlap:

    slabs           >= 20 atoms, in-plane cell 2.8-8.1 A, 14-26 A vacuum along c
    gas molecules   <= 3 atoms,  in-plane cell 15 A,      13.9 A vacuum

**2. A slab is periodic in two directions, not three.**

Repeating a slab along its surface normal would stack it on its own vacuum gap and
invent a second surface. In this sample the vacuum is at least 14.2 A against an
8 A cutoff, so no edge would actually cross it and the numerical difference is nil
-- but the two facts are independent, and a later change to the cutoff or a
thinner slab would turn a harmless approximation into a wrong one silently.
``pbc=(True, True, False)`` is correct for the reason, not for the number.

**3. MAX_SITES = 30 is a bulk-cell cap and must not be applied here.**

It exists so that a handful of 100-atom bulk crystals do not dominate every epoch
on a laptop. A slab plus adsorbate is simply bigger: 20 to 114 atoms, median 34,
with 67% of them over 30. Applying the bulk cap would silently discard two thirds
of the data and keep the small cells -- which are the small surface unit cells,
that is, the high-symmetry close-packed facets. The survivors would be a biased
sample of easy surfaces, and the bias would be invisible in the results.
"""

from __future__ import annotations

import json

import numpy as np

# Above this many atoms a system is a slab; at or below it is a gas-phase
# molecule. The sample has no system between 3 and 20 atoms, so this threshold
# sits in an empty region rather than cutting through a distribution.
MOLECULE_MAX_ATOMS = 3

# Slabs are large. This cap is not MAX_SITES; see the module docstring.
SLAB_MAX_ATOMS = 200


def _ndarray(x):
    """ASE's JSON encodes arrays as {'__ndarray__': [shape, dtype, flat]}."""
    if isinstance(x, dict):
        if "array" in x:                      # ASE Cell objects nest one level
            x = x["array"]
        if "__ndarray__" in x:
            shape, _dtype, flat = x["__ndarray__"]
            return np.array(flat).reshape(shape)
    return np.array(x)


def parse_ase_json(blob) -> dict | None:
    """One `InputFile(format: "json")` payload -> numbers, positions, cell.

    Returns None rather than raising: one unreadable geometry is a row to skip,
    not a reason to lose a download that cost a day of request budget.
    """
    try:
        d = json.loads(blob) if isinstance(blob, str) else blob
        if not isinstance(d, dict):
            return None
        keys = [k for k in d if k.isdigit()]
        if not keys:
            return None
        entry = d[str(min(int(k) for k in keys))]
        numbers = _ndarray(entry["numbers"]).astype(np.int16)
        positions = _ndarray(entry["positions"]).astype(np.float64)
        cell = _ndarray(entry["cell"]).astype(np.float64)
        if numbers.ndim != 1 or positions.shape != (len(numbers), 3) \
                or cell.shape != (3, 3) or len(numbers) == 0:
            return None
        return {"numbers": numbers, "positions": positions, "cell": cell}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def is_molecule(atoms: dict) -> bool:
    """Gas-phase molecule, or slab? Decided on size, not on the stored flag.

    The stored flag says every CO molecule is periodic in three directions and
    most slabs are periodic in none. See the module docstring.
    """
    return len(atoms["numbers"]) <= MOLECULE_MAX_ATOMS


def pbc_for(atoms: dict) -> tuple[bool, bool, bool]:
    """Periodicity implied by what the system IS.

    A molecule floating in a box repeats in nothing. A slab repeats in its two
    surface directions and is finite along the normal.
    """
    return (False, False, False) if is_molecule(atoms) else (True, True, False)


def vacuum_gap(atoms: dict) -> float:
    """Empty space along the third cell vector, in angstrom.

    A slab is only a surface if the vacuum is wider than the cutoff. Below that
    the top of the slab sees the bottom of its own image and both surfaces are
    corrupted -- so this is checked rather than assumed.
    """
    c = float(np.linalg.norm(atoms["cell"][2]))
    z = atoms["positions"][:, 2]
    return c - float(z.max() - z.min())


def classify_systems(systems: list[dict]) -> dict:
    """Which of a reaction's three systems is which.

    Catalysis-Hub returns the whole thermodynamic cycle: the adsorbed slab, the
    clean slab, and the gas-phase molecule. Identified by composition rather than
    by position in the list, because list order is not a documented guarantee and
    silently mislabelling the clean slab as the adsorbed one would invert the
    quantity being learned.
    """
    parsed = []
    for s in systems:
        atoms = parse_ase_json(s.get("InputFile"))
        if atoms is not None:
            atoms["energy"] = s.get("energy")
            atoms["formula"] = s.get("Formula")
            parsed.append(atoms)

    gas = [a for a in parsed if is_molecule(a)]
    slabs = [a for a in parsed if not is_molecule(a)]
    if not slabs:
        return {}
    slabs.sort(key=lambda a: len(a["numbers"]))
    out = {"clean": slabs[0], "adsorbed": slabs[-1]}
    if gas:
        out["gas"] = gas[0]
    # Two slabs of equal size cannot be a clean/adsorbed pair: adsorption adds
    # atoms. Refuse rather than pick one arbitrarily.
    if len(slabs) >= 2 and len(slabs[0]["numbers"]) == len(slabs[-1]["numbers"]):
        return {}
    return out


def adsorbate_mask(clean: dict, adsorbed: dict) -> np.ndarray:
    """Which atoms of the adsorbed slab are the adsorbate.

    By composition difference, not by position: the adsorbate is not reliably
    last in the list, and the model is later asked which atoms it attended to, so
    a wrong mask would corrupt the interpretation rather than the prediction --
    the more expensive failure, because nothing would look broken.
    """
    from collections import Counter

    surplus = Counter(adsorbed["numbers"].tolist())
    surplus.subtract(Counter(clean["numbers"].tolist()))
    mask = np.zeros(len(adsorbed["numbers"]), dtype=bool)
    # Take the highest-z atoms of each surplus element: the adsorbate sits on top
    # of the surface, so among atoms of the same element it is the outermost.
    for z_num, count in surplus.items():
        if count <= 0:
            continue
        idx = np.flatnonzero(adsorbed["numbers"] == z_num)
        order = idx[np.argsort(-adsorbed["positions"][idx, 2])]
        mask[order[:count]] = True
    return mask


def build_slab_graph(atoms: dict, cutoff: float | None = None,
                     max_neighbours: int | None = None) -> dict | None:
    """One slab -> one graph, with in-plane periodicity honoured.

    Positions arrive cartesian; the neighbour list works in fractional
    coordinates, so they are converted here rather than in the caller, where the
    conversion could be forgotten for one code path and not another.
    """
    from ..config import CUTOFF_ANGSTROM, MAX_NEIGHBOURS
    from .graph_build import neighbour_list

    cutoff = CUTOFF_ANGSTROM if cutoff is None else cutoff
    max_neighbours = MAX_NEIGHBOURS if max_neighbours is None else max_neighbours

    numbers, cell = atoms["numbers"], atoms["cell"]
    if len(numbers) > SLAB_MAX_ATOMS:
        return None
    try:
        frac = np.linalg.solve(cell.T, atoms["positions"].T).T
    except np.linalg.LinAlgError:
        return None

    src, dst, dist = neighbour_list(cell, frac, cutoff, max_neighbours,
                                    pbc=pbc_for(atoms))
    if src.size == 0:
        return None
    return {"z": numbers.astype(np.int16), "src": src, "dst": dst, "dist": dist,
            "n_atoms": len(numbers)}


def coordination(graph: dict, n_atoms: int, r: float = 3.0) -> np.ndarray:
    """Neighbours within ``r`` angstrom of each atom.

    Not used by the model -- used to CHECK it. Coordination number is the
    strongest simple structural predictor of binding (Calle-Vallejo et al.,
    Science 350, 185, 2015), so if the graphs are built correctly, surface atoms
    must come out less coordinated than bulk ones. That is a physical statement
    the geometry can be tested against, which a reconstruction error would fail.
    """
    close = graph["dist"] <= r
    return np.bincount(graph["src"][close], minlength=n_atoms)
