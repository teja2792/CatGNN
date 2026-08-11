"""Turn crystals into graphs: periodic neighbour lists, cached to disk.

This is the step that decides what the model can possibly learn. Every later
result depends on these neighbour lists being right, and a wrong one raises no
error -- it just quietly trains on the wrong chemistry.

Two subtleties, both of which have bitten this project already:

**1. Count every periodic image, not just the nearest.**
The same pair of atoms can be neighbours through several different cell images
at once, and each is a separate physical contact. Rutile TiO2 is the clean
example: its unit cell holds only 4 oxygens, yet each Ti is octahedrally
coordinated by 6. Keeping one image per pair (the naive "minimum image
convention") returns Ti CN = 4 and silently destroys the octahedron.

**2. How many images you need depends on the cutoff AND the cell.**
With an 8 A cutoff, a 3 A cell needs images three deep in that direction; a 20 A
cell needs one. Hard-coding +/-1 is correct for a bonding-length cutoff and badly
wrong for CGCNN's 8 A, where it silently truncates the neighbour list of every
small cell -- which is most of the interesting ones. The required depth is
computed per axis from the perpendicular cell width.

What is stored, and what is not: edges carry a raw distance in angstrom.
Expanding those into Gaussian basis features happens in the model, not here.
Storing 41 floats per edge instead of one would turn a ~400 MB cache into ~6 GB
for no gain, since the expansion is cheap and its parameters are a modelling
choice we may want to change without rebuilding anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import CACHE, CUTOFF_ANGSTROM, MAX_NEIGHBOURS
from .mp_download import read_chunks

GRAPHS = CACHE / "graphs"

# Element symbol -> atomic number. Built once; the model embeds atomic number.
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
Z_OF = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def image_range(lattice: np.ndarray, cutoff: float, pbc=None) -> np.ndarray:
    """How many cell repeats are needed in each direction to cover ``cutoff``.

    The distance between opposite faces of the cell along axis *i* is not the
    length of vector *i* -- for a skewed cell it can be far smaller. It is the
    perpendicular width, volume / |a_j x a_k|. Using the vector length instead
    under-counts images for triclinic cells and silently truncates their
    neighbour lists.

    ``pbc`` marks which axes repeat. Bulk crystals repeat in all three and that
    is the default. A SLAB does not: it is periodic in the two surface directions
    and finite along the normal, with vacuum above it. Passing
    ``pbc=(True, True, False)`` there is not a refinement -- repeating a slab
    along its normal would stack it on top of its own vacuum and invent a second
    surface that does not exist.
    """
    a, b, c = lattice
    volume = abs(float(np.dot(a, np.cross(b, c))))
    if volume < 1e-8:
        raise ValueError("degenerate lattice (zero volume)")

    widths = np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(c, a)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])
    n = np.ceil(cutoff / widths).astype(int)
    if pbc is not None:
        n = np.where(np.asarray(pbc, dtype=bool), n, 0)
    return n


def _offsets(n: np.ndarray) -> np.ndarray:
    ranges = [np.arange(-k, k + 1) for k in n]
    grid = np.meshgrid(*ranges, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float64)


def neighbour_list(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    cutoff: float = CUTOFF_ANGSTROM,
    max_neighbours: int = MAX_NEIGHBOURS,
    pbc=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbours of every atom, honouring periodic boundaries.

    Returns ``(src, dst, dist)``: for each edge, the atom it points from, the
    atom it points to, and the separation in angstrom. Edges are directed and
    listed once per contact, so an atom bonded to the same neighbour through two
    different cell images gets two edges -- which is physically correct.

    Each atom keeps at most ``max_neighbours`` closest contacts, the CGCNN
    convention. This bounds the graph size for dense structures.

    ``pbc`` defaults to periodic in all three directions. Slabs must pass
    ``(True, True, False)``; see ``image_range``.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    frac = np.asarray(frac_coords, dtype=np.float64)
    n_atoms = len(frac)

    shifts = _offsets(image_range(lattice, cutoff, pbc)) @ lattice   # (S, 3) cartesian
    cart = frac @ lattice                                       # (N, 3)

    # (N_target, S, 3) -> every image of every atom, relative to each source atom
    images = cart[:, None, :] + shifts[None, :, :]              # (N, S, 3)
    flat = images.reshape(-1, 3)                                # (N*S, 3)
    owner = np.repeat(np.arange(n_atoms), len(shifts))          # which atom each image is

    src_list, dst_list, dist_list = [], [], []
    for i in range(n_atoms):
        d = np.linalg.norm(flat - cart[i], axis=1)
        # Exclude the atom's own image at zero displacement, but keep its OTHER
        # images: in a small cell an atom really is a neighbour of its own
        # periodic copy, and dropping that would under-count coordination.
        keep = (d <= cutoff) & (d > 1e-8)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue
        if idx.size > max_neighbours:
            idx = idx[np.argpartition(d[idx], max_neighbours)[:max_neighbours]]
        order = np.argsort(d[idx])
        idx = idx[order]

        src_list.append(np.full(idx.size, i, dtype=np.int32))
        dst_list.append(owner[idx].astype(np.int32))
        dist_list.append(d[idx].astype(np.float32))

    if not src_list:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    return (
        np.concatenate(src_list),
        np.concatenate(dst_list),
        np.concatenate(dist_list),
    )


def build_graph(structure: dict, cutoff: float = CUTOFF_ANGSTROM,
                max_neighbours: int = MAX_NEIGHBOURS) -> dict | None:
    """One compact structure dict -> one graph. None if it cannot be built."""
    try:
        lattice = np.asarray(structure["lattice"], dtype=np.float64)
        frac = np.asarray(structure["frac_coords"], dtype=np.float64)
        species = structure["species"]
        if len(frac) == 0 or len(frac) != len(species):
            return None

        z = np.array([Z_OF.get(s, 0) for s in species], dtype=np.int16)
        if (z == 0).any():
            return None  # unknown element symbol -- refuse rather than guess

        src, dst, dist = neighbour_list(lattice, frac, cutoff, max_neighbours)
        if src.size == 0:
            return None  # isolated atoms: nothing for message passing to do

        return {"z": z, "src": src, "dst": dst, "dist": dist}
    except (KeyError, ValueError, TypeError):
        return None


def gaussian_expand(dist: np.ndarray, dmin: float = 0.0, dmax: float = CUTOFF_ANGSTROM,
                    step: float = 0.2, sigma: float | None = None) -> np.ndarray:
    """Expand distances into a smooth basis (the CGCNN edge featurisation).

    A raw distance is one number, and a network struggles to learn a smooth,
    non-monotonic response to it. Projecting onto overlapping Gaussians centred
    every 0.2 A gives the model a soft "how far, roughly" signal it can weight
    freely per shell. Done at training time, not at cache time -- see the module
    docstring for why.
    """
    centres = np.arange(dmin, dmax + 1e-9, step)
    sigma = sigma if sigma is not None else step
    return np.exp(-((dist[:, None] - centres[None, :]) ** 2) / sigma**2).astype(np.float32)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

TARGETS = ("band_gap", "formation_energy_per_atom", "energy_above_hull")

META_COLUMNS = (
    "material_id", "formula_pretty", "nsites", "nelements", "spacegroup_number",
    "crystal_system", "density", "volume", "energy_type", "energy_type_ambiguous",
    "energy_type_resolution", "is_stable", "is_metal", "theoretical", "chemsys",
)


def cache_path(index: int) -> Path:
    return GRAPHS / f"graphs_{index:04d}.npz"


def write_graph_chunk(graphs: list[dict], meta: list[dict], index: int) -> Path:
    """Store a batch of graphs as flat arrays plus offsets.

    One .npz holding every graph concatenated, with ptr arrays marking where each
    starts. Loading is then a single decompress and a slice, rather than
    thousands of small file reads -- which is the difference between an epoch
    that is compute-bound and one that is disk-bound.
    """
    GRAPHS.mkdir(parents=True, exist_ok=True)

    node_ptr = np.zeros(len(graphs) + 1, dtype=np.int64)
    edge_ptr = np.zeros(len(graphs) + 1, dtype=np.int64)
    for i, g in enumerate(graphs):
        node_ptr[i + 1] = node_ptr[i] + g["z"].size
        edge_ptr[i + 1] = edge_ptr[i] + g["src"].size

    path = cache_path(index)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        z=np.concatenate([g["z"] for g in graphs]),
        src=np.concatenate([g["src"] for g in graphs]),
        dst=np.concatenate([g["dst"] for g in graphs]),
        dist=np.concatenate([g["dist"] for g in graphs]),
        node_ptr=node_ptr,
        edge_ptr=edge_ptr,
        **{t: np.array([m.get(t, np.nan) if m.get(t) is not None else np.nan
                        for m in meta], dtype=np.float32) for t in TARGETS},
        material_id=np.array([m["material_id"] for m in meta]),
    )
    tmp.replace(path)  # atomic

    (GRAPHS / f"meta_{index:04d}.json").write_text(
        json.dumps([{k: m.get(k) for k in META_COLUMNS} for m in meta]),
        encoding="utf-8",
    )
    return path


def load_graph_chunk(index: int) -> dict:
    with np.load(cache_path(index), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def iter_graphs(chunk: dict) -> Iterator[dict]:
    """Slice a loaded chunk back into individual graphs."""
    np_, ep = chunk["node_ptr"], chunk["edge_ptr"]
    for i in range(len(np_) - 1):
        yield {
            "material_id": str(chunk["material_id"][i]),
            "z": chunk["z"][np_[i]:np_[i + 1]],
            "src": chunk["src"][ep[i]:ep[i + 1]],
            "dst": chunk["dst"][ep[i]:ep[i + 1]],
            "dist": chunk["dist"][ep[i]:ep[i + 1]],
            **{t: float(chunk[t][i]) for t in TARGETS if t in chunk},
        }


def existing_chunk_indices() -> set[int]:
    return {int(p.stem.split("_")[1]) for p in GRAPHS.glob("graphs_*.npz")}


def build_all(limit: int | None = None, chunk_size: int = 2000,
              cutoff: float = CUTOFF_ANGSTROM,
              max_neighbours: int = MAX_NEIGHBOURS) -> dict:
    """Build graphs for every downloaded material, caching as we go."""
    GRAPHS.mkdir(parents=True, exist_ok=True)
    done = existing_chunk_indices()
    if done:
        print(f"  resuming: {len(done)} graph chunks already cached")

    t0 = time.perf_counter()
    graphs: list[dict] = []
    meta: list[dict] = []
    built = failed = seen = 0
    n_nodes = n_edges = 0

    # Graph chunk k covers source rows [k*chunk_size, (k+1)*chunk_size). Keying
    # the cache to input position rather than to output count is what makes a
    # resumed build produce byte-identical chunks: a row that fails to build
    # must not shift every later row into a different chunk.
    for row in read_chunks():
        index = seen // chunk_size
        seen += 1
        if limit and seen > limit:
            break
        if index in done:
            continue

        g = build_graph(row.get("structure") or {}, cutoff, max_neighbours)
        if g is None:
            failed += 1
            continue

        graphs.append(g)
        meta.append(row)
        n_nodes += g["z"].size
        n_edges += g["src"].size

        # Flush when we reach the end of this input block, not when the output
        # list happens to be full -- failures would otherwise misalign chunks.
        if seen % chunk_size == 0 and graphs:
            write_graph_chunk(graphs, meta, index)
            built += len(graphs)
            el = time.perf_counter() - t0
            print(f"    chunk {index:04d}  {built:,} graphs  "
                  f"{el / 60:.1f} min  {built / el if el else 0:.0f} structures/s")
            graphs, meta = [], []

    if graphs:
        write_graph_chunk(graphs, meta, (seen - 1) // chunk_size)
        built += len(graphs)

    elapsed = time.perf_counter() - t0
    summary = {
        "graphs_built": built,
        "failed": failed,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "mean_edges_per_atom": round(n_edges / n_nodes, 2) if n_nodes else 0,
        "cutoff_angstrom": cutoff,
        "max_neighbours": max_neighbours,
        "seconds": round(elapsed, 1),
        "structures_per_second": round(built / elapsed, 1) if elapsed else 0,
    }
    (GRAPHS / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
