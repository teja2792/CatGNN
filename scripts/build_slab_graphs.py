"""Build graphs from the downloaded slab geometries, and measure what they can beat.

Two jobs, deliberately in one script so the second cannot be skipped.

**Build.** Each reaction returns three systems -- the adsorbed slab, the clean
slab, and the gas molecule. The graph is built on the ADSORBED slab, because that
is the structure whose energy is being predicted. Periodicity comes from the
geometry rather than the stored flag, which is wrong for 584 of 794 slabs; see
src/data/slab_graph.py and LIMITATIONS 18.

**Measure the ceiling ON THIS SAMPLE.** The full CO table has a composition-only
floor of 0.806 eV. Quoting that as the number to beat would be a comparison
against a different dataset: these 400 rows are 40 surfaces x 10 sites, chosen to
be rich in the within-surface variation that composition cannot see, so they are
a HARDER set for a composition model than the full table is. Measured here, the
surface+facet floor is 1.189 eV, not 0.806, and surface+facet explains only 44%
of the variance rather than 57%.

Beating 1.189 is the honest claim. Beating 0.806 would be arithmetic performed on
two different populations, and it would flatter the result by about 0.4 eV.

Run:  python scripts/build_slab_graphs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.slab_graph import (  # noqa: E402
    adsorbate_mask, build_slab_graph, classify_systems, coordination, vacuum_gap)

GEOM = REPO / "data" / "raw" / "catalysis_hub" / "geometries.jsonl"
OUT = REPO / "data" / "cache" / "slab_graphs.npz"
META = REPO / "data" / "cache" / "slab_graphs_meta.json"


def ceiling(y: np.ndarray, groups: list) -> tuple[float, float]:
    """Best possible RMSE for a model that knows only the group, and its R2.

    A model given only the surface and facet can do no better than predict each
    group's mean. The residual is the within-group variance -- the part of the
    target that is WHERE the molecule sits rather than WHAT the surface is made
    of. That is the quantity geometry was bought to reach.

    Computed with the group means fitted on the same data they are evaluated on,
    which makes this an optimistic floor: a real composition model would do
    worse. Optimistic is the right direction for a number the graph model has to
    beat.
    """
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    resid = np.concatenate([y[np.array(v)] - y[np.array(v)].mean()
                            for v in idx.values()])
    rmse = float(np.sqrt((resid ** 2).mean()))
    total = float(((y - y.mean()) ** 2).sum())
    return rmse, (1.0 - float((resid ** 2).sum()) / total if total else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--max-neighbours", type=int, default=None)
    args = ap.parse_args()

    if not GEOM.exists():
        print(f"\n  {GEOM.relative_to(REPO)} not found. Fetch geometries first.\n")
        sys.exit(1)

    rows = [json.loads(line) for line in GEOM.open(encoding="utf-8") if line.strip()]
    print(f"\n{'=' * 76}\n  Slab graphs from Catalysis-Hub geometries\n{'=' * 76}")
    print(f"\n  {len(rows)} downloaded reactions")

    graphs, meta = [], []
    dropped = defaultdict(int)
    for r in rows:
        sysd = classify_systems(r["systems"])
        if not sysd or "adsorbed" not in sysd or "clean" not in sysd:
            dropped["systems unidentifiable"] += 1
            continue
        ads_atoms = sysd["adsorbed"]
        gap = vacuum_gap(ads_atoms)
        # A slab whose vacuum is thinner than the cutoff sees the bottom of its
        # own image: both surfaces are then corrupted and the row is not a
        # surface calculation at all.
        if gap < 8.0:
            dropped[f"vacuum < cutoff ({gap:.1f} A)"] += 1
            continue
        g = build_slab_graph(ads_atoms, args.cutoff, args.max_neighbours)
        if g is None:
            dropped["graph could not be built"] += 1
            continue
        mask = adsorbate_mask(sysd["clean"], ads_atoms)
        if not mask.any():
            dropped["no adsorbate atoms found"] += 1
            continue
        g["is_adsorbate"] = mask
        # Height above the lowest slab atom. Kept because "outermost layer" is a
        # statement about geometry, and the earlier version of the check below
        # used position in the array as a proxy for height -- which happens to
        # correlate here and would stop correlating without warning.
        z_coord = ads_atoms["positions"][:, 2]
        g["height"] = (z_coord - z_coord[~mask].min()).astype(np.float32)
        graphs.append(g)
        meta.append({
            "id": r["id"], "y": r["reactionEnergy"],
            "surface": r.get("surfaceComposition"), "facet": r.get("facet"),
            "sites": r.get("sites"), "pubId": r.get("pubId"),
            "n_atoms": int(g["n_atoms"]), "n_edges": int(g["src"].size),
            "vacuum": round(gap, 2),
            "n_adsorbate": int(mask.sum()),
        })

    print(f"  {len(graphs)} graphs built"
          + (f", {sum(dropped.values())} dropped" if dropped else ", none dropped"))
    for k, v in dropped.items():
        print(f"      {v:>4}  {k}")
    if not graphs:
        sys.exit(1)

    na = np.array([m["n_atoms"] for m in meta])
    ne = np.array([m["n_edges"] for m in meta])
    print(f"\n  atoms per slab : min {na.min()} median {int(np.median(na))} "
          f"max {na.max()}   ({(na > 30).mean():.0%} exceed the bulk MAX_SITES=30,")
    print("                    which is why slabs use their own cap -- dropping")
    print("                    them would keep only the small, easy facets)")
    print(f"  edges per slab : median {int(np.median(ne))}")

    # The physical check. Not a formality: it is the only evidence that
    # periodicity was reconstructed correctly after the stored flag proved wrong.
    surf_cn, bulk_cn = [], []
    for g, m in zip(graphs, meta):
        cn = coordination(g, m["n_atoms"])
        slab = ~g["is_adsorbate"]
        if slab.sum() < 4:
            continue
        h = g["height"][slab]
        # Top layer vs interior, by HEIGHT. The interior is everything below the
        # topmost layer and above the bottom one, so the lower surface -- which
        # is also under-coordinated -- does not contaminate the "bulk" number and
        # flatter the comparison.
        top = h >= h.max() - 1.0
        interior = (h < h.max() - 1.0) & (h > h.min() + 1.0)
        if not top.any() or not interior.any():
            continue
        surf_cn.append(cn[slab][top].mean())
        bulk_cn.append(cn[slab][interior].mean())
    ratio = np.mean(surf_cn) / np.mean(bulk_cn)
    print("\n  Physical check -- surface atoms must be under-coordinated")
    print(f"    top layer {np.mean(surf_cn):.2f} vs interior "
          f"{np.mean(bulk_cn):.2f} neighbours within 3 A   (n={len(surf_cn)})")
    print(f"    ratio {ratio:.2f}  {'PASS' if ratio < 1.0 else 'FAIL'}"
          "   under-coordination is why surfaces bind at all;")
    print("      a graph built with the wrong periodicity fails this")

    # ---- what the graph model has to beat, measured on THIS sample ----
    y = np.array([m["y"] for m in meta])
    groups = [f'{m["surface"]}|{m["facet"]}' for m in meta]
    r_sf, r2_sf = ceiling(y, groups)
    r_surf, _ = ceiling(y, [m["surface"] for m in meta])
    print(f"\n{'=' * 76}\n  What a graph model has to beat, ON THESE ROWS\n{'=' * 76}")
    print(f"\n  target spread (std)                       {y.std():.3f} eV")
    print(f"  predict the global mean                   {y.std():.3f} eV")
    print(f"  predict the surface mean                  {r_surf:.3f} eV")
    print(f"  predict the surface+facet mean            {r_sf:.3f} eV   <- the ceiling")
    print(f"    R2 of surface+facet alone               {r2_sf:.3f}")
    print(f"\n  {1 - r2_sf:.0%} of the variance here is WHICH SITE the CO sits on.")
    print("  No composition or formula model can reach it, however good. That is")
    print("  the part geometry was bought to explain.")
    print("\n  NOTE: the full 3,554-row CO table gives 0.806 eV. These 400 rows")
    print(f"  give {r_sf:.3f} eV because they were chosen to be site-rich, which makes")
    print("  them HARDER for a composition model. Beating 0.806 here would be a")
    print("  comparison between two different populations.")

    # ---- save ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    node_ptr = np.cumsum([0] + [g["z"].size for g in graphs])
    edge_ptr = np.cumsum([0] + [g["src"].size for g in graphs])
    np.savez_compressed(
        OUT,
        z=np.concatenate([g["z"] for g in graphs]).astype(np.int16),
        src=np.concatenate([g["src"] for g in graphs]).astype(np.int32),
        dst=np.concatenate([g["dst"] for g in graphs]).astype(np.int32),
        dist=np.concatenate([g["dist"] for g in graphs]).astype(np.float32),
        is_adsorbate=np.concatenate([g["is_adsorbate"] for g in graphs]),
        height=np.concatenate([g["height"] for g in graphs]).astype(np.float32),
        node_ptr=node_ptr.astype(np.int64), edge_ptr=edge_ptr.astype(np.int64),
        y=y.astype(np.float64),
    )
    META.write_text(json.dumps({
        "rows": len(meta),
        "cutoff": args.cutoff, "max_neighbours": args.max_neighbours,
        "ceiling_surface_facet_eV": round(r_sf, 4),
        "ceiling_r2": round(r2_sf, 4),
        "ceiling_note": "measured on THESE 400 rows, not the 3,554-row table "
                        "(0.806 eV). The sample is site-rich by design, which "
                        "makes it harder for a composition model.",
        "target_std_eV": round(float(y.std()), 4),
        "coordination_check": {
            "top_layer": round(float(np.mean(surf_cn)), 3),
            "interior": round(float(np.mean(bulk_cn)), 3),
            "ratio": round(float(ratio), 3),
            "passes": bool(ratio < 1.0),
        },
        "dropped": dict(dropped),
        "graphs": meta,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {META.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
