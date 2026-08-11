"""What the geometries reveal that the metadata could not, including one failure.

The metadata download could only check that a row's *equation* was a single-
adsorbate adsorption. With the structures in hand the row can be checked against
its own physics, which is a different and stronger question. Costs no requests.

FOUR CHECKS THAT WORKED
-----------------------
1. *Is the reported energy the arithmetic it claims?*  E(slab+CO) - E(slab) -
   E(CO gas) reproduces `reactionEnergy` for 400 of 400 rows to 0.0000 eV. There
   is no hidden correction, no potential reference, no zero-point term. Worth
   knowing before modelling a quantity, and it was not documented anywhere.

2. *Did the CO stay a CO?*  A row labelled `CO(g) + * -> CO*` can still relax
   into dissociated C* + O*, which is a different reaction with a far more
   negative energy. Measured C-O distances span 1.14-1.33 A against 1.13 A in the
   gas phase: every molecule is intact. This was a hypothesis for the extreme
   values and the geometry refuted it.

3. *Is the CO on the surface, or in it?*  14 rows (3.5%) have the adsorbate below
   the topmost slab atom -- absorption rather than adsorption, a different
   process. They are flagged. None of them are the extreme-energy rows, so this
   did not explain those either.

4. *Are the energies physically credible?*  7 rows report CO binding below -5 eV,
   as far as -11.38 eV. All 7 come from ONE surface, which also contains +4.50 and
   +5.23 eV. Molecular CO chemisorption is roughly -3 to +1 eV on metals and can
   reach about -4 eV on very reactive carbides and nitrides; -11 eV is not a
   credible molecular adsorption energy.

WHAT THE CLEAN-SLAB REFERENCE DOES AND DOES NOT EXPLAIN
-------------------------------------------------------
The natural suspect for the implausible rows is the clean-slab reference: if
E(slab) is wrong, every adsorption energy on that surface shifts with it. Two
measurements, at 847 rows:

*It cannot explain the within-surface spread.* Every row in a (surface, facet)
group references the SAME clean-slab energy -- 0 of 85 groups use more than one.
So for `WN2-mp-754629-D|100`, whose ten rows run from -11.38 to +5.23 eV against
one fixed reference of -333.191 eV, the entire 16.6 eV spread lives in the
ADSORBED structures. Those relaxations ended somewhere very different from each
other, which is the reconstruction question below, still untested.

*It may explain the group's overall offset.* That same clean slab is -9.255
eV/atom while the other facet of the same material is -9.601. Over 36 atoms the
difference is 12.5 eV, which is the right size to move a normal -1 eV adsorption
energy to -11 eV.

But the pattern does not generalise: `PdN2-mp-1019239-B` has clean slabs within
0.038 eV/atom on its two facets and only one of them is flagged. So there is no
single cause, and the rows stay flagged rather than corrected.

ONE CHECK THAT DID NOT WORK, RECORDED BECAUSE IT LOOKED LIKE IT DID
------------------------------------------------------------------
The obvious explanation for the extreme values is surface reconstruction: if the
clean slab is metastable and rearranges when CO lands, the reported energy quietly
includes the rearrangement. Testing that by comparing slab atom positions between
the clean and adsorbed structures gave "21% of surfaces reconstruct, RMSD up to
5.59 A" -- a clean-looking result that is wrong.

It assumes atom *i* of the clean slab is atom *i* of the adsorbed slab. Nothing
guarantees that ordering, and a 5.59 A RMSD in a slab whose layers are 2 A apart
is an index mismatch, not a moving atom. The correlation between that RMSD and
binding energy was r = -0.013, i.e. none, which is what a shuffled index gives.

Testing it properly needs structure matching, not index subtraction. Until that
exists the reconstruction hypothesis is untested, not refuted -- so the cause of
the extreme values is still UNKNOWN, and they are handled by a stated physical
window rather than by an explanation.

Run:  python scripts/inspect_slab_geometries.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.slab_graph import adsorbate_mask, classify_systems  # noqa: E402

GEOM = REPO / "data" / "raw" / "catalysis_hub" / "geometries.jsonl"
OUT = REPO / "data" / "cache" / "slab_quality.json"

# Beyond this a "molecular CO adsorption" label is not credible. Set from
# chemistry, not from the data: CO chemisorption is about -3 to +1 eV on metals
# and reaches roughly -4 eV on reactive carbides/nitrides. Stated as a judgement
# because that is what it is -- the cause of these rows was never identified, so
# the model is run BOTH ways and the sensitivity reported.
CREDIBLE_MIN, CREDIBLE_MAX = -5.0, 3.0


def min_image(vec, cell):
    best = np.full(len(np.atleast_2d(vec)), np.inf)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            d = np.linalg.norm(np.atleast_2d(vec) + i * cell[0] + j * cell[1], axis=1)
            best = np.minimum(best, d)
    return best


def main() -> None:
    rows = [json.loads(x) for x in GEOM.open(encoding="utf-8") if x.strip()]
    print(f"\n{'=' * 76}\n  What the geometries say about the rows\n{'=' * 76}")
    print(f"\n  {len(rows)} reactions with structures\n")

    recs = []
    for r in rows:
        d = classify_systems(r["systems"])
        if not d or not {"adsorbed", "clean", "gas"} <= set(d):
            continue
        a, c = d["adsorbed"], d["clean"]
        m = adsorbate_mask(c, a)
        e = [d[k].get("energy") for k in ("adsorbed", "clean", "gas")]
        idx = np.flatnonzero(m)
        nums = a["numbers"][idx]
        ci, oi = idx[nums == 6], idx[nums == 8]
        co = (float(min_image(a["positions"][ci[0]] - a["positions"][oi[0]], a["cell"])[0])
              if len(ci) == 1 and len(oi) == 1 else np.nan)
        z = a["positions"][:, 2]
        recs.append({
            "id": r["id"], "y": r["reactionEnergy"],
            "surface": r.get("surfaceComposition"), "facet": r.get("facet"),
            "recomputed": (e[0] - e[1] - e[2]) if None not in e else None,
            "co_bond": co,
            "height": float(z[m].min() - z[~m].max()),
        })

    y = np.array([r["y"] for r in recs])

    # 1 -- the arithmetic
    have = [r for r in recs if r["recomputed"] is not None]
    dif = np.abs(np.array([r["recomputed"] - r["y"] for r in have]))
    print("1. Is reactionEnergy really E(slab+CO) - E(slab) - E(CO gas)?")
    print(f"     {(dif < 0.01).sum()}/{len(have)} agree to <0.01 eV; "
          f"max difference {dif.max():.4f} eV")
    print("     -> no hidden correction, reference, or zero-point term\n")

    # 2 -- the molecule
    co = np.array([r["co_bond"] for r in recs if np.isfinite(r["co_bond"])])
    print("2. Did the CO stay a CO, or dissociate into C* + O*?")
    print(f"     C-O distance {co.min():.2f}-{co.max():.2f} A "
          f"(gas phase 1.13 A); dissociated would be >1.5 A")
    print(f"     {(co > 1.5).sum()}/{len(co)} dissociated -> all intact\n")

    # 3 -- on the surface or in it
    h = np.array([r["height"] for r in recs])
    sub = h < 0
    print("3. Is the CO on the surface, or absorbed into it?")
    print(f"     height above the top slab atom: median {np.median(h):.2f} A")
    print(f"     {sub.sum()}/{len(h)} ({sub.mean():.1%}) sit BELOW it -- absorption,")
    print("     a different process; flagged, not silently mixed in")
    if sub.any():
        for s, n in Counter(f'{r["surface"]}|{r["facet"]}'
                            for r, b in zip(recs, sub) if b).most_common():
            print(f"        {n:>3}  {s}")
    print()

    # 4 -- credibility
    bad = (y < CREDIBLE_MIN) | (y > CREDIBLE_MAX)
    print("4. Are the energies credible for MOLECULAR CO adsorption?")
    print(f"     window {CREDIBLE_MIN} to {CREDIBLE_MAX} eV, set from chemistry")
    print(f"     {bad.sum()}/{len(y)} ({bad.mean():.1%}) fall outside it")
    if bad.any():
        by = defaultdict(list)
        for r, b in zip(recs, bad):
            if b:
                by[f'{r["surface"]}|{r["facet"]}'].append(r["y"])
        for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"        {len(v):>3}  {k:<30} "
                  f"{' '.join(f'{x:.1f}' for x in sorted(v))}")
        print("\n     The cause was NOT identified. The CO is intact and on the")
        print("     surface, and the arithmetic is exact, so the suspicion falls")
        print("     on the clean-slab reference -- untested, because testing it")
        print("     needs structure matching this script does not do.")
        print("     Therefore: flagged, and the model is run BOTH ways.")

    # what dropping them would do
    keep = ~bad
    print(f"\n  Effect of excluding them: {len(y)} -> {keep.sum()} rows, "
          f"spread {y.std():.3f} -> {y[keep].std():.3f} eV")
    print("  A large drop in spread means these rows would otherwise dominate")
    print("  the RMSE, and a model could look good purely by fitting them.")

    OUT.write_text(json.dumps({
        "rows": len(recs),
        "energy_is_plain_arithmetic": bool((dif < 0.01).all()),
        "co_bond_range_A": [round(float(co.min()), 3), round(float(co.max()), 3)],
        "dissociated": int((co > 1.5).sum()),
        "subsurface": int(sub.sum()),
        "credible_window_eV": [CREDIBLE_MIN, CREDIBLE_MAX],
        "outside_window": int(bad.sum()),
        "outside_window_ids": [r["id"] for r, b in zip(recs, bad) if b],
        "subsurface_ids": [r["id"] for r, b in zip(recs, sub) if b],
        "untested_hypothesis": "surface reconstruction; needs structure matching, "
                               "not index subtraction. See module docstring.",
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
