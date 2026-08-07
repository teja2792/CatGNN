"""Report what was actually downloaded, and check it is usable.

Run after `fetch_mp.py`. This is the "look at your data before you model it" step
that gets skipped far too often. It answers, on the real download rather than on
assumptions:

  * how many materials, how many atoms, how big are the cells
  * how the DFT functionals are distributed -- and whether any chemical formula
    has entries under more than one, which is the trap that makes pooled band
    gaps meaningless
  * how much label coverage each target actually has
  * how much polymorphism there is, which is the whole premise of this repo
  * whether anything is malformed enough to break graph construction later

Prints to the terminal and writes a summary to results/mp_dataset_report.json.
Requires no network and no API key -- it only reads what is already on disk.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.config import RESULTS  # noqa: E402
from src.data.mp_download import DEST, read_chunks  # noqa: E402

TARGETS = ["band_gap", "formation_energy_per_atom", "energy_above_hull"]


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "  n/a"


def main() -> None:
    if not DEST.exists() or not any(DEST.glob("mp_chunk_*.jsonl.gz")):
        print(f"No download found in {DEST}.\n\n    python scripts/fetch_mp.py --probe\n")
        sys.exit(1)

    rows = list(read_chunks())
    n = len(rows)
    print(f"\n{'=' * 66}\n  {n:,} materials on disk\n{'=' * 66}\n")

    # --- cells -------------------------------------------------------------
    sites = np.array([r["nsites"] for r in rows])
    print("Cell size")
    print(f"  atoms per cell     min {sites.min()}  median {int(np.median(sites))}  "
          f"mean {sites.mean():.1f}  max {sites.max()}")
    print(f"  total atoms        {int(sites.sum()):,}")

    # --- functionals -------------------------------------------------------
    et = Counter(r.get("energy_type") or "MISSING" for r in rows)
    print("\nDFT functional  (band gaps from different functionals are NOT comparable)")
    for k, v in et.most_common():
        print(f"  {k:<22} {v:>8,}   {pct(v, n)}")

    # The trap: same formula, entries under different functionals. Pooling them
    # produces a plausible-looking number that means nothing.
    by_formula: dict[str, set] = {}
    for r in rows:
        by_formula.setdefault(r["formula_pretty"], set()).add(r.get("energy_type"))
    mixed = {f: s for f, s in by_formula.items() if len(s) > 1}
    print(f"\n  formulas spanning >1 functional: {len(mixed):,} "
          f"of {len(by_formula):,} ({pct(len(mixed), len(by_formula))})")
    if mixed:
        print("  -> every analysis MUST group by energy_type. Examples:")
        for f, s in list(mixed.items())[:5]:
            print(f"       {f:<14} {sorted(str(x) for x in s)}")

    # --- label coverage ----------------------------------------------------
    print("\nLabel coverage")
    coverage = {}
    for t in TARGETS:
        vals = [r.get(t) for r in rows]
        present = [v for v in vals if v is not None]
        coverage[t] = len(present)
        print(f"  {t:<28} {len(present):>8,}   {pct(len(present), n)}", end="")
        if present:
            a = np.array(present, dtype=float)
            print(f"   range {a.min():8.3f} .. {a.max():8.3f}   median {np.median(a):7.3f}")
        else:
            print()

    gaps = np.array([r["band_gap"] for r in rows if r.get("band_gap") is not None], dtype=float)
    if gaps.size:
        zero = int((gaps <= 1e-6).sum())
        print(f"\n  band_gap == 0 (metals): {zero:,} ({pct(zero, gaps.size)})")
        print("  -> a large metal fraction makes plain MAE a soft target. Report the "
              "non-metal subset too.")

    # --- polymorphism: the premise of this repo ---------------------------
    counts = Counter(r["formula_pretty"] for r in rows)
    multi = {f: c for f, c in counts.items() if c > 1}
    print("\nPolymorphism  (why structure should beat composition)")
    print(f"  unique formulas          {len(counts):,}")
    print(f"  formulas with >1 entry   {len(multi):,}   {pct(len(multi), len(counts))}")
    if multi:
        n_in_multi = sum(multi.values())
        print(f"  materials sharing a formula with another: {n_in_multi:,}  "
              f"{pct(n_in_multi, n)}")
        print("  most polymorphic:")
        for f, c in Counter(multi).most_common(8):
            entries = [r for r in rows if r["formula_pretty"] == f]
            g = [e["band_gap"] for e in entries if e.get("band_gap") is not None]
            span = f"gap {min(g):.2f}..{max(g):.2f} eV" if len(g) > 1 else ""
            print(f"    {f:<16} {c:>4} entries   {span}")

    # --- is this a sample of Materials Project, or a sample of the alphabet? --
    #
    # Added after a real incident. The first 2,000 documents Materials Project
    # returned were *every single one* an A formula (Ac, Ag, Al): 58% contained
    # aluminium, and the entire periodic table past Al was missing. Nothing in
    # the pipeline complained. Every statistic below it would have been wrong,
    # and a model trained on it would have been an aluminium model with a
    # general-purpose label.
    print("\nSample representativeness")
    first_letters = Counter(r["formula_pretty"][0] for r in rows if r.get("formula_pretty"))
    dominant_letter, dom_n = first_letters.most_common(1)[0]
    elements = Counter()
    for r in rows:
        elements.update(r.get("elements") or [])
    top_el, top_n = elements.most_common(1)[0]

    print(f"  distinct elements present   {len(elements)} of ~89 in Materials Project")
    print(f"  most common element         {top_el} in {top_n:,} materials ({pct(top_n, n)})")
    print(f"  formulas starting '{dominant_letter}'      {dom_n:,} ({pct(dom_n, n)})")

    skewed = []
    if len(first_letters) < 8:
        skewed.append(f"only {len(first_letters)} distinct first letters")
    if dom_n / n > 0.40:
        skewed.append(f"{pct(dom_n, n).strip()} of formulas start with '{dominant_letter}'")
    if top_n / n > 0.35:
        skewed.append(f"{top_el} appears in {pct(top_n, n).strip()} of materials")
    if len(elements) < 50:
        skewed.append(f"only {len(elements)} elements represented")

    if skewed:
        print("\n  *** THIS IS NOT A RANDOM SAMPLE ***")
        for s in skewed:
            print(f"    - {s}")
        print("  Materials Project returns ids in an order correlated with chemistry,")
        print("  so taking the first N gives a slice of the alphabet, not of the database.")
        print("  Delete data/raw/materials_project/ and re-download; --max-materials now")
        print("  draws a seeded random subset.")
    else:
        print("  -> looks broadly representative")

    # --- how confident are we in the functional labels? --------------------
    res = Counter(r.get("energy_type_resolution") or "not_recorded" for r in rows)
    if res and set(res) != {"not_recorded"}:
        print("\nHow the DFT functional was determined")
        for k, v in res.most_common():
            print(f"  {k:<28} {v:>8,}   {pct(v, n)}")
        weak = res.get("fallback_preference", 0) + res.get("not_found", 0)
        if weak:
            print(f"  -> {weak:,} materials have a guessed functional. Consider excluding")
            print("     them from any analysis that groups by functional.")

    amb = sum(1 for r in rows if r.get("energy_type_ambiguous"))
    print(f"\n  materials computed under >1 functional: {amb:,} ({pct(amb, n)})")

    # --- metals vs non-metals ---------------------------------------------
    if gaps.size:
        nm = gaps[gaps > 1e-6]
        print("\nBand gap, metals excluded")
        print(f"  non-metals {nm.size:,} ({pct(nm.size, gaps.size)})   "
              f"range {nm.min():.3f} .. {nm.max():.3f}   median {np.median(nm):.3f}")
        print(f"  predicting 0.0 for everything would score MAE {np.abs(gaps).mean():.3f} eV "
              f"on all materials,")
        print(f"  but {np.abs(nm - np.median(nm)).mean():.3f} eV on non-metals alone.")
        print("  -> report both, or a model that only learns 'is it a metal' will look good.")

    # --- integrity ---------------------------------------------------------
    print("\nIntegrity checks")
    problems = []
    for r in rows:
        s = r.get("structure") or {}
        nf, ns = len(s.get("frac_coords", [])), len(s.get("species", []))
        if nf == 0 or nf != ns:
            problems.append((r["material_id"], f"coords {nf} vs species {ns}"))
        elif nf != r["nsites"]:
            problems.append((r["material_id"], f"nsites {r['nsites']} vs coords {nf}"))
        elif np.abs(np.linalg.det(np.array(s["lattice"], dtype=float))) < 1e-6:
            problems.append((r["material_id"], "degenerate lattice"))
    ids = [r["material_id"] for r in rows]
    dupes = [m for m, c in Counter(ids).items() if c > 1]

    print(f"  malformed structures     {len(problems):,}")
    print(f"  duplicate material_ids   {len(dupes):,}")
    for mid, why in problems[:5]:
        print(f"    ! {mid}: {why}")
    if not problems and not dupes:
        print("  -> clean; ready for graph construction")

    # --- report ------------------------------------------------------------
    report = {
        "n_materials": n,
        "total_atoms": int(sites.sum()),
        "sites": {"min": int(sites.min()), "median": int(np.median(sites)),
                  "mean": float(sites.mean()), "max": int(sites.max())},
        "energy_type_counts": dict(et),
        "formulas_spanning_multiple_functionals": len(mixed),
        "label_coverage": coverage,
        "unique_formulas": len(counts),
        "formulas_with_multiple_entries": len(multi),
        "malformed_structures": len(problems),
        "duplicate_material_ids": len(dupes),
        "distinct_elements": len(elements),
        "most_common_element": [top_el, top_n],
        "sample_skew_warnings": skewed,
        "functional_resolution": dict(res),
        "materials_with_multiple_functionals": amb,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "mp_dataset_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
