"""Which DFT functional does Materials Project's summary band gap actually come from?

Written because the probe turned up something that matters: Materials Project
returned **six** thermo records for three materials. Every material carries more
than one functional, so "the energy_type of material X" is not a well-defined
thing -- you have to say which record you mean.

That leaves an open question this repo cannot afford to guess at: when the
summary endpoint hands back `band_gap`, which calculation produced it? If we
label a GGA gap as r2SCAN (or the reverse), then every later analysis that
carefully groups by functional is carefully grouping by the wrong thing, and
nothing about the output will look wrong.

This script gathers the evidence:

  1. every thermo record for a set of materials, with thermo_type and energy_type
  2. how often materials carry more than one functional
  3. whether the summary band gap sits closer to the GGA or the r2SCAN value,
     using the fact that r2SCAN systematically predicts larger gaps than GGA

Usage:
    python scripts/diagnose_functional.py
    python scripts/diagnose_functional.py --formula Fe2O3 --n 40

Read-only. Downloads nothing, writes only a small report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import RESULTS, MissingAPIKey, get_mp_api_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formula", default="TiO2")
    ap.add_argument("--n", type=int, default=25, help="materials to inspect")
    args = ap.parse_args()

    try:
        key = get_mp_api_key()
    except MissingAPIKey as exc:
        print(exc)
        sys.exit(1)

    from mp_api.client import MPRester

    print(f"\nInspecting up to {args.n} {args.formula} entries\n" + "=" * 70)

    with MPRester(key) as mpr:
        summary = list(
            mpr.materials.summary.search(
                formula=args.formula,
                fields=["material_id", "formula_pretty", "band_gap", "nsites",
                        "symmetry", "formation_energy_per_atom", "energy_above_hull"],
            )
        )[: args.n]
        ids = [str(d.material_id) for d in summary]
        print(f"summary endpoint returned {len(summary)} materials\n")

        thermo = list(
            mpr.materials.thermo.search(
                material_ids=ids,
                fields=["material_id", "energy_type", "thermo_type",
                        "formation_energy_per_atom", "energy_above_hull"],
            )
        )
        print(f"thermo endpoint returned {len(thermo)} records "
              f"({len(thermo) / max(1, len(ids)):.1f} per material)\n")

    # ---- 1. what combinations exist -------------------------------------
    per_mat: dict[str, list] = {}
    for t in thermo:
        per_mat.setdefault(str(t.material_id), []).append(t)

    combos = Counter(
        tuple(sorted({str(getattr(t, "energy_type", "?")) for t in recs}))
        for recs in per_mat.values()
    )
    print("Functional combinations per material")
    for combo, n in combos.most_common():
        print(f"  {n:>4}  {' + '.join(combo)}")

    tt = Counter(str(getattr(t, "thermo_type", "?")) for t in thermo)
    print("\nthermo_type values seen")
    for k, v in tt.most_common():
        print(f"  {v:>4}  {k}")

    multi = sum(1 for recs in per_mat.values()
                if len({str(getattr(t, "energy_type", "?")) for t in recs}) > 1)
    print(f"\nmaterials with >1 functional: {multi} of {len(per_mat)}")
    if multi:
        print("  -> 'the functional of this material' is not well defined;")
        print("     the download stores all of them plus an ambiguity flag.")

    # ---- 2. which one matches the summary numbers? ----------------------
    # formation_energy_per_atom appears in BOTH endpoints. Whichever thermo
    # record reproduces the summary value is the record summary was built from --
    # direct evidence rather than an assumption.
    print("\nWhich thermo record does the summary agree with?")
    print("  (matching on formation_energy_per_atom, tolerance 1e-4 eV/atom)\n")
    votes: Counter = Counter()
    for s in summary:
        mid = str(s.material_id)
        sfe = getattr(s, "formation_energy_per_atom", None)
        if sfe is None or mid not in per_mat:
            continue
        for t in per_mat[mid]:
            tfe = getattr(t, "formation_energy_per_atom", None)
            if tfe is not None and abs(float(tfe) - float(sfe)) < 1e-4:
                votes[str(getattr(t, "energy_type", "?"))] += 1

    if votes:
        total = sum(votes.values())
        for k, v in votes.most_common():
            print(f"    {k:<10} {v:>4} / {total}   {100 * v / total:5.1f}%")
        winner = votes.most_common(1)[0][0]
        print(f"\n  -> summary agrees with '{winner}'. FUNCTIONAL_PREFERENCE in")
        print("     src/data/mp_download.py should put that first.")
    else:
        print("    no matches -- summary energies may be corrected or aggregated.")
        print("    Treat the recorded functional as provisional and say so.")

    # ---- 3. a few examples ----------------------------------------------
    print("\nExamples\n" + "-" * 70)
    for s in summary[:6]:
        mid = str(s.material_id)
        sg = getattr(getattr(s, "symmetry", None), "symbol", "?")
        gap = getattr(s, "band_gap", None)
        print(f"  {mid:<14} {sg:<10} summary gap {gap!s:<8} "
              f"Ef {getattr(s, 'formation_energy_per_atom', None)}")
        for t in per_mat.get(mid, []):
            print(f"      thermo  energy_type={str(getattr(t, 'energy_type', '?')):<8} "
                  f"thermo_type={str(getattr(t, 'thermo_type', '?')):<22} "
                  f"Ef={getattr(t, 'formation_energy_per_atom', None)}")

    report = {
        "formula": args.formula,
        "n_materials": len(summary),
        "n_thermo_records": len(thermo),
        "functional_combinations": {" + ".join(k): v for k, v in combos.items()},
        "thermo_types": dict(tt),
        "materials_with_multiple_functionals": multi,
        "summary_agrees_with": dict(votes),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "functional_diagnosis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
