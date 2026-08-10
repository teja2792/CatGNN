"""Look at the adsorption data before modelling it. Costs no API requests.

Phase 1's equivalent for Materials Project caught two things that would otherwise
have been invisible: a "random sample" that was 100% formulas beginning with A,
and a DFT functional assigned by network response order. Both looked fine in
aggregate and were wrong in a way no training run would have complained about.

This asks the same kinds of question of the catalysis data, plus the ones that
are specific to it:

  WHAT IS THE TARGET'S RANGE, and are there values that cannot be binding
  energies? An adsorption energy of +40 eV is not a weak bond, it is a broken
  record.

  HOW MANY DFT FUNCTIONALS are mixed together? Phase 1 found Materials Project
  silently mixing GGA and r2SCAN. Reaction energies from RPBE and BEEF-vdW are
  not interchangeable either, and the difference is comparable to the accuracy a
  model is trying to reach.

  WHO PUBLISHED IT? This is the one with no Materials Project analogue and it
  matters most. Rows come from papers, and one paper contributes a whole family
  of surfaces computed with one code, one functional and one set of conventions.
  A random split puts rows from the same paper on both sides, so the model can
  score well by recognising a calculation's fingerprint rather than its
  chemistry -- the catalysis version of the formula leakage that made a random
  split flatter every model in Phase 2.

  IS THE SAME SYSTEM PRESENT TWICE? Same surface, same facet, same adsorbate,
  different energy means either different sites or duplicated work, and both
  change what a test set means.

Run with:  python scripts/inspect_catalysis_hub.py
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

ROWS_FILE = REPO / "data" / "raw" / "catalysis_hub" / "adsorption.jsonl"
REPORT = REPO / "results" / "catalysis_hub_report.json"

# Chemisorption energies live in roughly -10..+5 eV. Outside that a row is more
# likely a data-entry artefact or a reaction that slipped the filter than a real
# binding energy, and it should be looked at rather than trained on.
PLAUSIBLE_LO, PLAUSIBLE_HI = -12.0, 6.0


def load() -> list[dict]:
    if not ROWS_FILE.exists():
        print(f"{ROWS_FILE} missing. Run:\n"
              "    python scripts/fetch_catalysis_hub.py --adsorbates CO\n")
        sys.exit(1)
    rows = []
    with ROWS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def bar(n: int, total: int, width: int = 28) -> str:
    filled = int(round(width * n / max(total, 1)))
    return "█" * filled + "·" * (width - filled)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    rows = load()
    n = len(rows)
    print(f"\n{'=' * 78}\n  {n:,} single-adsorbate rows from Catalysis-Hub"
          f"\n{'=' * 78}")

    # -- the target ------------------------------------------------------
    section("Reaction energy — is it plausibly a binding energy?")
    e = np.array([r["reactionEnergy"] for r in rows
                  if isinstance(r.get("reactionEnergy"), (int, float))])
    print(f"  {len(e):,} rows carry a number "
          f"({n - len(e)} missing)")
    if len(e):
        print(f"  mean {e.mean():+.3f}   median {np.median(e):+.3f}   "
              f"sd {e.std():.3f}")
        print(f"  range {e.min():+.3f} to {e.max():+.3f} eV")
        for q in (1, 5, 25, 50, 75, 95, 99):
            print(f"    p{q:<3}{np.percentile(e, q):+8.3f}")
        odd = [r for r in rows
               if isinstance(r.get("reactionEnergy"), (int, float))
               and not (PLAUSIBLE_LO <= r["reactionEnergy"] <= PLAUSIBLE_HI)]
        print(f"\n  outside {PLAUSIBLE_LO:+g}..{PLAUSIBLE_HI:+g} eV: "
              f"{len(odd)} ({100 * len(odd) / max(len(e), 1):.2f}%)")
        for r in odd[:5]:
            print(f"    {r['reactionEnergy']:+9.3f}  {r.get('Equation', '')[:40]:<42}"
                  f"{r.get('surfaceComposition', '')}")

    # -- what kind of adsorption -----------------------------------------
    section("Adsorbate, and metal deposition versus molecular chemisorption")
    ads = Counter(r["adsorbate"] for r in rows)
    metal = sum(1 for r in rows if r.get("is_metal_atom_adsorption"))
    for a, c in ads.most_common(12):
        print(f"  {a:<8}{c:>7}  {bar(c, n)}")
    if len(ads) > 12:
        print(f"  ... and {len(ads) - 12} more")
    print(f"\n  metal-atom deposition: {metal:,} ({100 * metal / max(n, 1):.1f}%)")
    print("  Physically a different process from molecular chemisorption.")
    print("  Kept and flagged, so combining them stays a choice rather than an")
    print("  accident of the filter.")

    # -- the pre-filter's contamination, measured ------------------------
    section("What the server's substring filter actually returned")
    mismatch = Counter((r.get("fetched_under"), r["adsorbate"]) for r in rows
                       if r.get("fetched_under") != r["adsorbate"])
    if mismatch:
        print("  rows KEPT but labelled differently from the query that fetched them:")
        for (asked, got), c in mismatch.most_common(10):
            print(f"    asked {asked:<6} got {got:<8}{c:>6}")
        print("\n  Every one of these would have been mislabelled by trusting")
        print("  the query instead of parsing the row.")
    else:
        print("  none — every kept row matched the adsorbate it was fetched under")

    # -- the methodology mix ---------------------------------------------
    section("DFT functionals and codes — are these numbers comparable?")
    for field in ("dftFunctional", "dftCode"):
        c = Counter(r.get(field) or "unrecorded" for r in rows)
        print(f"\n  {field}: {len(c)} distinct")
        for v, k in c.most_common(8):
            print(f"    {str(v)[:34]:<36}{k:>7}  {bar(k, n, 20)}")
    if len(Counter(r.get("dftFunctional") for r in rows)) > 1:
        print("\n  More than one functional. RPBE and BEEF-vdW adsorption energies")
        print("  differ by amounts comparable to the accuracy a model aims for, so")
        print("  this belongs in the split design and in LIMITATIONS, not as a")
        print("  footnote.")

    # -- leakage ----------------------------------------------------------
    section("Publication concentration — the leakage risk with no MP analogue")
    pubs = Counter(r.get("pubId") or "unknown" for r in rows)
    print(f"  {len(pubs)} publications")
    for pub, c in pubs.most_common(8):
        print(f"    {str(pub)[:38]:<40}{c:>7}  {100 * c / n:>5.1f}%  {bar(c, n, 16)}")
    top = pubs.most_common(1)[0][1] if pubs else 0
    print(f"\n  largest single publication: {100 * top / max(n, 1):.1f}% of all rows")
    print("  One paper means one code, one functional and one set of surface")
    print("  conventions. A random split puts its rows on both sides and lets a")
    print("  model score by recognising the calculation rather than the chemistry.")
    print("  This is the catalysis analogue of the 42.6% formula leakage in Phase 2,")
    print("  and it argues for a publication-disjoint split.")

    # -- duplicates -------------------------------------------------------
    section("Is the same system present more than once?")
    key = Counter((r.get("surfaceComposition"), r.get("facet"), r["adsorbate"])
                  for r in rows)
    dupes = {k: v for k, v in key.items() if v > 1}
    print(f"  {len(key):,} distinct (surface, facet, adsorbate) combinations")
    print(f"  {len(dupes):,} appear more than once, covering "
          f"{sum(dupes.values()):,} rows "
          f"({100 * sum(dupes.values()) / max(n, 1):.1f}%)")
    if dupes:
        worst = sorted(dupes.items(), key=lambda kv: -kv[1])[:5]
        print("\n  most repeated:")
        for (surf, facet, a), c in worst:
            same = [r["reactionEnergy"] for r in rows
                    if (r.get("surfaceComposition"), r.get("facet"),
                        r["adsorbate"]) == (surf, facet, a)
                    and isinstance(r.get("reactionEnergy"), (int, float))]
            spread = (max(same) - min(same)) if len(same) > 1 else 0.0
            print(f"    {str(surf)[:16]:<18}{str(facet):<6}{a:<6}{c:>4} rows, "
                  f"energies span {spread:.2f} eV")
        print("\n  A spread here is not necessarily an error -- different binding")
        print("  sites on the same facet genuinely differ. But it does mean the")
        print("  surface alone does not determine the label, so a model given only")
        print("  the surface has an irreducible error floor.")

    # -- the ceiling ------------------------------------------------------
    section("What could a model possibly achieve knowing only these fields?")
    ok = [r for r in rows if isinstance(r.get("reactionEnergy"), (int, float))
          and PLAUSIBLE_LO <= r["reactionEnergy"] <= PLAUSIBLE_HI]
    y = np.array([r["reactionEnergy"] for r in ok])
    total_ss = float(((y - y.mean()) ** 2).sum())
    print(f"  {len(ok):,} usable rows, spread {y.std():.3f} eV\n")
    print(f"  {'known to the model':<40}{'groups':>8}{'floor':>9}{'R2 ceiling':>12}")
    print("  " + "-" * 70)

    def ceiling(keyfn, label):
        groups = {}
        for r, v in zip(ok, y):
            groups.setdefault(keyfn(r), []).append(v)
        ss = sum(float(((np.array(v) - np.array(v).mean()) ** 2).sum())
                 for v in groups.values())
        floor = float(np.sqrt(ss / len(ok)))
        r2 = 1.0 - ss / total_ss if total_ss else float("nan")
        # A grouping with nearly one member per row explains everything by
        # construction and means nothing. Say so rather than printing 0.999.
        degenerate = len(groups) > 0.8 * len(ok)
        note = "  <- one group per row; meaningless" if degenerate else ""
        print(f"  {label:<40}{len(groups):>8}{floor:>8.3f} eV{r2:>11.3f}{note}")
        return {"groups": len(groups), "floor_ev": floor, "r2_ceiling": r2,
                "degenerate": degenerate}

    ceilings = {
        "surface": ceiling(lambda r: r.get("surfaceComposition"),
                           "surface composition"),
        "surface_facet": ceiling(
            lambda r: (r.get("surfaceComposition"), r.get("facet")),
            "surface composition + facet"),
        "slab_facet": ceiling(
            lambda r: (r.get("chemicalComposition"), r.get("facet")),
            "full slab formula + facet"),
        "site_index": ceiling(
            lambda r: (r.get("surfaceComposition"), r.get("facet"), r.get("sites")),
            "... + the site index"),
        "publication": ceiling(lambda r: r.get("pubId"), "which paper it came from"),
        "functional": ceiling(lambda r: r.get("dftFunctional"),
                              "which DFT functional"),
    }

    sf = ceilings["surface_facet"]
    print("\n  THE NUMBER THAT DECIDES PHASE 7: knowing the surface and the facet")
    print(f"  and nothing else, the best achievable error is {sf['floor_ev']:.2f} eV")
    print(f"  against a spread of {y.std():.2f} eV — a ceiling of R2 = "
          f"{sf['r2_ceiling']:.2f}.")
    print("\n  The missing 40-odd percent is WHERE on the surface the molecule sits.")
    print("  The `sites` column names it only as an opaque index (site1 ... site47),")
    print("  which a model cannot use and which does not transfer between surfaces.")
    print("  The one honest description of a binding site is its geometry.")
    print("\n  So unlike band gap — where structure bought 2-3% — here composition")
    print("  cannot get there at all. Structures are not an enhancement for this")
    print("  target; they are the target's dominant variable.")

    # -- structures -------------------------------------------------------
    section("Do these rows carry the information a graph needs?")
    with_sites = sum(1 for r in rows if r.get("sites"))
    print(f"  sites recorded:     {with_sites:,} ({100 * with_sites / max(n, 1):.1f}%)")
    print("  Structures were NOT fetched in this pass — 5.3 kB each, and they")
    print("  reduce rows per request. Whether a graph model is possible at all")
    print("  depends on that second pass; composition baselines do not need it.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "rows": n,
        "adsorbates": dict(ads),
        "metal_atom_adsorptions": metal,
        "energy": ({"mean": float(e.mean()), "median": float(np.median(e)),
                    "sd": float(e.std()), "min": float(e.min()),
                    "max": float(e.max())} if len(e) else {}),
        "functionals": dict(Counter(r.get("dftFunctional") for r in rows)),
        "publications": len(pubs),
        "largest_publication_share": top / max(n, 1),
        "distinct_surface_facet_adsorbate": len(key),
        "repeated_combinations": len(dupes),
        "mislabelled_by_query": {f"{a}->{g}": c for (a, g), c in mismatch.items()},
        "ceilings": ceilings,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
