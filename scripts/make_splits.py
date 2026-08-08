"""Build the four train/validation/test splits and measure how much each leaks.

Run after `build_graphs.py`. No network, no API key.

    python scripts/make_splits.py

Also excludes structures whose graph is degenerate -- crystals so sparse that some
atom has no neighbour within the cutoff. Those are isolated clusters in a vacuum
box rather than periodic solids, they cannot receive messages, and there are few
enough to name rather than wave at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.config import CACHE, CUTOFF_ANGSTROM  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, build_all_splits  # noqa: E402


def collect_rows() -> tuple[list[dict], list[dict]]:
    """Metadata for every cached graph, split into usable and degenerate."""
    usable, degenerate = [], []
    for i in sorted(gb.existing_chunk_indices()):
        meta = json.loads((gb.GRAPHS / f"meta_{i:04d}.json").read_text(encoding="utf-8"))
        chunk = gb.load_graph_chunk(i)
        for k, g in enumerate(gb.iter_graphs(chunk)):
            n = g["z"].size
            isolated = int((np.bincount(g["src"], minlength=n) == 0).sum())
            row = dict(meta[k])
            row["n_isolated_atoms"] = isolated
            row.setdefault("elements", (row.get("chemsys") or "").split("-"))
            (degenerate if isolated else usable).append(row)
    return usable, degenerate


def main() -> None:
    if not gb.existing_chunk_indices():
        print("No graph cache found.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    print("\nReading cached graphs...")
    usable, degenerate = collect_rows()
    total = len(usable) + len(degenerate)

    print(f"  {total:,} graphs cached")
    print(f"  {len(degenerate):,} excluded: an atom with no neighbour within "
          f"{CUTOFF_ANGSTROM} A")
    if degenerate:
        vpa = [r["volume"] / max(1, r["nsites"]) for r in degenerate if r.get("volume")]
        keep_vpa = [r["volume"] / max(1, r["nsites"]) for r in usable[:5000] if r.get("volume")]
        print(f"    median volume per atom {np.median(vpa):.0f} A^3, against "
              f"{np.median(keep_vpa):.1f} for the rest")
        print("    -> isolated clusters in a vacuum box, not periodic crystals.")
        print("    -> an atom with no neighbours receives no messages and is invisible")
        print("       to the model, so these are excluded rather than silently kept.")
        print(f"    examples: {', '.join(r['material_id'] for r in degenerate[:5])}")
    print(f"  {len(usable):,} usable\n")

    out = build_all_splits(usable)

    print("=" * 78)
    print(f"  {'scheme':<10} {'train':>8} {'val':>7} {'test':>7}   "
          f"{'test formulas':>14} {'test chemsys':>13} {'all elements':>13}")
    print(f"  {'':<10} {'':>8} {'':>7} {'':>7}   {'seen in train':>14} "
          f"{'seen in train':>13} {'seen in train':>13}")
    print("-" * 78)
    for s in SCHEMES:
        z, lk = out[s]["sizes"], out[s]["leakage"]
        print(f"  {s:<10} {z['train']:>8,} {z['val']:>7,} {z['test']:>7,}   "
              f"{lk['test_with_formula_seen_pct']:>13.1f}% "
              f"{lk['test_with_chemsys_seen_pct']:>12.1f}% "
              f"{lk['test_with_all_elements_seen_pct']:>12.1f}%")
    print("=" * 78)

    rand = out["random"]["leakage"]["test_with_formula_seen_pct"]
    form = out["formula"]["leakage"]["test_with_formula_seen_pct"]
    print(f"\n  A random split lets {rand:.1f}% of test materials share a formula with")
    print(f"  something in training; the formula split lets {form:.1f}%. Every result in")
    print("  this repo is reported against all four, so the inflation is visible")
    print("  rather than assumed away.")

    if "held_out_elements" in out["element"]:
        el = out["element"]["held_out_elements"]
        print(f"\n  Elements held out entirely ({len(el)}): {', '.join(el)}")

    print(f"\nwrote {(CACHE / 'splits').relative_to(REPO)}/  "
          f"({', '.join(s + '.json' for s in SCHEMES)}, summary.json)")


if __name__ == "__main__":
    main()
