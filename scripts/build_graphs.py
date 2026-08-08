"""Turn the downloaded crystals into cached graphs.

Run after `fetch_mp.py`. Needs no network and no API key -- it only reads what is
already on disk.

    python scripts/build_graphs.py --limit 2000     # trial, ~10 seconds
    python scripts/build_graphs.py                  # everything
    python scripts/build_graphs.py --verify         # check the cache is sane

Graph construction happens once and is cached. Rebuilding neighbour lists inside
a training loop is the single most common way CPU training becomes unusable, and
at 12 edges per atom over 1.4 million atoms there is no reason to pay for it more
than once.

The build is resumable: chunks already cached are skipped, and chunk boundaries
are keyed to input position rather than output count, so a resumed build produces
the same chunks as an uninterrupted one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.config import CUTOFF_ANGSTROM, MAX_NEIGHBOURS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.mp_download import DEST  # noqa: E402


def verify() -> None:
    """Read the cache back and check it says what it should.

    Deliberately re-derives the answers from the stored arrays rather than
    trusting build_summary.json: a summary written by the same code that made the
    mistake is not evidence.
    """
    idx = sorted(gb.existing_chunk_indices())
    if not idx:
        print("No graph cache found. Run without --verify first.")
        sys.exit(1)

    print(f"\nVerifying {len(idx)} cached chunks\n" + "=" * 62)
    n_graphs = n_nodes = n_edges = 0
    bad_dist = empty = disconnected = 0
    dmax = 0.0
    z_seen: set[int] = set()

    for i in idx:
        chunk = gb.load_graph_chunk(i)
        for g in gb.iter_graphs(chunk):
            n_graphs += 1
            n_nodes += g["z"].size
            n_edges += g["src"].size
            z_seen.update(int(v) for v in np.unique(g["z"]))

            if g["src"].size == 0:
                empty += 1
                continue
            if g["dist"].size:
                dmax = max(dmax, float(g["dist"].max()))
                if g["dist"].min() <= 0 or g["dist"].max() > CUTOFF_ANGSTROM + 1e-4:
                    bad_dist += 1
            # Every atom should have at least one neighbour; an isolated atom
            # receives no messages and is invisible to the model.
            if len(np.unique(g["src"])) < g["z"].size:
                disconnected += 1

    print(f"  graphs                {n_graphs:,}")
    print(f"  atoms                 {n_nodes:,}")
    print(f"  edges                 {n_edges:,}   ({n_edges / max(1, n_nodes):.2f} per atom, "
          f"cap {MAX_NEIGHBOURS})")
    print(f"  distinct elements     {len(z_seen)}")
    print(f"  longest edge          {dmax:.3f} A   (cutoff {CUTOFF_ANGSTROM})")
    print(f"  graphs with no edges  {empty}")
    print(f"  graphs with an isolated atom  {disconnected}")
    print(f"  edges out of range    {bad_dist}")

    ok = bad_dist == 0 and empty == 0 and dmax <= CUTOFF_ANGSTROM + 1e-4
    print("\n  " + ("-> cache is consistent" if ok else "-> PROBLEMS FOUND, see above"))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=None, help="only process the first N materials")
    ap.add_argument("--cutoff", type=float, default=CUTOFF_ANGSTROM)
    ap.add_argument("--max-neighbours", type=int, default=MAX_NEIGHBOURS)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--verify", action="store_true", help="check an existing cache and exit")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    if not DEST.exists() or not any(DEST.glob("mp_chunk_*.jsonl.gz")):
        print(f"No download found in {DEST}.\n\n    python scripts/fetch_mp.py\n")
        sys.exit(1)

    print("\nBuilding crystal graphs")
    print(f"  cutoff          {args.cutoff} A")
    print(f"  max neighbours  {args.max_neighbours}")
    print(f"  limit           {args.limit or 'none'}\n")

    summary = gb.build_all(
        limit=args.limit,
        chunk_size=args.chunk_size,
        cutoff=args.cutoff,
        max_neighbours=args.max_neighbours,
    )

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("=" * 62)
    print("\nNext:  python scripts/build_graphs.py --verify")


if __name__ == "__main__":
    main()
