"""Regenerate every figure in the README from source data.

No figure in this repository is hand-edited. If a number in a figure looks wrong,
this script is where to start: it rebuilds all of them from the data in
``data/reference/`` and the structures hard-coded in the figure modules.

Usage
-----
    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --only 1   # just figure 1
    python scripts/make_figures.py --list     # show what would run
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIGURES = {
    1: ("src.figures.fig_polymorph_problem", "Why composition alone is not enough"),
    2: ("src.figures.fig_crystal_to_graph", "How a crystal becomes a graph"),
    3: ("src.figures.fig_data_provenance", "Where every number comes from"),
    4: ("src.figures.fig_roadmap", "Build plan and current status"),
    5: ("src.figures.fig_split_leakage", "Why a random split is not a fair test"),
    6: ("src.figures.fig_baselines", "The bar every neural network has to clear"),
    7: ("src.figures.fig_cgcnn", "Did the graph network beat the descriptors?"),
    8: ("src.figures.fig_overfitting", "Why it fails on an unseen element"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=int, action="append", help="figure number(s) to build")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    args = ap.parse_args()

    if args.list:
        for n, (mod, desc) in FIGURES.items():
            print(f"  {n}. {desc:<45} {mod}")
        return

    import matplotlib
    matplotlib.use("Agg")

    wanted = args.only or sorted(FIGURES)
    failed = []
    t_all = time.perf_counter()

    for n in wanted:
        if n not in FIGURES:
            print(f"  ?  no figure {n}")
            continue
        mod_name, desc = FIGURES[n]
        print(f"\n[{n}] {desc}")
        t0 = time.perf_counter()
        try:
            importlib.import_module(mod_name).main()
            print(f"    ok  ({time.perf_counter() - t0:.1f} s)")
        except Exception as exc:  # noqa: BLE001 - we want the whole batch to finish
            failed.append((n, exc))
            print(f"    FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'-' * 60}")
    print(f"{len(wanted) - len(failed)}/{len(wanted)} figures built "
          f"in {time.perf_counter() - t_all:.1f} s")
    if failed:
        for n, exc in failed:
            print(f"  figure {n}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
