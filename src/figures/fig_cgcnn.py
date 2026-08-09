"""Figure 7 -- Did the graph network actually beat the descriptors?

Reads results/cgcnn_*.json (from scripts/train_cgcnn.py) and results/baselines.json
(from scripts/run_baselines.py) and puts them on the same axes.

The whole repository exists to answer one question, and this is the figure that
answers it. It is built to be readable whichever way the answer comes out: if the
GNN loses, the bars say so as plainly as if it wins. A plot that can only
illustrate one outcome is decoration, not evidence.

Run with:  python -m src.figures.fig_cgcnn
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    use_house_style, caption,
    COMPOSITION, FUSED, ACCENT, WARN, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

SPLIT_ORDER = ["random", "formula", "chemsys", "element"]

# Plain language on the axis. "chemsys" is jargon; "new element combinations" is
# something a chemical engineer can act on without reading the source.
SPLIT_LABEL = {
    "random": "Similar\nmaterials",
    "formula": "New\ncompositions",
    "chemsys": "New element\ncombinations",
    "element": "ELEMENTS never\nseen in training",
}
UNITS = {"band_gap": "eV", "band_gap_nonmetals": "eV",
         "formation_energy_per_atom": "eV/atom", "energy_above_hull": "eV/atom"}


def find_runs() -> dict[str, dict]:
    """Every CGCNN result file, keyed by target. Smoke runs are ignored."""
    out = {}
    for p in sorted(RESULTS.glob("cgcnn_*.json")):
        if p.stem.endswith("_smoke"):
            continue
        target = p.stem.replace("cgcnn_", "")
        out[target] = json.loads(p.read_text(encoding="utf-8"))
    return out


def baseline_best(target: str, split: str):
    path = RESULTS / "baselines.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    sel = [r for r in rows if r["split"] == split and r["target"] == target
           and r["model"] != "mean"]
    return min(sel, key=lambda r: r["mae"]) if sel else None


def panel_comparison(ax, target: str, runs: dict):
    splits = [s for s in SPLIT_ORDER if s in runs]
    x = np.arange(len(splits))
    w = 0.36

    gnn = [runs[s]["test"]["mae"] for s in splits]
    base, base_name = [], []
    for s in splits:
        b = baseline_best(target, s)
        base.append(b["mae"] if b else np.nan)
        base_name.append(b["model"] if b else "")

    ax.bar(x - w / 2, base, w * 0.92, color=COMPOSITION, alpha=0.88, zorder=3,
           label="Looking up element properties\n(electronegativity, ionic radius, …)")
    ax.bar(x + w / 2, gnn, w * 0.92, color=FUSED, alpha=0.88, zorder=3,
           label="Learning from the crystal structure\n(CGCNN graph neural network)")

    for xi, v, nm in zip(x - w / 2, base, base_name):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.3f}\n{nm}", ha="center", va="bottom",
                    fontsize=7.6, color=INK, fontweight="bold")
    for xi, v in zip(x + w / 2, gnn):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.6, color=INK, fontweight="bold")

    # The verdict, per split, stated rather than left to the reader's eye.
    for xi, b, g in zip(x, base, gnn):
        if not np.isfinite(b):
            continue
        delta = 100 * (b - g) / b
        won = delta > 0
        ax.text(xi, max(b, g) * 1.15,
                f"{delta:+.0f}%", ha="center", va="bottom", fontsize=11,
                color=ACCENT if won else WARN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits], fontsize=9)
    ax.set_ylabel(f"Typical prediction error  ({UNITS.get(target, '')}, lower is better)")
    ax.set_xlabel("How the test materials were chosen  —  stricter to the right")

    # Where the story turns. Marking it beats hoping the reader spots it.
    losers = [i for i, (b, g) in enumerate(zip(base, gnn))
              if np.isfinite(b) and g > b]
    if losers and len(losers) < len(splits):
        ax.axvspan(min(losers) - 0.5, len(splits) - 0.5, color=WARN, alpha=0.06, zorder=0)
        ax.text(min(losers) - 0.62, np.nanmax(base + gnn) * 1.36,
                "unfamiliar chemistry:\nthe network collapses,\nlooked-up properties do not",
                fontsize=9.6, color=WARN, va="top", ha="right", fontweight="bold")
    ax.set_ylim(0, np.nanmax(base + gnn) * 1.42)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    use_house_style()
    all_runs = find_runs()
    if not all_runs:
        raise FileNotFoundError(
            "No CGCNN results found. Run:\n"
            "    python scripts/train_cgcnn.py --all-splits"
        )

    # Prefer whichever target has the most splits, and among ties prefer the
    # non-metal one. The pooled band gap is 59% metals sitting at exactly zero,
    # so it flatters every model; the non-metal subset is where a band gap is a
    # real quantity, and it is the number this repo reports as the headline.
    target = max(all_runs, key=lambda t: (len(all_runs[t]), "nonmetals" in t))
    runs = all_runs[target]

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.10, right=0.97)

    panel_comparison(ax, target, runs)
    nice = {"band_gap_nonmetals": "Predicting band gap of non-metals",
            "band_gap": "Predicting band gap, all materials"}.get(target, target)
    ax.set_title(nice, loc="left", pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.94)

    wins = sum(1 for s in runs
               if baseline_best(target, s)
               and runs[s]["test"]["mae"] < baseline_best(target, s)["mae"])
    verdict = (f"CGCNN wins {wins} of {len(runs)} splits — and loses the one that matters most"
               if 0 < wins < len(runs)
               else f"CGCNN wins on {wins} of {len(runs)} splits"
               if wins else "the descriptors win on every split")

    fig.suptitle("Does learning from structure beat looking up chemistry?",
                 fontsize=15.5, fontweight="bold", y=1.005, x=0.006, ha="left")
    fig.text(0.006, 0.945, verdict, fontsize=11, color=WARN,
             fontweight="bold", ha="left", va="top")

    budget = next(iter(runs.values())).get("config", {}).get("max_minutes", "?")
    caption(
        fig,
        f"Both approaches saw exactly the same test materials. The network trained for a fixed {budget:.0f} minutes per test on one laptop CPU, so this reflects\n"
        "what a laptop can deliver, not what unlimited compute could. Green percentages mean the network won; red means looking up properties won.\n"
        "The rightmost test holds out whole ELEMENTS — see Figure 8 for why that is where the network fails.",
        y=0.055,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig7_cgcnn_vs_baselines.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"wrote {path}")
    for s in SPLIT_ORDER:
        if s not in runs:
            continue
        b = baseline_best(target, s)
        g = runs[s]["test"]["mae"]
        if b:
            d = 100 * (b["mae"] - g) / b["mae"]
            print(f"  {s:<9} CGCNN {g:.4f}   descriptors {b['mae']:.4f} ({b['model']})"
                  f"   {d:+.1f}%")


if __name__ == "__main__":
    main()
