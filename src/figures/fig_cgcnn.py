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
SPLIT_LABEL = {"random": "Random", "formula": "New\nformula",
               "chemsys": "New\nsystem", "element": "New\nelement"}
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

    ax.bar(x - w / 2, base, w * 0.92, color=COMPOSITION, alpha=0.88,
           label="Best descriptor model\n(Phase 2)", zorder=3)
    ax.bar(x + w / 2, gnn, w * 0.92, color=FUSED, alpha=0.88,
           label="CGCNN\n(learned from structure)", zorder=3)

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
        ax.text(xi, max(b, g) * 1.14,
                f"{delta:+.0f}%", ha="center", va="bottom", fontsize=9,
                color=ACCENT if won else WARN, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits], fontsize=9)
    ax.set_ylabel(f"Mean absolute error  ({UNITS.get(target, '')})")

    # Where the story turns. Marking it beats hoping the reader spots it.
    losers = [i for i, (b, g) in enumerate(zip(base, gnn))
              if np.isfinite(b) and g > b]
    if losers and len(losers) < len(splits):
        ax.axvspan(min(losers) - 0.5, len(splits) - 0.5, color=WARN, alpha=0.06, zorder=0)
        ax.text(min(losers) - 0.42, ax.get_ylim()[1] * 0.955,
                "unseen chemistry:\nthe GNN collapses,\nthe descriptors do not",
                fontsize=8.6, color=WARN, va="top", ha="left", fontweight="bold")
    ax.set_ylim(0, np.nanmax(base + gnn) * 1.35)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_curves(ax, runs: dict):
    """Validation error against wall-clock, which is the budget that binds."""
    colours = [WARN, COMPOSITION, FUSED, ACCENT]
    for (s, c) in zip(SPLIT_ORDER, colours):
        if s not in runs:
            continue
        h = runs[s].get("history", [])
        if not h:
            continue
        mins = [e["minutes"] for e in h]
        val = [e["val_mae"] for e in h]
        ax.plot(mins, val, color=c, lw=1.7, label=SPLIT_LABEL[s].replace("\n", " "))
        b = runs[s].get("best_epoch")
        if b is not None and 0 <= b < len(h):
            ax.scatter([h[b]["minutes"]], [h[b]["val_mae"]], color=c, s=34, zorder=5)

    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel("Validation MAE")
    ax.set_title("B.  Learning curves, and where the budget stopped them",
                 loc="left", pad=10)
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=8.4, title="split", title_fontsize=8.4)


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

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.30)

    panel_comparison(axes[0], target, runs)
    nice = {"band_gap_nonmetals": "band gap, non-metals",
            "band_gap": "band gap, all materials"}.get(target, target)
    axes[0].set_title(f"A.  {nice}", loc="left", pad=10)
    axes[0].legend(loc="upper left", fontsize=8.4)
    panel_curves(axes[1], runs)

    wins = sum(1 for s in runs
               if baseline_best(target, s)
               and runs[s]["test"]["mae"] < baseline_best(target, s)["mae"])
    verdict = (f"CGCNN wins {wins} of {len(runs)} splits — and loses the one that matters most"
               if 0 < wins < len(runs)
               else f"CGCNN wins on {wins} of {len(runs)} splits"
               if wins else "the descriptors win on every split")

    fig.suptitle(f"Does learning from structure beat looking up chemistry?  —  {verdict}",
                 fontsize=15, fontweight="bold", y=1.0, x=0.006, ha="left")

    budget = next(iter(runs.values())).get("config", {}).get("max_minutes", "?")
    caption(
        fig,
        f"Both sides evaluated on identical test sets. CGCNN trained under a fixed {budget}-minute CPU budget per split, so the comparison reflects what\n"
        "a laptop can actually deliver rather than what unlimited compute could. Percentages above the bars are the change in error: green means the graph\n"
        "network won, red means the descriptors did. The last split holds out whole ELEMENTS, so the model meets chemistry it has never seen -- and a\n"
        "learned atom embedding has nothing to say about an element it was never trained on, while a looked-up electronegativity still does.",
        y=0.045,
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
