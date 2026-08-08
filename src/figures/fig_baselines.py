"""Figure 6 -- The bar every neural network has to clear.

Reads results/baselines.json, produced by scripts/run_baselines.py.

Two things this figure is for. First, to put a number on what chemistry alone
achieves, so that a GNN result later has something to be compared against.
Second, to show what the honest splits cost: the same model, same features,
evaluated four ways.

If the file was produced with --subsample, the figure says so on its face. A
plot that does not disclose it was made from a sanity run is how provisional
numbers end up in a talk.

Run with:  python -m src.figures.fig_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, WARN, MUTED, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "baselines.json"
OUT = REPO / "results" / "figures"

SPLIT_ORDER = ["random", "formula", "chemsys", "element"]
SPLIT_LABEL = {"random": "Random", "formula": "New\nformula",
               "chemsys": "New\nsystem", "element": "New\nelement"}

BLOCK_COLOUR = {"composition": COMPOSITION, "structure_lite": STRUCTURE, "both": FUSED}
BLOCK_LABEL = {
    "composition": "Chemistry only\n(the formula)",
    "structure_lite": "Cheap structure\n(density, symmetry)",
    "both": "Both",
}

# Xie & Grossman, PRL 120, 145301 (2018), Materials Project band gap.
CGCNN_GAP_MAE = 0.388


def load() -> dict:
    if not RESULTS.exists():
        raise FileNotFoundError(
            f"{RESULTS} missing. Run:\n    python scripts/run_baselines.py"
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def best(rows, **where):
    sel = [r for r in rows
           if all(r.get(k) == v for k, v in where.items()) and r["model"] != "mean"]
    return min(sel, key=lambda r: r["mae"]) if sel else None


def floor_for(rows, **where):
    sel = [r for r in rows if all(r.get(k) == v for k, v in where.items())
           and r["model"] == "mean"]
    return min(sel, key=lambda r: r["mae"]) if sel else None


def panel(ax, rows, target, title, show_cgcnn=False):
    blocks = [b for b in ("composition", "structure_lite", "both")
              if any(r["block"] == b and r["target"] == target for r in rows)]
    splits = [s for s in SPLIT_ORDER
              if any(r["split"] == s and r["target"] == target for r in rows)]
    if not blocks or not splits:
        ax.text(0.5, 0.5, "no results for\n" + target, ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        ax.axis("off")
        return

    x = np.arange(len(splits))
    width = 0.8 / len(blocks)

    for k, b in enumerate(blocks):
        vals, labels = [], []
        for s in splits:
            r = best(rows, split=s, target=target, block=b)
            vals.append(r["mae"] if r else np.nan)
            labels.append(r["model"] if r else "")
        pos = x + (k - (len(blocks) - 1) / 2) * width
        ax.bar(pos, vals, width * 0.9, color=BLOCK_COLOUR[b], alpha=0.88,
               label=BLOCK_LABEL[b], zorder=3)
        for xi, v, lb in zip(pos, vals, labels):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.8, color=INK, fontweight="bold", rotation=90)

    # The trivial baseline: always predict the training median. Anything at or
    # below this line has learned nothing at all.
    fl = [floor_for(rows, split=s, target=target, block=blocks[0]) for s in splits]
    fv = [f["mae"] if f else np.nan for f in fl]
    if np.isfinite(fv).any():
        ax.plot(x, fv, ls=":", marker="o", ms=4, color=WARN, lw=1.4, zorder=5,
                label="Predict the median\n(learned nothing)")

    if show_cgcnn:
        ax.axhline(CGCNN_GAP_MAE, color=INK, ls="--", lw=1.2, alpha=0.55, zorder=4)
        ax.text(len(splits) - 0.45, CGCNN_GAP_MAE, f" CGCNN, published ({CGCNN_GAP_MAE})",
                fontsize=8, color=INK, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits], fontsize=9)
    ax.set_ylabel("Mean absolute error  (eV)")
    ax.set_title(title, loc="left", pad=10)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    top = np.nanmax([r["mae"] for r in rows if r["target"] == target])
    ax.set_ylim(0, top * 1.28)


def main() -> None:
    use_house_style()
    blob = load()
    rows = blob["results"]

    targets = [t for t in ("band_gap", "band_gap_nonmetals",
                           "formation_energy_per_atom", "energy_above_hull")
               if any(r["target"] == t for r in rows)]
    n = len(targets)

    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.8), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(bottom=0.26, top=0.84, wspace=0.28)

    titles = {
        "band_gap": "A.  Band gap, all materials",
        "band_gap_nonmetals": "B.  Band gap, non-metals only",
        "formation_energy_per_atom": "C.  Formation energy",
        "energy_above_hull": "D.  Stability (energy above hull)",
    }
    for ax, t in zip(axes, targets):
        panel(ax, rows, t, titles.get(t, t), show_cgcnn=(t == "band_gap"))

    axes[0].legend(loc="upper left", fontsize=8.2, framealpha=0.9)

    provisional = blob.get("subsampled_train_to")
    head = "What chemistry alone already achieves"
    if provisional:
        head += f"   [PROVISIONAL — trained on {provisional:,} materials, not the full set]"
    fig.suptitle(head, fontsize=15.5, fontweight="bold", y=1.0, x=0.006, ha="left",
                 color=WARN if provisional else INK)

    caption(
        fig,
        "Every bar is the best of ridge regression, random forest and gradient boosting on looked-up element properties -- no crystal structure in the\n"
        "first block at all. The same model scores differently under each split, because each is a different question. Note that a stricter split is not\n"
        "automatically harder: it also changes which materials are being asked about. A neural network has to beat these bars to earn its training time.",
        y=0.055,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig6_baselines.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"wrote {path}")
    for t in targets:
        for s in SPLIT_ORDER:
            r = best(rows, split=s, target=t, block="composition")
            r2 = best(rows, split=s, target=t, block="both")
            if r:
                gain = f"{100 * (r['mae'] - r2['mae']) / r['mae']:+.1f}%" if r2 else "  n/a"
                print(f"  {t:<26} {s:<8} composition {r['mae']:.4f}  "
                      f"both {r2['mae']:.4f}  gain {gain}" if r2
                      else f"  {t:<26} {s:<8} composition {r['mae']:.4f}")


if __name__ == "__main__":
    main()
