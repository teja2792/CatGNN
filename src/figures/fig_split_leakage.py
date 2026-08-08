"""Figure 5 -- How much a random split lets the model see the answers.

Built from the real splits in data/cache/splits/summary.json, over all 102,815
usable Materials Project crystals.

The point: a random split is the default in most materials ML, and on a database
full of polymorph families it hands the model a large slice of the test set's
chemistry during training. The score that comes back is partly recall, not
generalisation -- and nothing about it looks wrong.

Run with:  python -m src.figures.fig_split_leakage
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    use_house_style, caption, source_stamp,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "data" / "cache" / "splits" / "summary.json"
OUT = REPO / "results" / "figures"

SCHEMES = ["random", "formula", "chemsys", "element"]

LABELS = {
    "random":  ("Random", "the default"),
    "formula": ("Formula-disjoint", "new formula"),
    "chemsys": ("System-disjoint", "new system"),
    "element": ("Element-disjoint", "new element"),
}

QUESTIONS = {
    "random":  "How well does it do on\nmore of the same?",
    "formula": "…on a formula it has\nnever seen?",
    "chemsys": "…on a chemical system\nit has never seen?",
    "element": "…on an element it has\nnever seen?",
}

AXES = [
    ("test_with_formula_seen_pct", "Same formula\nalready in training", COMPOSITION),
    ("test_with_chemsys_seen_pct", "Same chemical system\nalready in training", FUSED),
    ("test_with_all_elements_seen_pct", "Every element\nalready in training", STRUCTURE),
]


def load() -> dict:
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"{SUMMARY} missing. Run:\n"
            "    python scripts/build_graphs.py\n"
            "    python scripts/make_splits.py"
        )
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def panel_leakage(ax, data):
    schemes = data["schemes"]
    x = np.arange(len(SCHEMES))
    width = 0.26

    for k, (key, label, colour) in enumerate(AXES):
        vals = [schemes[s]["leakage"][key] for s in SCHEMES]
        pos = x + (k - 1) * width
        ax.bar(pos, vals, width * 0.92, color=colour, alpha=0.88,
               label=label, zorder=3)
        for xi, v in zip(pos, vals):
            if v >= 1:
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{LABELS[s][0]}\n{LABELS[s][1]}" for s in SCHEMES], fontsize=9.5
    )
    ax.tick_params(axis="x", pad=8)

    ax.set_ylabel("% of the test set the model has\nalready met, in some form")
    ax.set_ylim(0, 132)
    ax.set_title("A.  What the test set gives away", loc="left", pad=12)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005),
              fontsize=8.6, ncol=3, columnspacing=1.0, handlelength=1.2)

    rand = schemes["random"]["leakage"]["test_with_formula_seen_pct"]
    ax.annotate(
        f"{rand:.0f}% of a random test set shares\na formula with something\nin training",
        xy=(0 - width, rand + 2), xytext=(0.52, 0.52), textcoords="axes fraction",
        fontsize=9.2, color=WARN, fontweight="bold", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.4,
                        connectionstyle="arc3,rad=0.28", shrinkB=4),
    )


def panel_questions(ax, data):
    ax.set_title("B.  Four questions, four honest answers", loc="left", pad=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    schemes = data["schemes"]
    colours = [WARN, COMPOSITION, FUSED, ACCENT]
    top, step = 8.9, 2.15

    for i, (s, colour) in enumerate(zip(SCHEMES, colours)):
        y = top - i * step
        z = schemes[s]["sizes"]
        ax.add_patch(plt.Rectangle((0.15, y - 0.82), 9.6, 1.62, facecolor=colour,
                                   alpha=0.09, edgecolor=colour, lw=1.4, zorder=1))
        ax.text(0.45, y + 0.42, LABELS[s][0], fontsize=10.5, fontweight="bold",
                color=INK, va="center")
        ax.text(0.45, y - 0.05, QUESTIONS[s].replace("\n", " "), fontsize=9,
                color=MUTED, va="center")
        ax.text(0.45, y - 0.52,
                f"train {z['train']:,}   ·   val {z['val']:,}   ·   test {z['test']:,}",
                fontsize=8.4, color=MUTED, va="center")

    held = schemes["element"].get("held_out_elements", [])
    if held:
        ax.text(0.15, 0.55,
                f"Elements held out entirely for the last split ({len(held)}):\n"
                + ", ".join(held),
                fontsize=8.6, color=ACCENT, va="center", style="italic")


def main() -> None:
    use_house_style()
    data = load()

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.24)

    panel_leakage(axes[0], data)
    panel_questions(axes[1], data)

    fig.suptitle(
        "A random split is not a fair test",
        fontsize=15.5, fontweight="bold", y=1.015, x=0.006, ha="left",
    )
    caption(
        fig,
        "Materials Project is full of polymorph families -- Li7Mn2(CoO4)3 alone has 221 entries at identical cell size, the same lattice with different\n"
        "cation orderings. A random split scatters those across training and test, so the model can memorise one and be graded on its near-twin.\n"
        "Every result in this repository is reported against all four splits, so the inflation is visible instead of assumed away.",
        y=0.045,
    )
    fig.text(
        0.006, 0.955,
        f"All {data['n_materials']:,} usable Materials Project crystals  ·  seed {data['seed']}  ·  data/cache/splits/summary.json",
        fontsize=8.4, color=MUTED, ha="left", va="top",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig5_split_leakage.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"wrote {path}")
    for s in SCHEMES:
        lk = data["schemes"][s]["leakage"]
        print(f"  {s:<9} formula {lk['test_with_formula_seen_pct']:5.1f}%   "
              f"chemsys {lk['test_with_chemsys_seen_pct']:5.1f}%   "
              f"elements {lk['test_with_all_elements_seen_pct']:5.1f}%")


if __name__ == "__main__":
    main()
