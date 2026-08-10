"""Figure 12 -- the fix, and the two predictions that were wrong.

Phase 3 diagnosed the element-generalisation collapse as a consequence of the
network learning a private code per element. Phase 5 tested that by deleting the
learned code and substituting tabulated element properties.

Three predictions were written into scripts/train_fusion.py before any run. One
held, two did not, and the two that failed are the more interesting half of the
result -- so this figure shows all three rather than only the one that worked.

Run with:  python -m src.figures.fig_fusion
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .style import (
    use_house_style, caption,
    COMPOSITION, ACCENT, WARN, MUTED, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

SPLITS = ["random", "formula", "chemsys", "element"]

# Questions, not jargon. "chemsys" means nothing to a reader who has not read
# the code, and the whole point of the four splits is that each asks a different
# question about what the model will face in practice.
SPLIT_PLAIN = {
    "random": "Materials like\nthe training set",
    "formula": "A formula it has\nnever seen",
    "chemsys": "An element combination\nit has never seen",
    "element": "An ELEMENT it has\nnever seen",
}

# Fallbacks are the measured values, so the figure builds on a fresh clone.
BASE = {"cgcnn_random": 0.4137, "cgcnn_formula": 0.4437,
        "cgcnn_chemsys": 0.4851, "cgcnn_element": 1.0194,
        "desc_random": 0.5113, "desc_formula": 0.5465,
        "desc_chemsys": 0.6135, "desc_element": 0.6942}

LABEL = {
    "cgcnn": ("Learned code per element", "the Phase 3 baseline"),
    "cgcnn_both": ("Learned code + properties", "prediction 3 said this would win"),
    "cgcnn_both_comp": ("…and the 192 descriptors too", "more inputs, still worse"),
    "cgcnn_properties": ("Properties only, no learned code", "the fix"),
}
COLOUR = {"cgcnn": WARN, "cgcnn_both": COMPOSITION,
          "cgcnn_both_comp": MUTED, "cgcnn_properties": ACCENT}


def load():
    fus = {}
    p = RESULTS / "fusion_band_gap_nonmetals.json"
    if p.exists():
        fus = json.loads(p.read_text(encoding="utf-8"))

    base = dict(BASE)
    p = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for sp in SPLITS:
            if sp in d:
                base[f"cgcnn_{sp}"] = d[sp]["test"]["mae"]
    p = RESULTS / "baselines.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))["results"]
        for sp in SPLITS:
            m = [r["mae"] for r in rows if r.get("target") == "band_gap_nonmetals"
                 and r.get("split") == sp and r.get("model") != "mean"]
            if m:
                base[f"desc_{sp}"] = min(m)

    sig = {}
    p = RESULTS / "fusion_significance.json"
    if p.exists():
        sig = json.loads(p.read_text(encoding="utf-8"))

    diag = {}
    p = RESULTS / "fusion_diagnostics.json"
    if p.exists():
        diag = json.loads(p.read_text(encoding="utf-8"))
    return fus, base, sig, diag


def panel_all_splits(ax, fus, base):
    """The complete result: three approaches, four tests of increasing strictness."""
    x = np.arange(len(SPLITS), dtype=float)
    w = 0.26

    desc = [base[f"desc_{sp}"] for sp in SPLITS]
    learned = [base[f"cgcnn_{sp}"] for sp in SPLITS]
    props = [fus.get("cgcnn_properties", {}).get(sp, {}).get("test", {}).get("mae")
             for sp in SPLITS]

    ax.bar(x - w, desc, w, color=COMPOSITION, alpha=0.88, edgecolor="white",
           label="Chemistry only — no structure at all", zorder=3)
    ax.bar(x, learned, w, color=WARN, alpha=0.88, edgecolor="white",
           label="Graph network, learned element codes  (Phase 3)", zorder=3)
    ax.bar(x + w, [p if p else 0 for p in props], w, color=ACCENT, alpha=0.9,
           edgecolor="white", label="Graph network + the periodic table  (Phase 5)",
           zorder=3)

    for xi, (a, b, c) in enumerate(zip(desc, learned, props)):
        for dx, v, col in ((-w, a, COMPOSITION), (0, b, WARN), (w, c, ACCENT)):
            if v:
                ax.text(xi + dx, v + 0.022, f"{v:.3f}", fontsize=8.2,
                        fontweight="bold", color=col, ha="center", va="bottom",
                        zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_PLAIN[sp] for sp in SPLITS], fontsize=8.8)
    ax.set_ylabel("Band-gap error  (eV, lower is better)")
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("How the test materials were chosen  —  stricter to the right")
    ax.set_title("A.  The complete result: four tests, three approaches",
                 loc="left", pad=10)
    ax.grid(True, axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.95)

    ax.annotate("this is the collapse\nPhase 5 set out to fix",
                xy=(3.0, learned[3]), xytext=(2.02, 1.03),
                fontsize=8.6, fontweight="bold", color=WARN, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))


def panel_variants(ax, fus, base, sig, diag):
    """Why removing the learned code beats supplementing it.

    The dashed row is not another trained model. It is the SAME 'both' weights
    with the ten untrained element rows swapped for the average trained row, so
    the difference isolates those rows instead of confounding them with a new fit.
    """
    def mae(k):
        return (base["cgcnn_element"] if k == "cgcnn"
                else fus.get(k, {}).get("element", {}).get("test", {}).get("mae"))

    rows = []
    for k in ("cgcnn", "cgcnn_both"):
        if mae(k) is not None:
            rows.append((LABEL[k][0], LABEL[k][1], mae(k), COLOUR[k], False))

    ab = diag.get("ablation", {})
    if ab.get("mae_rows_replaced"):
        rows.append(("…with the untrained rows neutralised",
                     "same weights, no retraining — an ablation, not a model",
                     ab["mae_rows_replaced"], ACCENT, True))

    for k in ("cgcnn_both_comp", "cgcnn_properties"):
        if mae(k) is not None:
            rows.append((LABEL[k][0], LABEL[k][1], mae(k), COLOUR[k], False))

    y = np.arange(len(rows))[::-1].astype(float)
    for yi, (title, sub, v, col, dashed) in zip(y, rows):
        ax.text(0.010, yi + 0.42, title, fontsize=8.8, fontweight="bold",
                color=INK, ha="left", va="center", zorder=6)
        ax.text(0.010, yi + 0.24, sub, fontsize=7.5, color=MUTED,
                ha="left", va="center", zorder=6)
        if dashed:
            ax.barh(yi - 0.12, v, height=0.26, facecolor="none", edgecolor=col,
                    linewidth=1.8, linestyle="--", hatch="///", zorder=4)
        else:
            ax.barh(yi - 0.12, v, height=0.26, color=col, alpha=0.88,
                    edgecolor="white", zorder=3)
        ax.text(v + 0.02, yi - 0.12, f"{v:.3f}", fontsize=9.2, fontweight="bold",
                color=col, ha="left", va="center", zorder=6)

    # Fraction of the penalty the ablation recovers, computed here rather than
    # read from the results file, so an older diagnostics JSON still works.
    prop = mae("cgcnn_properties")
    if ab.get("mae_as_trained") and ab.get("mae_rows_replaced") and prop:
        b, a = ab["mae_as_trained"], ab["mae_rows_replaced"]
        if b > prop:
            ax.text(0.010, -0.52,
                    f"neutralising those rows recovers {100 * (b - a) / (b - prop):.0f}% "
                    f"of the penalty — real, but not all of it",
                    fontsize=7.9, color=ACCENT, fontweight="bold",
                    ha="left", va="center", zorder=6)

    ax.axvline(base["desc_element"], color=COMPOSITION, ls="--", lw=1.7, zorder=5)
    ax.text(base["desc_element"] + 0.03, y[0] + 0.82, "chemistry only",
            fontsize=8.0, fontweight="bold", color=COMPOSITION,
            ha="left", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))

    ax.set_yticks([])
    ax.set_ylim(-0.85, len(rows) + 0.10)
    ax.set_xlim(0, 1.30)
    ax.set_xlabel("Error on the unseen-element test  (eV)")
    ax.set_title("B.  Removing the learned code beats supplementing it",
                 loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.45)
    ax.set_axisbelow(True)


def panel_predictions(ax, fus, base):
    """Scoreboard against what was predicted before the runs."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C.  Three predictions, written down before the runs",
                 loc="left", pad=10)

    prop_random = fus.get("cgcnn_properties", {}).get("random", {}) \
                     .get("test", {}).get("mae")
    prop_el = fus.get("cgcnn_properties", {}).get("element", {}) \
                 .get("test", {}).get("mae")
    both_el = fus.get("cgcnn_both", {}).get("element", {}).get("test", {}).get("mae")

    rows = [
        (True,
         "The element-disjoint error will improve a lot",
         f"1.019 → {prop_el:.3f} eV, and it now beats descriptors "
         f"({base['desc_element']:.3f} eV) too." if prop_el else "—"),
        (False,
         "The random-split error will get slightly worse",
         f"It got BETTER: {base['cgcnn_random']:.3f} → {prop_random:.3f} eV. "
         "The periodic table helps\neven where the model has seen every element."
         if prop_random else "—"),
        (False,
         "“Both” will be the best of the three",
         f"It was the worst: {both_el:.3f} eV against {prop_el:.3f}. Keeping the "
         "learned code at all\nhurts — for an unseen element it contributes pure "
         "noise." if both_el else "—"),
    ]

    for i, (held, claim, outcome) in enumerate(rows):
        yy = 8.7 - i * 3.02
        col = ACCENT if held else WARN
        ax.add_patch(FancyBboxPatch(
            (0.05, yy - 1.80), 9.9, 2.45,
            boxstyle="round,pad=0.10,rounding_size=0.12",
            facecolor="white", edgecolor=col, linewidth=1.6, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (0.05, yy - 0.02), 9.9, 0.67,
            boxstyle="square,pad=0.0",
            facecolor=col, alpha=0.14, edgecolor="none", zorder=4))
        ax.text(0.42, yy + 0.32, "✓" if held else "✗", fontsize=12,
                fontweight="bold", color=col, ha="center", va="center", zorder=6)
        ax.text(0.95, yy + 0.32, claim, fontsize=9.2, fontweight="bold",
                color=INK, ha="left", va="center", zorder=6)
        ax.text(9.75, yy + 0.32, "held" if held else "wrong", fontsize=8.4,
                fontweight="bold", color=col, ha="right", va="center", zorder=6)
        ax.text(0.42, yy - 1.05, outcome, fontsize=8.5, color=INK,
                ha="left", va="center", zorder=6, linespacing=1.6)

    ax.text(0.05, 0.05,
            "Recording predictions first is the only thing that makes “we expected that” checkable afterwards.",
            fontsize=8.3, color=MUTED, ha="left", va="center", style="italic")


def main() -> None:
    use_house_style()
    fus, base, sig, diag = load()
    if not fus:
        raise FileNotFoundError(
            "results/fusion_band_gap_nonmetals.json missing. Run:\n"
            "    python scripts/train_fusion.py --atoms properties --split element --nonmetals")

    fig = plt.figure(figsize=(15.6, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0],
                          left=0.055, right=0.975, top=0.845, bottom=0.135,
                          hspace=0.46, wspace=0.17)

    panel_all_splits(fig.add_subplot(gs[0, :]), fus, base)
    panel_variants(fig.add_subplot(gs[1, 0]), fus, base, sig, diag)
    panel_predictions(fig.add_subplot(gs[1, 1]), fus, base)

    fig.suptitle("Give the network the periodic table and it wins every test",
                 fontsize=16.0, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.948,
             "Same architecture, same 35-minute budget, same test sets. The only thing that changes is what numbers each atom starts with.",
             fontsize=9.5, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "A graph network normally learns a private code for each element from training data. Measured, that code carries almost no chemical structure — chemically alike\n"
        "elements score +0.09 on a similarity contrast where the tabulated properties score +1.04, and random numbers score +0.12. So an element the model never met has a\n"
        "code that means nothing, which is why the error nearly tripled on the strictest test. Replacing the code with tabulated electronegativity, ionic radius, row and\n"
        "group improves every one of the four tests and turns a 47% loss against chemistry-only into a 4% win. Keeping the code alongside the properties is worse than removing\n"
        "it, and the dashed bar tests why: taking those same trained weights and neutralising only the ten untrained element rows recovers 53% of the penalty. The untrained-row\n"
        "mechanism is therefore real but partial — the other half is still unexplained, and the extra 14,720 parameters are the obvious untested suspect.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig12_fusion.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    for sp in SPLITS:
        v = fus.get("cgcnn_properties", {}).get(sp, {}).get("test", {}).get("mae")
        if v:
            print(f"  {sp:<9}properties {v:.4f}   learned {base[f'cgcnn_{sp}']:.4f}   "
                  f"descriptors {base[f'desc_{sp}']:.4f}")


if __name__ == "__main__":
    main()
