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
    STRUCTURE, COMPOSITION, ACCENT, WARN, MUTED, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

# Fallbacks are the measured values, so the figure builds on a fresh clone.
BASE = {"cgcnn_random": 0.4137, "cgcnn_element": 1.0194,
        "desc_random": 0.511, "desc_element": 0.694}

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
        if "random" in d:
            base["cgcnn_random"] = d["random"]["test"]["mae"]
        if "element" in d:
            base["cgcnn_element"] = d["element"]["test"]["mae"]
    p = RESULTS / "baselines.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))["results"]
        for split in ("random", "element"):
            m = [r["mae"] for r in rows if r.get("target") == "band_gap_nonmetals"
                 and r.get("split") == split and r.get("model") != "mean"]
            if m:
                base[f"desc_{split}"] = min(m)

    sig = {}
    p = RESULTS / "fusion_significance.json"
    if p.exists():
        sig = json.loads(p.read_text(encoding="utf-8"))
    return fus, base, sig


def panel_element(ax, fus, base, sig):
    """The split the whole phase exists for."""
    order = ["cgcnn", "cgcnn_both", "cgcnn_both_comp", "cgcnn_properties"]
    vals, names = [], []
    for k in order:
        v = (base["cgcnn_element"] if k == "cgcnn"
             else fus.get(k, {}).get("element", {}).get("test", {}).get("mae"))
        if v is not None:
            vals.append(v)
            names.append(k)

    # Labels above the bars, never on them: text on a coloured bar is unreadable
    # in half the palette and unreadable in all of it once screenshotted.
    y = np.arange(len(names))[::-1].astype(float)
    for yi, k, v in zip(y, names, vals):
        title, sub = LABEL[k]
        ax.text(0.010, yi + 0.46, title, fontsize=9.6, fontweight="bold",
                color=INK, ha="left", va="center", zorder=6)
        ax.text(0.010, yi + 0.26, sub, fontsize=8.2, color=MUTED,
                ha="left", va="center", zorder=6)
        ax.barh(yi - 0.10, v, height=0.32, color=COLOUR[k], alpha=0.88,
                edgecolor="white", zorder=3)
        ax.text(v + 0.018, yi - 0.10, f"{v:.3f} eV", fontsize=10.4,
                fontweight="bold", color=COLOUR[k], ha="left", va="center",
                zorder=6)

    # The bar the graph network has to clear: chemistry with no structure at all.
    ax.axvline(base["desc_element"], color=STRUCTURE, ls="--", lw=1.8, zorder=5)
    ax.text(base["desc_element"] + 0.03, y[0] + 0.90,
            f"descriptors alone, no structure  ({base['desc_element']:.3f} eV)",
            fontsize=8.4, fontweight="bold", color=STRUCTURE,
            ha="left", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95))

    ax.set_yticks([])
    ax.set_ylim(-0.75, len(names) + 0.20)
    ax.set_xlim(0, 1.30)
    ax.set_xlabel("Band-gap error on materials containing ELEMENTS never seen in "
                  "training  (eV, lower is better)")
    ax.set_title("A.  The test that broke the graph network in Phase 3",
                 loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.45)
    ax.set_axisbelow(True)

    if sig:
        pair = next((p for p in sig.get("pairs", [])
                     if {p["a"], p["b"]} == {"cgcnn", "cgcnn_properties"}), None)
        if pair:
            d = abs(pair["delta_mae"])
            lo, hi = sorted(abs(x) for x in (pair["ci_low"], pair["ci_high"]))
            ax.text(1.28, -0.52,
                    f"deleting the learned code is worth {d:.3f} eV  "
                    f"[95% range {lo:.3f} – {hi:.3f}]",
                    fontsize=8.4, fontweight="bold", color=ACCENT,
                    ha="right", va="center")


def panel_predictions(ax, fus, base):
    """Scoreboard against what was predicted before the runs."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("B.  Three predictions, written down before the runs",
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
         f"It was the worst: {both_el:.3f} eV against {prop_el:.3f}. Given a "
         "memorisable route and\na chemical one, the model took the memorisable "
         "route." if both_el else "—"),
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
    fus, base, sig = load()
    if not fus:
        raise FileNotFoundError(
            "results/fusion_band_gap_nonmetals.json missing. Run:\n"
            "    python scripts/train_fusion.py --atoms properties --split element --nonmetals")

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.9),
                             gridspec_kw={"width_ratios": [1.06, 1.0]})
    fig.subplots_adjust(left=0.045, right=0.975, top=0.79, bottom=0.24, wspace=0.16)

    panel_element(axes[0], fus, base, sig)
    panel_predictions(axes[1], fus, base)

    fig.suptitle("Give the network the periodic table and the collapse mostly goes away",
                 fontsize=15.5, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.938,
             "Same architecture, same budget, same test set. The only thing that changes is what numbers each atom starts with.",
             fontsize=9.4, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "A graph network normally learns a private code for each element from the training data, so an element it never met has a code that was never learned — which is why\n"
        "the error nearly tripled on this test. Replacing that code with tabulated electronegativity, ionic radius, row and group cuts the error by 35% and finally beats a\n"
        "structure-blind descriptor model on its own strongest ground. The failed predictions matter as much: adding properties ALONGSIDE the learned code made things worse,\n"
        "because a free per-element vector is the easier way to fit the training set and the model takes it. The shortcut has to be removed, not merely supplemented.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig12_fusion.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
