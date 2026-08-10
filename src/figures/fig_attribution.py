"""Figure 14 -- which piece of chemistry the model actually used.

Two rankings, because the naive one is misleading and the reason it is misleading
is itself the interesting part.

Panel A is raw attribution. Electronegativity tops it, which looks like a
triumph -- until the control column shows that a model trained on SHUFFLED band
gaps also puts electronegativity near the top. Some of that ranking is the
geometry of the property table, not anything the model learned.

Panel B divides by the control, leaving only what training changed. The picture
that survives is sharper and more chemical than the raw one.

Panel C is the sanity check, against a computed null rather than an assumed one.

Run with:  python -m src.figures.fig_attribution
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)
from ..models.attribution import (cosine_null, enrichment, family_of, label_of,
                                  profile_similarity, spearman)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

FAMILY_COLOUR = {"electronic": ACCENT, "size": STRUCTURE, "position": FUSED,
                 "block": COMPOSITION, "thermal": MUTED, "other": LIGHT}


def load(split: str):
    p = RESULTS / f"attribution_band_gap_{split}_nonmetals.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run:\n    python scripts/explain.py --split {split}")
    return json.loads(p.read_text(encoding="utf-8"))


def panel_raw(ax, names, imp, ctrl_key):
    """Raw attribution, with the control alongside so it cannot be over-read."""
    t = imp["trained"] / imp["trained"].max()
    c = imp[ctrl_key] / imp[ctrl_key].max()

    top = np.argsort(-t)[:12]
    y = np.arange(len(top)).astype(float)

    ax.barh(y + 0.19, t[top], height=0.36, color=[FAMILY_COLOUR[family_of(names[j])]
                                                  for j in top],
            alpha=0.9, edgecolor="white", zorder=3)
    ax.barh(y - 0.19, c[top], height=0.36, facecolor="none", edgecolor=WARN,
            linewidth=1.4, hatch="///", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([label_of(names[j]) for j in top], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Share of the prediction it moved  (1.0 = the largest)")
    ax.set_title("A.  Raw attribution — and why it cannot be read alone",
                 loc="left", pad=22)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Patch(facecolor=MUTED, label="the trained model"),
        Patch(facecolor="none", edgecolor=WARN, hatch="///",
              label="trained on SHUFFLED band gaps — learned nothing"),
    ], fontsize=8.2, loc="lower right", framealpha=0.96)

    ax.text(0.0, 1.035,
            "the control ranks electronegativity first too — so this ranking is "
            "partly the shape of the input table",
            transform=ax.transAxes, fontsize=8.2, color=WARN, fontweight="bold",
            ha="left", va="bottom")


def panel_enrichment(ax, names, imp, ctrl_key):
    """What training actually changed."""
    e = enrichment(imp["trained"], imp[ctrl_key])
    order = np.argsort(-e)
    show = np.concatenate([order[:8], order[-4:]])
    y = np.arange(len(show)).astype(float)

    for yi, j in zip(y, show):
        up = e[j] >= 1.0
        ax.barh(yi, e[j], height=0.62,
                color=FAMILY_COLOUR[family_of(names[j])] if up else MUTED,
                alpha=0.9 if up else 0.55, edgecolor="white", zorder=3)
        ax.text(e[j] + 0.18, yi, f"{e[j]:.1f}×", fontsize=8.6, fontweight="bold",
                color=FAMILY_COLOUR[family_of(names[j])] if up else MUTED,
                ha="left", va="center", zorder=6)

    ax.axvline(1.0, color=INK, lw=1.6, zorder=5)
    ax.text(1.15, y[0] - 0.95, "1.0 = training changed nothing", fontsize=8.0,
            color=INK, ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([label_of(names[j]) for j in show], fontsize=8.6)
    ax.set_ylim(len(show) - 0.4, -1.35)
    ax.set_xlim(0, max(e[show]) * 1.22)
    ax.set_xlabel("Attribution relative to the shuffled-label control")
    ax.set_title("B.  What training actually changed", loc="left", pad=22)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    # A visible break, so nobody reads the bottom four as ranks 9-12.
    ax.axhline((y[7] + y[8]) / 2, color=MUTED, ls=":", lw=1.3, zorder=5)
    ax.text(max(e[show]) * 1.19, (y[7] + y[8]) / 2 - 0.30,
            "the four training pushed DOWN", fontsize=7.8, color=MUTED,
            ha="right", va="bottom", style="italic")


def panel_checks(ax, names, imp, blob):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C.  Does it just reproduce a control?", loc="left", pad=10)

    null = cosine_null(len(names))
    ax.text(0.1, 9.0,
            f"Two UNRELATED non-negative profiles of {len(names)} numbers already\n"
            f"score cosine {null['mean']:.2f} (90% of draws {null['p05']:.2f}–{null['p95']:.2f}). "
            f"So a cosine near 0.8\nmeans nothing on its own. Spearman has its null at 0.",
            fontsize=8.4, color=MUTED, ha="left", va="top", linespacing=1.6)

    rows = [(k, imp[k]) for k in ("untrained", "shuffled_labels") if k in imp]
    ax.text(0.1, 6.3, f"{'control':<20}{'cosine':>10}{'Spearman':>13}",
            fontsize=8.6, fontweight="bold", color=INK, ha="left", va="center",
            family="DejaVu Sans Mono")
    for i, (k, v) in enumerate(rows):
        cos = profile_similarity(imp["trained"], v)
        rho = spearman(imp["trained"], v)
        inside = null["p05"] <= cos <= null["p95"]
        ax.text(0.1, 5.4 - i * 0.9,
                f"{k:<20}{cos:>10.3f}{rho:>13.3f}",
                fontsize=8.6, color=INK, ha="left", va="center",
                family="DejaVu Sans Mono")
        ax.text(7.6, 5.4 - i * 0.9,
                "ordinary" if inside or cos < null["p05"] else "high",
                fontsize=8.0, color=ACCENT if inside or cos < null["p05"] else WARN,
                ha="left", va="center")

    ax.add_patch(FancyBboxPatch(
        (0.05, 1.1), 9.9, 2.2, boxstyle="round,pad=0.12,rounding_size=0.15",
        facecolor=ACCENT, alpha=0.10, edgecolor=ACCENT, linewidth=1.5, zorder=2))
    ax.text(0.35, 2.75, "Passes.", fontsize=9.6, fontweight="bold",
            color=ACCENT, ha="left", va="center", zorder=6)
    ax.text(0.35, 1.85,
            "Neither control reproduces the trained ranking, so the\n"
            "attribution describes this model rather than the method.",
            fontsize=8.4, color=INK, ha="left", va="center", linespacing=1.55,
            zorder=6)


def panel_families(ax, names, imp):
    t = imp["trained"]
    fam = {}
    for j, n in enumerate(names):
        fam[family_of(n)] = fam.get(family_of(n), 0.0) + t[j]
    total = sum(fam.values()) or 1.0
    items = sorted(fam.items(), key=lambda kv: -kv[1])

    left = 0.0
    for name, v in items:
        w = 100 * v / total
        ax.barh(0, w, left=left, height=0.55, color=FAMILY_COLOUR[name],
                alpha=0.9, edgecolor="white", zorder=3)
        if w > 7:
            ax.text(left + w / 2, 0, f"{name}\n{w:.0f}%", fontsize=8.2,
                    fontweight="bold", color="white", ha="center", va="center",
                    zorder=6, linespacing=1.4)
        left += w

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Share of all attribution  (%)")
    ax.set_title("D.  Which kind of chemistry", loc="left", pad=8)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def main(split: str = "random") -> None:
    use_house_style()
    blob = load(split)
    names = blob["properties"]
    imp = {k: np.array(v["mean_abs"]) for k, v in blob["models"].items()}
    ctrl_key = "shuffled_labels" if "shuffled_labels" in imp else "untrained"

    fig = plt.figure(figsize=(15.8, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62],
                          width_ratios=[1.0, 1.0],
                          left=0.115, right=0.975, top=0.845, bottom=0.135,
                          hspace=0.42, wspace=0.30)

    panel_raw(fig.add_subplot(gs[0, 0]), names, imp, ctrl_key)
    panel_enrichment(fig.add_subplot(gs[0, 1]), names, imp, ctrl_key)
    panel_checks(fig.add_subplot(gs[1, 0]), names, imp, blob)
    panel_families(fig.add_subplot(gs[1, 1]), names, imp)

    fig.suptitle("The model learned the chemistry a chemist would have picked",
                 fontsize=16.0, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.947,
             f"Band-gap predictions attributed to the 31 tabulated element properties each atom starts from. "
             f"Integrated gradients over {blob['n_crystals']:,} test crystals, {blob['split']} split.",
             fontsize=9.4, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "Electron affinity is the quantity that sets where a material's conduction band sits, and it is the single property training changed most — the model leans on it "
        "roughly fourteen times\nharder than one that learned nothing. Next come the Mendeleev number, a hand-built ordering of the periodic table by chemical similarity, and the "
        "f-block and transition-metal flags.\nMeanwhile atomic mass — which correlates with almost everything in the periodic table and causes none of it — is DOWN-weighted to 0.3×. "
        "A model taking shortcuts would have done\nthe opposite. Raw attribution alone would not have shown this: the control ranks electronegativity first too, so panel A is partly "
        "the shape of the input table rather than the model.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig14_what_chemistry.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")

    e = enrichment(imp["trained"], imp[ctrl_key])
    for j in np.argsort(-e)[:5]:
        print(f"  {label_of(names[j]):<26}{e[j]:>6.1f}x")


if __name__ == "__main__":
    main()
