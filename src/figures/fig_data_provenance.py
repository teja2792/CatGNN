"""Figure 3 -- Where every number in this repository comes from.

The repo's first rule is that nothing is invented. This figure is that rule drawn
out: each prediction target traced back to the open database it came from, and
colour-coded by what kind of number it is.

Kept in sync by hand with DATA_GROUNDING.md and SOURCES.md.

Run with:  python -m src.figures.fig_data_provenance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .style import use_house_style, caption, source_stamp, TIER_COLORS, MUTED, INK, LIGHT

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

# (label, sub-label, y, tier)
SOURCES = [
    ("Materials Project", "~150k crystals · mp-api", 8.55, "calculated"),
    ("JARVIS-DFT  (NIST)", "~80k crystals · jarvis-tools", 7.35, "calculated"),
    ("matminer  expt_gap", "~4.6k measured gaps", 6.15, "measured"),
    ("Catalysis-Hub  (SUNCAT)", ">100k adsorption energies", 4.55, "calculated"),
    ("Open Catalyst  OC20", "IS2RE 10k subset · stretch goal", 3.35, "calculated"),
]

# (label, y, tier, [source indices])
TARGETS = [
    ("Band gap", 8.55, "calculated", [0, 1]),
    ("Band gap, measured", 7.35, "measured", [2]),
    ("Formation energy", 6.35, "calculated", [0, 1]),
    ("Stability  (E above hull)", 5.35, "calculated", [0]),
    ("Adsorption energy", 4.05, "calculated", [3, 4]),
]

DERIVED = ("Catalytic activity descriptor", 2.35, "derived")

TIER_TEXT = {
    "measured": "Measured in a lab",
    "calculated": "Calculated by DFT, by someone else, cited",
    "derived": "Computed here from the row above, via a published equation",
    "synthetic": "Synthetic — verification harness only, never a claim",
}


def box(ax, x, y, w, h, text, sub, color, fontsize=10, align="left"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, alpha=0.13, edgecolor=color, linewidth=1.6, zorder=2,
        )
    )
    tx = x + 0.18 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(tx, y + (0.14 if sub else 0), text, fontsize=fontsize,
            fontweight="bold", color=INK, va="center", ha=ha, zorder=3)
    if sub:
        ax.text(tx, y - 0.20, sub, fontsize=8.3, color=MUTED, va="center", ha=ha, zorder=3)


def main() -> None:
    use_house_style()
    fig, ax = plt.subplots(figsize=(14.2, 7.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(1.2, 9.9)
    ax.axis("off")

    SX, SW = 0.2, 5.0
    TX, TW = 8.6, 5.2
    H = 0.86

    ax.text(SX + 0.1, 9.55, "OPEN DATABASES", fontsize=10, fontweight="bold", color=MUTED)
    ax.text(TX + 0.1, 9.55, "WHAT THIS REPO PREDICTS", fontsize=10, fontweight="bold", color=MUTED)

    src_y = {}
    for i, (name, sub, y, tier) in enumerate(SOURCES):
        box(ax, SX, y, SW, H, name, sub, TIER_COLORS[tier])
        src_y[i] = y

    for name, y, tier, srcs in TARGETS:
        box(ax, TX, y, TW, H, name, None, TIER_COLORS[tier])
        for s in srcs:
            ax.add_patch(
                FancyArrowPatch(
                    (SX + SW + 0.08, src_y[s]), (TX - 0.08, y),
                    arrowstyle="-|>", mutation_scale=12,
                    color=TIER_COLORS[tier], lw=1.3, alpha=0.5,
                    connectionstyle="arc3,rad=0.12", zorder=1,
                )
            )

    # The derived target hangs off adsorption energy, not off a database.
    dname, dy, dtier = DERIVED
    box(ax, TX, dy, TW, H, dname, None, TIER_COLORS[dtier])
    ax.add_patch(
        FancyArrowPatch(
            (TX + TW / 2, 4.05 - H / 2 - 0.05), (TX + TW / 2, dy + H / 2 + 0.05),
            arrowstyle="-|>", mutation_scale=13, color=TIER_COLORS[dtier], lw=1.8, zorder=1,
        )
    )
    ax.text(
        TX + TW / 2 + 0.25, (4.05 + dy) / 2,
        "scaling relations / volcano equation\n(Nørskov-type, cited in DATA_GROUNDING.md)",
        fontsize=8.6, color=TIER_COLORS[dtier], va="center", ha="left", style="italic",
    )

    ax.text(
        TX, 1.55,
        "NOT predicted: measured catalytic rates (turnover frequency, current density).\n"
        "No structure→measured-activity dataset exists at the scale a model like this needs.",
        fontsize=9, color=MUTED, va="center", ha="left",
    )

    for k, (tier, label) in enumerate(TIER_TEXT.items()):
        y = 9.55 - k * 0.0
        ax.add_patch(
            FancyBboxPatch(
                (14.15, 8.9 - k * 0.62), 0.28, 0.28,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=TIER_COLORS[tier], edgecolor="none", zorder=3,
            )
        )
        ax.text(14.55, 9.04 - k * 0.62, label, fontsize=8.4, color=INK, va="center", ha="left")
    ax.text(14.15, 9.5, "WHAT KIND OF NUMBER", fontsize=9, fontweight="bold", color=MUTED)
    ax.text(14.15, 6.3, "Full detail, including licences,\naccess dates and exact queries:\nDATA_GROUNDING.md · SOURCES.md",
            fontsize=8.4, color=MUTED, va="top", ha="left", style="italic")

    fig.suptitle(
        "Every number traced back to an open, citable source",
        fontsize=15.5, fontweight="bold", y=0.985, x=0.006, ha="left",
    )
    caption(
        fig,
        "Nothing in this repository is invented. Four of the five prediction targets are downloaded directly from open databases. The fifth —\n"
        "catalytic activity — is computed from real adsorption energies through a published thermodynamic relation, and is labelled a descriptor\n"
        "rather than a rate everywhere it appears. Measured catalytic activity is explicitly out of scope, for the reason given at the bottom.",
        y=0.045,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig3_data_provenance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
