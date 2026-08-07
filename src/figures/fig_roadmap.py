"""Figure 4 -- Build plan and current status.

A reader landing on the README should be able to see in one glance what is
finished, what is running, and what is not built yet. Repos that quietly imply
completeness are worse than repos that show their seams.

Edit ``PHASES`` as work lands; this is the single source of truth for the status
badges in the README.

Run with:  python -m src.figures.fig_roadmap
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .style import use_house_style, caption, STRUCTURE, COMPOSITION, FUSED, ACCENT, MUTED, INK, LIGHT

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

DONE, ACTIVE, TODO = "done", "active", "todo"

STATUS_STYLE = {
    DONE: dict(face=ACCENT, label="built"),
    ACTIVE: dict(face=COMPOSITION, label="in progress"),
    TODO: dict(face=MUTED, label="not started"),
}

# (block, phase number, title, one-line outcome, status)
PHASES = [
    ("Foundations", 0, "Scope, sourcing and hardware budget",
     "Governance docs, motivating figures, laptop benchmark", DONE),
    ("Foundations", 1, "Data layer",
     "Download, build periodic graphs, cache, leakage-aware splits", ACTIVE),
    ("Foundations", 2, "Chemical descriptors and baselines",
     "The bar every neural network has to clear", TODO),
    ("Models", 3, "CGCNN, written from scratch",
     "One architecture understood completely, with invariance tests", TODO),
    ("Models", 4, "Controlled architecture comparison",
     "MPNN, MEGNet, ALIGNN, GATv2 under one identical budget", TODO),
    ("The question", 5, "Descriptor–GNN fusion",
     "Does chemistry added to structure help? Where, and by how much?", TODO),
    ("The question", 6, "Interpretability, done correctly",
     "Real attention, attribution, and a randomisation sanity check", TODO),
    ("The question", 7, "Catalysis targets",
     "Adsorption energy, then a derived activity descriptor", TODO),
    ("Surface", 8, "Interactive app",
     "Crystal → graph → embedding → prediction, precomputed", TODO),
    ("Surface", 9, "Rigor pass",
     "Error analysis, calibration, out-of-distribution, model cards", TODO),
    ("Surface", 10, "Portfolio integration",
     "Feed learned embeddings into CatalystBO's search loop", TODO),
]

BLOCK_COLOR = {
    "Foundations": STRUCTURE,
    "Models": COMPOSITION,
    "The question": FUSED,
    "Surface": MUTED,
}


def main() -> None:
    use_house_style()
    fig, ax = plt.subplots(figsize=(13.6, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(len(PHASES) - 0.2 - (len(PHASES) - 1) * 0.80 - 0.55, len(PHASES) + 1.35)
    ax.axis("off")

    row_h = 0.80
    top = len(PHASES) - 0.2

    seen_blocks: list[str] = []
    for i, (block, num, title, outcome, status) in enumerate(PHASES):
        y = top - i * row_h
        col = BLOCK_COLOR[block]
        st = STATUS_STYLE[status]

        if block not in seen_blocks:
            seen_blocks.append(block)
            ax.text(0.05, y, block.upper(), fontsize=9.5,
                    fontweight="bold", color=col, va="center", ha="left")

        ax.add_patch(
            FancyBboxPatch(
                (2.35, y - row_h * 0.34), 10.05, row_h * 0.70,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=col, alpha=0.07 if status == TODO else 0.14,
                edgecolor=col, linewidth=1.1 if status == TODO else 1.6, zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (1.55, y - row_h * 0.28), 0.62, row_h * 0.58,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=col, alpha=0.9 if status != TODO else 0.35,
                edgecolor="none", zorder=3,
            )
        )
        ax.text(1.86, y, str(num), fontsize=11, fontweight="bold",
                color="white", va="center", ha="center", zorder=4)

        ax.text(2.60, y + 0.13, title, fontsize=10.5, fontweight="bold",
                color=INK if status != TODO else MUTED, va="center", ha="left", zorder=4)
        ax.text(2.60, y - 0.16, outcome, fontsize=8.8, color=MUTED,
                va="center", ha="left", zorder=4)

        ax.add_patch(
            FancyBboxPatch(
                (12.55, y - 0.15), 1.32, 0.30,
                boxstyle="round,pad=0.01,rounding_size=0.14",
                facecolor=st["face"], alpha=0.90 if status != TODO else 0.22,
                edgecolor="none", zorder=3,
            )
        )
        ax.text(13.21, y, st["label"], fontsize=8.2, fontweight="bold",
                color="white" if status != TODO else MUTED,
                va="center", ha="center", zorder=4)

    ax.text(
        0.05, top + 0.95,
        "Minimum useful version = phases 0–3 and 5.  Phases 4, 6 and 8 are amplifiers, not prerequisites.\n"
        "Every model is trained on one Ryzen 5 laptop CPU, under an identical fixed compute budget.",
        fontsize=9.5, color=MUTED, va="bottom", ha="left",
    )

    fig.suptitle(
        "Build plan — and what is actually finished",
        fontsize=15.5, fontweight="bold", y=0.985, x=0.006, ha="left",
    )
    caption(
        fig,
        "Status badges are generated from a single list in src/figures/fig_roadmap.py, so this figure cannot drift out of date relative to the README.",
        y=0.035,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig4_roadmap.png"
    fig.savefig(path)
    plt.close(fig)

    counts = {k: sum(1 for p in PHASES if p[4] == k) for k in (DONE, ACTIVE, TODO)}
    print(f"wrote {path}")
    print(f"  phases: {counts[DONE]} built, {counts[ACTIVE]} in progress, {counts[TODO]} not started")


if __name__ == "__main__":
    main()
