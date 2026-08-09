"""Figure 10 -- a crystal graph network drawn as a network.

Textbook neural-network diagrams are columns of circles with lines between them,
and the reason they are drawn that way is that it works: you can see the shape of
the computation without reading a single equation.

A graph network can be drawn the same way, and almost never is. The trick is that
the "layers" are not different sets of units -- they are the SAME atoms, redrawn
once per round of message passing, with a line wherever two atoms are bonded. Once
it is unrolled like that, the thing that makes a graph network different from an
ordinary network becomes visible: the wiring between layers is not all-to-all and
arbitrary, it is the actual bond network of the crystal.

Everything to the right of the pooling step is an ordinary neural network, which
is the second point worth seeing. The graph part is a featuriser; the prediction
is just a regression on what it produces.

Run with:  python -m src.figures.fig_gnn
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

# A five-atom toy crystal. Small enough that every line is followable, big
# enough that the neighbourhoods differ from atom to atom.
BONDS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
SPECIES = ["Ti", "O", "O", "Ti", "O"]
ATOM_COLOUR = {"Ti": STRUCTURE, "O": COMPOSITION}

N_ROUNDS = 3


def main() -> None:
    use_house_style()
    fig, ax = plt.subplots(figsize=(17.0, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 78)
    ax.axis("off")

    ys = np.array([66.0, 55.5, 45.0, 34.5, 24.0])       # one row per atom
    xs = [20.0, 33.0, 46.0, 59.0]                        # input + 3 rounds
    x_pool, x_hidden, x_out = 71.0, 82.5, 93.0

    # ---------------- the crystal itself --------------------------------
    ax.text(7.5, 71.5, "the crystal", fontsize=9.6, fontweight="bold",
            color=INK, ha="center", va="center")
    cpos = np.array([[0.0, 1.05], [-1.15, 0.15], [1.15, 0.15],
                     [-0.72, -1.05], [0.80, -1.05]])
    cx, cy, s = 7.5, 52.0, 5.4
    for a, b in BONDS:
        ax.plot([cx + cpos[a, 0] * s * 0.62, cx + cpos[b, 0] * s * 0.62],
                [cy + cpos[a, 1] * s, cy + cpos[b, 1] * s],
                color=MUTED, lw=1.8, alpha=0.65, zorder=3)
    for i, (dx, dy) in enumerate(cpos):
        ax.scatter([cx + dx * s * 0.62], [cy + dy * s], s=430,
                   c=[ATOM_COLOUR[SPECIES[i]]], edgecolors="white",
                   linewidths=1.6, zorder=5)
        ax.text(cx + dx * s * 0.62, cy + dy * s, SPECIES[i], fontsize=7.6,
                fontweight="bold", color="white", ha="center", va="center",
                zorder=6)
    ax.text(7.5, 32.0, "atoms and the\nbonds between them",
            fontsize=8.2, color=MUTED, ha="center", va="center", linespacing=1.5)

    ax.add_patch(FancyArrowPatch((13.5, 45.0), (16.4, 45.0), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=2.0, color=MUTED,
                                 zorder=4))

    # ---------------- wiring between rounds ------------------------------
    # A line is drawn only where a real bond exists. That is the whole
    # difference from an ordinary dense network, where every unit in one layer
    # connects to every unit in the next.
    for k in range(len(xs) - 1):
        for i in range(5):
            ax.plot([xs[k], xs[k + 1]], [ys[i], ys[i]], color=ACCENT,
                    lw=1.5, alpha=0.55, zorder=2)          # the atom itself
        for a, b in BONDS:
            for p, q in ((a, b), (b, a)):
                ax.plot([xs[k], xs[k + 1]], [ys[p], ys[q]], color=MUTED,
                        lw=1.0, alpha=0.42, zorder=2)

    # ---------------- the atom columns -----------------------------------
    titles = ["starting numbers", "after round 1", "after round 2", "after round 3"]
    subs = ["one row per atom", "knows its neighbours",
            "knows 2 bonds away", "knows 3 bonds away"]
    for k, x in enumerate(xs):
        ax.text(x, 71.5, titles[k], fontsize=9.2, fontweight="bold",
                color=INK, ha="center", va="center")
        ax.text(x, 68.4, subs[k], fontsize=7.8, color=MUTED,
                ha="center", va="center")
        for i in range(5):
            ax.scatter([x], [ys[i]], s=330,
                       c=[ATOM_COLOUR[SPECIES[i]] if k == 0 else ACCENT],
                       alpha=1.0 if k == 0 else 0.45 + 0.18 * k,
                       edgecolors="white", linewidths=1.5, zorder=6)
        ax.text(x, 18.6, f"({len(ys)}, 64)", fontsize=8.0, color=MUTED,
                ha="center", va="center", family="DejaVu Sans Mono")

    ax.annotate("", xy=(xs[3] + 2.0, 14.6), xytext=(xs[0] - 2.0, 14.6),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
    ax.text((xs[0] + xs[3]) / 2, 11.4,
            f"{N_ROUNDS} rounds of message passing — the same atoms every time, "
            "wired by the actual bonds",
            fontsize=8.6, fontweight="bold", color=ACCENT, ha="center", va="center")

    # ---------------- pooling --------------------------------------------
    for i in range(5):
        ax.plot([xs[3], x_pool], [ys[i], 45.0], color=COMPOSITION, lw=1.3,
                alpha=0.5, zorder=2)
    ax.scatter([x_pool], [45.0], s=760, c=[COMPOSITION], edgecolors="white",
               linewidths=2.0, zorder=6)
    ax.text(x_pool, 45.0, "avg", fontsize=8.4, fontweight="bold", color="white",
            ha="center", va="center", zorder=7)
    ax.text(x_pool, 71.5, "average", fontsize=9.2, fontweight="bold", color=INK,
            ha="center", va="center")
    ax.text(x_pool, 68.4, "one vector per crystal", fontsize=7.8, color=MUTED,
            ha="center", va="center")
    ax.text(x_pool, 18.6, "(1, 64)", fontsize=8.0, color=MUTED,
            ha="center", va="center", family="DejaVu Sans Mono")
    ax.text(x_pool, 36.0, "mean, not sum —\ndouble the cell and\nthe answer must\nnot move",
            fontsize=7.5, color=MUTED, ha="center", va="center", linespacing=1.5)

    # ---------------- dense head -----------------------------------------
    hidden_y = np.linspace(58.0, 32.0, 6)
    for hy in hidden_y:
        ax.plot([x_pool, x_hidden], [45.0, hy], color=WARN, lw=1.1,
                alpha=0.45, zorder=2)
        ax.plot([x_hidden, x_out], [hy, 45.0], color=WARN, lw=1.1,
                alpha=0.45, zorder=2)
        ax.scatter([x_hidden], [hy], s=250, c=[WARN], alpha=0.8,
                   edgecolors="white", linewidths=1.3, zorder=6)
    ax.text(x_hidden, 71.5, "dense layer", fontsize=9.2, fontweight="bold",
            color=INK, ha="center", va="center")
    ax.text(x_hidden, 68.4, "128 units", fontsize=7.8, color=MUTED,
            ha="center", va="center")
    ax.text(x_hidden, 27.0, "an ordinary\nneural network\nfrom here on",
            fontsize=7.8, color=WARN, fontweight="bold", ha="center",
            va="center", linespacing=1.5)

    ax.scatter([x_out], [45.0], s=560, c=[INK], edgecolors="white",
               linewidths=2.0, zorder=6)
    ax.text(x_out, 71.5, "output", fontsize=9.2, fontweight="bold", color=INK,
            ha="center", va="center")
    ax.text(x_out, 68.4, "one number", fontsize=7.8, color=MUTED,
            ha="center", va="center")
    ax.text(x_out, 39.4, "band gap\nin eV", fontsize=8.6, fontweight="bold",
            color=INK, ha="center", va="center", linespacing=1.5)

    # ---------------- what one arrow means -------------------------------
    ax.add_patch(FancyBboxPatch(
        (3.0, 1.0), 94.0, 8.4, boxstyle="round,pad=0.25,rounding_size=0.7",
        facecolor=LIGHT, alpha=0.55, edgecolor=MUTED, linewidth=1.2, zorder=2))
    ax.text(5.5, 7.2, "What one round actually does to one atom:",
            fontsize=9.0, fontweight="bold", color=INK, ha="left", va="center",
            zorder=6)
    ax.text(5.5, 3.6,
            "collect a message from every bonded neighbour  →  add them all up  →  add the atom's own numbers back in  →  that is its new row.",
            fontsize=8.6, color=INK, ha="left", va="center", zorder=6)
    ax.text(96.5, 3.6, "the four architectures differ\nonly in how that message is built  →",
            fontsize=8.0, color=FUSED, fontweight="bold", ha="right",
            va="center", zorder=6, linespacing=1.5)

    fig.suptitle("A crystal graph network, drawn as a network",
                 fontsize=16.5, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.951,
             "The columns are not different layers of units — they are the same five atoms, redrawn after each round of message passing.",
             fontsize=9.6, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "In an ordinary neural network every unit in one column connects to every unit in the next, and the wiring carries no meaning. Here a line is drawn only where two\n"
        "atoms are actually bonded, so the network's structure IS the crystal's structure — a different crystal is a differently wired network. Green lines are each atom keeping\n"
        "its own information; grey lines are information arriving from a neighbour. Everything right of the orange node is a conventional regression on a 64-number summary.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig10_gnn_as_a_network.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
