"""Figure 11 -- the four architectures, which differ in exactly one place.

Figure 10 shows that a round of message passing is: collect a message from every
bonded neighbour, add them up, add the atom's own numbers back in. Every
architecture in this repository agrees on that. They differ only in how loudly
each neighbour gets to speak.

That is a small enough difference to draw honestly on one page, and drawing it is
worth doing because the difference between a *gate* and *attention* -- which is
the distinction the whole field is casual about -- is entirely visible here. A
gate scores each bond on its own, so every bond can be wide open at once and the
weights need not sum to anything. Attention makes bonds compete for a fixed
budget, so the weights sum to one and can honestly be read as "the model looked
here rather than there".

The weights drawn are illustrative values chosen to show the mechanism. The
errors underneath each panel are measured.

Run with:  python -m src.figures.fig_message_rules
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

NEIGHBOURS = [(1.7, 7.6), (1.7, 4.0), (8.5, 5.8)]
CENTRE = (5.0, 5.8)

PANELS = [
    dict(key="mpnn", colour=MUTED, name="MPNN",
         headline="No weighting at all",
         weights=[1.0, 1.0, 1.0], show_numbers=False,
         rule="Every neighbour's message is added exactly as it arrives.",
         note="The control. Whatever CGCNN gains\nover this is what its gate is worth."),
    dict(key="cgcnn", colour=STRUCTURE, name="CGCNN",
         headline="A volume knob on each bond",
         weights=[0.90, 0.75, 0.30], show_numbers=True,
         rule="Each bond gets its own value between 0 and 1, decided on its own.",
         note="They do NOT compete — all three\ncan be wide open at once."),
    dict(key="gatv2", colour=FUSED, name="GATv2",
         headline="Bonds compete for one budget",
         weights=[0.55, 0.30, 0.15], show_numbers=True,
         rule="The weights are forced to add up to exactly 1 for every atom.",
         note="This is what “attention” means.\nOnly this one is a share-out."),
    dict(key="megnet", colour=COMPOSITION, name="MEGNet",
         headline="Plus a whole-crystal summary",
         weights=[1.0, 1.0, 1.0], show_numbers=False, global_state=True,
         rule="Unweighted, but every atom also sees a crystal-wide summary.",
         note="Lets an atom know it is in a dense\noxide without deducing it hop by hop."),
]


def measured() -> dict:
    p = RESULTS / "architecture_significance.json"
    if not p.exists():
        return {}
    blob = json.loads(p.read_text(encoding="utf-8"))
    return {"mae": blob["mae"], "params": blob["n_parameters"]}


def draw_panel(ax, spec, stats):
    colour = spec["colour"]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    ax.text(0.1, 10.1, spec["name"], fontsize=11.0, fontweight="bold",
            color=colour, ha="left", va="center")
    ax.text(0.1, 9.1, spec["headline"], fontsize=9.6, fontweight="bold",
            color=INK, ha="left", va="center")

    total = sum(spec["weights"])
    if spec["show_numbers"]:
        ax.text(9.9, 9.4, f"they add up to {total:.2f}", fontsize=8.4,
                fontweight="bold",
                color=WARN if abs(total - 1.0) > 1e-6 else ACCENT,
                ha="right", va="center")
        ax.text(9.9, 8.5,
                "— nothing forces this" if abs(total - 1.0) > 1e-6
                else "— always, by construction",
                fontsize=7.6, color=MUTED, ha="right", va="center")

    for (nx, ny), w in zip(NEIGHBOURS, spec["weights"]):
        ax.add_patch(FancyArrowPatch(
            (nx, ny), CENTRE, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.1 + 4.2 * w, color=colour,
            alpha=0.35 + 0.55 * w, zorder=4, shrinkA=12, shrinkB=21))
        ax.scatter([nx], [ny], s=260, c=[MUTED], alpha=0.85,
                   edgecolors="white", linewidths=1.4, zorder=6)
        if spec["show_numbers"]:
            mx, my = (nx + CENTRE[0]) / 2, (ny + CENTRE[1]) / 2
            ax.text(mx, my + 0.38, f"{w:.2f}", fontsize=9.0, fontweight="bold",
                    color=colour, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.20", fc="white",
                              ec="none", alpha=0.95))

    if spec.get("global_state"):
        # Below the centre, clear of both the rule text and the result box.
        ax.add_patch(FancyBboxPatch(
            (3.35, 2.60), 3.3, 1.20, boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor=colour, alpha=0.16, edgecolor=colour, linewidth=1.4,
            zorder=4))
        ax.text(5.0, 3.20, "whole-crystal\nsummary", fontsize=7.2,
                color=colour, fontweight="bold", ha="center", va="center",
                zorder=6, linespacing=1.35)
        ax.add_patch(FancyArrowPatch(
            (5.0, 3.90), (5.0, 4.90), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.9, color=colour, linestyle="--", zorder=4))

    ax.scatter([CENTRE[0]], [CENTRE[1]], s=980, c=[colour],
               edgecolors="white", linewidths=2.0, zorder=7)
    ax.text(CENTRE[0], CENTRE[1], "this\natom", fontsize=7.8,
            fontweight="bold", color="white", ha="center", va="center",
            zorder=8, linespacing=1.3)

    ax.text(0.1, 1.75, spec["rule"], fontsize=8.4, color=INK,
            ha="left", va="center")
    ax.text(0.1, 0.65, spec["note"], fontsize=7.8, color=MUTED,
            ha="left", va="center", linespacing=1.5)

    if stats and spec["key"] in stats.get("mae", {}):
        mae = stats["mae"][spec["key"]]
        par = stats["params"].get(spec["key"])
        ax.add_patch(FancyBboxPatch(
            (6.75, 0.15), 3.15, 1.85,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor="white", edgecolor=colour, linewidth=1.5, zorder=5))
        ax.text(8.32, 1.42, f"{mae:.3f} eV", fontsize=10.4, fontweight="bold",
                color=colour, ha="center", va="center", zorder=6)
        ax.text(8.32, 0.62, f"{par:,} parameters" if par else "measured",
                fontsize=7.0, color=MUTED, ha="center", va="center", zorder=6)


def main() -> None:
    use_house_style()
    stats = measured()

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.6))
    fig.subplots_adjust(left=0.035, right=0.975, top=0.83, bottom=0.13,
                        hspace=0.26, wspace=0.13)

    for ax, spec in zip(axes.ravel(), PANELS):
        draw_panel(ax, spec, stats)

    fig.suptitle("The four networks differ in exactly one place: how loudly each neighbour speaks",
                 fontsize=15.5, fontweight="bold", y=0.985, x=0.006, ha="left")
    fig.text(0.006, 0.930,
             "Everything else — the graph, the three rounds, the averaging, the prediction head — is identical across all four. "
             "Errors are measured on the same 4,308 test materials under the same 35-minute budget.",
             fontsize=9.4, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "The distinction that matters is between CGCNN and GATv2. CGCNN scores each bond on its own, so the weights need not add up to anything and every bond can be\n"
        "wide open at once — that is a GATE. GATv2 forces an atom's weights to sum to one, so bonds compete and the result really is a share-out — that is ATTENTION. CGCNN's\n"
        "gate is routinely called attention in write-ups, which licenses reading a picture of gate values as “where the model looked”. It does not support that reading.\n"
        "Measured outcome: deleting the gate entirely (MPNN) changes the error by 0.001 eV, and the two mechanisms that DO change the answer both make it worse.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig11_message_rules.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
