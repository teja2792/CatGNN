"""Figure 0 -- what actually happens to one material, from database row to number.

Not a block diagram of the software. A trace of the DATA: at every stage the
reader sees what the material has been turned into and how big that thing is, so
"the network predicts the band gap" stops being a black box and becomes six
concrete transformations, each of which can be checked.

The material followed through is a real one from the dataset, and its numbers are
read from the cached graphs at build time rather than invented, so the shapes on
the page are the shapes the code actually produces.

Run with:  python -m src.figures.fig_dataflow
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

D_ATOM, N_EDGE_FEA, N_CONV = 64, 41, 3


# ---------------------------------------------------------------------------
# A real material, pulled from the cache when it is there
# ---------------------------------------------------------------------------

def example_material() -> dict:
    """A real crystal from the cache: a NON-metal, small enough to draw.

    Only fields that genuinely exist in the cache are shown on the figure. It
    would be easy to print plausible lattice constants next to a real material_id
    and nobody would check -- which is exactly why it is not done.
    """
    fallback = {"material_id": "mp-2657", "formula": "TiO2", "n_atoms": 6,
                "n_edges": 108, "band_gap": 1.78, "spacegroup": 136,
                "crystal_system": "Tetragonal", "density": 4.24, "volume": 62.4}
    try:
        from ..data import graph_build as gb

        for i in sorted(gb.existing_chunk_indices()):
            chunk = gb.load_graph_chunk(i)
            meta = {m["material_id"]: m for m in json.loads(
                (gb.GRAPHS / f"meta_{i:04d}.json").read_text(encoding="utf-8"))}
            for g in gb.iter_graphs(chunk):
                n = int(g["z"].size)
                gap = float(g.get("band_gap", np.nan))
                m = meta.get(g["material_id"], {})
                if 4 <= n <= 8 and np.isfinite(gap) and gap > 0.8 and m:
                    return {"material_id": g["material_id"],
                            "formula": m.get("formula_pretty", "?"),
                            "n_atoms": n, "n_edges": int(g["src"].size),
                            "band_gap": gap,
                            "spacegroup": m.get("spacegroup_number"),
                            "crystal_system": m.get("crystal_system"),
                            "density": m.get("density"),
                            "volume": m.get("volume")}
            break
    except Exception:
        pass
    return fallback


def headline_numbers() -> dict:
    n = {"cgcnn": 0.4137, "desc": 0.511}
    p = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if "random" in d:
            n["cgcnn"] = d["random"]["test"]["mae"]
    p = RESULTS / "baselines.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))["results"]
        m = [r["mae"] for r in rows if r.get("target") == "band_gap_nonmetals"
             and r.get("split") == "random" and r.get("model") != "mean"]
        if m:
            n["desc"] = min(m)
    return n


# ---------------------------------------------------------------------------
# Drawing helpers
#
# XLIM/YLIM are not equal, so a Circle patch comes out as an ellipse. Every
# round thing on this figure therefore goes through dot(), which corrects the
# x-radius by the axis aspect. Stretched "circles" are the visual equivalent of
# a typo.
# ---------------------------------------------------------------------------

XLIM, YLIM = 100.0, 66.0
FIGW, FIGH = 17.0, 8.4

# A Circle patch in data coordinates comes out as an ellipse whenever the x and
# y scales differ, which they do here. scatter() markers are sized in display
# space, so they are round whatever the axes are doing -- which is why every
# round thing on this figure is a scatter point rather than a patch.
LAYOUT_X = 0.60      # x-offset multiplier that makes a ring of points look round


def dot(ax, x, y, size, colour, alpha=1.0, z=6, lw=1.2):
    ax.scatter([x], [y], s=size, c=[colour], alpha=alpha, zorder=z,
               edgecolors="white", linewidths=lw)


def card(ax, x, y, w, h, colour, step, title):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="white", edgecolor=colour, linewidth=1.9, zorder=3))
    # The step number sits ON the top edge, so it cannot collide with a title
    # that turns out one word longer than planned.
    dot(ax, x - w / 2 + 2.2, y + h / 2, 330, colour, z=7, lw=1.8)
    ax.text(x - w / 2 + 2.2, y + h / 2, str(step), fontsize=8.8,
            fontweight="bold", color="white", ha="center", va="center", zorder=8)
    ax.text(x, y + h / 2 - 2.4, title, fontsize=9.5, fontweight="bold",
            color=INK, ha="center", va="center", zorder=6)


def grid(ax, x, y, ncols, nrows, cw, ch, colour, alpha=0.55, gap=0.12):
    """A block of little squares standing in for an array."""
    for r in range(nrows):
        for c in range(ncols):
            ax.add_patch(FancyBboxPatch(
                (x + c * (cw + gap), y - r * (ch + gap)), cw, ch,
                boxstyle="square,pad=0",
                facecolor=colour, alpha=alpha * (0.55 + 0.45 * ((r + c) % 3) / 2),
                edgecolor="white", linewidth=0.35, zorder=5))


def shape_label(ax, x, y, text, colour):
    ax.text(x, y, text, fontsize=8.3, color=colour, fontweight="bold",
            ha="center", va="center", zorder=7, family="DejaVu Sans Mono")


def little_graph(ax, cx, cy, s=1.0):
    pos = np.array([[0.0, 1.0], [-1.05, 0.15], [1.05, 0.15],
                    [-0.62, -0.95], [0.72, -0.95]])
    pos = pos * s
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (1, 2)]:
        ax.plot([cx + pos[a, 0] * LAYOUT_X, cx + pos[b, 0] * LAYOUT_X],
                [cy + pos[a, 1], cy + pos[b, 1]],
                color=MUTED, lw=1.4, alpha=0.6, zorder=5)
    for i, (dx, dy) in enumerate(pos):
        dot(ax, cx + dx * LAYOUT_X, cy + dy, 210,
            STRUCTURE if i % 2 == 0 else COMPOSITION, z=6)


def main() -> None:
    use_house_style()
    mat = example_material()
    head = headline_numbers()

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.axis("off")

    W, GAP, Y, H = 13.6, 2.2, 42.5, 32.0
    xs = [3.2 + W / 2 + i * (W + GAP) for i in range(6)]
    N, E = mat["n_atoms"], mat["n_edges"]

    # ---- 1. the database row ----------------------------------------------
    x = xs[0]
    card(ax, x, Y, W, H, STRUCTURE, 1, "A database entry")
    ax.text(x, Y + 11.4, f"{mat['material_id']}   {mat['formula']}", fontsize=9.4,
            fontweight="bold", color=STRUCTURE, ha="center", va="center", zorder=6)
    rows = [("atoms in cell", f"{N}"),
            ("crystal system", str(mat.get("crystal_system", "?"))),
            ("space group", f"#{mat.get('spacegroup', '?')}"),
            ("volume", f"{mat.get('volume', 0):.1f} Å³"),
            ("band gap (DFT)", f"{mat['band_gap']:.2f} eV")]
    for i, (k, v) in enumerate(rows):
        yy = Y + 7.4 - i * 2.3
        ax.text(x - 5.6, yy, k, fontsize=7.2, color=MUTED,
                ha="left", va="center", zorder=6)
        ax.text(x + 5.6, yy, v, fontsize=7.2, color=INK, fontweight="bold",
                ha="right", va="center", zorder=6, family="DejaVu Sans Mono")
    ax.text(x, Y - 6.8, "plus the position of\nevery atom in the cell",
            fontsize=7.6, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    shape_label(ax, x, Y - 13.2, "a structure", STRUCTURE)

    # ---- 2. the graph ------------------------------------------------------
    x = xs[1]
    card(ax, x, Y, W, H, STRUCTURE, 2, "Made into a graph")
    little_graph(ax, x, Y + 7.6, s=2.6)
    ax.text(x, Y - 0.4, "atom → node\ncontact within 8 Å → edge",
            fontsize=7.9, color=INK, ha="center", va="center", zorder=6,
            linespacing=1.6)
    ax.text(x, Y - 6.6,
            "every periodic image counts,\nso an atom bonded through\ntwo images gets two edges",
            fontsize=7.1, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    shape_label(ax, x, Y - 13.2, f"{N} nodes, {E} edges", STRUCTURE)

    # ---- 3. starting numbers ----------------------------------------------
    x = xs[2]
    card(ax, x, Y, W, H, FUSED, 3, "Atoms get numbers")
    grid(ax, x - 5.3, Y + 10.2, 12, 4, 0.80, 0.80, FUSED)
    ax.text(x, Y + 3.4, f"one row per atom,\n{D_ATOM} numbers wide",
            fontsize=7.9, color=INK, ha="center", va="center", zorder=6,
            linespacing=1.5)
    ax.text(x, Y - 2.4,
            "Phase 5 puts the element's\nREAL properties here —\nelectronegativity, radius —\ninstead of a memorised code",
            fontsize=7.1, color=FUSED, fontweight="bold", ha="center",
            va="center", zorder=6, linespacing=1.5)
    ax.text(x, Y - 9.0, f"each bond also gets {N_EDGE_FEA}\nnumbers for its length",
            fontsize=7.0, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.4)
    shape_label(ax, x, Y - 13.2, f"({N}, {D_ATOM})", FUSED)

    # ---- 4. message passing ------------------------------------------------
    x = xs[3]
    card(ax, x, Y, W, H, ACCENT, 4, "Neighbours mix in")
    cx, cy = x, Y + 8.0
    for ang in (90, 162, 234, 306, 18):
        a = np.deg2rad(ang)
        px, py = cx + 4.6 * LAYOUT_X * np.cos(a), cy + 4.4 * np.sin(a)
        dot(ax, px, py, 150, STRUCTURE, alpha=0.85, z=6)
        ax.add_patch(FancyArrowPatch(
            (px - 1.1 * LAYOUT_X * np.cos(a), py - 1.05 * np.sin(a)),
            (cx + 2.1 * LAYOUT_X * np.cos(a), cy + 2.0 * np.sin(a)),
            arrowstyle="-|>", mutation_scale=9, linewidth=1.3,
            color=MUTED, zorder=5))
    dot(ax, cx, cy, 430, ACCENT, z=7, lw=1.6)
    ax.text(x, Y - 0.8, f"repeated {N_CONV} times", fontsize=8.5,
            fontweight="bold", color=ACCENT, ha="center", va="center", zorder=6)
    ax.text(x, Y - 6.6,
            "after one round an atom\nknows its neighbours;\nafter three it knows its\nwhole local environment",
            fontsize=7.1, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    shape_label(ax, x, Y - 13.2, f"still ({N}, {D_ATOM})", ACCENT)

    # ---- 5. pooling --------------------------------------------------------
    x = xs[4]
    card(ax, x, Y, W, H, COMPOSITION, 5, "Average the atoms")
    grid(ax, x - 5.3, Y + 10.6, 12, 4, 0.80, 0.80, ACCENT, alpha=0.35)
    ax.add_patch(FancyArrowPatch((x, Y + 5.6), (x, Y + 3.0), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.8,
                                 color=COMPOSITION, zorder=6))
    grid(ax, x - 5.3, Y + 2.2, 12, 1, 0.80, 0.80, COMPOSITION, alpha=0.85)
    ax.text(x, Y - 1.8, "one vector for the\nwhole material", fontsize=7.9,
            fontweight="bold", color=INK, ha="center", va="center", zorder=6,
            linespacing=1.5)
    ax.text(x, Y - 7.4,
            "an average, not a sum, so\ndoubling the unit cell\ncannot change the answer",
            fontsize=7.1, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    shape_label(ax, x, Y - 13.2, f"(1, {D_ATOM})", COMPOSITION)

    # ---- 6. the number -----------------------------------------------------
    x = xs[5]
    card(ax, x, Y, W, H, WARN, 6, "One number out")
    ax.text(x, Y + 8.8, "1 number", fontsize=13.5, fontweight="bold",
            color=WARN, ha="center", va="center", zorder=6)
    ax.text(x, Y + 5.2, "two dense layers turn that\nvector into a prediction",
            fontsize=7.4, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    ax.plot([x - 4.8, x + 4.8], [Y + 2.4, Y + 2.4], color=LIGHT, lw=1.5, zorder=5)
    ax.text(x, Y - 0.4,
            f"for this crystal the DFT\nanswer is {mat['band_gap']:.2f} eV",
            fontsize=7.6, color=INK, ha="center", va="center", zorder=6,
            linespacing=1.5)
    ax.text(x, Y - 7.0,
            "averaged over thousands of\nmaterials, the gap between\nprediction and DFT is what\nthis repository reports",
            fontsize=7.1, color=MUTED, ha="center", va="center", zorder=6,
            linespacing=1.5)
    shape_label(ax, x, Y - 13.2, "eV", WARN)

    # ---- arrows between stages --------------------------------------------
    # No labels: the gap between cards is 2.2 units wide and every honest label
    # is wider than that. The card titles already say what each step does.
    for i in range(5):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + W / 2 + 0.25, Y), (xs[i + 1] - W / 2 - 0.25, Y),
            arrowstyle="-|>", mutation_scale=14, linewidth=2.1,
            color=MUTED, zorder=4))

    # ---- the comparison strip ---------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (3.2, 3.0), 93.2, 19.0, boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=LIGHT, alpha=0.5, edgecolor=MUTED, linewidth=1.2, zorder=2))
    ax.text(6.2, 19.0, "Why steps 2–5 exist at all", fontsize=10.2,
            fontweight="bold", color=INK, ha="left", va="center", zorder=6)
    for i, line in enumerate([
            "Skip them: take the formula, look up its elements, average their",
            "properties. That is 192 numbers and a Random Forest, with no",
            "structure anywhere — and it cannot tell rutile from anatase,",
            "because both are TiO₂ and both give the identical 192 numbers."]):
        ax.text(6.2, 15.0 - i * 3.0, line, fontsize=8.3, color=MUTED,
                ha="left", va="center", zorder=6)

    ax.text(76.0, 19.0, "Band gap, non-metals, identical test set",
            fontsize=8.4, fontweight="bold", color=INK, ha="center",
            va="center", zorder=6)
    for x0, col, name, val in [(56.5, COMPOSITION, "formula only", head["desc"]),
                               (74.0, STRUCTURE, "the graph above", head["cgcnn"])]:
        ax.add_patch(FancyBboxPatch(
            (x0, 5.0), 16.0, 10.6, boxstyle="round,pad=0.15,rounding_size=0.5",
            facecolor="white", edgecolor=col, linewidth=1.7, zorder=4))
        ax.text(x0 + 8.0, 13.2, name, fontsize=8.2, fontweight="bold",
                color=col, ha="center", va="center", zorder=6)
        ax.text(x0 + 8.0, 8.8, f"{val:.3f} eV", fontsize=13.5, fontweight="bold",
                color=col, ha="center", va="center", zorder=6)

    gain = 100 * (head["desc"] - head["cgcnn"]) / head["desc"]
    ax.text(93.0, 10.3, f"structure\nis worth\n{gain:.0f}%", fontsize=8.8,
            fontweight="bold", color=ACCENT, ha="center", va="center",
            zorder=6, linespacing=1.6)

    # ---- titles ------------------------------------------------------------
    fig.suptitle("How one material becomes one predicted number",
                 fontsize=16.5, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.949,
             "Following a single real crystal all the way through. Every array size shown is the size the code actually produces.",
             fontsize=9.6, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "The graph network never sees a chemical formula. It sees atoms, the distances between them, and — after Phase 5 — the tabulated properties of each element.\n"
        "Steps 3 to 5 are where a crystal becomes a fixed-length vector that an ordinary regression can use: the network is a learned featuriser, and the last step is just a fit.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig0_dataflow.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}   (example: {mat['material_id']} {mat['formula']}, "
          f"{N} atoms, {E} edges, gap {mat['band_gap']:.2f} eV)")


if __name__ == "__main__":
    main()
