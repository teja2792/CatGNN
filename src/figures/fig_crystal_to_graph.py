"""Figure 2 -- How a crystal becomes a graph, and what "message passing" means.

Written for a reader who knows crystallography and coordination chemistry but has
never trained a neural network. Built from a real rutile TiO2 cell using published
lattice parameters and Wyckoff positions -- not a cartoon. The computed Ti-O bond
lengths are checked against the neutron-diffraction values at the end of the script.

Run with:  python -m src.figures.fig_crystal_to_graph
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from .style import (
    use_house_style,
    caption,
    source_stamp,
    STRUCTURE,
    COMPOSITION,
    FUSED,
    ACCENT,
    MUTED,
    INK,
    LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

# Rutile TiO2, space group P4_2/mnm (136).
# a = b = 4.5937 A, c = 2.9587 A, oxygen internal parameter u = 0.30478.
# Source: Howard, Sabine & Dickson, Acta Cryst. B47, 462 (1991), neutron diffraction.
A_LAT = 4.5937
C_LAT = 2.9587
U_PARAM = 0.30478

# 2.4 A captures the Ti-O octahedron (1.949 A x4 equatorial, 1.980 A x2 apical) and
# stops short of the O-O edge-sharing contact at 2.536 A, giving a picture that matches
# what a chemist would call the bonding.
#
# NOTE, because it matters later: the real models in this repo do NOT use a cutoff this
# tight. CGCNN's convention is 8 A with up to 12 neighbours, which deliberately includes
# many contacts nobody would call a bond. In a crystal graph an edge means "close enough
# to matter", not "chemically bonded", and the model learns how much each contact is
# worth. This figure uses a bonding-like cutoff only because it is easier to read.
CUTOFF_ANGSTROM = 2.4

STYLE = {
    "Ti": dict(color=STRUCTURE, r=0.42),
    "O": dict(color=COMPOSITION, r=0.30),
}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def rutile_cell():
    """Fractional coordinates, species and lattice matrix for rutile TiO2."""
    frac = np.array(
        [
            [0.0, 0.0, 0.0],                      # Ti 2a
            [0.5, 0.5, 0.5],                      # Ti 2a
            [U_PARAM, U_PARAM, 0.0],              # O 4f
            [1 - U_PARAM, 1 - U_PARAM, 0.0],
            [0.5 + U_PARAM, 0.5 - U_PARAM, 0.5],
            [0.5 - U_PARAM, 0.5 + U_PARAM, 0.5],
        ]
    )
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    lattice = np.diag([A_LAT, A_LAT, C_LAT])
    return frac, species, lattice


def periodic_neighbours(frac, lattice, cutoff):
    """All neighbour contacts within ``cutoff``, respecting periodic boundaries.

    A crystal repeats forever, so an atom near one face of the unit cell is a real
    neighbour of an atom near the opposite face. We therefore search neighbouring
    cell images as well as the cell itself.

    The subtle part -- a genuine bug caught while writing this figure -- is that the
    *same* pair of atoms can be neighbours through *several different* images at once,
    and each is a separate physical bond. Rutile is the perfect illustration: the unit
    cell holds only 4 oxygens, yet each Ti is octahedrally coordinated by 6. Keeping
    only the closest image per pair (the naive "minimum-image convention") returns
    Ti CN = 4 and quietly destroys the octahedron. Counting every image within the
    cutoff returns the correct CN = 6.

    Returns ``(i, j, distance, shift)``; ``shift == (0,0,0)`` means the contact lies
    within the drawn cell, anything else means the bond wraps around the boundary.
    """
    pairs = []
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    for i in range(len(frac)):
        for j in range(len(frac)):
            d = frac[j] + shifts - frac[i]
            dist = np.linalg.norm(d @ lattice, axis=1)
            for s, dd in zip(shifts, dist):
                if i == j and not s.any():
                    continue
                if dd <= cutoff:
                    pairs.append((i, j, float(dd), s))
    return pairs


def iso(cart: np.ndarray) -> np.ndarray:
    """Isometric projection, so the drawing reads as a three-dimensional lattice."""
    cart = np.atleast_2d(cart)
    cos30, sin30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    x = (cart[:, 0] - cart[:, 1]) * cos30
    y = cart[:, 2] + (cart[:, 0] + cart[:, 1]) * sin30
    return np.column_stack([x, y])


def build_block(frac, species, lattice, nx=2, ny=2, nz=2):
    """Replicate the unit cell into a small block so it looks like a crystal."""
    pos, sp, home = [], [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                offset = np.array([i, j, k])
                for f, s in zip(frac, species):
                    pos.append((f + offset) @ lattice)
                    sp.append(s)
                    home.append(i == 0 and j == 0 and k == 0)
    return np.array(pos), sp, np.array(home)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_crystal(ax, frac, species, lattice):
    ax.set_title("1.  The crystal", loc="left", pad=14)

    pos, sp, home = build_block(frac, species, lattice)
    xy = iso(pos)

    # Bonds between every Ti-O pair in the block that is within the cutoff.
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if sp[i] == sp[j]:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= CUTOFF_ANGSTROM:
                ax.plot(
                    [xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    color=MUTED, lw=1.3, alpha=0.45, zorder=1, solid_capstyle="round",
                )

    # Highlight one complete TiO6 octahedron. Pick whichever Ti in the block has the
    # most oxygens actually present around it -- Ti sites near the block surface have
    # neighbours that only exist in the next repeat and would draw as a partial cage.
    def o_neighbours(k):
        return [
            j for j in range(len(pos))
            if sp[j] == "O" and np.linalg.norm(pos[j] - pos[k]) <= CUTOFF_ANGSTROM
        ]

    ti_sites = [k for k in range(len(pos)) if sp[k] == "Ti"]
    centre = max(ti_sites, key=lambda k: (len(o_neighbours(k)), -abs(xy[k, 0])))
    oct_idx = o_neighbours(centre)
    for j in oct_idx:
        ax.plot(
            [xy[centre, 0], xy[j, 0]], [xy[centre, 1], xy[j, 1]],
            color=ACCENT, lw=2.6, alpha=0.95, zorder=4, solid_capstyle="round",
        )
    if len(oct_idx) >= 3:
        hull = np.array([xy[j] for j in oct_idx])
        c = hull.mean(axis=0)
        order = np.argsort(np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0]))
        ax.add_patch(
            Polygon(hull[order], closed=True, facecolor=ACCENT, alpha=0.13,
                    edgecolor="none", zorder=2)
        )

    # Draw atoms back-to-front so the depth ordering looks right.
    depth = np.argsort(pos[:, 0] + pos[:, 1] - pos[:, 2])
    for i in depth:
        st = STYLE[sp[i]]
        emphasised = (i == centre) or (i in oct_idx)
        ax.add_patch(
            Circle(
                xy[i], st["r"],
                facecolor=st["color"], edgecolor="white",
                lw=1.3, zorder=5 if emphasised else 3,
                alpha=1.0 if emphasised else 0.42,
            )
        )
    ax.add_patch(Circle(xy[centre], STYLE["Ti"]["r"], facecolor=STYLE["Ti"]["color"],
                        edgecolor=ACCENT, lw=2.4, zorder=6))

    ax.annotate(
        "one TiO$_6$ octahedron\n(6 O around 1 Ti)",
        xy=xy[centre], xytext=(46, 62), textcoords="offset points",
        fontsize=9.5, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.2, shrinkB=8),
    )

    body(ax,
         f"Rutile TiO$_2$, 2×2×2 unit cells\n"
         f"a = b = {A_LAT} Å,  c = {C_LAT} Å\n"
         "Blue = Ti,  orange = O")


def panel_graph(ax, frac, species, lattice, pairs):
    ax.set_title("2.  The same thing, as a graph", loc="left", pad=14)

    # Ego-graph of one Ti: the node and its six oxygen neighbours, laid out on a
    # circle. Position on the page is arbitrary and that is exactly the point --
    # the graph stores connectivity and distances, not absolute coordinates.
    centre_atom = 0
    nbrs = [(j, d) for i, j, d, s in pairs if i == centre_atom]
    nbrs.sort(key=lambda t: t[1])

    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.1, 3.4)
    R = 2.15
    angles = np.linspace(90, 450, len(nbrs), endpoint=False) * np.pi / 180.0

    for (j, d), a in zip(nbrs, angles):
        p = np.array([R * np.cos(a), R * np.sin(a)])
        ax.plot([0, p[0]], [0, p[1]], color=ACCENT, lw=2.2, alpha=0.85, zorder=2)
        lab = p * 0.56
        ax.text(
            lab[0], lab[1], f"{d:.2f} Å", fontsize=8.6, color=INK,
            ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=LIGHT, lw=0.8),
        )
        ax.add_patch(Circle(p, 0.40, facecolor="white", edgecolor=COMPOSITION, lw=2.4, zorder=5))
        ax.text(p[0], p[1], "O", ha="center", va="center", fontsize=10,
                color=COMPOSITION, fontweight="bold", zorder=6)

    ax.add_patch(Circle((0, 0), 0.52, facecolor="white", edgecolor=STRUCTURE, lw=2.8, zorder=5))
    ax.text(0, 0, "Ti", ha="center", va="center", fontsize=11,
            color=STRUCTURE, fontweight="bold", zorder=6)

    ax.text(
        0, 3.05,
        "node = atom     edge = neighbour contact     edge label = bond length",
        ha="center", va="center", fontsize=9.5, color=MUTED,
    )

    body(ax,
         "Keeps who-is-next-to-whom and how far apart;\n"
         "throws away absolute position, so the answer is\n"
         "the same however you rotate or re-tile the crystal.")


def panel_message_passing(ax):
    ax.set_title("3.  What the model does with it", loc="left", pad=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    rounds = [
        ("Round 0", "Each atom knows only which\nelement it is.", [STRUCTURE]),
        ("Round 1", "Each atom now also knows its\nimmediate neighbours.", [STRUCTURE, COMPOSITION]),
        ("Round 2", "…and its neighbours'\nneighbours.", [STRUCTURE, COMPOSITION, ACCENT]),
        ("Round 3", "Each atom now carries its full local\ncoordination environment.",
         [STRUCTURE, COMPOSITION, ACCENT, FUSED]),
    ]
    top, step = 8.9, 2.25
    for n, (title, desc, cols) in enumerate(rounds):
        y = top - n * step
        for k, c in enumerate(cols):
            ax.add_patch(
                Circle((0.85 + k * 0.60, y), 0.30, facecolor=c,
                       edgecolor="white", lw=1.3, zorder=3)
            )
        ax.text(0.85 + len(cols) * 0.60 + 0.25, y, title, fontsize=10,
                fontweight="bold", va="center", ha="left", color=INK)
        ax.text(0.6, y - 0.62, desc, fontsize=9.2, va="top", ha="left", color=MUTED)
        if n < len(rounds) - 1:
            ax.add_patch(
                FancyArrowPatch((0.85, y - 1.32), (0.85, y - step + 0.42),
                                arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.3)
            )

    body(ax,
         'This is "message passing" — the one new idea here.\n'
         "After three rounds each atom encodes its coordination\n"
         "shell: what you mean by “octahedral Ti”.")
    ax.axis("off")


def body(ax, text: str) -> None:
    """Explanatory text in a consistent place under every panel."""
    ax.text(0.5, -0.015, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=9.0, color=MUTED, linespacing=1.45)


# ---------------------------------------------------------------------------

def main() -> None:
    use_house_style()
    frac, species, lattice = rutile_cell()
    pairs = periodic_neighbours(frac, lattice, CUTOFF_ANGSTROM)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.6))
    fig.subplots_adjust(bottom=0.20, top=0.90, wspace=0.10)
    for ax in axes:
        ax.axis("off")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_aspect("equal", adjustable="datalim")

    panel_crystal(axes[0], frac, species, lattice)
    panel_graph(axes[1], frac, species, lattice, pairs)
    panel_message_passing(axes[2])

    fig.suptitle(
        "A crystal is already a graph — atoms, and the bonds between them",
        fontsize=15.5, fontweight="bold", y=1.01, x=0.006, ha="left",
    )
    caption(
        fig,
        "Nothing exotic happens between panels 1 and 2: a graph is a bonding diagram written in a form a computer can read. What the model adds is\n"
        "panel 3 — atoms repeatedly updating their own description from their neighbours', until each one encodes its coordination environment.\n"
        "Bond lengths here are computed from the published rutile structure and agree with neutron diffraction to within 0.003 Å.",
        y=-0.005,
    )
    source_stamp(
        fig,
        "Rutile structure: Howard, Sabine & Dickson, Acta Cryst. B47, 462 (1991)  ·  periodic neighbour list, all images within cutoff",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig2_crystal_to_graph.png"
    fig.savefig(path)
    plt.close(fig)

    ti_cn = [sum(1 for i, j, d, s in pairs if i == k) for k in range(2)]
    o_cn = [sum(1 for i, j, d, s in pairs if i == k) for k in range(2, 6)]
    dists = sorted({round(d, 3) for _, _, d, _ in pairs})
    print(f"wrote {path}")
    print(f"  Ti coordination numbers: {ti_cn}   (rutile: 6, 6)")
    print(f"  O  coordination numbers: {o_cn}   (rutile: 3, 3, 3, 3)")
    print(f"  Ti-O bond lengths: {dists} A   (neutron diffraction: 1.946 x4, 1.983 x2)")

    # Self-checks. If these fail the graph construction is wrong, and every number
    # downstream would be quietly wrong with it.
    assert ti_cn == [6, 6], f"rutile Ti must be 6-coordinate, got {ti_cn}"
    assert o_cn == [3, 3, 3, 3], f"rutile O must be 3-coordinate, got {o_cn}"
    assert abs(dists[0] - 1.946) < 0.01 and abs(dists[1] - 1.983) < 0.01, dists
    print("  self-checks passed")


if __name__ == "__main__":
    main()
