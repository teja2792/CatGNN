"""Shared plotting style for every figure in this repo.

One place to change colours and fonts so all figures look like they belong
to the same project. Imported by every script in ``src/figures/``.
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palette
#
# Chosen to stay distinguishable in greyscale and for the most common forms of
# colour vision deficiency (deuteranopia / protanopia). Figures in a README get
# printed, screenshotted, and viewed on bad monitors; the palette should survive
# all three.
# ---------------------------------------------------------------------------

INK = "#1b1b1f"        # near-black, body text and axes
MUTED = "#6b7280"      # secondary text, gridlines
STRUCTURE = "#1f6f8b"  # teal  -> anything about crystal STRUCTURE
COMPOSITION = "#c2571a"  # orange -> anything about COMPOSITION / chemistry
FUSED = "#6a4c93"      # purple -> structure + chemistry combined
ACCENT = "#2f8f5b"     # green  -> measured / experimental data
WARN = "#b3282d"       # red    -> caveats, failure cases, error floors
LIGHT = "#e8eaed"      # fills

# Provenance tiers -- used by the data-provenance figure and kept consistent
# with DATA_GROUNDING.md so the colours mean the same thing everywhere.
TIER_COLORS = {
    "measured": ACCENT,      # Tier 1e -- someone measured this in a lab
    "calculated": STRUCTURE,  # Tier 1  -- DFT, computed by a cited source
    "derived": FUSED,        # Tier 2  -- computed by us from Tier 1 via a cited equation
    "synthetic": WARN,       # Tier 3  -- verification harness only, never a claim
}


def use_house_style() -> None:
    """Apply the repo-wide matplotlib style. Call once at the top of a figure script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.8,
        }
    )


def caption(fig, text: str, y: float = -0.02) -> None:
    """Attach a plain-language caption under a figure.

    Every figure in this repo carries one. A chemical engineer should be able to
    read the caption alone and know what they are looking at, without the README
    body text around it.
    """
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
        wrap=True,
    )


def source_stamp(fig, text: str) -> None:
    """Stamp the data provenance onto the figure itself.

    Figures get copied out of READMEs and pasted into slides, where they lose
    their surrounding context. The provenance should travel with the pixels.
    """
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7.5, color=MUTED)


__all__ = [
    "use_house_style",
    "caption",
    "source_stamp",
    "INK",
    "MUTED",
    "STRUCTURE",
    "COMPOSITION",
    "FUSED",
    "ACCENT",
    "WARN",
    "LIGHT",
    "TIER_COLORS",
]
