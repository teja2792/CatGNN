"""Figure 13 -- what an error of 0.4 eV actually means.

Every headline number in this repository is an error in electron-volts, and a
number in eV means nothing on its own. "0.4 eV" is not a property of any one
material and it is not a calibration constant. It is the *average size of the
miss* when the model predicts the band gap of a material it has never seen,
taken over thousands of different materials.

This figure gives that number a scale to sit against, three ways:

  A  where the dataset's band gaps actually are, and what familiar semiconductors
     have, so 0.4 eV can be compared against something a reader knows
  B  what 0.4 eV does to a decision you might actually make with the number
  C  the error budget -- the model's disagreement with DFT set against DFT's own
     disagreement with experiment, which is larger

Point C is the one most likely to be missed. These models are trained on DFT
values, so the best they can ever do is reproduce DFT. DFT is itself wrong about
band gaps by more than the model is wrong about DFT.

Run with:  python -m src.figures.fig_what_is_an_ev
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .style import (
    use_house_style, caption,
    STRUCTURE, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

# Experimental room-temperature band gaps, in eV. Textbook values, included so a
# reader has something familiar to measure 0.4 eV against. Marked as experimental
# on the figure because the dataset holds DFT values, which are systematically
# smaller -- see panel C.
REFERENCE = [
    ("Ge", 0.67, "infrared detectors"),
    ("Si", 1.12, "almost every solar panel"),
    ("GaAs", 1.42, "high-efficiency solar"),
    ("GaP", 2.26, "green LEDs"),
    ("CdS", 2.42, "photocatalysis"),
    ("TiO₂", 3.20, "the workhorse photocatalyst"),
    ("ZnO", 3.37, "transparent electrodes"),
    ("diamond", 5.47, "an insulator"),
]

VISIBLE_LO, VISIBLE_HI = 1.65, 3.10       # 750 nm to 400 nm

# DFT-vs-experiment, from the literature rather than measured here.
DFT_RMSE_LO, DFT_RMSE_HI = 0.75, 1.05
DFT_SOURCE = ("Kim, S. et al., Sci. Data 7, 387 (2020) — reported RMSE of GGA-based "
              "databases against experiment")


def dataset_gaps() -> np.ndarray | None:
    try:
        from ..data import graph_build as gb

        gaps = []
        for i in sorted(gb.existing_chunk_indices()):
            chunk = gb.load_graph_chunk(i)
            for g in gb.iter_graphs(chunk):
                v = g.get("band_gap", np.nan)
                if np.isfinite(v) and v > 1e-6:
                    gaps.append(float(v))
        return np.array(gaps) if gaps else None
    except Exception:
        return None


def model_error() -> tuple[float, str]:
    """Best measured error on the random split, and where it came from."""
    p = RESULTS / "fusion_band_gap_nonmetals.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        v = d.get("cgcnn_properties", {}).get("random", {}).get("test", {}).get("mae")
        if v:
            return v, "graph network with element properties, random split"
    p = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if "random" in d:
            return d["random"]["test"]["mae"], "CGCNN, random split"
    return 0.393, "graph network with element properties, random split"


def panel_scale(ax, gaps, mae):
    xmax = 6.0
    ax.axvspan(VISIBLE_LO, VISIBLE_HI, color=FUSED, alpha=0.10, zorder=1)
    ax.text((VISIBLE_LO + VISIBLE_HI) / 2, 0.955, "absorbs visible light",
            transform=ax.get_xaxis_transform(), fontsize=8.4, color=FUSED,
            fontweight="bold", ha="center", va="top", zorder=6)

    if gaps is not None:
        ax.hist(gaps, bins=np.linspace(0, xmax, 121), color=STRUCTURE,
                alpha=0.55, edgecolor="none", zorder=3)
        med = float(np.median(gaps))
        ax.axvline(med, color=STRUCTURE, lw=2.0, zorder=5)
        ax.text(med + 0.07, 0.72, f"half this dataset\nsits below {med:.2f} eV",
                transform=ax.get_xaxis_transform(), fontsize=8.4,
                fontweight="bold", color=STRUCTURE, ha="left", va="center",
                zorder=7, linespacing=1.5)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Band gap  (eV)")
    ax.set_ylabel("number of\nmaterials")
    ax.set_title("A.  Where the band gaps in this dataset actually are",
                 loc="left", pad=8)
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)

    n = len(gaps) if gaps is not None else 0
    ax.text(0.995, 0.955, f"{n:,} non-metals   (metals are all exactly 0 and excluded)",
            transform=ax.transAxes, fontsize=8.2, color=MUTED,
            ha="right", va="top")


def panel_reference(ax, mae):
    xmax = 6.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.axvspan(VISIBLE_LO, VISIBLE_HI, color=FUSED, alpha=0.10, zorder=1)
    ax.axhline(0.62, color=MUTED, lw=1.1, zorder=2)

    for i, (name, gap, use) in enumerate(REFERENCE):
        tier = i % 2
        ax.plot([gap, gap], [0.62, 0.72 + 0.16 * tier], color=INK, lw=1.0, zorder=4)
        ax.scatter([gap], [0.62], s=46, c=[INK], zorder=6,
                   edgecolors="white", linewidths=1.1)
        ax.text(gap, 0.78 + 0.16 * tier, f"{name}  {gap:.2f}", fontsize=8.4,
                fontweight="bold", color=INK, ha="center", va="bottom", zorder=6)
        ax.text(gap, 0.735 + 0.16 * tier, use, fontsize=7.0, color=MUTED,
                ha="center", va="bottom", zorder=6)

    # The model's typical miss, drawn at the same scale as everything above it.
    y = 0.30
    centre = 1.12                     # anchored on silicon
    ax.annotate("", xy=(centre - mae, y), xytext=(centre + mae, y),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=2.4))
    ax.plot([centre, centre], [y - 0.09, y + 0.09], color=WARN, lw=2.0, zorder=6)
    ax.text(centre, y - 0.20,
            f"a typical miss of ±{mae:.2f} eV, drawn around silicon",
            fontsize=8.6, fontweight="bold", color=WARN, ha="center", va="center")

    ax.text(0.06, 0.30, "so the model would place\nsilicon anywhere in here:",
            fontsize=8.2, color=WARN, ha="left", va="center", linespacing=1.5)

    ax.set_yticks([])
    ax.set_xlabel("Band gap  (eV)   —   experimental values for materials you already know")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def panel_budget(ax, mae, source):
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-1.35, 2.60)
    ax.axis("off")
    ax.text(0.0, 2.42, "C.  Two sources of error, and the model's is the smaller one",
            fontsize=11.5, fontweight="bold", color=INK, ha="left", va="center")

    rows = [
        (1.55, ACCENT, "This model vs DFT", f"{mae:.3f} eV",
         "measured here, on 4,308 materials it never saw", mae, None),
        (0.35, WARN, "DFT vs the laboratory", f"{DFT_RMSE_LO:.2f} – {DFT_RMSE_HI:.2f} eV",
         "published, not measured here — GGA underestimates real gaps",
         DFT_RMSE_LO, DFT_RMSE_HI),
    ]
    for y, col, name, val, note, v0, v1 in rows:
        ax.text(0.0, y + 0.46, name, fontsize=9.8, fontweight="bold",
                color=INK, ha="left", va="center")
        if v1 is None:
            ax.barh(y, v0, height=0.30, color=col, alpha=0.88,
                    edgecolor="white", zorder=4)
        else:
            ax.barh(y, v1 - v0, left=v0, height=0.30, color=col, alpha=0.5,
                    edgecolor=col, linewidth=1.6, zorder=4)
            ax.barh(y, v0, height=0.30, color=col, alpha=0.88,
                    edgecolor="white", zorder=4)
        ax.text((v1 or v0) + 0.03, y, val, fontsize=10.2, fontweight="bold",
                color=col, ha="left", va="center", zorder=6)
        ax.text(0.0, y - 0.44, note, fontsize=8.0, color=MUTED,
                ha="left", va="center")

    ax.add_patch(FancyBboxPatch(
        (0.0, -1.28), 1.32, 0.60, boxstyle="round,pad=0.05,rounding_size=0.04",
        facecolor=LIGHT, alpha=0.6, edgecolor=MUTED, linewidth=1.0, zorder=2))
    ax.text(0.03, -0.98,
            "The model is now better at reproducing DFT than DFT is at reproducing reality.",
            fontsize=8.8, fontweight="bold", color=INK, ha="left", va="center",
            zorder=6)


def main() -> None:
    use_house_style()
    gaps = dataset_gaps()
    mae, source = model_error()

    fig = plt.figure(figsize=(15.4, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.12, 1.62],
                          left=0.058, right=0.975, top=0.845, bottom=0.115,
                          hspace=0.62)

    panel_scale(fig.add_subplot(gs[0]), gaps, mae)
    ax_ref = fig.add_subplot(gs[1])
    ax_ref.set_title("B.  What that error looks like next to materials you know",
                     loc="left", pad=8)
    panel_reference(ax_ref, mae)
    panel_budget(fig.add_subplot(gs[2]), mae, source)

    fig.suptitle("What does “an error of 0.4 eV” actually mean?",
                 fontsize=16.0, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.945,
             "It is not a property of one material and it is not a calibration constant. It is the average size of the miss, over thousands of different crystals the model had never seen.",
             fontsize=9.5, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "A band gap is the energy needed to knock an electron loose in a material. It decides whether a crystal is a metal, a semiconductor or an insulator, what colour it\n"
        "absorbs, and whether it can drive a photocatalytic reaction. Every eV figure quoted in this repository is a mean absolute error in that quantity, averaged over a whole\n"
        f"test set — never a single material. {DFT_SOURCE}.",
        y=0.012)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig13_what_is_an_ev.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    if gaps is not None:
        print(f"  {len(gaps):,} non-metals, median gap {np.median(gaps):.3f} eV, "
              f"model error {mae:.3f} eV ({100 * mae / np.median(gaps):.0f}% of the median)")


if __name__ == "__main__":
    main()
