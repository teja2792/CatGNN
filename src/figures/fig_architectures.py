"""Figure 9 -- Four graph networks, one budget, and a result that is mostly a null.

The point of this figure is not a leaderboard. It is that the most-cited design
choice in crystal graph networks -- CGCNN's gate -- makes no measurable
difference, and that saying so requires showing the uncertainty rather than
quoting four numbers to four decimal places.

Panel A is the comparison. Panel B is the part most benchmark tables leave out:
how big the differences are relative to how much they could move by chance.

Run with:  python -m src.figures.fig_architectures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MODELS = REPO / "models"
OUT = RESULTS / "figures"

TAG = "band_gap_random_nonmetals"

# What each model actually changes, in words a chemical engineer can use. The
# model NAME carries no information for that reader, so the name is the small
# text and the mechanism is the large text.
DESCRIPTION = {
    "mpnn": ("Plain message passing",
             "Each bond passes a message. No weighting."),
    "cgcnn": ("Message passing + a bond “volume knob”",
              "Each bond gets its own 0–1 importance, set independently."),
    "gatv2": ("Message passing + competing bonds",
              "An atom splits 100% of its attention among its neighbours."),
    "megnet": ("Message passing + a whole-crystal summary",
               "Every atom also sees a running description of the whole material."),
}
COLOUR = {"mpnn": MUTED, "cgcnn": STRUCTURE, "gatv2": FUSED, "megnet": COMPOSITION}


def load():
    sig = RESULTS / "architecture_significance.json"
    if not sig.exists():
        raise FileNotFoundError(
            f"{sig} missing. Run:\n    python scripts/compare_architectures.py")
    stats = json.loads(sig.read_text(encoding="utf-8"))

    errs = {}
    for a in stats["mae"]:
        d = np.load(MODELS / a / f"{TAG}_predictions.npz")
        errs[a] = np.abs(d["y_true"] - d["y_pred"])
    return stats, errs


def bootstrap_ci(err: np.ndarray, n_boot: int = 20_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    means = err[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def panel_bars(ax, stats, errs):
    """Bars with every label in clear space above them.

    Text placed on top of a coloured bar is unreadable in half the palette and
    unreadable in all of it once the figure is screenshotted. The bars are thin
    and the words sit above them.
    """
    order = sorted(stats["mae"], key=lambda a: stats["mae"][a])
    y = np.arange(len(order))[::-1].astype(float)
    xmax = 0.86

    for yi, a in zip(y, order):
        mae = stats["mae"][a]
        lo, hi = bootstrap_ci(errs[a])

        title, sub = DESCRIPTION[a]
        ax.text(0.004, yi + 0.40, title, fontsize=10.3, fontweight="bold",
                color=INK, va="center", ha="left", zorder=6)
        ax.text(0.004, yi + 0.21, sub, fontsize=8.8, color=MUTED,
                va="center", ha="left", zorder=6)

        ax.barh(yi - 0.06, mae, height=0.26, color=COLOUR[a], alpha=0.88,
                edgecolor="white", zorder=3)
        ax.errorbar(mae, yi - 0.06, xerr=[[mae - lo], [hi - mae]], fmt="none",
                    ecolor=INK, elinewidth=1.5, capsize=4, capthick=1.5, zorder=5)

        ax.text(mae + 0.018, yi - 0.06, f"{mae:.3f} eV", fontsize=10.5,
                fontweight="bold", color=COLOUR[a], va="center", ha="left", zorder=6)
        ax.text(0.004, yi - 0.32,
                f"{a.upper()}  ·  {stats['n_parameters'][a]:,} tunable numbers",
                fontsize=8.1, color=MUTED, va="center", ha="left", zorder=6)

    # Brace joining the two indistinguishable bars, drawn in the free space to
    # the right of every bar so it cannot land on top of anything.
    bx = 0.60
    ytop, ybot = y[0] - 0.06, y[1] - 0.06
    ax.plot([bx, bx], [ybot, ytop], color=WARN, lw=1.6, zorder=6)
    for yy in (ytop, ybot):
        ax.plot([bx - 0.018, bx], [yy, yy], color=WARN, lw=1.6, zorder=6)
    ax.text(bx + 0.022, (ytop + ybot) / 2,
            "same to within\nmeasurement noise",
            fontsize=9.4, fontweight="bold", color=WARN,
            va="center", ha="left", zorder=6)

    ax.set_yticks([])
    ax.set_ylim(-0.62, len(order) - 0.35)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.set_xlabel("Typical band-gap error on 4,308 unseen materials  (eV — lower is better)\n"
                  "black whiskers are the 95% range, not a single lucky number")
    ax.set_title("A.  Same data, same 35-minute budget, same laptop",
                 loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_bounds(0, 0.6)


def panel_differences(ax, stats):
    """Every pairwise gap with its uncertainty. Zero means 'no difference'."""
    pairs = stats["pairs"]
    labels, deltas, los, his, sig = [], [], [], [], []
    for p in pairs:
        labels.append(f"{p['a'].upper()}  vs  {p['b'].upper()}")
        deltas.append(-p["delta_mae"])          # positive = first model better
        los.append(-p["ci_high"])
        his.append(-p["ci_low"])
        sig.append(p["significant"])

    y = np.arange(len(labels))[::-1].astype(float)

    # Reserve a right-hand column for the verdicts so they cannot land on a
    # confidence interval, and a blank row at the bottom for the legend.
    data_max = max(his)
    verdict_x = data_max * 1.14
    ax.set_xlim(-data_max * 0.42, data_max * 1.82)
    ax.set_ylim(-2.05, len(labels) - 0.25)

    # The "too small to matter" band. 0.02 eV is well inside DFT's own
    # disagreement with experiment, so a gap smaller than this is not one a
    # chemical engineer could act on even if it were real.
    ax.axvspan(-0.02, 0.02, color=LIGHT, zorder=1)
    ax.axvline(0, color=INK, lw=1.3, zorder=4)
    ax.text(0.0, -0.92, "gaps inside this band are smaller than DFT's own error",
            fontsize=8.2, color=MUTED, ha="center", va="bottom", style="italic")

    for yi, d, lo, hi, s in zip(y, deltas, los, his, sig):
        col = ACCENT if s else WARN
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.8, solid_capstyle="round",
                zorder=5)
        ax.scatter([d], [yi], s=62, color=col, zorder=6, edgecolor="white",
                   linewidth=1.4)
        ax.text(verdict_x, yi, "real difference" if s else "TOO CLOSE TO CALL",
                fontsize=8.7, fontweight="normal" if s else "bold",
                color=col, ha="left", va="center", zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.02, data_max + 0.021, 0.02))
    ax.set_xlabel("How much better the FIRST model is  (eV)\n"
                  "bar = range the true difference is 95% likely to lie in")
    ax.set_title("B.  Which gaps are real, and which are noise", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_bounds(-0.02, data_max)

    ax.legend(handles=[
        Patch(facecolor=ACCENT, label="range excludes zero — a real gap"),
        Patch(facecolor=WARN, label="range includes zero — no evidence of any gap"),
    ], fontsize=8.4, loc="lower left", bbox_to_anchor=(-0.02, -0.01),
        framealpha=0.95, borderpad=0.5, handlelength=1.4)


def main() -> None:
    use_house_style()
    stats, errs = load()

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.5),
                             gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.30)

    panel_bars(axes[0], stats, errs)
    panel_differences(axes[1], stats)

    fig.suptitle(
        "The most-copied idea in crystal graph networks does not do anything",
        fontsize=15.5, fontweight="bold", y=1.005, x=0.006, ha="left",
    )
    fig.text(
        0.006, 0.945,
        "Four architectures predicting band gap for non-metals, trained under an identical fixed wall-clock budget on one laptop CPU. "
        "The only thing that differs between them is how a bond's message is weighted.",
        fontsize=9.2, color=MUTED, ha="left", va="top",
    )

    caption(
        fig,
        "CGCNN's defining feature is a learned “volume knob” on every bond. Removing it entirely (MPNN) changes the error by 0.001 eV — a gap whose 95% range,\n"
        "[-0.013, +0.010] eV, comfortably contains zero. The gate is not doing measurable work on this task. Two designs that DO change the answer make it worse:\n"
        "forcing an atom's bonds to compete for a fixed attention budget (GATv2, +0.025 eV) and adding a whole-crystal summary vector (MEGNet, +0.062 eV) both\n"
        "cost accuracy here, and MEGNet does so while carrying twice MPNN's tunable numbers and fitting fewer passes through the data into the same 35 minutes.\n"
        "Uncertainty is from resampling the 4,308 test materials 20,000 times; it does not include the variation from retraining with a different random seed.",
        y=0.015,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig9_architecture_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    for a, m in stats["mae"].items():
        print(f"  {a:<8}{m:.4f} eV   {stats['n_parameters'][a]:>8,} params")


if __name__ == "__main__":
    main()
