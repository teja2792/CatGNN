"""Figure 8 -- Why the graph network fails on an element it has never seen.

This is the clearest single picture in the repository, so it gets its own figure
rather than sharing space with the results bars.

The story it tells, in words a chemical engineer can act on: a neural network
that learns a numerical fingerprint for each element gets better and better at
the elements it has examples of, and learns nothing whatever about the ones it
does not. You can watch that happen -- the model's performance on *practice*
materials keeps improving while its performance on *unseen* materials gets worse.

Run with:  python -m src.figures.fig_overfitting
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, ACCENT, WARN, MUTED, INK,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

SPLIT_ORDER = ["random", "formula", "chemsys", "element"]

# Plain language. "chemsys" means nothing to a reader who has not read the code.
SPLIT_PLAIN = {
    "random": "Tested on similar materials",
    "formula": "Tested on new compositions",
    "chemsys": "Tested on new element combinations",
    "element": "Tested on ELEMENTS never seen in training",
}
SPLIT_COLOUR = {"random": STRUCTURE, "formula": COMPOSITION,
                "chemsys": FUSED, "element": WARN}


def load() -> dict:
    p = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run:\n"
            "    python scripts/train_cgcnn.py --split random --nonmetals   (and the other three)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def panel_all_splits(ax, runs: dict):
    """Every split's error on held-out materials, against training time."""
    for s in SPLIT_ORDER:
        if s not in runs:
            continue
        h = runs[s]["history"]
        mins = [e["minutes"] for e in h]
        val = [e["val_mae"] for e in h]
        is_bad = s == "element"
        ax.plot(mins, val, color=SPLIT_COLOUR[s], lw=2.6 if is_bad else 1.8,
                label=SPLIT_PLAIN[s], zorder=4 if is_bad else 3,
                alpha=1.0 if is_bad else 0.85)

        b = runs[s].get("best_epoch")
        if b is not None and 0 <= b < len(h):
            ax.scatter([h[b]["minutes"]], [h[b]["val_mae"]],
                       color=SPLIT_COLOUR[s], s=52, zorder=6,
                       edgecolor="white", linewidth=1.4)

    ax.annotate(
        "This one gets WORSE\nthe longer it trains",
        xy=(runs["element"]["history"][-1]["minutes"],
            runs["element"]["history"][-1]["val_mae"]),
        xytext=(0.40, 0.90), textcoords="axes fraction",
        fontsize=10, color=WARN, fontweight="bold", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.6,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.annotate(
        "These three keep improving",
        xy=(28, 0.47), xytext=(0.42, 0.30), textcoords="axes fraction",
        fontsize=9.5, color=INK, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                        connectionstyle="arc3,rad=0.2"),
    )

    ax.set_xlabel("Training time  (minutes)")
    ax.set_ylabel("Band-gap error on materials the model\nhas never seen  (eV, lower is better)")
    ax.set_title("A.  Four ways of testing the same model", loc="left", pad=10)
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="center right", framealpha=0.94,
              title="How the test set was chosen", title_fontsize=9)


def panel_divergence(ax, runs: dict):
    """Practice error vs exam error, for the split that fails."""
    h = runs["element"]["history"]
    mins = [e["minutes"] for e in h]
    train = [e["train_loss"] for e in h]
    val = [e["val_mae"] for e in h]

    # Training loss is on the normalised target; rescale to eV so both curves are
    # in the same physical units and the divergence is a fair visual comparison.
    std = runs["element"]["normaliser"]["std"]
    train_ev = [t * std for t in train]

    ax.plot(mins, train_ev, color=ACCENT, lw=2.4, marker="o", ms=3.4,
            label="On materials it is LEARNING from\n(“practice questions”)")
    ax.plot(mins, val, color=WARN, lw=2.4, marker="s", ms=3.4,
            label="On materials with NEW elements\n(“the real exam”)")

    ax.fill_between(mins, train_ev, val, color=WARN, alpha=0.10, zorder=1)

    best = min(range(len(val)), key=lambda i: val[i])
    ax.axvline(mins[best], color=MUTED, ls="--", lw=1.2, zorder=2)
    ax.annotate(
        f"best it ever gets:\n{val[best]:.2f} eV at {mins[best]:.0f} min.\n"
        "Every minute after this\nmade it worse.",
        xy=(mins[best], val[best]), xytext=(mins[best] + 0.6, max(val) * 1.14),
        fontsize=9, color=MUTED, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                        connectionstyle="arc3,rad=-0.2"),
    )

    # A label placed inside the shaded band needs no arrow, and an arrow that has
    # to cross the plot to reach its target is a sign the label is in the wrong
    # place rather than a sign it needs a longer arrow.
    mid = len(mins) // 2
    ax.text(
        mins[mid], (train_ev[mid] + val[mid]) / 2,
        "this widening gap is the model\nmemorising the elements it has",
        fontsize=9.2, color=WARN, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=WARN, lw=1.0, alpha=0.92),
    )

    ax.set_xlabel("Training time  (minutes)")
    ax.set_ylabel("Band-gap error  (eV, lower is better)")
    ax.set_title("B.  The same model, practice vs exam", loc="left", pad=10)
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(val) * 1.30)
    ax.legend(fontsize=8.8, loc="lower center", framealpha=0.94, ncol=2,
              columnspacing=1.0)


def main() -> None:
    use_house_style()
    runs = load()
    if "element" not in runs:
        raise FileNotFoundError("the element-disjoint run is missing")

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.4))
    fig.subplots_adjust(bottom=0.25, top=0.84, wspace=0.26)

    panel_all_splits(axes[0], runs)
    panel_divergence(axes[1], runs)

    fig.suptitle(
        "A neural network cannot guess at an element it has never met",
        fontsize=15.5, fontweight="bold", y=1.0, x=0.006, ha="left",
    )
    fig.text(
        0.006, 0.945,
        "CGCNN predicting band gap. Every curve is the same model and the same training procedure — only the test set differs.",
        fontsize=9.2, color=MUTED, ha="left", va="top",
    )

    caption(
        fig,
        "Left: when the test materials resemble the training ones, longer training keeps helping. When the test materials contain elements that were held\n"
        "out of training entirely, longer training makes the model WORSE. Right: the same failing run, split into how it does on the materials it is learning\n"
        "from (green) versus the ones with unfamiliar elements (red). The green curve keeps falling while the red one climbs — the model is getting better\n"
        "at what it has already seen and no better at anything else. A network learns a numeric fingerprint per element from data, so an element that was\n"
        "never in the training set has no meaningful fingerprint. A descriptor model looks up electronegativity and ionic radius, which exist for every\n"
        "element in the periodic table, and so degrades gently instead of collapsing.",
        y=0.055,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig8_element_generalisation.png"
    fig.savefig(path)
    plt.close(fig)

    h = runs["element"]["history"]
    best = min(h, key=lambda e: e["val_mae"])
    print(f"wrote {path}")
    print(f"  element split: best val {best['val_mae']:.4f} eV at epoch {best['epoch']}, "
          f"final {h[-1]['val_mae']:.4f} eV")
    print(f"  training loss over the same span: {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f}")


if __name__ == "__main__":
    main()
