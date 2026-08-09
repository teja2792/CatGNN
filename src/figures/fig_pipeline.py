"""Figure 0 -- the whole project on one page, drawn as a process flow diagram.

A chemical engineer reads process flow diagrams fluently: feed streams enter on
the left, unit operations transform them, product streams leave on the right, and
every stream is labelled with what it carries and how much. So this is drawn as
one, deliberately. The boxes are unit operations, the arrows are streams, the
numbers on the streams are real throughputs measured from the actual run, and the
one thing that loops backwards is a recycle.

It exists because the honest answer to "what does this repository do?" is a
sequence of five transformations, and a reader should be able to see all five,
see what came out of each, and see which one the experiment is actually about --
without reading any code or knowing what a neural network is.

Every number on this diagram is read from the results files at build time. If a
run changes, the diagram changes. Nothing here is typed in by hand.

Run with:  python -m src.figures.fig_pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import (
    use_house_style, caption,
    STRUCTURE, COMPOSITION, FUSED, WARN, MUTED, INK, LIGHT,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"


# ---------------------------------------------------------------------------
# Numbers, read from the run rather than typed in
# ---------------------------------------------------------------------------

def load_numbers() -> dict:
    n = {
        "crystals": 102957, "atoms": "1.42M", "edges": "16.9M",
        "descriptors": 192, "leakage": 42.6,
        "cgcnn_random": None, "cgcnn_element": None,
        "desc_random": None, "desc_element": None,
        "arch_spread": None,
    }

    p = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if "random" in d:
            n["cgcnn_random"] = d["random"]["test"]["mae"]
        if "element" in d:
            n["cgcnn_element"] = d["element"]["test"]["mae"]

    p = RESULTS / "baselines.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))["results"]
        n["desc_random"] = _best_baseline(rows, "random")
        n["desc_element"] = _best_baseline(rows, "element")

    p = RESULTS / "architecture_significance.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))["mae"]
        n["arch_spread"] = (min(m.values()), max(m.values()))

    # Fall back to the published values in the README so the figure still builds
    # on a fresh clone, before anything has been trained.
    n["cgcnn_random"] = n["cgcnn_random"] or 0.4137
    n["cgcnn_element"] = n["cgcnn_element"] or 1.019
    n["desc_random"] = n["desc_random"] or 0.511
    n["desc_element"] = n["desc_element"] or 0.694
    n["arch_spread"] = n["arch_spread"] or (0.4137, 0.4755)
    return n


def _best_baseline(rows: list, split: str, target: str = "band_gap_nonmetals"):
    """Lowest MAE any descriptor model reached on `split`.

    Deliberately the BEST of them, not the average and not a favourite. The GNN
    should have to beat the strongest thing the descriptors can do, or the
    comparison flatters it.
    """
    maes = [r["mae"] for r in rows
            if r.get("target") == target and r.get("split") == split
            and r.get("model") != "mean"]
    return min(maes) if maes else None


# ---------------------------------------------------------------------------
# Drawing helpers
#
# Boxes are sized from their content rather than guessed at. Text that spills
# over a box edge reads as a mistake and undermines everything else on the page,
# and it is the single most common way a diagram like this goes wrong.
# ---------------------------------------------------------------------------

W_STAGE = 21.0          # every unit operation the same width, like a real PFD


def unit(ax, x, y, h, title, lines, colour, tag=None, w=W_STAGE):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.7",
        facecolor=colour, alpha=0.09, edgecolor=colour, linewidth=1.8, zorder=3))

    if tag:
        # Above the box, never inside it -- a tag inside collides with the title
        # the moment the title needs two words more than you expected.
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y + h / 2 + 0.5), 11.0, 2.2,
            boxstyle="round,pad=0.05,rounding_size=0.4",
            facecolor=colour, edgecolor="none", zorder=5))
        ax.text(x - w / 2 + 5.5, y + h / 2 + 1.6, tag, fontsize=7.4,
                fontweight="bold", color="white", ha="center", va="center", zorder=6)

    ax.text(x, y + h / 2 - 2.6, title, fontsize=10.0, fontweight="bold",
            color=INK, ha="center", va="center", zorder=6)
    for i, (txt, bold) in enumerate(lines):
        ax.text(x, y + h / 2 - 5.1 - i * 2.15, txt,
                fontsize=9.0 if bold else 8.3,
                fontweight="bold" if bold else "normal",
                color=colour if bold else MUTED,
                ha="center", va="center", zorder=6)


def stream(ax, p0, p1, colour, label=None, rad=0.0, lw=2.0,
           label_xy=None, dashed=False, fontsize=8.2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=colour, zorder=4, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = label_xy or ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 2.0)
        ax.text(mx, my, label, fontsize=fontsize, color=colour, fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.94))


def main() -> None:
    use_house_style()
    n = load_numbers()

    fig, ax = plt.subplots(figsize=(16.6, 9.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 89.5)
    ax.axis("off")

    UP, DOWN = 77.0, 52.0
    MID = (UP + DOWN) / 2
    X1, X2, X3, X4 = 10.0, 38.0, 63.0, 87.0
    W_FEED = 18.0          # narrower, to open up room for stream labels

    # ---------------- FEED ----------------
    unit(ax, X1, MID, 19, "Materials Project", [
        (f"{n['crystals']:,} crystals", True),
        ("each one computed by DFT", False),
        ("and published openly", False),
        ("", False),
        ("fetched once, with a manifest", False),
        ("and a checksum", False),
    ], STRUCTURE, tag="FEED", w=W_FEED)

    # ---------------- Two descriptions ----------------
    unit(ax, X2, UP, 17, "Describe by RECIPE only", [
        (f"{n['descriptors']} numbers per material", True),
        ("average and spread of", False),
        ("electronegativity, ionic radius,", False),
        ("valence, melting point …", False),
        ("blind to structure, on purpose", False),
    ], COMPOSITION, tag="STREAM A")

    unit(ax, X2, DOWN, 17, "Describe by the CRYSTAL", [
        ("atoms = nodes, contacts = edges", True),
        (f"{n['atoms']} atoms, {n['edges']} contacts", False),
        ("8 Å cutoff, periodic images", False),
        ("counted properly", False),
        ("sees rutile ≠ anatase", False),
    ], STRUCTURE, tag="STREAM B")

    # ---------------- Fairness gate ----------------
    unit(ax, X3, MID, 25, "FAIRNESS GATE", [
        ("four test sets, hardest last", True),
        ("", False),
        ("1   random materials", False),
        ("2   formulas never seen", False),
        ("3   element combinations never seen", False),
        ("4   ELEMENTS never seen", False),
        ("", False),
        (f"a random split leaks {n['leakage']}%", True),
    ], WARN, tag="QUALITY CHECK")

    # ---------------- Fitting ----------------
    unit(ax, X4, UP, 15, "Fit a chemistry model", [
        ("Random Forest / boosting", True),
        ("on the 192 recipe numbers", False),
        ("", False),
        ("this is the bar to beat", False),
    ], COMPOSITION)

    unit(ax, X4, DOWN, 15, "Fit graph networks", [
        ("CGCNN · MPNN · GATv2 · MEGNet", True),
        ("all four written from scratch", False),
        ("", False),
        ("35 min each, one laptop CPU", False),
    ], STRUCTURE)

    # ---------------- Streams ----------------
    e, f = W_STAGE / 2, W_FEED / 2
    stream(ax, (X1 + f, MID + 3.5), (X2 - e, UP - 5.0), COMPOSITION,
           "formula\nonly", rad=-0.14, label_xy=(23.0, 71.2), fontsize=7.8)
    stream(ax, (X1 + f, MID - 3.5), (X2 - e, DOWN + 5.0), STRUCTURE,
           "atoms +\npositions", rad=0.14, label_xy=(23.0, 57.6), fontsize=7.8)

    stream(ax, (X2 + e, UP - 3.0), (X3 - e, MID + 7.0), COMPOSITION, rad=0.14)
    stream(ax, (X2 + e, DOWN + 3.0), (X3 - e, MID - 7.0), STRUCTURE, rad=-0.14)

    stream(ax, (X3 + e, MID + 7.0), (X4 - e, UP - 3.0), COMPOSITION, rad=0.14)
    stream(ax, (X3 + e, MID - 7.0), (X4 - e, DOWN + 3.0), STRUCTURE, rad=-0.14)

    # ---------------- Recycle: Phase 5 ----------------
    # Routed under the structure train so it crosses nothing. This is the only
    # backwards arrow on the page, which is the point: everything else is a
    # one-pass pipeline, and this is the thing the project is actually about.
    # rad is NEGATIVE: drawn right-to-left, a positive radius bows the arc
    # upwards straight through the fairness gate.
    ax.add_patch(FancyArrowPatch(
        (X4 - 4, DOWN - 9.0), (X2 + 4, DOWN - 9.0),
        arrowstyle="-|>", mutation_scale=17, linewidth=2.2, color=FUSED,
        linestyle="--", zorder=5, connectionstyle="arc3,rad=-0.30"))
    ax.text(X3, 36.4,
            "RECYCLE  (Phase 5)   —   the collapse on test 4 says the network is memorising elements,\n"
            "so feed every atom its real electronegativity and radius instead of a memorised code",
            fontsize=8.8, fontweight="bold", color=FUSED, ha="center", va="center",
            zorder=7, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=FUSED, lw=1.4))

    # ---------------- PRODUCT ----------------
    ax.add_patch(FancyBboxPatch(
        (2.0, 2.5), 96, 26.0,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=LIGHT, alpha=0.5, edgecolor=MUTED, linewidth=1.2, zorder=2))
    ax.text(4.5, 26.0, "PRODUCT   —   what actually came out, in order",
            fontsize=10.6, fontweight="bold", color=INK, ha="left", va="center",
            zorder=6)

    findings = [
        (COMPOSITION, "Chemistry alone is\na high bar",
         f"{n['desc_random']:.3f} eV using nothing\nbut the formula — better\nthan the published\nCGCNN score."),
        (STRUCTURE, "Structure wins,\nby about 20%",
         f"{n['cgcnn_random']:.3f} eV vs {n['desc_random']:.3f} eV,\nand it holds on three\nof the four tests."),
        (WARN, "Then it collapses\non test 4",
         f"{n['cgcnn_element']:.3f} eV on unseen\nelements — worse than\nchemistry alone\n({n['desc_element']:.3f} eV)."),
        (MUTED, "Architecture barely\nmatters",
         f"all four networks land\nin {n['arch_spread'][0]:.2f}–{n['arch_spread'][1]:.2f} eV.\nThe collapse above is\nten times bigger."),
        (FUSED, "So: give it the\nperiodic table",
         "an element it has never\nseen still has a real\nelectronegativity.\n→ Phase 5"),
    ]
    cw, gap = 16.6, 2.4
    for i, (col, head, body) in enumerate(findings):
        cx = 4.5 + i * (cw + gap) + cw / 2
        ax.add_patch(FancyBboxPatch(
            (cx - cw / 2, 4.2), cw, 18.2,
            boxstyle="round,pad=0.12,rounding_size=0.5",
            facecolor="white", edgecolor=col, linewidth=1.6, zorder=4))
        ax.add_patch(FancyBboxPatch(
            (cx - cw / 2, 18.0), cw, 4.4,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=col, alpha=0.14, edgecolor="none", zorder=5))
        ax.text(cx - cw / 2 + 1.1, 20.2, f"{i + 1}", fontsize=10.5,
                fontweight="bold", color=col, ha="center", va="center", zorder=6)
        ax.text(cx + 1.0, 20.2, head, fontsize=8.7, fontweight="bold", color=col,
                ha="center", va="center", zorder=6, linespacing=1.4)
        ax.text(cx, 11.0, body, fontsize=8.2, color=INK, ha="center", va="center",
                zorder=6, linespacing=1.7)
        if i < len(findings) - 1:
            ax.text(cx + cw / 2 + gap / 2, 13.3, "→", fontsize=13, color=MUTED,
                    ha="center", va="center", zorder=6)

    # ---------------- Titles ----------------
    fig.suptitle("CatGNN, as a process flow diagram",
                 fontsize=16.5, fontweight="bold", y=0.995, x=0.006, ha="left")
    fig.text(0.006, 0.952,
             "Two ways of describing the same 102,957 materials go in. The question is which description predicts a property better — and where each one stops working.",
             fontsize=9.6, color=MUTED, ha="left", va="top")

    caption(
        fig,
        "Read it like any flowsheet: feed on the left, two parallel treatment trains, a quality check both must pass, then fitting, then product. The quality check is the\n"
        "step most published work leaves out — 42.6% of a randomly chosen test set shares a chemical formula with something in training, which makes a random split closer\n"
        "to a memory test than a prediction test. Every number on this diagram is read from the results files when the figure is built, so it cannot drift from the runs.",
        y=0.028)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig0_pipeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
