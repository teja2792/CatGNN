"""Figure 1 -- The polymorph problem: why composition alone is not enough.

This is the motivating figure for the whole repository, and every number in it
comes from real DFT data downloaded from the Materials Project (snapshot in
``data/reference/mp_summary_snapshot.csv``, provenance in ``SOURCES.md``).

The argument, in one sentence: a model that only knows the chemical formula
must give the same answer for every polymorph of that formula, so the spread
between polymorphs is an error it can never remove -- no matter how good the
model is.

Run with:  python -m src.figures.fig_polymorph_problem
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import (
    use_house_style,
    caption,
    source_stamp,
    STRUCTURE,
    COMPOSITION,
    WARN,
    MUTED,
    INK,
)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "reference" / "mp_summary_snapshot.csv"
OUT = REPO / "results" / "figures"

# Published CGCNN test errors on Materials Project, for scale.
# Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018).
CGCNN_GAP_MAE_EV = 0.388
CGCNN_FE_MAE_EV_PER_ATOM = 0.039

# Well-known TiO2 polymorphs, identified by space group, for annotation.
NAMED_POLYMORPHS = {
    "I4_1/amd": "anatase",
    "P4_2/mnm": "rutile",
    "Pbca": "brookite",
}

# One TiO2 entry is a very high-energy, low-density structure (mp-1445035,
# -0.76 eV/atom vs. about -3.3 for everything else). It is a real Materials
# Project entry, not a data error, but it is not a material anyone would make.
# We report the numbers both with and without it rather than quietly dropping it.
OUTLIER_FE_CUTOFF = -2.0


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. See SOURCES.md for how this snapshot was produced."
        )
    return pd.read_csv(DATA)


def composition_only_floor(values: np.ndarray) -> float:
    """Smallest mean-absolute-error any composition-only model can achieve.

    A model that only sees the formula gets one input and must emit one number
    for every polymorph sharing that formula. The single number minimising mean
    absolute error is the median, so the resulting error is the mean absolute
    deviation about the median. No amount of model capacity reduces it: the
    information required is simply not in the input.
    """
    return float(np.abs(values - np.median(values)).mean())


def panel_a(ax, tio2: pd.DataFrame) -> dict:
    """Every TiO2 polymorph in formation-energy / band-gap space."""
    # More negative formation energy = more stable = the normal population.
    keep = tio2[tio2.formation_energy_per_atom_eV < OUTLIER_FE_CUTOFF]
    drop = tio2[tio2.formation_energy_per_atom_eV >= OUTLIER_FE_CUTOFF]

    ax.scatter(
        keep.formation_energy_per_atom_eV,
        keep.band_gap_eV,
        s=46,
        color=STRUCTURE,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        label=f"TiO$_2$ polymorphs in Materials Project (n = {len(keep)})",
    )

    # The single point a composition-only model is forced to predict.
    med_fe = float(np.median(keep.formation_energy_per_atom_eV))
    med_gap = float(np.median(keep.band_gap_eV))
    ax.scatter(
        [med_fe],
        [med_gap],
        marker="*",
        s=560,
        color=COMPOSITION,
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
        label='"TiO$_2$" — the one answer a formula-only model can give',
    )

    # Error band: how far the real polymorphs sit from that single answer.
    gap_floor = composition_only_floor(keep.band_gap_eV.values)
    ax.axhspan(med_gap - gap_floor, med_gap + gap_floor, color=COMPOSITION, alpha=0.10, zorder=1)
    ax.axhline(med_gap, color=COMPOSITION, lw=1.0, ls="--", alpha=0.6, zorder=2)

    # Zoom on the physically sensible population; the outlier is reported in
    # text rather than allowed to compress the interesting region to a sliver.
    lo = float(keep.formation_energy_per_atom_eV.min())
    hi = float(keep.formation_energy_per_atom_eV.max())
    pad = (hi - lo) * 0.10
    ax.set_xlim(lo - pad, hi + pad * 5.0)
    ax.set_ylim(-0.25, 3.85)

    # Named polymorphs, fanned out to the right with leader lines.
    label_offsets = {"brookite": (78, 46), "anatase": (86, 4), "rutile": (78, -42)}
    for _, r in keep.iterrows():
        name = NAMED_POLYMORPHS.get(r.space_group_symbol)
        if not name:
            continue
        if name == "brookite" and r.material_id != "mp-1840":
            continue  # two Pbca entries; label the lower-energy one only
        txt = f"{name}  ({r.band_gap_eV:.2f} eV)"
        if bool(r.is_stable):
            txt += "\nlowest-energy polymorph in DFT"
        ax.annotate(
            txt,
            (r.formation_energy_per_atom_eV, r.band_gap_eV),
            textcoords="offset points",
            xytext=label_offsets[name],
            fontsize=9,
            color=INK,
            fontweight="bold" if bool(r.is_stable) else "normal",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4),
            zorder=7,
        )

    ax.set_xlabel("Formation energy  (eV per atom)     ← more stable")
    ax.set_ylabel("Band gap  (eV)")
    ax.set_title("A.  One formula, 44 structures, 44 different band gaps", loc="left", pad=12)
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    ax.annotate(
        f"shaded band = ±{gap_floor:.2f} eV\nthe error a formula-only\nmodel can never remove",
        xy=(0.03, 0.955),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color=COMPOSITION,
        fontweight="bold",
    )
    if len(drop):
        r = drop.iloc[0]
        ax.annotate(
            f"+ 1 high-energy outlier off-scale\n({r.material_id}, {r.formation_energy_per_atom_eV:.2f} eV/atom).\nExcluded here, reported in the text.",
            xy=(0.03, 0.10),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
            style="italic",
        )
    return {"gap_floor": gap_floor, "n_kept": len(keep), "n_total": len(tio2)}


def panel_b(ax, df: pd.DataFrame) -> dict:
    """The same error floor, computed for each material family in the snapshot."""
    rows = []
    for formula, sub in df.groupby("formula_pretty"):
        # Never mix DFT functionals: GGA and GGA+U and r2SCAN gaps are not
        # comparable numbers. Use whichever functional has the most entries.
        run_type = sub.dft_run_type.value_counts().idxmax()
        sub = sub[sub.dft_run_type == run_type]
        if len(sub) < 2:
            rows.append((formula, run_type, len(sub), np.nan))
            continue
        rows.append(
            (formula, run_type, len(sub), composition_only_floor(sub.band_gap_eV.values))
        )
    res = pd.DataFrame(rows, columns=["formula", "run_type", "n", "floor"])
    res = res.sort_values("floor", na_position="first").reset_index(drop=True)

    xmax = max(1.05, float(np.nanmax(res.floor)) * 1.5)
    ypos = np.arange(len(res))
    for y, r in zip(ypos, res.itertuples()):
        if np.isnan(r.floor):
            ax.text(
                CGCNN_GAP_MAE_EV + xmax * 0.06,
                y,
                f"only {r.n} entr{'y' if r.n == 1 else 'ies'} in this snapshot —\nno spread can be measured",
                va="center",
                fontsize=8.5,
                color=MUTED,
                style="italic",
            )
            continue
        ax.barh(y, r.floor, height=0.5, color=STRUCTURE, alpha=0.85, zorder=3)
        ax.text(
            r.floor + xmax * 0.015,
            y,
            f"{r.floor:.2f} eV",
            va="center",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )

    ax.axvline(CGCNN_GAP_MAE_EV, color=WARN, lw=1.6, ls="--", zorder=4)
    ax.annotate(
        f"← total error of a published\n     structure-aware model\n     (CGCNN, {CGCNN_GAP_MAE_EV:.2f} eV)",
        xy=(CGCNN_GAP_MAE_EV + xmax * 0.02, 0.62),
        fontsize=8.5,
        color=WARN,
        va="center",
        ha="left",
    )

    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [
            f"{r.formula}\n{r.n} entr{'y' if r.n == 1 else 'ies'} · {r.run_type}"
            for r in res.itertuples()
        ],
        fontsize=9.5,
    )
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.set_xlabel("Band-gap error a formula-only model cannot remove  (eV)")
    ax.set_title("B.  The same problem in every family we checked", loc="left", pad=12)
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    return {"per_formula": res}


def main() -> None:
    use_house_style()
    df = load()
    tio2 = df[(df.formula_pretty == "TiO2") & (df.dft_run_type == "GGA")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), gridspec_kw={"width_ratios": [1.35, 1]})
    a = panel_a(axes[0], tio2)
    b = panel_b(axes[1], df)

    fig.suptitle(
        "Two materials can share a formula and behave nothing alike",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        x=0.008,
        ha="left",
    )
    caption(
        fig,
        "Every point is a real Materials Project entry. Anatase, rutile and brookite are all TiO$_2$ — same formula, different crystal\n"
        "structures, band gaps differing by more than 0.5 eV. A model given only the formula “TiO$_2$” has no way to tell them apart, so it must\n"
        "predict the orange star for all of them. Panel B: that unavoidable error is around the same size as the *entire* error of a published\n"
        "structure-aware model. This is the gap this repository sets out to measure properly.",
        y=-0.03,
    )
    source_stamp(
        fig,
        "Data: Materials Project (GGA / GGA+U), snapshot in data/reference/  ·  CGCNN reference: Xie & Grossman, PRL 120, 145301 (2018)",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig1_polymorph_problem.png"
    fig.savefig(path)
    plt.close(fig)

    print(f"wrote {path}")
    print(f"  TiO2 polymorphs: {a['n_total']} total, {a['n_kept']} after flagging outlier")
    print(f"  band-gap floor (TiO2, outlier excluded): {a['gap_floor']:.4f} eV")
    print(b["per_formula"].to_string(index=False))


if __name__ == "__main__":
    main()
