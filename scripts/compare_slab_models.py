"""Compare the readout variants the way they should be compared: paired by seed.

WHAT EIGHT SEEDS SAID (847 rows, surface-disjoint, implausible rows dropped)
----------------------------------------------------------------------------
Paired by seed, differences in R2 against the mean-readout control:

    site      +0.048 -0.023 +0.382 +0.075 +0.031 +0.543 -0.084 +0.286
    site_feat -0.010 +0.034 +0.631 +0.142 +0.031 +0.346 -0.022 +0.296

    site       mean +0.157  t(7) = 2.01  p = 0.084   better on 6/8, sign p = 0.29
    site_feat  mean +0.181  t(7) = 2.24  p = 0.060   better on 6/8, sign p = 0.29

THE ACCURACY IMPROVEMENT IS NOT ESTABLISHED. Neither variant separates from the
control at p < 0.05, and the sign test -- which a collapsed run cannot skew -- is
nowhere near it at 0.29. Two large positive differences are doing most of the
work, and both are seeds where the control failed rather than seeds where the
site readout excelled.

WHAT IS ESTABLISHED IS STABILITY:

    cgcnn      sd 0.188   worst -0.206   epochs 18-65
    site       sd 0.051   worst +0.173   epochs 18-31
    site_feat  sd 0.097   worst +0.140   epochs 19-31

    cgcnn vs site       F = 13.59   p = 0.0027   <- real
    cgcnn vs site_feat  F =  3.76   p = 0.102

The control collapses on roughly a quarter of seeds and the site readout never
does. That is a genuine result at p = 0.003, and it is a different claim from
"the model is more accurate", which the data does not support.

THE FEATURES DO NOT HELP, AND THEY COST STABILITY
--------------------------------------------------
site_feat against site, paired: mean +0.024, t(7) = 0.53, p = 0.61, better on 5
of 8. No effect. And the variance advantage over the control weakens from
p = 0.003 to p = 0.10 when they are added.

So is_adsorbate, is_site, height and coordination -- all four, chosen on physical
grounds and defended at length -- buy nothing measurable here and make the model
slightly less reliable. The default was changed from site_feat to site on that
basis. Recorded rather than quietly dropped, because a negative result about
one's own idea is worth exactly as much as a positive one and is easier to lose.

Run:  python scripts/compare_slab_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models.seed_stats import paired_test, variance_ratio  # noqa: E402

RESULTS = REPO / "results" / "catalysis"
MODELS = ["cgcnn", "site", "site_feat"]


def load(model: str, split: str, clean: bool) -> dict[int, dict]:
    """Every seed's metrics for one variant, keyed by seed."""
    out = {}
    stem = f"{model}_{split}" + ("_clean" if clean else "")
    for d in sorted(RESULTS.glob(f"{stem}_seed*")):
        f = d / f"adsorption_energy_{split}.json"
        if not f.exists():
            continue
        try:
            seed = int(d.name.rsplit("seed", 1)[1])
        except (IndexError, ValueError):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        out[seed] = {"r2": float(r["test"]["r2"]),
                     "rmse": float(r["test"]["rmse"]),
                     "mae": float(r["test"]["mae"]),
                     "epochs": int(r.get("epochs_run", 0))}
    return out


def paired(a: dict, b: dict, key: str):
    """Difference on seeds BOTH variants were run with. Seeds present in only
    one are dropped rather than compared against a different draw."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return np.array([]), []
    return np.array([a[s][key] - b[s][key] for s in shared]), shared


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "surface"
    clean = "--raw" not in sys.argv

    runs = {m: load(m, split, clean) for m in MODELS}
    have = {m: r for m, r in runs.items() if r}
    if len(have) < 2:
        print("\n  Need at least two variants run with --seeds. e.g.\n"
              "    python scripts/train_slab.py --split surface --model cgcnn "
              "--seeds 0 1 2 3 4 5 6 7 --drop-implausible\n")
        sys.exit(1)

    print(f"\n{'=' * 76}\n  Readout comparison, {split}-disjoint"
          f"{', implausible rows dropped' if clean else ''}\n{'=' * 76}")

    seeds = sorted(set().union(*[set(r) for r in have.values()]))
    print(f"\n  {'seed':>6}" + "".join(f"{m:>12}" for m in have) + "     (R2)")
    for s in seeds:
        row = "".join(f"{have[m][s]['r2']:>12.3f}" if s in have[m] else f"{'-':>12}"
                      for m in have)
        print(f"  {s:>6}{row}")
    print("  " + "-" * (6 + 12 * len(have)))
    for label, fn in (("mean", np.mean), ("sd", lambda v: np.std(v, ddof=1)),
                      ("worst", np.min)):
        row = "".join(f"{fn([v['r2'] for v in have[m].values()]):>12.3f}"
                      for m in have)
        print(f"  {label:>6}{row}")

    if len(seeds) < 5:
        print(f"\n  WARNING: {len(seeds)} seeds. On 13 held-out surfaces that is")
        print("  not enough to separate these. Treat every gap below as unproven.")

    base = have.get("cgcnn")
    if base:
        print("\n  PAIRED against the mean-readout control")
        print("    A t-test assumes the differences are roughly normal; a")
        print("    collapsed run is a heavy outlier that breaks that. The sign")
        print("    test only counts wins, so one catastrophe cannot create a")
        print("    result. When they disagree, believe the sign test.")
        for m in have:
            if m == "cgcnn":
                continue
            d, _ = paired(have[m], base, "r2")
            if d.size < 2:
                continue
            r = paired_test(d)
            print(f"\n    {m}")
            print(f"      differences  {' '.join(f'{x:+.3f}' for x in d)}")
            print(f"      mean {r['mean']:+.3f}   t({r['n'] - 1}) = {r['t']:.2f}   "
                  f"p = {r['p_t']:.3f}")
            print(f"      better on {r['wins']}/{r['n']} seeds        "
                  f"sign test p = {r['p_sign']:.3f}")
            verdict = ("accuracy gain IS established"
                       if r["p_t"] < 0.05 and r["p_sign"] < 0.05 else
                       "accuracy gain is NOT established" if r["p_sign"] >= 0.05
                       else "mixed: t and sign test disagree")
            print(f"      -> {verdict}")

    print("\n  STABILITY, tested rather than asserted")
    r2 = {m: [v["r2"] for v in have[m].values()] for m in have}
    ep = {m: [v["epochs"] for v in have[m].values()] for m in have}
    for m in have:
        print(f"    {m:<12} sd {np.std(r2[m], ddof=1):.3f}   "
              f"worst {min(r2[m]):+.3f}   epochs {min(ep[m])}-{max(ep[m])}")
    if base:
        for m in have:
            if m == "cgcnn":
                continue
            shared = sorted(set(base) & set(have[m]))
            v = variance_ratio([base[s]["r2"] for s in shared],
                               [have[m][s]["r2"] for s in shared])
            flag = "REAL" if v["p"] < 0.05 else "not established"
            print(f"    cgcnn vs {m:<12} F = {v['F']:5.2f}   p = {v['p']:.4f}   {flag}")
    print("\n    A collapsed run shows up as few epochs AND low R2: it stopped")
    print("    early because it never improved, not because it converged.")

    if len(have) == 3 and "site" in have and "site_feat" in have:
        d, _ = paired(have["site_feat"], have["site"], "r2")
        if d.size >= 2:
            r = paired_test(d)
            print("\n  DO THE FEATURES ADD ANYTHING over the readout alone?")
            print(f"    mean {r['mean']:+.3f}   t({r['n'] - 1}) = {r['t']:.2f}   "
                  f"p = {r['p_t']:.3f}   better on {r['wins']}/{r['n']}")
            if r["p_t"] >= 0.05:
                print("    -> No. The four descriptors buy nothing measurable.")
    print()


if __name__ == "__main__":
    main()
