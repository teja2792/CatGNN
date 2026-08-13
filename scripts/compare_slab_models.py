"""Compare the readout variants the way they should be compared: paired by seed.

WHY NOT JUST COMPARE THE MEANS
-------------------------------
Round 2 produced these mean R2 over three seeds on the surface-disjoint split:

    cgcnn (mean readout)     0.090
    site  (site readout)     0.207
    site_feat (+ features)   0.308

Read as means, that is a clean 3x improvement and confirmation of the diagnosis.
Read PAIRED -- same seed, same split, model against model -- it says something
different:

    seed        cgcnn     site   site_feat
       0        0.246    0.238       0.234
       1        0.195    0.173       0.229
       2       -0.172    0.211       0.460

On seeds 0 and 1 the site readout is very slightly WORSE than the control. The
entire mean improvement comes from seed 2, where the control collapsed to
R2 = -0.172 and stopped after 17 epochs having never found the signal.

So the demonstrated effect of the site readout is not better typical accuracy.
It is that the control sometimes fails completely and the site readout does not:
seed-to-seed sd falls from 0.228 to 0.033, a factor of 7, and the worst seed goes
from -0.172 to +0.173.

That is a revision of the stated mechanism, not a confirmation of it. The
prediction was that mean-pooling introduces a systematic distortion confounded
with the split. What the runs show is that it makes OPTIMISATION fragile --
consistent with 16x signal dilution producing weak gradients through the atoms
that matter, so some initialisations never escape predicting the mean. Coherent,
but it is the revised story and is labelled as such.

Comparing unpaired means would have reported a mechanism that the per-seed
numbers do not support.

Run:  python scripts/compare_slab_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
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

    print("\n  Paired differences vs the mean-readout control (same seed):")
    base = have.get("cgcnn")
    if base:
        for m in have:
            if m == "cgcnn":
                continue
            d, shared = paired(have[m], base, "r2")
            if not d.size:
                continue
            wins = int((d > 0).sum())
            print(f"    {m:<12} {' '.join(f'{x:+.3f}' for x in d)}"
                  f"   mean {d.mean():+.3f}   better on {wins}/{len(d)} seeds")
            if wins < len(d):
                print("      ^ NOT a uniform win. The mean is carried by the"
                      " seed where the control failed.")

    print("\n  Stability, which is what the numbers actually support:")
    for m in have:
        v = [x["r2"] for x in have[m].values()]
        e = [x["epochs"] for x in have[m].values()]
        print(f"    {m:<12} sd {np.std(v, ddof=1):.3f}   worst {min(v):+.3f}   "
              f"epochs {min(e)}-{max(e)}")
    print("\n  A collapsed run shows up as few epochs AND low R2: the model")
    print("  stopped early because it never improved, not because it converged.\n")


if __name__ == "__main__":
    main()
