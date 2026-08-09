"""Is the gap between two architectures larger than the noise?

    python scripts/compare_architectures.py

A table of four numbers to four decimal places invites the reader to rank them.
Whether that ranking means anything depends on how much the numbers would move
if nothing changed but the random seed, and a single training run cannot say.

Retraining each model with five seeds would answer it properly and costs about
twelve hours on this laptop. This does the cheaper thing that is still honest:
every model was evaluated on the SAME 4,308 test materials, so the comparison
can be made *paired* -- resample the test set, and recompute both models' errors
on the same resample. That removes the variation caused by which materials
happen to be in the test set, which is the larger of the two noise sources, and
leaves the question "does model A beat model B on this data".

What it does NOT capture is seed-to-seed variation in training itself. So a
confidence interval that excludes zero here means "A really is better on these
materials", not "A would beat B if both were retrained". The second claim needs
seeds, and this script says so rather than letting the reader assume it.

Paired, because the models' errors are strongly correlated -- a material with an
odd DFT band gap is hard for all of them. Comparing two independent intervals
would hugely overstate the uncertainty in their *difference*.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODELS = REPO / "models"
RESULTS = REPO / "results"

N_BOOT = 20_000
SEED = 42


def load(arch: str, tag: str):
    p = MODELS / arch / f"{tag}_predictions.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_true"], d["y_pred"]


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = N_BOOT):
    """95% CI on mean(err_a) - mean(err_b), resampling materials not models."""
    rng = np.random.default_rng(SEED)
    n = len(err_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p: how often does the resampled difference cross zero?
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(err_a.mean() - err_b.mean()), float(lo), float(hi), float(p)


def main() -> None:
    tag = "band_gap_random_nonmetals"
    archs = ["cgcnn", "mpnn", "gatv2", "megnet"]

    loaded = {}
    for a in archs:
        got = load(a, tag)
        if got is None:
            print(f"  missing predictions for {a} -- skipping")
            continue
        loaded[a] = got

    if len(loaded) < 2:
        print("Need at least two architectures. Run scripts/train_arch.py first.")
        sys.exit(1)

    # Every model must have been scored on the identical test set, or pairing is
    # meaningless. This is worth asserting rather than trusting.
    ref = next(iter(loaded.values()))[0]
    for a, (y, _) in loaded.items():
        if len(y) != len(ref) or not np.allclose(y, ref):
            print(f"  {a} was evaluated on a different test set -- cannot pair")
            sys.exit(1)

    errs = {a: np.abs(y - p) for a, (y, p) in loaded.items()}
    n = len(ref)

    print(f"\n{'=' * 78}")
    print(f"  Paired bootstrap on {n:,} identical test materials, "
          f"{N_BOOT:,} resamples")
    print(f"{'=' * 78}\n")

    order = sorted(errs, key=lambda a: errs[a].mean())
    print(f"  {'model':<9}{'MAE (eV)':>11}{'median':>10}{'params':>10}")
    print("  " + "-" * 40)
    cfgpath = RESULTS / "architectures_band_gap_nonmetals.json"
    sizes = {}
    if cfgpath.exists():
        for k, v in json.loads(cfgpath.read_text(encoding="utf-8")).items():
            sizes[k] = v["random"].get("n_parameters")
    cg = RESULTS / "cgcnn_band_gap_nonmetals.json"
    if cg.exists():
        sizes["cgcnn"] = json.loads(cg.read_text(encoding="utf-8"))["random"].get("n_parameters")
    for a in order:
        s = sizes.get(a)
        print(f"  {a:<9}{errs[a].mean():>11.4f}{np.median(errs[a]):>10.4f}"
              f"{(f'{s:,}' if s else '?'):>10}")

    print("\n  Every pair, best-first. CI is on the DIFFERENCE in MAE.\n")
    print(f"  {'comparison':<20}{'Δ MAE':>9}{'95% CI':>20}{'p':>9}   verdict")
    print("  " + "-" * 74)

    rows = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            d, lo, hi, p = paired_bootstrap(errs[a], errs[b])
            sig = hi < 0 or lo > 0
            verdict = (f"{a} better" if sig else "indistinguishable")
            print(f"  {a + ' vs ' + b:<20}{d:>+9.4f}  [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"{p:>9.4f}   {verdict}")
            rows.append({"a": a, "b": b, "delta_mae": d, "ci_low": lo,
                         "ci_high": hi, "p": p, "significant": bool(sig)})

    out = {
        "test_materials": int(n), "n_bootstrap": N_BOOT, "seed": SEED,
        "mae": {a: float(errs[a].mean()) for a in order},
        "median_ae": {a: float(np.median(errs[a])) for a in order},
        "n_parameters": {a: sizes.get(a) for a in order},
        "pairs": rows,
        "caveat": ("Paired bootstrap over test materials only. Does not capture "
                   "seed-to-seed variation in training; a significant result "
                   "means 'better on these materials', not 'better if retrained'."),
    }
    path = RESULTS / "architecture_significance.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n  This resamples MATERIALS, not training runs. It answers 'is the")
    print("  difference real on this test set', not 'would it survive retraining'.")
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
