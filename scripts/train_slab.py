"""Train a graph network on CO adsorption energies, against an honest baseline.

PRE-REGISTERED PREDICTIONS, written before any run
--------------------------------------------------
1. On the RANDOM split the model beats the surface-mean baseline comfortably,
   and this means almost nothing: 100% of test rows sit on a surface whose other
   nine sites are in training, so memorising surface means is available.
2. On the SURFACE-DISJOINT split the model does markedly worse than on random.
   The gap between the two is the leakage, and it is the point of running both.
3. Beating the surface-disjoint baseline (0.868 eV) is NOT expected to be easy
   with 280 training rows. The honest possible outcomes include "the graph model
   does not beat it", and that will be reported if it happens.

Prediction 3 deserves emphasis. 280 training rows is very little for a network
with tens of thousands of parameters, and the baseline on this split is strong
precisely because it degenerates to the global mean -- which is hard to beat when
the target has no surface-level signal left to exploit.

WHY TWO DATA VARIANTS
---------------------
9 rows from one surface report CO binding between -11.4 and +5.2 eV, which is not
credible for molecular chemisorption. The cause was never identified (see
scripts/inspect_slab_geometries.py). They are 2.2% of the rows and 40% of the
variance, so including them lets a model post a good RMSE by fitting nine points,
and excluding them without saying so would be quiet data cleaning.

Both are run. `--drop-implausible` excludes them. If the conclusion depends on
which is chosen, that dependence is the result.

Run:
    python scripts/train_slab.py --split surface
    python scripts/train_slab.py --split random
    python scripts/train_slab.py --split surface --drop-implausible
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.slab_splits import (  # noqa: E402
    group_mean_baseline, leakage_report, random_split, surface_split)

RESULTS = REPO / "results" / "catalysis"


def bootstrap_rmse(err: np.ndarray, n: int = 10_000, seed: int = 0):
    """Confidence interval on the test RMSE, resampling TEST ROWS.

    With 60 test rows a bare RMSE is not a number anyone should act on: two
    models differing by 0.1 eV may be indistinguishable. Resampling rows (not
    training runs) answers "how much of this is which rows landed in the test
    set", which is the dominant uncertainty at this sample size.
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(err), size=(n, len(err)))
    boots = np.sqrt((err[draws] ** 2).mean(axis=1))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["surface", "random"], default="surface")
    ap.add_argument("--drop-implausible", action="store_true",
                    help="exclude the 9 rows outside the credible energy window")
    ap.add_argument("--drop-subsurface", action="store_true",
                    help="exclude the 14 rows where CO sits below the surface")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=15.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="control: permute targets. Any skill left is a bug.")
    args = ap.parse_args()

    import torch

    from src.models.cgcnn import CGCNN
    from src.models.slab_dataset import SlabStore
    from src.models.train import TrainConfig, train

    store = SlabStore()
    rows = store.rows()
    y = store.y["adsorption_energy"]
    flags = store.flags()

    drop = np.zeros(len(rows), bool)
    if args.drop_implausible:
        drop |= flags["implausible"]
    if args.drop_subsurface:
        drop |= flags["subsurface"]
    keep_ids = {r["id"] for r, d in zip(rows, drop) if not d}
    kept = [r for r in rows if r["id"] in keep_ids]

    print(f"\n{'=' * 76}\n  CO adsorption energy: {args.split}-disjoint split"
          f"\n{'=' * 76}")
    print(f"\n  {len(rows)} rows, {len(kept)} kept"
          + (f" ({drop.sum()} dropped: "
             f"{'implausible ' if args.drop_implausible else ''}"
             f"{'subsurface' if args.drop_subsurface else ''})" if drop.any() else ""))

    if args.shuffle_labels:
        # Permuted BEFORE splitting, so the control breaks the row-to-target
        # correspondence everywhere rather than only in the test set.
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(y))
        store.y["adsorption_energy"] = y[perm].copy()
        y = store.y["adsorption_energy"]
        print("  LABELS SHUFFLED -- this is a control. Skill here means a leak.")

    split = (surface_split if args.split == "surface" else random_split)(
        kept, seed=args.seed)
    L = leakage_report(kept, split)
    y_kept = np.array([r["y"] for r in kept])
    base = group_mean_baseline(y_kept, kept, split)

    print(f"\n  train/val/test           {len(split['train'])}/{len(split['val'])}"
          f"/{len(split['test'])} rows")
    print(f"  test surfaces            {L['test_surfaces']}")
    print(f"  test rows on a training surface: "
          f"{L['test_rows_on_a_training_surface']}/{L['test_rows']} "
          f"= {L['leak_fraction']:.0%}")
    if args.split == "surface":
        assert L["leak_fraction"] == 0.0, "surface split leaked; do not trust this run"
    print(f"\n  BASELINE to beat (surface mean from training rows): {base:.3f} eV")
    print(f"  test-set spread (std)                              "
          f"{y_kept[[k for k, r in enumerate(kept) if r['id'] in set(split['test'])]].std():.3f} eV")

    cfg = TrainConfig(target="adsorption_energy", split=args.split,
                      max_epochs=args.epochs, max_minutes=args.max_minutes,
                      batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                      notes="catalysis CO adsorption, Phase 7")
    out = RESULTS / (f"{args.split}"
                     + ("_clean" if args.drop_implausible else "")
                     + ("_shuffled" if args.shuffle_labels else ""))
    out.mkdir(parents=True, exist_ok=True)

    model = CGCNN()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  model parameters                                   {n_par:,}")
    print(f"    ^ {n_par / max(1, len(split['train'])):,.0f} parameters per training row.")
    print("      Overfitting is the expected failure mode here, not underfitting.\n")

    res = train(model, store, split, cfg, torch, out)

    rmse = float(res["test"]["rmse"])
    # Per-row errors come from the predictions train() saved, so the interval is
    # computed on the same rows the RMSE is, not on a re-run.
    pred_file = out / f"adsorption_energy_{args.split}_predictions.npz"
    err = np.array([])
    if pred_file.exists():
        d = np.load(pred_file, allow_pickle=True)
        err = d["y_true"].astype(float) - d["y_pred"].astype(float)

    print(f"\n{'=' * 76}\n  Result\n{'=' * 76}")
    print(f"\n  graph model     {rmse:.3f} eV   "
          f"(MAE {res['test']['mae']:.3f}, R2 {res['test']['r2']:.3f})")
    lo = hi = None
    if err.size:
        lo, hi = bootstrap_rmse(err)
        print(f"                  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"(resampling {err.size} test rows)")
    print(f"  baseline        {base:.3f} eV   (surface mean from training rows)")
    delta = base - rmse
    print(f"  improvement     {delta:+.3f} eV  ({delta / base:+.0%} of the baseline)")
    if lo is not None:
        if hi >= base:
            print("\n  The interval INCLUDES the baseline: on this test set the model")
            print("  is not distinguishable from predicting the surface mean.")
        else:
            print("\n  The interval excludes the baseline.")
    if args.shuffle_labels:
        print("\n  Labels were shuffled. This number is the floor, not a result.")
    print()

    (out / "summary.json").write_text(json.dumps({
        "split": args.split, "rows_kept": len(kept),
        "dropped_implausible": bool(args.drop_implausible),
        "dropped_subsurface": bool(args.drop_subsurface),
        "shuffled_labels": bool(args.shuffle_labels),
        "leakage": L, "baseline_eV": round(base, 4),
        "test_rmse_eV": round(rmse, 4),
        "test_rmse_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "test_mae_eV": round(float(res["test"]["mae"]), 4),
        "test_r2": round(float(res["test"]["r2"]), 4),
        "parameters": n_par,
        "parameters_per_training_row": round(n_par / max(1, len(split["train"])), 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
