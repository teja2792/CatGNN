"""The baseline the graph network should have been measured against all along.

Phase 7 has compared the network to "predict this surface's mean binding energy".
On a surface-disjoint split that baseline collapses to the global mean, because a
surface never seen has no mean to borrow -- so it is a very low bar and the
network has been clearing it by ~17% and calling that progress.

This runs ridge regression on eight hand-written site descriptors
(src/features/site_descriptors.py) over the SAME splits and the SAME seeds, so
the two numbers are directly comparable. Nothing is learned about chemistry; the
features are what a surface chemist would write down in a minute.

Run:  python scripts/baseline_site_ridge.py --seeds 0 1 2 3 4 5 6 7
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
    group_mean_baseline, random_split, surface_split)
from src.features.site_descriptors import (  # noqa: E402
    FEATURE_NAMES, descriptor_matrix, ridge_fit)
from src.models.seed_stats import paired_test  # noqa: E402

RESULTS = REPO / "results" / "catalysis"


def metrics(y_true, y_pred):
    err = y_true - y_pred
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": 1.0 - float((err ** 2).sum()) / ss if ss else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["surface", "random"], default="surface")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--keep-implausible", action="store_true")
    args = ap.parse_args()

    from src.models.slab_dataset import SlabStore

    store = SlabStore()
    rows = store.rows()
    flags = store.flags()
    keep = np.ones(len(rows), bool) if args.keep_implausible else ~flags["implausible"]
    kept = [r for r, k in zip(rows, keep) if k]
    pos = {r["id"]: i for i, r in enumerate(rows)}

    X = descriptor_matrix(store)
    y = store.y["adsorption_energy"]

    print(f"\n{'=' * 76}\n  Ridge on {len(FEATURE_NAMES)} site descriptors, "
          f"{args.split}-disjoint\n{'=' * 76}")
    print(f"\n  {len(kept)} rows, {len(FEATURE_NAMES)} features, "
          f"{len(FEATURE_NAMES) + 1} parameters")
    print("  (the graph network uses 81,345 on the same rows)")

    splitter = surface_split if args.split == "surface" else random_split
    r2s, rmses, ws = [], [], []
    print(f"\n  {'seed':>6}{'RMSE':>10}{'MAE':>9}{'R2':>9}")
    for sd in args.seeds:
        sp = splitter(kept, seed=sd)
        tr = np.array([pos[i] for i in sp["train"] + sp["val"] if i in pos])
        te = np.array([pos[i] for i in sp["test"] if i in pos])
        # Train+val together: ridge has no early stopping, so holding out a
        # validation set would hand the network more data than the baseline and
        # make the comparison unfair in the network's favour.
        predict, w = ridge_fit(X[tr], y[tr], args.lam)
        m = metrics(y[te], predict(X[te]))
        r2s.append(m["r2"]); rmses.append(m["rmse"]); ws.append(w)
        print(f"  {sd:>6}{m['rmse']:>10.3f}{m['mae']:>9.3f}{m['r2']:>9.3f}")

    r2s, rmses = np.array(r2s), np.array(rmses)
    print(f"  {'mean':>6}{rmses.mean():>10.3f}{'':>9}{r2s.mean():>9.3f}")
    print(f"  {'sd':>6}{rmses.std(ddof=1):>10.3f}{'':>9}{r2s.std(ddof=1):>9.3f}")

    sm = [group_mean_baseline(np.array([r["y"] for r in kept]), kept,
                              splitter(kept, seed=sd)) for sd in args.seeds]
    print(f"\n  surface-mean baseline, same splits: {np.mean(sm):.3f} eV")
    print(f"  ridge on site descriptors:          {rmses.mean():.3f} eV")

    W = np.array(ws)[:, :-1]
    order = np.argsort(-np.abs(W.mean(0)))
    print("\n  Standardised coefficients (mean over seeds), largest first:")
    for k in order:
        print(f"    {FEATURE_NAMES[k]:<16}{W[:, k].mean():+7.3f}  "
              f"+- {W[:, k].std(ddof=1):.3f}")
    print("\n  These are the chemistry the network has to at least match:")
    print("  site type, coordination of the binding atoms, and how far the")
    print("  molecule sits from the surface.")

    # ---- compare against the network, if it has been run on the same seeds ----
    net = {}
    for model in ("cgcnn", "site", "site_feat"):
        stem = f"{model}_{args.split}" + ("" if args.keep_implausible else "_clean")
        per = {}
        for d in RESULTS.glob(f"{stem}_seed*"):
            f = d / f"adsorption_energy_{args.split}.json"
            if not f.exists():
                continue
            try:
                s = int(d.name.rsplit("seed", 1)[1])
            except (IndexError, ValueError):
                continue
            per[s] = float(json.loads(f.read_text(encoding="utf-8"))["test"]["r2"])
        if per:
            net[model] = per

    if net:
        print(f"\n{'=' * 76}\n  Ridge versus the graph network, paired by seed"
              f"\n{'=' * 76}")
        mine = dict(zip(args.seeds, r2s))
        for model, per in net.items():
            shared = sorted(set(mine) & set(per))
            if len(shared) < 2:
                continue
            d = np.array([per[s] - mine[s] for s in shared])
            r = paired_test(d)
            print(f"\n  {model} minus ridge, on {len(shared)} shared seeds")
            print(f"    {' '.join(f'{x:+.3f}' for x in d)}")
            print(f"    mean {r['mean']:+.3f}   t({r['n'] - 1}) = {r['t']:.2f}   "
                  f"p = {r['p_t']:.3f}   network better on {r['wins']}/{r['n']}")
            if r["mean"] < 0:
                print("    -> the NETWORK LOSES to a linear model on 8 numbers.")

    out = RESULTS / f"ridge_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "split": args.split, "rows": len(kept), "lambda": args.lam,
        "features": FEATURE_NAMES, "seeds": args.seeds,
        "rmse_mean": round(float(rmses.mean()), 4),
        "rmse_sd": round(float(rmses.std(ddof=1)), 4),
        "r2_mean": round(float(r2s.mean()), 4),
        "r2_sd": round(float(r2s.std(ddof=1)), 4),
        "surface_mean_baseline_eV": round(float(np.mean(sm)), 4),
        "coefficients": {n: round(float(W[:, k].mean()), 4)
                         for k, n in enumerate(FEATURE_NAMES)},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {(out / 'summary.json').relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
