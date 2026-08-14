"""Train a graph network on CO adsorption energies, against an honest baseline.

ROUND 1 (mean readout, 847 rows, one seed) -- WHAT HAPPENED
------------------------------------------------------------
    shuffled labels   R2 -0.013   control passed, no leak in the pipeline
    random split      R2 +0.612   but 100% leakage: memorising surface means
    surface-disjoint  R2 -0.110   WORSE than predicting the mean
    surface, cleaned  R2 +0.183   dropping 30 suspect rows recovers real skill

Predictions 1 and 2 held. Prediction 3 -- "the model may fail to beat the
baseline" -- came true on the honest split.

WHY IT FAILED, AND WHY IT WAS NOT SIMPLY LACK OF DATA
------------------------------------------------------
Diagnosed rather than assumed. `CGCNN.pool` averages over every atom. Adsorption
energy is a property of ONE bond at ONE site, so on a median 34-atom slab the two
atoms that form the bond get about 6% of the readout and 32 spectators -- bulk
interior, and the entire opposite bare face -- get 94%.

Dilution alone would be survivable. The disqualifying part is that slabs run from
22 to 114 atoms, so the adsorbate's share of the mean runs 9.1% down to 1.8%, a
5.2x range, and slab size is a property of the SURFACE. The distortion is
therefore perfectly confounded with the surface-disjoint split: every held-out
surface presents a signal scaling never seen in training, and no quantity of
extra rows on other surfaces teaches it. That is a sufficient mechanism for a
negative R2, and it is a modelling error, not a data shortage.

ROUND 2 PRE-REGISTERED PREDICTIONS, AND HOW THEY CAME OUT (8 seeds)
-------------------------------------------------------------------
4. "The site readout beats the mean readout on the surface-disjoint split."
   PARTLY. Mean R2 difference +0.157, but t(7) = 2.01, p = 0.084, and the sign
   test is 6/8 at p = 0.29. The accuracy claim is NOT established. What is
   established is stability: seed sd 0.188 -> 0.051, F = 13.59, p = 0.0027.
   The control collapses on ~1 seed in 4; the site readout never does.
5. Untested so far -- the random split was not re-run across seeds.
6. HELD, emphatically. Round 1's R2 = -0.110 was one draw from a distribution
   whose sd is 0.188. Single-seed numbers on this split are not interpretable.

And one result that was not predicted: the four per-atom descriptors added in
round 2 do nothing. site_feat against site, paired: mean +0.024, p = 0.61,
better on 5 of 8 seeds. They also weaken the stability advantage (F = 3.76,
p = 0.10 versus 13.59, p = 0.003). The default is therefore `site`, not
`site_feat`. See scripts/compare_slab_models.py.

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
    python scripts/train_slab.py --split surface --model cgcnn --seeds 0 1 2
    python scripts/train_slab.py --split surface --model site  --seeds 0 1 2
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
    # lr 3e-3 was inherited from the 100k-crystal band-gap phase. On 566 rows
    # the validation MAE jumped 0.34 -> 1.43 between adjacent epochs, which is a
    # step size far too large for the gradient noise at this sample size.
    ap.add_argument("--lr", type=float, default=1e-3)
    # weight_decay was 0.0, also inherited. Training loss fell 0.87 -> 0.18 while
    # validation stalled at 0.38: textbook overfitting with 143 parameters per
    # training row and nothing holding them back.
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=64,
                    help="atom feature width. 64 gives 81k parameters on ~590 "
                         "rows; 32 gives ~22k and is the obvious thing to try.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run several seeds and report spread. A single seed on "
                         "13 held-out surfaces cannot distinguish a result from "
                         "noise, and every earlier number here was one seed.")
    ap.add_argument("--model", choices=["cgcnn", "site", "site_feat"],
                    default="site",
                    help="cgcnn = mean over ALL atoms, no extra features (the "
                         "round-1 control); site = site readout only; "
                         "site_feat = site readout + 4 per-atom descriptors. "
                         "Three variants so the readout and the features can be "
                         "attributed separately.")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="control: permute targets. Any skill left is a bug.")
    ap.add_argument("--delta", action="store_true",
                    help="DELTA LEARNING. Fit ridge on the site descriptors "
                         "first, then train the network on what ridge got "
                         "wrong. Final prediction = ridge + network. If the "
                         "network learns nothing it outputs zero and the result "
                         "IS ridge, so this cannot score worse than the "
                         "baseline -- which the plain network does, on 7 of 8 "
                         "seeds. Ramakrishnan et al., JCTC 11, 2087 (2015).")
    args = ap.parse_args()

    import torch

    from src.models.cgcnn import CGCNN
    from src.models.site_cgcnn import SiteCGCNN
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

    base_pred = None
    if args.delta:
        # Ridge is fitted on TRAIN + VAL only. Fitting it on everything would
        # leak the test rows into the baseline the network is corrected against,
        # and the leak would be invisible because the network never sees y.
        from src.features.site_descriptors import descriptor_matrix, ridge_fit
        X = descriptor_matrix(store)
        pos_all = {r["id"]: i for i, r in enumerate(rows)}
        fit_idx = np.array([pos_all[i] for i in split["train"] + split["val"]
                            if i in pos_all])
        predict, _ = ridge_fit(X[fit_idx], y[fit_idx], 1.0)
        base_pred = predict(X)
        te_idx = np.array([pos_all[i] for i in split["test"] if i in pos_all])
        rb = float(np.sqrt(((base_pred[te_idx] - y[te_idx]) ** 2).mean()))
        print("\n  DELTA LEARNING")
        print(f"    ridge on 8 site descriptors, fitted on train+val: "
              f"{rb:.3f} eV on test")
        print("    the network now predicts ridge's RESIDUAL, and the reported")
        print("    number is ridge + network. It cannot come out worse than")
        print(f"    {rb:.3f} unless the network actively adds noise.")
        store.y["adsorption_energy"] = y - base_pred
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

    seeds = args.seeds if args.seeds else [args.seed]
    from src.models.cgcnn import CGCNNConfig
    mcfg = CGCNNConfig(atom_fea_len=args.hidden, h_fea_len=2 * args.hidden,
                       dropout=args.dropout)
    if args.model == "cgcnn":
        def Model(): return CGCNN(mcfg)
    elif args.model == "site":
        def Model(): return SiteCGCNN(mcfg, use_features=False)
    else:
        def Model(): return SiteCGCNN(mcfg, use_features=True)
    print("\n  " + {"cgcnn": "readout: mean over ALL atoms   features: none   (round-1 control)",
                     "site": "readout: binding site + global   features: none",
                     "site_feat": "readout: binding site + global   features: "
                                  "is_adsorbate, is_site, height, coordination",
                     }[args.model])

    runs = []
    for si, sd in enumerate(seeds):
        cfg = TrainConfig(target="adsorption_energy", split=args.split,
                          max_epochs=args.epochs, max_minutes=args.max_minutes,
                          batch_size=args.batch_size, lr=args.lr, seed=sd,
                          weight_decay=args.weight_decay,
                          notes="catalysis CO adsorption, Phase 7")
        tag = "" if args.hidden == 64 else f"_h{args.hidden}"
        out = RESULTS / (f"{args.model}{tag}_{args.split}"
                         + ("_clean" if args.drop_implausible else "")
                         + ("_shuffled" if args.shuffle_labels else "")
                         + (f"_seed{sd}" if len(seeds) > 1 else ""))
        out.mkdir(parents=True, exist_ok=True)

        model = Model()
        n_par = sum(p.numel() for p in model.parameters())
        if si == 0:
            print(f"  model parameters                                   {n_par:,}")
            print(f"    ^ {n_par / max(1, len(split['train'])):,.0f} parameters per training row.")
            print("      Overfitting is the expected failure mode here.\n")
        if len(seeds) > 1:
            print(f"\n  ---- seed {sd} ({si + 1}/{len(seeds)}) ----")
        r = train(model, store, split, cfg, torch, out)
        runs.append((sd, r, out))

    if len(runs) > 1:
        rs = np.array([float(r["test"]["rmse"]) for _, r, _ in runs])
        r2 = np.array([float(r["test"]["r2"]) for _, r, _ in runs])
        print(f"\n{'=' * 76}\n  Across {len(runs)} seeds\n{'=' * 76}")
        print(f"\n  {'seed':>6}{'RMSE':>10}{'R2':>9}")
        for (sd, r, _), a, b in zip(runs, rs, r2):
            print(f"  {sd:>6}{a:>10.3f}{b:>9.3f}")
        print(f"\n  RMSE {rs.mean():.3f} +- {rs.std(ddof=1):.3f}   "
              f"R2 {r2.mean():.3f} +- {r2.std(ddof=1):.3f}")
        print("  ^ seed spread. Any single-seed number below is one draw from this.")

    _, res, out = runs[0]
    rmse = float(res["test"]["rmse"])
    # Per-row errors come from the predictions train() saved, so the interval is
    # computed on the same rows the RMSE is, not on a re-run.
    pred_file = out / f"adsorption_energy_{args.split}_predictions.npz"
    err = np.array([])
    if pred_file.exists():
        d = np.load(pred_file, allow_pickle=True)
        yt = d["y_true"].astype(float)
        yp = d["y_pred"].astype(float)
        if base_pred is not None:
            # Undo the delta transform. Aligned by material_id, never by row
            # order: the prediction file is written in test-set order and
            # base_pred is in store order.
            pos_all = {r["id"]: i for i, r in enumerate(rows)}
            b = np.array([base_pred[pos_all[m]] for m in d["material_id"]])
            yt, yp = yt + b, yp + b
            ss = float(((yt - yt.mean()) ** 2).sum())
            rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
            res["test"] = {"rmse": rmse, "mae": float(np.abs(yt - yp).mean()),
                           "medae": float(np.median(np.abs(yt - yp))),
                           "r2": 1.0 - float(((yt - yp) ** 2).sum()) / ss}
            print("\n  (metrics below are ridge + network, back on the real"
                  " energy scale)")
        err = yt - yp

    print(f"\n{'=' * 76}\n  Result\n{'=' * 76}")
    print(f"\n  graph model     {rmse:.3f} eV   "
          f"(MAE {res['test']['mae']:.3f}, medAE {res['test']['medae']:.3f}, "
          f"R2 {res['test']['r2']:.3f})")
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
        "split": args.split, "model": args.model, "rows_kept": len(kept),
        "seeds": seeds, "lr": args.lr, "weight_decay": args.weight_decay,
        "hidden": args.hidden, "dropout": args.dropout,
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
