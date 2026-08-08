"""Run every baseline against every feature block, split and target.

This produces the table the neural networks have to beat. Run after
`make_splits.py`. No network, no API key.

    python scripts/run_baselines.py --quick     # gbm only, ~2 minutes
    python scripts/run_baselines.py             # full sweep

Two reporting choices worth knowing about:

**Metals are reported separately for band gap.** 58.9% of the dataset has a band
gap of exactly zero, so predicting zero for everything already scores 0.739 eV.
A model that has only learned "is this a metal" would look like a decent band-gap
model on the pooled number. Both are printed.

**MAE and median absolute error together.** The dataset contains solid helium at
17.9 eV. A handful of real extremes drags the mean, and the gap between the two
numbers tells you how much of the score is a few hard cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.config import CACHE, RANDOM_SEED, RESULTS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, load_split  # noqa: E402
from src.features.descriptors import BLOCKS, feature_names, featurise  # noqa: E402
from src.models.baselines import MODELS, evaluate, permutation_importance_fast  # noqa: E402

TARGETS = ("band_gap", "formation_energy_per_atom", "energy_above_hull")
UNITS = {"band_gap": "eV", "formation_energy_per_atom": "eV/atom",
         "energy_above_hull": "eV/atom"}

FEATURE_CACHE = CACHE / "features"


def build_features(block: str) -> tuple[np.ndarray, list[str], dict]:
    """Featurise every cached graph, caching the matrix to disk."""
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    path = FEATURE_CACHE / f"{block}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            X, ids = d["X"], [str(v) for v in d["material_id"]]
        targets = json.loads((FEATURE_CACHE / f"{block}_targets.json").read_text())
        return X, ids, {"ids": ids, "targets": targets}

    print(f"  featurising ({block})...", end="", flush=True)
    t0 = time.perf_counter()
    rows, ids, y = [], [], {t: [] for t in TARGETS}

    for i in sorted(gb.existing_chunk_indices()):
        meta = json.loads((gb.GRAPHS / f"meta_{i:04d}.json").read_text(encoding="utf-8"))
        chunk = gb.load_graph_chunk(i)
        for k, g in enumerate(gb.iter_graphs(chunk)):
            m = meta[k]
            # Species come back from the graph's atomic numbers, so the features
            # describe exactly the crystal the GNN will see -- not a separate
            # reading of the source file that could drift out of step with it.
            species = [ELEMENT_OF_Z.get(int(z), "") for z in g["z"]]
            rows.append(featurise(m, species, block))
            ids.append(g["material_id"])
            for t in TARGETS:
                y[t].append(g.get(t, np.nan))

    X = np.vstack(rows).astype(np.float32)
    np.savez_compressed(path, X=X, material_id=np.array(ids))
    (FEATURE_CACHE / f"{block}_targets.json").write_text(
        json.dumps({t: [float(v) for v in y[t]] for t in TARGETS}))
    print(f" {X.shape[0]:,} x {X.shape[1]} in {time.perf_counter() - t0:.0f}s")
    return X, ids, {"ids": ids, "targets": {t: y[t] for t in TARGETS}}


ELEMENT_OF_Z = {v: k for k, v in gb.Z_OF.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="gbm only, band gap only")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--splits", nargs="+", default=list(SCHEMES))
    ap.add_argument("--subsample", type=int, default=None,
                    help="train on at most N materials per split (fast sanity run)")
    ap.add_argument("--importance", action="store_true",
                    help="also compute permutation importance (slow)")
    args = ap.parse_args()

    if args.quick:
        args.models, args.blocks, args.targets = ["gbm"], ["composition", "both"], ["band_gap"]

    if not gb.existing_chunk_indices():
        print("No graph cache.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    print("\nBuilding descriptor matrices")
    data = {b: build_features(b) for b in args.blocks}
    ids = data[args.blocks[0]][1]
    targets = json.loads((FEATURE_CACHE / f"{args.blocks[0]}_targets.json").read_text())
    pos = {m: i for i, m in enumerate(ids)}

    results = []
    t_all = time.perf_counter()

    for split_name in args.splits:
        print(f"\n  --- split: {split_name} ---", flush=True)
        split = load_split(split_name)
        tr = np.array([pos[m] for m in split["train"] if m in pos])
        te = np.array([pos[m] for m in split["test"] if m in pos])

        if args.subsample:
            # Subsample the TRAINING set only. Shrinking the test set as well
            # would make the reported error noisier without making the run
            # meaningfully faster, and the point of a sanity run is to check the
            # pipeline, not to produce a number anyone will quote.
            rng = np.random.default_rng(RANDOM_SEED)
            if tr.size > args.subsample:
                tr = np.sort(rng.choice(tr, args.subsample, replace=False))

        for target in args.targets:
            y = np.asarray(targets[target], dtype=np.float64)
            ok = np.isfinite(y)
            tr_i, te_i = tr[ok[tr]], te[ok[te]]

            for block in args.blocks:
                X = data[block][0]
                for model in args.models:
                    r = evaluate(model, X[tr_i], y[tr_i], X[te_i], y[te_i])
                    r.update(split=split_name, target=target, block=block,
                             n_train=len(tr_i))
                    results.append(r)
                    print(f"  {split_name:<8} {target:<26} {block:<14} "
                          f"{model:<6} MAE {r['mae']:7.4f}  med {r['medae']:7.4f}  "
                          f"R2 {r['r2']:6.3f}  ({r['fit_seconds']}s)", flush=True)

            # Non-metals only. Predicting zero everywhere is a strong-looking
            # band-gap model on the pooled set and a useless one in practice.
            if target == "band_gap":
                nm = ok & (y > 1e-6)
                tr_n, te_n = tr[nm[tr]], te[nm[te]]
                for block in args.blocks:
                    X = data[block][0]
                    for model in args.models:
                        r = evaluate(model, X[tr_n], y[tr_n], X[te_n], y[te_n])
                        r.update(split=split_name, target="band_gap_nonmetals",
                                 block=block, n_train=len(tr_n))
                        results.append(r)
                        print(f"  {split_name:<8} {'band_gap (non-metals)':<26} "
                              f"{block:<14} {model:<6} MAE {r['mae']:7.4f}  "
                              f"med {r['medae']:7.4f}  R2 {r['r2']:6.3f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "baselines.json"
    out.write_text(json.dumps({
        "results": results,
        "seconds": round(time.perf_counter() - t_all, 1),
        "subsampled_train_to": args.subsample,
        "complete": args.subsample is None and set(args.splits) == set(SCHEMES),
    }, indent=2), encoding="utf-8")
    if args.subsample:
        print(f"\n  NOTE: training sets were capped at {args.subsample:,}. These numbers")
        print("  are a pipeline check, not results. Re-run without --subsample.")

    # ---- the comparison this repo exists to make -------------------------
    print("\n" + "=" * 78)
    print("  Does structure help? Best model per block, MAE")
    print("=" * 78)
    for target in sorted({r["target"] for r in results}):
        print(f"\n  {target}  ({UNITS.get(target.replace('_nonmetals', ''), '')})")
        print(f"    {'split':<10}" + "".join(f"{b:>17}" for b in args.blocks) + "     gain")
        for split_name in args.splits:
            cells, best = [], {}
            for b in args.blocks:
                sel = [r for r in results
                       if r["split"] == split_name and r["target"] == target
                       and r["block"] == b and r["model"] != "mean"]
                if not sel:
                    cells.append(f"{'-':>17}")
                    continue
                bm = min(sel, key=lambda r: r["mae"])
                best[b] = bm["mae"]
                cells.append(f"{bm['mae']:>12.4f} ({bm['model'][:3]})")
            gain = ""
            if "composition" in best and "both" in best:
                d = 100 * (best["composition"] - best["both"]) / best["composition"]
                gain = f"  {d:+5.1f}%"
            print(f"    {split_name:<10}" + "".join(cells) + gain)

    print("\n  'gain' = how much adding cheap structural features improves on")
    print("  composition alone. That is the number a GNN has to beat to justify")
    print("  itself, and it is measured, not assumed.")

    if args.importance:
        print("\n  Permutation importance (gbm, both, random split, band gap)...")
        from src.models.baselines import build_model
        X = data["both"][0]
        y = np.asarray(targets["band_gap"], dtype=np.float64)
        split = load_split("random")
        tr = np.array([pos[m] for m in split["train"] if m in pos])[:20000]
        te = np.array([pos[m] for m in split["test"] if m in pos])[:5000]
        model = build_model("gbm")
        model.fit(X[tr], y[tr])
        imp = permutation_importance_fast(model, X[te].copy(), y[te],
                                          feature_names("both"), top_k=15)
        for row in imp:
            print(f"    {row['feature']:<40} +{row['mae_increase']:.4f} eV")
        (RESULTS / "baseline_importance.json").write_text(json.dumps(imp, indent=2))

    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
