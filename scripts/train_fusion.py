"""Phase 5 -- does giving the network the periodic table fix the collapse?

    python scripts/train_fusion.py --selftest       # ~1 min, no training
    python scripts/train_fusion.py --atoms properties --split element --nonmetals
    python scripts/train_fusion.py --atoms properties --split random  --nonmetals
    python scripts/train_fusion.py --atoms both      --split element --nonmetals
    python scripts/train_fusion.py --atoms both --composition --split element --nonmetals

THE PREDICTION, WRITTEN DOWN BEFORE THE RUNS
--------------------------------------------
Phase 3 measured a graph network collapsing from 0.414 eV on a random split to
1.019 eV when the test set contains elements training never saw -- while a plain
descriptor model, which reads electronegativity out of a table, degraded only to
0.694 eV. The diagnosis was that the network's per-element vectors are *learned*,
so an unseen element gets a randomly initialised row that training never touched.

If that diagnosis is right, then replacing the learned lookup with tabulated
element properties should:

    1. improve the element-disjoint result substantially -- towards 0.694 eV or
       better, because the model now has what the descriptor model has;
    2. leave the random-split result roughly unchanged, or slightly worse, since
       31 fixed properties are less expressive than 64 free numbers per element
       when you do have training data for every element;
    3. make 'both' the best of the three on strict splits and the best or equal
       on easy ones, since it can memorise where it has data and fall back on
       chemistry where it does not.

If instead 'properties' fails to help on the element split, the diagnosis was
wrong and the collapse is about something else -- most likely that unseen
elements come with unseen *structures* too. Either outcome is reportable; the
prediction is recorded here so it cannot be quietly rewritten afterwards.

Run `--selftest` first. It checks the element table against chemistry that is
true independently of this code, and confirms the held-out elements of the
element-disjoint split are actually covered by the table -- if they were not,
'properties' would be handing the model zeros and the experiment would measure
nothing.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import RESULTS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, SPLITS, load_split  # noqa: E402
from src.features.element_features import element_feature_table  # noqa: E402

MODELS_DIR = REPO / "models"


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("PyTorch is not installed.\n\n"
              "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
        sys.exit(1)


def variant_name(atoms: str, composition: bool, backbone: str) -> str:
    tag = f"{backbone}_{atoms}"
    return tag + "_comp" if composition else tag


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def selftest() -> None:
    torch = require_torch()

    from src.models import cgcnn_reference as ref
    from src.models.fusion import AtomFeaturiser, FusedGNN, FusionConfig

    print("\nSelf-test: the periodic table, and the models that use it")
    print("=" * 70)

    table, known, names = element_feature_table()
    print(f"\n  table {table.shape[0]} rows x {table.shape[1]} properties, "
          f"{int(known.sum())} elements covered")
    assert not np.isnan(table).any() and not np.isinf(table).any()
    assert np.allclose(table[known].mean(axis=0), 0, atol=1e-5)
    assert np.allclose(table[known].std(axis=0), 1, atol=1e-5)
    print("  standardised, no holes")

    # Checked against chemistry, not against this code's own output. A table off
    # by one row would still be standardised and still have no holes.
    print("\n  Chemical sanity — similarity of element vectors:")
    checks = [
        ("Cl vs Br  (both halogens)", 17, 35, "high"),
        ("Na vs K   (both alkali)", 11, 19, "high"),
        ("S  vs Se  (both chalcogens)", 16, 34, "high"),
        ("Cl vs Na  (halogen vs alkali)", 17, 11, "low"),
        ("Cl vs Fe  (halogen vs transition metal)", 17, 26, "low"),
    ]
    for label, a, b, want in checks:
        c = cos(table[a], table[b])
        ok = (c > 0.75) if want == "high" else (c < 0.50)
        print(f"    {label:<42}{c:+.3f}   {'ok' if ok else 'FAIL'}")
        assert ok, f"{label} came out at {c:+.3f}; the table may be misaligned"

    # The elements actually held out by the element-disjoint split must be in the
    # table, or 'properties' hands the model zeros and the experiment measures
    # nothing at all.
    summary = SPLITS / "summary.json"
    if summary.exists():
        blob = json.loads(summary.read_text(encoding="utf-8"))
        held = sorted(set(blob["schemes"].get("element", {}).get("held_out_elements", [])))
        if held:
            from src.features.descriptors import element_table as raw
            elements, _ = raw()
            missing = [e for e in held
                       if e not in elements or not known[int(elements[e]["Z"])]]
            print(f"\n  element-disjoint split holds out {len(held)} elements:")
            print(f"    {', '.join(held)}")
            if missing:
                print(f"    NOT IN THE TABLE: {missing}  <-- experiment is invalid")
                sys.exit(1)
            print("    every one is covered by the element table")
    else:
        print("\n  (no splits built yet — skipping the held-out element check)")

    print("\n  Held-out chemistry survives featurisation:")
    for mode in ("learned", "properties"):
        f = AtomFeaturiser(mode, d=32).double().eval()
        with torch.no_grad():
            v = f(torch.tensor([16, 34, 29])).numpy()      # S, Se, Cu
        print(f"    {mode:<11} Se~S {cos(v[1], v[0]):+.3f}   Se~Cu {cos(v[1], v[2]):+.3f}")

    # Invariances, for every combination that will be trained.
    rng = np.random.default_rng(0)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    bidx = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    u = ref.gaussian_expand(rng.uniform(1.5, 7.5, size=len(src)))
    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))

    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.full(4, 2.0))

    print(f"\n  {'backbone':<10}{'atoms':<12}{'params':>9}{'permutation':>13}"
          f"{'supercell':>12}")
    print("  " + "-" * 56)
    for backbone in ("cgcnn", "mpnn", "gatv2"):
        for atoms in ("learned", "properties", "both"):
            m = FusedGNN(FusionConfig(
                backbone=backbone, atom_features=atoms, atom_fea_len=16,
                n_conv=2, h_fea_len=32, use_batch_norm=False)).double().eval()

            def run(zz, ss, dd, uu, bb, n):
                with torch.no_grad():
                    return m({"z": torch.from_numpy(zz), "src": torch.from_numpy(ss),
                              "dst": torch.from_numpy(dd), "u": torch.from_numpy(uu),
                              "batch_index": torch.from_numpy(bb),
                              "n_graphs": n}).numpy()

            base = run(z, src, dst, u, bidx, 2)
            p = run(np.concatenate([z[:4][perm], z[4:]]), remap(src), remap(dst),
                    u, bidx, 2)
            one = run(z1, s1, d1, u1, np.zeros(3, dtype=np.int64), 1)
            two = run(np.concatenate([z1, z1]), np.concatenate([s1, s1 + 3]),
                      np.concatenate([d1, d1 + 3]), np.concatenate([u1, u1]),
                      np.zeros(6, dtype=np.int64), 1)

            dp, ds = float(np.max(np.abs(base - p))), float(np.max(np.abs(one - two)))
            print(f"  {backbone:<10}{atoms:<12}{m.n_parameters():>9,}{dp:>13.1e}{ds:>12.1e}")
            assert dp < 1e-9 and ds < 1e-9, f"{backbone}/{atoms} violates an invariance"

    print("\n  All good. The experiment is measuring what it claims to.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--atoms", default="properties",
                    choices=["learned", "properties", "both"],
                    help="how each atom's starting vector is produced")
    ap.add_argument("--composition", action="store_true",
                    help="also concatenate the 192-feature composition descriptor")
    ap.add_argument("--backbone", default="cgcnn", choices=["cgcnn", "mpnn", "gatv2"])
    ap.add_argument("--target", default="band_gap")
    ap.add_argument("--split", default="element", choices=list(SCHEMES))
    ap.add_argument("--nonmetals", action="store_true")
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2000 crystals, 3 minutes")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="Phase 6 control: train on randomly permuted band gaps")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    torch = require_torch()
    if not gb.existing_chunk_indices():
        print("No graph cache.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    from src.models.dataset import GraphStore, N_EDGE_FEATURES
    from src.models.fusion import FusedGNN, FusionConfig
    from src.models.train import TrainConfig, select_indices, train

    print("\nLoading cached graphs into memory...", flush=True)
    store = GraphStore()
    print(f"  {len(store):,} crystals, {store.z.size:,} atoms, {store.src.size:,} edges")

    if args.shuffle_labels:
        # The sanity check for Phase 6. Permuting the targets destroys every
        # relationship between a crystal and its band gap while leaving the
        # distribution of targets untouched, so a model trained on this CANNOT
        # have learned chemistry -- but it will still fit, and an attribution
        # method will still produce a confident-looking ranking from it. If that
        # ranking resembles the real model's, the attribution is measuring the
        # method rather than the model.
        #
        # Permuted once, globally, before splitting: permuting within a split
        # would leave the train/test relationship subtly intact.
        rng = np.random.default_rng(args.seed)
        y = store.y[args.target]
        finite = np.where(np.isfinite(y))[0]
        y[finite] = y[rng.permutation(finite)]
        print(f"  LABELS SHUFFLED — {len(finite):,} targets permuted (seed "
              f"{args.seed}). This is a control, not a model.")

    split = load_split(args.split)
    tcfg = TrainConfig(
        target=args.target, split=args.split,
        max_minutes=3.0 if args.smoke else args.minutes,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        num_threads=args.threads, exclude_metals=args.nonmetals,
        subsample_train=2000 if args.smoke else None,
        notes=f"atoms={args.atoms} composition={args.composition} "
              f"backbone={args.backbone}"
              + (" LABELS-SHUFFLED-CONTROL" if args.shuffle_labels else ""),
    )

    n_comp = 198
    if args.composition:
        # Fitted on exactly the rows training will use -- see select_indices.
        tr, _, _ = select_indices(store, split, tcfg)
        n_comp = store.attach_composition(tr)
        print(f"  attached {n_comp} composition descriptors "
              f"(scaler fitted on {len(tr):,} training crystals, "
              f"{store.comp_scaler['dead_columns']} constant columns zeroed)")

    cfg = FusionConfig(
        backbone=args.backbone, atom_features=args.atoms,
        use_composition=args.composition, n_composition=n_comp,
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        n_edge_fea=N_EDGE_FEATURES,
    )
    model = FusedGNN(cfg)

    name = variant_name(args.atoms, args.composition, args.backbone)
    if args.shuffle_labels:
        name += "_shuffled"
    print(f"\n{'=' * 70}\n  {name.upper()} — {args.target}"
          f"{' (non-metals)' if args.nonmetals else ''} — {args.split} split\n{'=' * 70}")

    # An element in the data that the table does not cover would be silently fed
    # zeros -- the very failure this phase exists to remove.
    if args.atoms in ("properties", "both"):
        unknown = model.featuriser.unknown_elements(
            torch.from_numpy(np.unique(store.z).astype(np.int64)))
        unknown = [int(u) for u in unknown if int(u) != 0]
        if unknown:
            print(f"  WARNING: Z={unknown} are in the data but not in the element "
                  f"table; those atoms get the average element.")

    res = train(model, store, split, tcfg, torch, MODELS_DIR / name)
    res["variant"] = name
    res["atom_features"] = args.atoms
    res["use_composition"] = args.composition
    res["backbone"] = args.backbone
    res["shuffled_labels"] = args.shuffle_labels
    res["model_config"] = cfg.to_dict()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"fusion_{args.target}{'_nonmetals' if args.nonmetals else ''}"
    if args.smoke:
        tag += "_smoke"
    path = RESULTS / f"{tag}.json"

    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    merged.setdefault(name, {})[args.split] = res
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    del model
    gc.collect()

    print(f"\n{'=' * 70}\n  Phase 5 so far — {args.target}"
          f"{' (non-metals)' if args.nonmetals else ''}\n{'=' * 70}")
    splits = [s for s in SCHEMES if any(s in v for v in merged.values())]
    print(f"  {'variant':<22}" + "".join(f"{s:>12}" for s in splits))
    print("  " + "-" * (22 + 12 * len(splits)))
    for v, runs in merged.items():
        cells = "".join(f"{runs[s]['test']['mae']:>12.4f}" if s in runs else f"{'—':>12}"
                        for s in splits)
        print(f"  {v:<22}{cells}")

    # The two numbers this phase is trying to beat, printed every time so the
    # result is never read without its baseline.
    print("\n  What Phase 5 is trying to beat, on the element-disjoint split:")
    print("    CGCNN with a learned element table   1.019 eV   (collapses)")
    print("    Composition descriptors alone        0.694 eV   (degrades gracefully)")
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
