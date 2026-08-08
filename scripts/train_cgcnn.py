"""Train CGCNN. Needs PyTorch; no network, no API key.

    python scripts/train_cgcnn.py --selftest              # ~1 min, checks correctness
    python scripts/train_cgcnn.py --smoke                 # ~3 min, 2000 crystals
    python scripts/train_cgcnn.py                         # band gap, random split
    python scripts/train_cgcnn.py --all-splits            # the honest four

Run `--selftest` first. It checks the hand-written convolution against an
independent NumPy implementation and asserts the physical invariances the model
must have, in about a minute. A training run that starts from a wrong layer
converges perfectly happily to the wrong answer, and nothing in the loss curve
looks unusual.

The bar this has to clear, from Phase 2 (`results/baselines.json`):

    band gap, random split, chemistry-only descriptors ...... 0.342 eV
    band gap, non-metals, element-disjoint split ............ 0.694 eV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import RESULTS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, load_split  # noqa: E402

MODELS_DIR = REPO / "models" / "cgcnn"


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print(
            "PyTorch is not installed.\n\n"
            "  CPU-only build (much smaller than the default CUDA wheel):\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        )
        sys.exit(1)


def selftest() -> None:
    """Check the hand-written layer before trusting anything it produces."""
    torch = require_torch()
    import numpy as np

    from src.models.cgcnn import CGCNN, CGCNNConfig
    from src.models import cgcnn_reference as ref

    print("\nSelf-test: hand-written convolution vs independent NumPy reference")
    print("=" * 66)

    cfg = CGCNNConfig(atom_fea_len=16, n_conv=3, h_fea_len=32, use_batch_norm=False)
    model = CGCNN(cfg).double().eval()

    rng = np.random.default_rng(0)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    batch_index = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    dist = rng.uniform(1.5, 7.5, size=len(src))
    u = ref.gaussian_expand(dist)

    batch = {
        "z": torch.from_numpy(z),
        "src": torch.from_numpy(src),
        "dst": torch.from_numpy(dst),
        "u": torch.from_numpy(u),
        "batch_index": torch.from_numpy(batch_index),
        "n_graphs": 2,
    }
    with torch.no_grad():
        got = model(batch).numpy()

    node_ptr = np.array([0, 4, 6])
    want = ref.forward(z, src, dst, u, node_ptr, model.export_numpy_params())

    delta = float(np.max(np.abs(got - want)))
    print(f"  torch      {np.round(got, 10)}")
    print(f"  numpy      {np.round(want, 10)}")
    print(f"  max |diff| {delta:.2e}")
    assert delta < 1e-9, "PyTorch and NumPy implementations disagree"
    print("  -> the two independent implementations agree\n")

    # Permutation of atom labels must not change the prediction.
    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    zp = np.concatenate([z[:4][perm], z[4:]])
    remap = lambda a: np.array([inv[int(v)] if v < 4 else int(v) for v in a])  # noqa: E731
    with torch.no_grad():
        p2 = model({**batch, "z": torch.from_numpy(zp),
                    "src": torch.from_numpy(remap(src)),
                    "dst": torch.from_numpy(remap(dst))}).numpy()
    print(f"  permutation invariance   max |diff| {np.max(np.abs(got - p2)):.2e}")
    assert np.allclose(got, p2, atol=1e-10)

    # A 2x replica of a crystal is the same material: an intensive property must
    # not change. This is what makes the mean readout the right choice.
    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.array([2.0, 2.0, 2.0, 2.0]))
    single = {"z": torch.from_numpy(z1), "src": torch.from_numpy(s1),
              "dst": torch.from_numpy(d1), "u": torch.from_numpy(u1),
              "batch_index": torch.zeros(3, dtype=torch.long), "n_graphs": 1}
    double = {"z": torch.from_numpy(np.concatenate([z1, z1])),
              "src": torch.from_numpy(np.concatenate([s1, s1 + 3])),
              "dst": torch.from_numpy(np.concatenate([d1, d1 + 3])),
              "u": torch.from_numpy(np.concatenate([u1, u1])),
              "batch_index": torch.zeros(6, dtype=torch.long), "n_graphs": 1}
    with torch.no_grad():
        a, b = model(single).numpy(), model(double).numpy()
    print(f"  supercell invariance     max |diff| {np.max(np.abs(a - b)):.2e}")
    assert np.allclose(a, b, atol=1e-10), "mean readout broken -- see docs/cgcnn_math.md"

    print("\n  All checks passed. The layer computes what the equations say.\n")


def estimate(args) -> None:
    """Time one real epoch, then say how many the budget actually buys.

    Same principle as scripts/benchmark_hardware.py: a compute budget chosen
    without measuring it is a guess, and a model reported at 14 epochs when it
    needed 25 is a result about the clock being presented as a result about the
    architecture.
    """
    import time

    torch = require_torch()
    import numpy as np

    from src.models.cgcnn import CGCNN, CGCNNConfig
    from src.models.dataset import GraphStore, N_EDGE_FEATURES, Normaliser
    from src.models.train import physical_cores

    print("\nTiming one epoch on the real training set...")
    store = GraphStore()
    split = load_split(args.split)
    tr = store.indices_for(split["train"])
    y = store.y[args.target]
    tr = tr[np.isfinite(y[tr])]

    torch.set_num_threads(args.threads or physical_cores())
    model = CGCNN(CGCNNConfig(atom_fea_len=args.atom_fea_len,
                              n_conv=args.n_conv, n_edge_fea=N_EDGE_FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    norm = Normaliser(y[tr])
    loss_fn = torch.nn.L1Loss()

    n_probe = min(len(tr), 4000)
    idx = tr[:n_probe]
    t0 = time.perf_counter()
    model.train()
    for i in range(0, n_probe, args.batch_size):
        b = store.collate(idx[i:i + args.batch_size], args.target, torch)
        target = torch.from_numpy(
            norm.encode(b["y"].numpy().astype(np.float64)).astype(np.float32))
        opt.zero_grad()
        loss_fn(model(b), target).backward()
        opt.step()
    per_graph = (time.perf_counter() - t0) / n_probe

    val = store.indices_for(split["val"])
    val = val[np.isfinite(y[val])]
    epoch_min = (per_graph * len(tr) + per_graph * 0.3 * len(val)) / 60.0

    print(f"  {torch.get_num_threads()} threads, {model.n_parameters():,} parameters")
    print(f"  {per_graph * 1000:.2f} ms per training graph")
    print(f"  train {len(tr):,}  val {len(val):,}")
    print(f"  -> about {epoch_min:.1f} min per epoch\n")
    for budget in (20, 35, 60, 90):
        print(f"     {budget:>3} min budget  ->  ~{budget / epoch_min:.0f} epochs"
              + ("   (default)" if budget == 35 else ""))
    print("\n  The smoke run was still improving at epoch 11, so aim for 25+.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="verify the layer and exit")
    ap.add_argument("--smoke", action="store_true", help="2000 crystals, 3 minutes")
    ap.add_argument("--estimate", action="store_true",
                    help="time one real epoch and report what the budget buys")
    ap.add_argument("--target", default="band_gap")
    ap.add_argument("--split", default="random", choices=list(SCHEMES))
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--nonmetals", action="store_true", help="exclude metals (band gap)")
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.estimate:
        estimate(args)
        return

    torch = require_torch()
    if not gb.existing_chunk_indices():
        print("No graph cache.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    from src.models.cgcnn import CGCNN, CGCNNConfig
    from src.models.dataset import GraphStore, N_EDGE_FEATURES
    from src.models.train import TrainConfig, train

    print("\nLoading cached graphs into memory...", flush=True)
    store = GraphStore()
    print(f"  {len(store):,} crystals, {store.z.size:,} atoms, {store.src.size:,} edges")

    splits = list(SCHEMES) if args.all_splits else [args.split]
    all_results = {}

    for split_name in splits:
        print(f"\n{'=' * 66}\n  CGCNN — {args.target}"
              f"{' (non-metals)' if args.nonmetals else ''} — {split_name} split\n{'=' * 66}")

        model = CGCNN(CGCNNConfig(
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            n_edge_fea=N_EDGE_FEATURES,
        ))
        cfg = TrainConfig(
            target=args.target,
            split=split_name,
            max_minutes=3.0 if args.smoke else args.minutes,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            num_threads=args.threads,
            exclude_metals=args.nonmetals,
            subsample_train=2000 if args.smoke else None,
            notes="smoke run" if args.smoke else "",
        )
        res = train(model, store, load_split(split_name), cfg, torch, MODELS_DIR)
        all_results[split_name] = res

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"cgcnn_{args.target}{'_nonmetals' if args.nonmetals else ''}"
    if args.smoke:
        tag += "_smoke"
    out = RESULTS / f"{tag}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # The comparison that matters: did it beat the descriptors?
    base_path = RESULTS / "baselines.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())["results"]
        target = args.target + ("_nonmetals" if args.nonmetals else "")
        print(f"\n{'=' * 66}\n  CGCNN vs the Phase 2 descriptor baselines\n{'=' * 66}")
        print(f"  {'split':<10}{'CGCNN':>10}{'best descriptor':>18}{'verdict':>14}")
        for s, r in all_results.items():
            sel = [b for b in base if b["split"] == s and b["target"] == target
                   and b["model"] != "mean"]
            if not sel:
                continue
            best = min(sel, key=lambda b: b["mae"])
            d = 100 * (best["mae"] - r["test"]["mae"]) / best["mae"]
            verdict = f"{d:+.1f}%" + ("  WINS" if d > 0 else "")
            print(f"  {s:<10}{r['test']['mae']:>10.4f}"
                  f"{best['mae']:>13.4f} ({best['model']:<3}){verdict:>14}")

    if args.smoke:
        print("\n  NOTE: smoke run (2,000 crystals, 3 minutes). Not a result.")
    print(f"\nwrote {out.relative_to(REPO)} and {MODELS_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
