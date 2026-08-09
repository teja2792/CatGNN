"""Train any of the four architectures, under one shared budget.

    python scripts/train_arch.py --selftest                     # ~1 min, all four
    python scripts/train_arch.py --arch mpnn   --split random --nonmetals
    python scripts/train_arch.py --arch megnet --split random --nonmetals
    python scripts/train_arch.py --arch gatv2  --split random --nonmetals

Run `--selftest` first. It checks each model against the physical invariances it
must have -- relabel the atoms, replicate the cell, batch two crystals together --
and against the NumPy attention reference. A model that fails any of those trains
perfectly happily and reports a number that means nothing.

One architecture per process is deliberate: a fresh process returns every byte to
the OS when it exits, which matters on a memory-constrained laptop. Results merge
into a single file, so running them separately still builds one comparison table.

Each model answers one question CGCNN alone cannot:

    mpnn    is CGCNN's gate worth anything over generic message passing?
    megnet  does an explicit whole-crystal global state help?
    gatv2   real attention, so Phase 6's attention maps mean what they say
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import RESULTS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, load_split  # noqa: E402

MODELS_DIR = REPO / "models"


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("PyTorch is not installed.\n\n"
              "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
        sys.exit(1)


def selftest() -> None:
    """Every architecture must satisfy the same physical contract."""
    torch = require_torch()
    import numpy as np

    from src.models import cgcnn_reference as ref
    from src.models.architectures import ARCHITECTURES, ArchConfig, GATv2, build

    print("\nSelf-test: physical invariances, all architectures\n" + "=" * 66)

    rng = np.random.default_rng(0)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    bidx = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    u = ref.gaussian_expand(rng.uniform(1.5, 7.5, size=len(src)))
    batch = {"z": torch.from_numpy(z), "src": torch.from_numpy(src),
             "dst": torch.from_numpy(dst), "u": torch.from_numpy(u),
             "batch_index": torch.from_numpy(bidx), "n_graphs": 2}

    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))

    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.full(4, 2.0))

    print(f"  {'model':<9}{'params':>10}{'permutation':>14}{'supercell':>13}{'batching':>11}")
    print("  " + "-" * 55)

    for name in ARCHITECTURES:
        m = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double().eval()

        def run(zz, ss, dd, uu, bb, n):
            with torch.no_grad():
                return m({"z": torch.from_numpy(zz), "src": torch.from_numpy(ss),
                          "dst": torch.from_numpy(dd), "u": torch.from_numpy(uu),
                          "batch_index": torch.from_numpy(bb), "n_graphs": n}).numpy()

        base = run(z, src, dst, u, bidx, 2)
        p = run(np.concatenate([z[:4][perm], z[4:]]), remap(src), remap(dst), u, bidx, 2)
        one = run(z1, s1, d1, u1, np.zeros(3, dtype=np.int64), 1)
        two = run(np.concatenate([z1, z1]), np.concatenate([s1, s1 + 3]),
                  np.concatenate([d1, d1 + 3]), np.concatenate([u1, u1]),
                  np.zeros(6, dtype=np.int64), 1)
        a = run(z[:4], src[:6], dst[:6], u[:6], np.zeros(4, dtype=np.int64), 1)
        b = run(z[4:], src[6:] - 4, dst[6:] - 4, u[6:], np.zeros(2, dtype=np.int64), 1)

        d_perm = float(np.max(np.abs(base - p)))
        d_super = float(np.max(np.abs(one - two)))
        d_batch = float(np.max(np.abs(base - np.concatenate([a, b]))))
        print(f"  {name:<9}{m.n_parameters():>10,}{d_perm:>14.1e}{d_super:>13.1e}{d_batch:>11.1e}")

        for label, delta in (("permutation", d_perm), ("supercell", d_super),
                             ("batching", d_batch)):
            assert delta < 1e-9, f"{name} fails {label} invariance ({delta:.2e})"

    # GATv2's attention must be a real distribution, and must match the NumPy
    # reference. That property is the entire reason it is in the comparison.
    g = GATv2(ArchConfig(atom_fea_len=16, n_conv=2, n_heads=4,
                         use_batch_norm=False)).double().eval()
    with torch.no_grad():
        w = g.attention_weights(batch)
    worst = max(abs(float(w[L][src == a0].sum(dim=0).min()) - 1.0)
                for L in range(len(w)) for a0 in np.unique(src))
    print(f"\n  GATv2 attention sums to 1 over each atom's neighbours "
          f"(worst deviation {worst:.1e})")
    assert worst < 1e-9

    scores = rng.normal(scale=3.0, size=(6, 4))
    s2 = np.array([0, 0, 0, 1, 1, 2])
    from src.models.architectures import GATv2Conv
    got = GATv2Conv._softmax_by_source(torch.from_numpy(scores),
                                       torch.from_numpy(s2), 3).numpy()
    want = ref.softmax_by_source(scores, s2, 3)
    print(f"  torch attention vs NumPy reference: max |diff| "
          f"{np.max(np.abs(got - want)):.1e}")
    assert np.allclose(got, want, atol=1e-12)

    print("\n  All architectures pass. Safe to train.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arch", default="cgcnn",
                    choices=["cgcnn", "mpnn", "megnet", "gatv2"])
    ap.add_argument("--target", default="band_gap")
    ap.add_argument("--split", default="random", choices=list(SCHEMES))
    ap.add_argument("--nonmetals", action="store_true")
    ap.add_argument("--minutes", type=float, default=35.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--atom-fea-len", type=int, default=64)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2000 crystals, 3 minutes")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    torch = require_torch()
    if not gb.existing_chunk_indices():
        print("No graph cache.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    from src.models.architectures import ArchConfig, build
    from src.models.dataset import GraphStore, N_EDGE_FEATURES
    from src.models.train import TrainConfig, train

    print("\nLoading cached graphs into memory...", flush=True)
    store = GraphStore()
    mem = (store.z.nbytes + store.src.nbytes + store.dst.nbytes + store.dist.nbytes) / 1e6
    print(f"  {len(store):,} crystals, {store.z.size:,} atoms, "
          f"{store.src.size:,} edges  ({mem:.0f} MB)")

    cfg = ArchConfig(atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
                     n_edge_fea=N_EDGE_FEATURES)
    model = build(args.arch, cfg)

    print(f"\n{'=' * 66}\n  {args.arch.upper()} — {args.target}"
          f"{' (non-metals)' if args.nonmetals else ''} — {args.split} split\n{'=' * 66}")

    tcfg = TrainConfig(
        target=args.target, split=args.split,
        max_minutes=3.0 if args.smoke else args.minutes,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        num_threads=args.threads, exclude_metals=args.nonmetals,
        subsample_train=2000 if args.smoke else None,
        notes=f"architecture={args.arch}",
    )
    res = train(model, store, load_split(args.split), tcfg, torch,
                MODELS_DIR / args.arch)
    res["architecture"] = args.arch
    res["model_config"] = cfg.to_dict()

    # Merge into one file per (target, subset), keyed by architecture then split,
    # so any order of separate runs builds the same comparison table.
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"architectures_{args.target}{'_nonmetals' if args.nonmetals else ''}"
    if args.smoke:
        tag += "_smoke"
    path = RESULTS / f"{tag}.json"

    merged = {}
    if path.exists():
        try:
            merged = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    merged.setdefault(args.arch, {})[args.split] = res
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    del model
    gc.collect()

    # Everything gathered so far, so progress through the sweep is visible.
    print(f"\n{'=' * 66}\n  All architectures so far — {args.target}"
          f"{' (non-metals)' if args.nonmetals else ''}\n{'=' * 66}")
    splits = [s for s in SCHEMES if any(s in v for v in merged.values())]
    print(f"  {'model':<9}" + "".join(f"{s:>12}" for s in splits))
    for arch, runs in merged.items():
        cells = "".join(f"{runs[s]['test']['mae']:>12.4f}" if s in runs else f"{'—':>12}"
                        for s in splits)
        print(f"  {arch:<9}{cells}")
    print(f"\nwrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
