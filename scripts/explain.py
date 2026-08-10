"""Phase 6 -- which piece of chemistry did the model use?

    python scripts/explain.py --selftest        # seconds, checks the maths
    python scripts/explain.py                   # ~2 min, needs a trained model
    python scripts/explain.py --split element

Attributes band-gap predictions to the 31 named element properties each atom
starts from, using integrated gradients, and reports the result against two
controls that must NOT reproduce it.

This is only possible because of Phase 5. When an atom started from a learned
64-number code there was nothing to attribute to: the numbers had no names, and
the diagnostic showed they had no chemical structure either. Now the inputs are
electronegativity, ionic radius, valence count and 28 other named quantities, and
"which one mattered" is a question with a checkable answer.

WHAT TO LOOK FOR
----------------
Band gaps are dominated by electronegativity in every textbook account: the
larger the electronegativity difference between the elements, the more ionic the
bonding and the wider the gap. If the attribution puts electronegativity and the
valence-electron count near the top, the model has recovered something a chemist
would recognise. If it puts atomic mass or thermal conductivity on top, it has
found a shortcut through a correlated quantity, and that is worth knowing too.

THE CONTROLS ARE THE POINT
--------------------------
Attribution methods return confident-looking rankings for models that have
learned nothing (Adebayo et al., NeurIPS 2018). So the same procedure is run on:

    untrained   the same architecture at random initialisation
    shuffled    the same architecture trained on SHUFFLED band gaps, which fits
                noise thoroughly and cannot have learned chemistry
                (train_fusion.py --shuffle-labels)

If the trained profile resembles either, the attribution is measuring the method
and the data rather than the model, and that is what gets reported.
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

from src.config import RESULTS  # noqa: E402
from src.data import graph_build as gb  # noqa: E402
from src.data.splits import SCHEMES, load_split  # noqa: E402
from src.features.element_features import element_feature_table  # noqa: E402
from src.features.property_groups import family_of, label_of  # noqa: E402

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
    """Check the arithmetic before trusting any ranking it produces."""
    torch = require_torch()

    from src.models import cgcnn_reference as ref
    from src.models.attribution import (aggregate, completeness_error,
                                        integrated_gradients)
    from src.models.fusion import FusedGNN, FusionConfig

    print("\nSelf-test: integrated gradients\n" + "=" * 62)

    rng = np.random.default_rng(0)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    batch = {
        "z": torch.from_numpy(z),
        "src": torch.from_numpy(np.array([0, 0, 1, 2, 3, 1, 4, 5])),
        "dst": torch.from_numpy(np.array([1, 2, 0, 0, 1, 3, 5, 4])),
        "u": torch.from_numpy(ref.gaussian_expand(
            rng.uniform(1.5, 7.5, size=8)).astype(np.float32)),
        "batch_index": torch.from_numpy(np.array([0, 0, 0, 0, 1, 1])),
        "n_graphs": 2,
    }

    model = FusedGNN(FusionConfig(atom_features="properties", atom_fea_len=16,
                                  n_conv=2, h_fea_len=32, use_batch_norm=False))

    print(f"\n  {'steps':>7}{'completeness error':>22}")
    print("  " + "-" * 29)
    for steps in (4, 16, 64, 256):
        a = integrated_gradients(model, batch, steps=steps)
        err = completeness_error(model, batch, a)
        print(f"  {steps:>7}{err:>22.3e}")

    a = integrated_gradients(model, batch, steps=256)
    err = completeness_error(model, batch, a)
    assert err < 1e-3, f"completeness violated ({err:.2e}) — the maths is wrong"
    print("\n  Attributions sum to the change in the prediction, as they must.")
    print("  The error falls as the integration is refined, which is the")
    print("  signature of a discretisation error rather than a bug.")

    # An input the model cannot see must receive no attribution. Constant columns
    # of the property table are exactly that, and they are a free trap.
    table, known, names = element_feature_table()
    agg = aggregate(a, batch["batch_index"].numpy(), 2)
    spread = table[known].std(axis=0)
    dead = [i for i, sd in enumerate(spread) if sd < 1e-9]
    if dead:
        assert np.allclose(agg["mean_abs"][dead], 0, atol=1e-6)
        print(f"  {len(dead)} constant properties receive zero attribution.")

    print(f"\n  {len(names)} properties, all mapped to a family: "
          f"{sorted({family_of(n) for n in names})}")
    print("\n  Maths checks out. Rankings from it are worth reading.\n")


def attribute(model, store, idx, torch, steps=32, batch_size=64, log=print):
    """Run IG over a set of crystals, in batches, and pool the result."""
    from src.models.attribution import (aggregate, completeness_error,
                                        integrated_gradients)

    n_props = model.featuriser.table.shape[1]
    total_abs = np.zeros(n_props)
    total_signed = np.zeros(n_props)
    n_seen = 0
    worst_completeness = 0.0

    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        batch = store.collate(chunk, "band_gap", torch)
        a = integrated_gradients(model, batch, steps=steps)
        worst_completeness = max(
            worst_completeness,
            completeness_error(model, batch, a) / max(len(chunk), 1))

        agg = aggregate(a, batch["batch_index"].numpy(), len(chunk))
        total_abs += agg["mean_abs"] * len(chunk)
        total_signed += agg["mean_signed"] * len(chunk)
        n_seen += len(chunk)

        if start % (batch_size * 20) == 0:
            log(f"    {n_seen:>6,} / {len(idx):,} crystals", flush=True)

    return {"mean_abs": total_abs / n_seen, "mean_signed": total_signed / n_seen,
            "n_crystals": n_seen, "worst_completeness_per_crystal": worst_completeness}


def load_model(name, tag, torch):
    path = MODELS_DIR / name / f"{tag}.pt"
    if not path.exists():
        return None
    from src.models.fusion import FusedGNN, FusionConfig

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FusionConfig(**{k: v for k, v in ck["model_config"].items()
                          if k in FusionConfig.__dataclass_fields__})
    model = FusedGNN(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split", default="random", choices=list(SCHEMES))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-crystals", type=int, default=2000,
                    help="attribution is expensive; a sample is enough")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    torch = require_torch()
    if not gb.existing_chunk_indices():
        print("No graph cache.\n\n    python scripts/build_graphs.py\n")
        sys.exit(1)

    from src.models.attribution import profile_similarity
    from src.models.dataset import GraphStore
    from src.models.fusion import FusedGNN, FusionConfig
    from src.models.train import TrainConfig, select_indices

    tag = f"band_gap_{args.split}_nonmetals"
    trained = load_model("cgcnn_properties", tag, torch)
    if trained is None:
        print(f"models/cgcnn_properties/{tag}.pt missing. Run:\n"
              f"    python scripts/train_fusion.py --atoms properties "
              f"--split {args.split} --nonmetals\n")
        sys.exit(1)

    print("\nLoading cached graphs into memory...", flush=True)
    store = GraphStore()
    tcfg = TrainConfig(target="band_gap", split=args.split, exclude_metals=True)
    _, _, te = select_indices(store, load_split(args.split), tcfg)

    rng = np.random.default_rng(args.seed)
    if len(te) > args.max_crystals:
        te = np.sort(rng.choice(te, args.max_crystals, replace=False))

    _, _, names = element_feature_table()
    out = {"split": args.split, "steps": args.steps,
           "n_crystals": int(len(te)), "properties": names, "models": {}}

    # The trained model, and two controls it must not resemble.
    untrained = FusedGNN(FusionConfig(**trained.cfg.to_dict()))
    untrained.eval()
    candidates = [("trained", trained), ("untrained", untrained)]

    shuffled = load_model("cgcnn_properties_shuffled", tag, torch)
    if shuffled is not None:
        candidates.append(("shuffled_labels", shuffled))
    else:
        print(f"\n  (no shuffled-label control — run\n"
              f"     python scripts/train_fusion.py --atoms properties "
              f"--split {args.split} --nonmetals --shuffle-labels)")

    for label, model in candidates:
        print(f"\n  attributing: {label}", flush=True)
        out["models"][label] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in attribute(model, store, te, torch, steps=args.steps).items()
        }

    imp = {k: np.array(v["mean_abs"]) for k, v in out["models"].items()}

    print(f"\n{'=' * 78}\n  Which property moved the prediction most — "
          f"{args.split} split, {len(te):,} crystals\n{'=' * 78}\n")
    print(f"  {'#':>3}  {'property':<26}{'family':<12}{'trained':>10}"
          f"{'untrained':>11}{'shuffled':>10}")
    print("  " + "-" * 74)

    order = np.argsort(-imp["trained"])
    scale = imp["trained"][order[0]] or 1.0
    for rank, j in enumerate(order[:12], 1):
        row = f"  {rank:>3}  {label_of(names[j]):<26}{family_of(names[j]):<12}" \
              f"{imp['trained'][j] / scale:>10.3f}"
        for key in ("untrained", "shuffled_labels"):
            row += (f"{imp[key][j] / (imp[key].max() or 1):>11.3f}"
                    if key in imp else f"{'—':>11}")
        print(row)

    print(f"\n  {'family':<14}{'share of total attribution':>28}")
    print("  " + "-" * 42)
    fam = {}
    for j, n in enumerate(names):
        fam[family_of(n)] = fam.get(family_of(n), 0.0) + imp["trained"][j]
    tot = sum(fam.values()) or 1.0
    for f, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"  {f:<14}{100 * v / tot:>27.1f}%")

    print("\n  Sanity checks — the trained profile must NOT match a control:")
    for key in ("untrained", "shuffled_labels"):
        if key in imp:
            sim = profile_similarity(imp["trained"], imp[key])
            verdict = ("SUSPICIOUS — too similar" if sim > 0.9
                       else "ok, clearly different" if sim < 0.7
                       else "borderline")
            print(f"    trained vs {key:<18}cosine {sim:+.3f}   {verdict}")
            out.setdefault("similarity", {})[key] = sim

    worst = max(v["worst_completeness_per_crystal"] for v in out["models"].values())
    print(f"\n  worst completeness error, per crystal: {worst:.2e}")
    print("  (attributions must sum to the change in the prediction; large means")
    print("   the integration was too coarse and the ranking is unreliable)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"attribution_band_gap_{args.split}_nonmetals.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
