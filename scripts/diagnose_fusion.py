"""Why did the Phase 5 result come out the way it did? Ask the weights.

    python scripts/diagnose_fusion.py        # seconds, no training

Two of the three predictions written into scripts/train_fusion.py before the runs
turned out wrong. This script checks the mechanism behind both, using the trained
checkpoints rather than a plausible-sounding story.

QUESTION 1 -- is the Phase 3 diagnosis literally true?
The claim was that a learned element table leaves an unseen element's row exactly
where random initialisation put it, because no gradient ever reaches a row for an
element that never appears in training. That is checkable: compare the rows of
the trained embedding for elements the model saw against the rows for the ten
elements held out of the element-disjoint split. If the claim is right, the
held-out rows should still look like fresh noise while the seen rows have moved.

QUESTION 2 -- why is 'both' WORSE than 'properties'?
Prediction 3 said combining a learned table with tabulated properties would beat
either alone: memorise where you have data, fall back on chemistry where you do
not. It came out 0.836 eV against 0.663 eV -- decisively worse.

The suspicion is shortcut learning. Given a free per-element vector AND real
element properties, gradient descent will lean on whichever pathway lowers
training loss fastest, and a free vector is strictly more flexible. If that is
what happened, the learned pathway should dominate the mixing layer -- and the
model will have rebuilt the very failure mode Phase 5 exists to remove.

Both questions are answered by looking at parameters, not by re-running anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.splits import SPLITS  # noqa: E402
from src.features.descriptors import element_table  # noqa: E402

MODELS = REPO / "models"
RESULTS = REPO / "results"
TAG = "band_gap_element_nonmetals"


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("PyTorch is not installed.\n\n"
              "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
        sys.exit(1)


def held_out_elements() -> list[str]:
    p = SPLITS / "summary.json"
    if not p.exists():
        return []
    blob = json.loads(p.read_text(encoding="utf-8"))
    return blob["schemes"].get("element", {}).get("held_out_elements", [])


def z_of(symbols) -> list[int]:
    elements, _ = element_table()
    return [int(elements[s]["Z"]) for s in symbols if s in elements]


def load(name: str, torch):
    p = MODELS / name / f"{TAG}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    torch = require_torch()
    out: dict = {}

    held = held_out_elements()
    held_z = set(z_of(held))
    print(f"\n{'=' * 74}\n  Phase 5 diagnostics — element-disjoint split\n{'=' * 74}")
    print(f"\n  held-out elements ({len(held)}): {', '.join(held)}")

    # ------------------------------------------------------------------
    # Q1. Does an unseen element's learned row ever move?
    # ------------------------------------------------------------------
    ck = load("cgcnn", torch)
    if ck is None:
        print("\n  models/cgcnn/… missing — run train_cgcnn.py --split element --nonmetals")
    else:
        emb = None
        for k, v in ck["state_dict"].items():
            if k.endswith("embedding.weight"):
                emb = v.numpy()
                break
        if emb is None:
            print("\n  no embedding in the CGCNN checkpoint (unexpected)")
        else:
            # Only rows for elements that exist at all; row 0 is a padding slot.
            all_z = [z for z in range(1, emb.shape[0]) if np.any(emb[z] != 0)]
            seen = [z for z in all_z if z not in held_z]
            unseen = [z for z in all_z if z in held_z]

            n_seen = np.linalg.norm(emb[seen], axis=1)
            n_unseen = np.linalg.norm(emb[unseen], axis=1)

            print("\n  Q1  Learned element table, after training on the element split")
            print(f"      {'':22}{'rows':>7}{'mean ‖row‖':>13}{'std':>9}")
            print("      " + "-" * 51)
            print(f"      {'elements it trained on':22}{len(seen):>7}"
                  f"{n_seen.mean():>13.4f}{n_seen.std():>9.4f}")
            print(f"      {'elements held out':22}{len(unseen):>7}"
                  f"{n_unseen.mean():>13.4f}{n_unseen.std():>9.4f}")

            ratio = float(n_seen.mean() / max(n_unseen.mean(), 1e-12))
            print(f"\n      trained rows are {ratio:.2f}x the length of untouched ones")
            if ratio > 1.15:
                print("      → the held-out rows are still at their random starting values.")
                print("        The model is not making a poor guess for those elements;")
                print("        it is reading numbers that were never learned.")
            else:
                print("      → the two look alike, so the Phase 3 story is NOT the whole")
                print("        explanation and the collapse needs another cause.")

            out["q1"] = {"n_seen": len(seen), "n_unseen": len(unseen),
                         "norm_seen": float(n_seen.mean()),
                         "norm_unseen": float(n_unseen.mean()),
                         "ratio": ratio}

    # ------------------------------------------------------------------
    # Q2. In 'both', which pathway does the model actually lean on?
    # ------------------------------------------------------------------
    ck = load("cgcnn_both", torch)
    if ck is None:
        print("\n  models/cgcnn_both/… missing — run train_fusion.py --atoms both …")
    else:
        sd = ck["state_dict"]
        mix_w = next(v.numpy() for k, v in sd.items() if k.endswith("mix.weight"))
        emb_w = next(v.numpy() for k, v in sd.items()
                     if k.endswith("featuriser.embedding.weight"))
        proj_w = next(v.numpy() for k, v in sd.items() if k.endswith("project.weight"))
        proj_b = next(v.numpy() for k, v in sd.items() if k.endswith("project.bias"))
        table = next(v.numpy() for k, v in sd.items() if k.endswith("featuriser.table"))

        d = emb_w.shape[1]
        w_emb, w_prop = mix_w[:, :d], mix_w[:, d:]

        # What actually reaches the mixing layer, per pathway, across the
        # elements present in training. Comparing raw weight norms would be
        # misleading -- what matters is the size of the signal each pathway
        # delivers, which depends on the inputs too.
        all_z = [z for z in range(1, emb_w.shape[0]) if np.any(emb_w[z] != 0)]
        seen = [z for z in all_z if z not in held_z]

        softplus = lambda a: np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0)  # noqa: E731
        p_seen = softplus(table[seen] @ proj_w.T + proj_b)

        a_emb = emb_w[seen] @ w_emb.T
        a_prop = p_seen @ w_prop.T

        # Spread ACROSS elements is what carries element identity; a pathway with
        # a large constant offset but no variation tells the network nothing
        # about which element it is looking at.
        s_emb = float(a_emb.std(axis=0).mean())
        s_prop = float(a_prop.std(axis=0).mean())
        share = s_emb / max(s_emb + s_prop, 1e-12)

        print("\n  Q2  In 'both', how much of the element signal comes from each route")
        print(f"      {'':30}{'spread across elements':>24}")
        print("      " + "-" * 54)
        print(f"      {'learned free vector':30}{s_emb:>24.4f}")
        print(f"      {'tabulated properties':30}{s_prop:>24.4f}")
        print(f"\n      the learned route carries {100 * share:.0f}% of it")
        if share > 0.6:
            print("      → the model leaned on the memorisable route, which is exactly")
            print("        the one that is blank for an unseen element. Offering both")
            print("        and hoping it picks the right one does not work.")
        elif share < 0.4:
            print("      → the properties dominate, so 'both' underperforming needs a")
            print("        different explanation than shortcut learning.")
        else:
            print("      → roughly balanced; the shortcut story is not clearly supported.")

        out["q2"] = {"spread_learned": s_emb, "spread_properties": s_prop,
                     "learned_share": share}

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "fusion_diagnostics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
