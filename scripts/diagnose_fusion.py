"""Why did the Phase 5 result come out the way it did? Ask the weights.

    python scripts/diagnose_fusion.py        # seconds, no training

Two explanations were offered for the Phase 5 results. The first version of this
script tested both and **refuted both**, which is recorded here rather than
quietly replaced, because a diagnostic that only ever confirms the story it was
written to support is not a diagnostic.

WHAT WAS CLAIMED, AND WHAT THE FIRST TEST FOUND
-----------------------------------------------
Claim 1 -- "an unseen element's learned row is never touched by training, so the
model is reading its random initialisation." Test: compare the LENGTH of embedding
rows for elements the model trained on against the ten held out.
Result: 8.06 against 8.02, a ratio of 1.005. No difference at all.

That does not refute the claim, but it does not support it either -- the test was
simply too weak. A vector can rotate a long way while keeping its length, and
nn.Embedding initialises from a unit normal, so a 64-dimensional row starts with
a length near sqrt(64) = 8 whether or not it later moves. Row length was the wrong
statistic. Replaced below by a test of whether the learned table encodes CHEMISTRY,
which is the property that actually matters and which rotation cannot fake.

Claim 2 -- "'both' loses because the model leans on the memorisable route and lets
the chemistry route underdevelop." Test: how much of the variation between elements
arrives through each pathway.
Result: learned 1.06, properties 1.74. The properties carry 62%, not the learned
route. **The shortcut-learning story is wrong**, and the README has been corrected.

WHAT THIS VERSION TESTS INSTEAD
--------------------------------
A. Does the learned element table encode chemistry, or arbitrary codes? Checked
   against pairs whose relationship is true independently of this model: Cl/Br
   should be close if the table learned chemistry, Cl/Na should not.

B. The corrected explanation for claim 2. A pathway does not have to DOMINATE to
   do damage -- it only has to contribute. For a held-out element the learned row
   was never trained, so whatever it contributes is noise. The question is what
   share of an unseen atom's starting vector is built from that noise. If it is
   substantial, 'both' is worse than 'properties' not because the model preferred
   the shortcut but because it could not avoid the shortcut's garbage.

Neither requires retraining.
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
from src.features.element_features import element_feature_table  # noqa: E402

MODELS = REPO / "models"
RESULTS = REPO / "results"
TAG = "band_gap_element_nonmetals"

# Pairs whose chemical relationship is true regardless of any model.
SIMILAR = [("Cl", "Br"), ("Na", "K"), ("S", "Se"), ("Mg", "Ca"), ("Ni", "Pd")]
DIFFERENT = [("Cl", "Na"), ("Cl", "Fe"), ("O", "Cs"), ("F", "Ba"), ("H", "U")]


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
    return json.loads(p.read_text(encoding="utf-8"))["schemes"] \
        .get("element", {}).get("held_out_elements", [])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load(name: str, torch):
    p = MODELS / name / f"{TAG}.pt"
    return None if not p.exists() else torch.load(p, map_location="cpu",
                                                  weights_only=False)


def contrast(vectors, z_of, held_z):
    """Mean similarity of chemically alike pairs minus unalike pairs."""
    def mean_for(pairs):
        vals = []
        for a, b in pairs:
            za, zb = z_of.get(a), z_of.get(b)
            if za is None or zb is None or za in held_z or zb in held_z:
                continue
            vals.append(cos(vectors[za], vectors[zb]))
        return float(np.mean(vals)) if vals else float("nan")

    s, d = mean_for(SIMILAR), mean_for(DIFFERENT)
    return s, d, s - d


def main() -> None:
    torch = require_torch()
    out: dict = {}

    elements, _ = element_table()
    z_of = {s: int(v["Z"]) for s, v in elements.items()}
    held = held_out_elements()
    held_z = {z_of[e] for e in held if e in z_of}

    print(f"\n{'=' * 76}\n  Phase 5 diagnostics — element-disjoint split\n{'=' * 76}")
    print(f"\n  held-out elements ({len(held)}): {', '.join(held)}")

    # ------------------------------------------------------------------
    # A. Does the learned table encode chemistry, or arbitrary codes?
    # ------------------------------------------------------------------
    print("\n  A  Does each element table encode CHEMISTRY?")
    print("     Similar pairs: " + ", ".join(f"{a}/{b}" for a, b in SIMILAR))
    print("     Unalike pairs: " + ", ".join(f"{a}/{b}" for a, b in DIFFERENT))
    print(f"\n     {'table':<34}{'similar':>10}{'unalike':>10}{'contrast':>11}")
    print("     " + "-" * 65)

    table, known, _ = element_feature_table()
    s, d, c = contrast(table, z_of, held_z)
    print(f"     {'tabulated properties (Phase 5)':<34}{s:>10.3f}{d:>10.3f}{c:>11.3f}")
    out["properties_contrast"] = {"similar": s, "different": d, "contrast": c}

    ck = load("cgcnn", torch)
    if ck is None:
        print("     models/cgcnn/… missing — run train_cgcnn.py --split element --nonmetals")
    else:
        emb = next(v.numpy() for k, v in ck["state_dict"].items()
                   if k.endswith("embedding.weight"))
        s2, d2, c2 = contrast(emb, z_of, held_z)
        print(f"     {'learned codes (Phase 3), trained':<34}{s2:>10.3f}{d2:>10.3f}{c2:>11.3f}")
        out["learned_contrast"] = {"similar": s2, "different": d2, "contrast": c2}

        rng = np.random.default_rng(0)
        fresh = rng.normal(size=emb.shape)
        s3, d3, c3 = contrast(fresh, z_of, held_z)
        print(f"     {'random numbers, for comparison':<34}{s3:>10.3f}{d3:>10.3f}{c3:>11.3f}")
        out["random_contrast"] = {"contrast": c3}

        n_seen = np.linalg.norm(emb[[z for z in range(1, emb.shape[0])
                                     if z not in held_z and np.any(emb[z] != 0)]], axis=1)
        n_unseen = np.linalg.norm(emb[sorted(held_z)], axis=1)
        print(f"\n     row lengths: trained {n_seen.mean():.3f}, held out "
              f"{n_unseen.mean():.3f}  (ratio {n_seen.mean() / n_unseen.mean():.3f})")
        print("     Row length cannot separate them — a vector rotates without changing")
        print("     length, and a 64-D unit-normal row starts near sqrt(64) = 8 anyway.")

        out["row_norms"] = {"seen": float(n_seen.mean()),
                            "unseen": float(n_unseen.mean())}

        if not np.isnan(c2):
            if c2 < 0.5 * c:
                print(f"\n     → the learned table carries far less chemical structure "
                      f"({c2:+.3f}) than\n       the tabulated one ({c:+.3f}). It holds "
                      "codes, not chemistry — which is\n       exactly why a code it "
                      "never learned is worthless.")
            else:
                print(f"\n     → the learned table DID pick up chemical structure "
                      f"({c2:+.3f}). The Phase 5\n       gain then needs an explanation "
                      "other than 'it learned arbitrary codes'.")

    # ------------------------------------------------------------------
    # B. In 'both', how much of an UNSEEN atom's vector is untrained noise?
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
        tab = next(v.numpy() for k, v in sd.items() if k.endswith("featuriser.table"))

        d_atom = emb_w.shape[1]
        w_emb, w_prop = mix_w[:, :d_atom], mix_w[:, d_atom:]

        def softplus(a):
            return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0)

        print("\n  B  In 'both', how much of an atom's starting vector comes from")
        print("     the learned route — the one that is untrained for a held-out element?")
        print(f"\n     {'elements':<26}{'from learned':>14}{'from properties':>17}"
              f"{'learned share':>15}")
        print("     " + "-" * 72)

        rows = {}
        for label, zs in (("the model trained on",
                           [z for z in range(1, emb_w.shape[0])
                            if z not in held_z and np.any(emb_w[z] != 0)]),
                          ("HELD OUT of training", sorted(held_z))):
            if not zs:
                continue
            a_emb = emb_w[zs] @ w_emb.T
            a_prop = softplus(tab[zs] @ proj_w.T + proj_b) @ w_prop.T
            m_emb = float(np.abs(a_emb).mean())
            m_prop = float(np.abs(a_prop).mean())
            share = m_emb / max(m_emb + m_prop, 1e-12)
            print(f"     {label:<26}{m_emb:>14.3f}{m_prop:>17.3f}{100 * share:>14.0f}%")
            rows[label] = {"learned": m_emb, "properties": m_prop, "share": share}

        out["both_pathways"] = rows

        held_share = rows.get("HELD OUT of training", {}).get("share")
        if held_share is not None:
            print(f"\n     → for an element it never saw, {100 * held_share:.0f}% of the "
                  "starting vector is built\n       from a row that training never "
                  "touched. The learned route does not have to\n       DOMINATE to do "
                  "damage; it only has to contribute, and it cannot be\n       switched "
                  "off for the elements where it is meaningless.")
            print("\n       That is the corrected explanation for why 'both' loses to")
            print("       'properties'. The earlier claim — that the model prefers the")
            print("       memorisable route — is not supported: the properties carry the")
            print("       larger share of the between-element signal.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "fusion_diagnostics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO)}\n")


if __name__ == "__main__":
    main()
