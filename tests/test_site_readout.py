"""Tests for the adsorption readout.

The bug this guards against is not a crash: it is a readout that quietly behaves
like the one it was meant to replace, so the comparison between them measures
nothing and reports a difference of zero as evidence.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from src.models.cgcnn import CGCNN, CGCNNConfig  # noqa: E402
from src.models.site_cgcnn import SiteCGCNN  # noqa: E402


def toy_batch(n_atoms=10, n_site=2, seed=0):
    """One graph: a ring, with the first `n_site` atoms flagged as the site."""
    rng = np.random.default_rng(seed)
    src = np.arange(n_atoms)
    dst = (src + 1) % n_atoms
    src = np.concatenate([src, dst])
    dst = np.concatenate([dst, np.arange(n_atoms)])
    site = np.zeros(n_atoms, bool)
    site[:n_site] = True
    return {
        "z": torch.from_numpy(rng.integers(1, 90, n_atoms).astype(np.int64)),
        "src": torch.from_numpy(src.astype(np.int64)),
        "dst": torch.from_numpy(dst.astype(np.int64)),
        "u": torch.from_numpy(rng.random((len(src), 41)).astype(np.float32)),
        "batch_index": torch.zeros(n_atoms, dtype=torch.int64),
        "n_graphs": 1,
        "site_mask": torch.from_numpy(site),
    }


# ---------------------------------------------------------------------------
# The readout must actually be different from the one it replaces
# ---------------------------------------------------------------------------

def test_the_site_readout_is_not_secretly_the_mean_readout():
    """If these agree, the whole experiment is comparing a model with itself."""
    v = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    bidx = torch.zeros(10, dtype=torch.int64)
    site = torch.zeros(10, dtype=torch.bool)
    site[:2] = True
    g = SiteCGCNN.pool_site(v, bidx, site, 1)
    mean = CGCNN.pool(v, bidx, 1)
    assert not torch.allclose(g[:, :2], mean)


def test_the_site_half_averages_only_the_site_atoms():
    v = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    site = torch.zeros(10, dtype=torch.bool)
    site[[0, 3]] = True
    g = SiteCGCNN.pool_site(v, torch.zeros(10, dtype=torch.int64), site, 1)
    assert torch.allclose(g[0, :2], (v[0] + v[3]) / 2)


def test_the_global_half_still_averages_everything():
    """Global context is kept, so the site model can represent anything the
    control can and a loss against it would be a real result."""
    v = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    site = torch.zeros(10, dtype=torch.bool)
    site[0] = True
    g = SiteCGCNN.pool_site(v, torch.zeros(10, dtype=torch.int64), site, 1)
    assert torch.allclose(g[0, 2:], v.mean(dim=0))


def test_spectator_atoms_cannot_move_the_site_half():
    """The point of the change. Adding bulk atoms far from the CO must not
    dilute the binding-site representation -- under mean pooling it would."""
    v = torch.cat([torch.ones(2, 2), torch.zeros(30, 2)])
    site = torch.zeros(32, dtype=torch.bool)
    site[:2] = True
    bidx = torch.zeros(32, dtype=torch.int64)
    g = SiteCGCNN.pool_site(v, bidx, site, 1)
    assert torch.allclose(g[0, :2], torch.ones(2))
    assert torch.allclose(CGCNN.pool(v, bidx, 1)[0], torch.full((2,), 2 / 32))


def test_graphs_in_a_batch_do_not_leak_into_each_other():
    v = torch.cat([torch.ones(4, 2), 5 * torch.ones(4, 2)])
    bidx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    site = torch.tensor([True, False, False, False, True, False, False, False])
    g = SiteCGCNN.pool_site(v, bidx, site, 2)
    assert torch.allclose(g[0, :2], torch.ones(2))
    assert torch.allclose(g[1, :2], 5 * torch.ones(2))


def test_a_graph_with_no_site_atoms_degrades_instead_of_producing_nan():
    """Cannot happen for a real row, but NaNs surface only as silent training
    failure, so the clamp is tested rather than trusted."""
    v = torch.ones(4, 2)
    g = SiteCGCNN.pool_site(v, torch.zeros(4, dtype=torch.int64),
                            torch.zeros(4, dtype=torch.bool), 1)
    assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# Invariances: one that must hold, one that must NOT
# ---------------------------------------------------------------------------

def test_prediction_is_unchanged_by_reordering_the_atoms():
    """The site mask has to permute WITH the atoms. If it does not, the model
    reads the wrong atoms as the binding site and nothing looks broken."""
    torch.manual_seed(0)
    model = SiteCGCNN(CGCNNConfig(n_conv=2)).eval()
    b = toy_batch()
    with torch.no_grad():
        a = model(b)

    n = b["z"].numel()
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    pb = dict(b)
    pb["z"] = b["z"][perm]
    pb["site_mask"] = b["site_mask"][perm]
    pb["src"] = inv[b["src"]]
    pb["dst"] = inv[b["dst"]]
    with torch.no_grad():
        c = model(pb)
    assert torch.allclose(a, c, atol=1e-5)


def test_the_site_model_refuses_a_batch_with_no_mask():
    """Falling back to the mean readout would make this model silently identical
    to its own control."""
    model = SiteCGCNN(CGCNNConfig(n_conv=1)).eval()
    b = toy_batch()
    del b["site_mask"]
    with pytest.raises(KeyError):
        model(b)


def test_the_extra_parameters_are_a_small_fraction_of_the_total():
    """A win must not be explainable by capacity alone."""
    base = CGCNN().n_parameters()
    site = SiteCGCNN().n_parameters()
    assert site > base
    assert (site - base) / base < 0.10


# ---------------------------------------------------------------------------
# The store has to deliver the mask aligned to the batch
# ---------------------------------------------------------------------------

def test_the_mask_lines_up_with_the_atoms_in_the_batch():
    """Off-by-one here flags the wrong atoms as the site on every row after the
    first, and the model still trains."""
    from src.models.slab_dataset import SlabStore
    try:
        store = SlabStore()
    except FileNotFoundError:
        pytest.skip("slab graphs not built")
    idx = np.array([3, 0, 7])
    b = store.collate(idx, "adsorption_energy", torch)
    expected = np.concatenate([
        store.site_mask[store.node_ptr[i]:store.node_ptr[i + 1]] for i in idx])
    assert np.array_equal(b["site_mask"].numpy(), expected.astype(bool))
    assert b["site_mask"].numel() == b["z"].numel()


def test_every_adsorbate_atom_is_in_its_own_site():
    from src.models.slab_dataset import SlabStore
    try:
        store = SlabStore()
    except FileNotFoundError:
        pytest.skip("slab graphs not built")
    assert bool(store.site_mask[store.is_adsorbate].all())


# ---------------------------------------------------------------------------
# The four per-atom descriptors
# ---------------------------------------------------------------------------

def feat_batch(n=6):
    b = toy_batch(n_atoms=n, n_site=2)
    b["is_adsorbate"] = torch.zeros(n, dtype=torch.bool)
    b["is_adsorbate"][:1] = True
    b["height"] = torch.arange(n, dtype=torch.float32)
    b["coordination"] = torch.full((n,), 6.0)
    return b


def test_the_feature_columns_are_in_the_documented_order():
    """An insertion that shifts these would feed coordination into the height
    slot. The model would still train, on a different physical quantity."""
    b = feat_batch()
    f = SiteCGCNN.node_features(b)
    from src.models.site_cgcnn import COORDINATION_SCALE, HEIGHT_SCALE
    assert torch.allclose(f[:, 0], b["is_adsorbate"].float())
    assert torch.allclose(f[:, 1], b["site_mask"].float())
    assert torch.allclose(f[:, 2], b["height"] / HEIGHT_SCALE)
    assert torch.allclose(f[:, 3], b["coordination"] / COORDINATION_SCALE)


def test_the_scales_are_fixed_constants_not_fitted_to_the_data():
    """A scaler fitted over the whole set would put test statistics into the
    training inputs. Small leak, real, and invisible."""
    a = SiteCGCNN.node_features(feat_batch(6))
    b = feat_batch(6)
    b["coordination"] = torch.full((6,), 12.0)      # a different "dataset"
    assert torch.allclose(SiteCGCNN.node_features(b)[:, 3], torch.ones(6))
    assert torch.allclose(a[:, 3], torch.full((6,), 0.5))


def test_the_feature_model_refuses_a_batch_without_them():
    """Zero-filling would give a model that appears to use these and does not."""
    model = SiteCGCNN(CGCNNConfig(n_conv=1), use_features=True).eval()
    b = toy_batch()
    with pytest.raises(KeyError):
        model(b)


def test_features_can_be_switched_off_so_the_readout_can_be_attributed_alone():
    model = SiteCGCNN(CGCNNConfig(n_conv=1), use_features=False).eval()
    with torch.no_grad():
        model(toy_batch())          # must not raise: no features needed


def test_features_change_the_prediction():
    """If they did not, the ablation would report a real difference of zero."""
    torch.manual_seed(0)
    model = SiteCGCNN(CGCNNConfig(n_conv=2), use_features=True).eval()
    b = feat_batch()
    c = dict(b)
    c["coordination"] = torch.full((b["z"].numel(),), 11.0)
    with torch.no_grad():
        assert not torch.allclose(model(b), model(c))


def test_features_permute_with_the_atoms():
    torch.manual_seed(0)
    model = SiteCGCNN(CGCNNConfig(n_conv=2), use_features=True).eval()
    b = feat_batch(8)
    with torch.no_grad():
        a = model(b)
    n = b["z"].numel()
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    pb = {**b, "z": b["z"][perm], "src": inv[b["src"]], "dst": inv[b["dst"]]}
    for k in ("site_mask", "is_adsorbate", "height", "coordination"):
        pb[k] = b[k][perm]
    with torch.no_grad():
        assert torch.allclose(a, model(pb), atol=1e-5)


def test_the_store_rejects_a_stale_cache():
    """A cache written before a feature was added would misalign every atom."""
    from src.models.slab_dataset import SlabStore
    try:
        store = SlabStore()
    except FileNotFoundError:
        pytest.skip("slab graphs not built")
    assert store.coordination.shape[0] == store.z.shape[0]
    assert store.height.shape[0] == store.z.shape[0]
