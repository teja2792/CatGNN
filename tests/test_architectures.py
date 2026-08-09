"""Tests for the Phase 4 architectures.

Every model here must satisfy the same physical contract CGCNN does -- relabel
the atoms or replicate the cell and the prediction must not move -- because a
comparison between architectures is only meaningful if they are all solving the
same problem correctly.

The GATv2 tests additionally check the property that distinguishes attention from
CGCNN's gate: the weights must sum to one over each atom's neighbours. That is
the whole reason GATv2 is in the comparison, so it is worth asserting rather
than assuming.

Skipped where PyTorch is absent. Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models import cgcnn_reference as ref  # noqa: E402


def _has_torch() -> bool:
    try:
        __import__("torch")
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
ARCH_NAMES = ["cgcnn", "mpnn", "megnet", "gatv2"]


def toy_batch(torch, dtype=None):
    """Two crystals: TiO3 (4 atoms) and TiO (2 atoms)."""
    rng = np.random.default_rng(0)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    batch_index = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    u = ref.gaussian_expand(rng.uniform(1.5, 7.5, size=len(src)))
    if dtype is not None:
        u = u.astype(dtype)
    return {
        "z": torch.from_numpy(z), "src": torch.from_numpy(src),
        "dst": torch.from_numpy(dst), "u": torch.from_numpy(u),
        "batch_index": torch.from_numpy(batch_index), "n_graphs": 2,
    }, z, src, dst, u


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_forward_returns_one_number_per_crystal(name):
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double().eval()
    batch, *_ = toy_batch(T)
    with T.no_grad():
        out = model(batch)

    assert out.shape == (2,)
    assert T.isfinite(out).all()


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_permutation_invariance(name):
    """Renaming atom 0 to atom 2 must not change the material's prediction."""
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double().eval()
    batch, z, src, dst, u = toy_batch(T)
    with T.no_grad():
        base = model(batch).numpy()

    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))
    with T.no_grad():
        got = model({**batch,
                     "z": T.from_numpy(np.concatenate([z[:4][perm], z[4:]])),
                     "src": T.from_numpy(remap(src)),
                     "dst": T.from_numpy(remap(dst))}).numpy()

    assert np.allclose(base, got, atol=1e-9), f"{name} is not permutation invariant"


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_supercell_invariance(name):
    """A 2x replica is the same material; band gap is intensive and must not move."""
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double().eval()

    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.full(4, 2.0))

    def run(z, s, d, u):
        with T.no_grad():
            return model({"z": T.from_numpy(z), "src": T.from_numpy(s),
                          "dst": T.from_numpy(d), "u": T.from_numpy(u),
                          "batch_index": T.zeros(len(z), dtype=T.long),
                          "n_graphs": 1}).numpy()

    single = run(z1, s1, d1, u1)
    double = run(np.concatenate([z1, z1]), np.concatenate([s1, s1 + 3]),
                 np.concatenate([d1, d1 + 3]), np.concatenate([u1, u1]))
    assert np.allclose(single, double, atol=1e-9), f"{name} readout is not a mean"


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_batching_matches_individual(name):
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double().eval()
    batch, z, src, dst, u = toy_batch(T)
    with T.no_grad():
        both = model(batch).numpy()

    def run(zz, ss, dd, uu):
        with T.no_grad():
            return model({"z": T.from_numpy(zz), "src": T.from_numpy(ss),
                          "dst": T.from_numpy(dd), "u": T.from_numpy(uu),
                          "batch_index": T.zeros(len(zz), dtype=T.long),
                          "n_graphs": 1}).numpy()

    a = run(z[:4], src[:6], dst[:6], u[:6])
    b = run(z[4:], src[6:] - 4, dst[6:] - 4, u[6:])
    assert np.allclose(both, np.concatenate([a, b]), atol=1e-9), \
        f"{name}: batching changes the answer"


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_gradients_reach_every_parameter(name):
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=16, n_conv=2, h_fea_len=32,
                                   use_batch_norm=False)).double()
    batch, *_ = toy_batch(T)
    model(batch).sum().backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or T.all(p.grad == 0))
            and "embedding" not in n]
    assert not dead, f"{name}: no gradient reaches {dead}"


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_can_overfit_a_tiny_batch(name):
    """If it cannot memorise two graphs, the training loop is not connected."""
    import torch as T

    from src.models.architectures import ArchConfig, build

    model = build(name, ArchConfig(atom_fea_len=32, n_conv=2, use_batch_norm=False))
    opt = T.optim.Adam(model.parameters(), lr=0.02)
    batch, *_ = toy_batch(T, dtype=np.float32)
    y = T.tensor(np.random.default_rng(0).normal(size=2).astype(np.float32))

    first = float(T.nn.functional.l1_loss(model(batch), y))
    for _ in range(250):
        opt.zero_grad()
        loss = T.nn.functional.l1_loss(model(batch), y)
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.2, f"{name}: loss barely moved"


@needs_torch
@pytest.mark.parametrize("name", ARCH_NAMES)
def test_parameter_counts_are_comparable(name):
    """The comparison is only fair if the models are of similar size.

    Phase 4 asks which message-passing rule is better, not which model is bigger.
    A 5x parameter gap would answer a different question than the one asked.
    """
    from src.models.architectures import ArchConfig, build

    n = build(name, ArchConfig()).n_parameters()
    assert 40_000 < n < 250_000, f"{name} has {n:,} parameters, out of band"


# ---------------------------------------------------------------------------
# The property that makes GATv2 attention rather than gating
# ---------------------------------------------------------------------------

@needs_torch
def test_gatv2_attention_sums_to_one_over_neighbours():
    """The defining difference from CGCNN's gate.

    A softmax across an atom's neighbours makes the weights a distribution: they
    compete for a fixed budget and can honestly be read as "which bond did the
    model attend to". CGCNN's sigmoid gate scores each bond independently and
    carries no such guarantee.
    """
    import torch as T

    from src.models.architectures import ArchConfig, GATv2

    model = GATv2(ArchConfig(atom_fea_len=16, n_conv=2, n_heads=4,
                             use_batch_norm=False)).double().eval()
    batch, z, src, *_ = toy_batch(T)
    with T.no_grad():
        weights = model.attention_weights(batch)

    assert len(weights) == 2
    for layer, w in enumerate(weights):
        for atom in np.unique(src):
            total = w[src == atom].sum(dim=0).numpy()   # per head
            assert np.allclose(total, 1.0, atol=1e-8), \
                f"layer {layer}, atom {atom}: attention sums to {total}, not 1"


@needs_torch
def test_gatv2_attention_is_stable_for_large_scores():
    """A single overflow would turn one atom's whole neighbourhood into NaN."""
    import torch as T

    from src.models.architectures import GATv2Conv

    scores = T.tensor([[1000.0], [1001.0], [-1000.0]], dtype=T.float64)
    src = T.tensor([0, 0, 0])
    alpha = GATv2Conv._softmax_by_source(scores, src, 1)

    assert T.isfinite(alpha).all()
    assert float(alpha.sum()) == pytest.approx(1.0)


@needs_torch
def test_megnet_global_state_actually_changes_the_output():
    """If the global vector were inert, MEGNet would just be a slower MPNN."""
    import torch as T

    from src.models.architectures import ArchConfig, MEGNet

    cfg = ArchConfig(atom_fea_len=16, n_conv=2, use_batch_norm=False)
    model = MEGNet(cfg).double().eval()
    batch, *_ = toy_batch(T)

    with T.no_grad():
        base = model(batch).numpy()
        for p in model.convs[0].global_mlp.parameters():
            p.mul_(0.0)
        muted = model(batch).numpy()

    assert not np.allclose(base, muted), "the global state has no effect"


# ---------------------------------------------------------------------------
# The attention softmax, checked without PyTorch
#
# These run everywhere. The three architectures were written on a machine where
# torch could not be installed, so the piece with the most scope for a silent
# bug -- normalising over a variable number of neighbours with scatter ops --
# gets a NumPy reference that can be verified independently.
# ---------------------------------------------------------------------------

def test_numpy_attention_sums_to_one():
    src = np.array([0, 0, 0, 1])
    scores = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [5.0, 5.0]])
    a = ref.softmax_by_source(scores, src, 2)

    for atom in (0, 1):
        assert np.allclose(a[src == atom].sum(axis=0), 1.0)


def test_numpy_attention_survives_overflow_scale_scores():
    """exp(1000) is inf, and one inf makes the whole neighbourhood NaN."""
    a = ref.softmax_by_source(np.array([[1000.0], [1001.0], [-1000.0]]),
                              np.array([0, 0, 0]), 1)
    assert np.isfinite(a).all()
    assert float(a.sum()) == pytest.approx(1.0)
    assert a[1, 0] > a[0, 0] > a[2, 0]      # ordering preserved


def test_numpy_attention_is_uniform_for_equal_scores():
    src = np.array([0, 0, 0, 0])
    a = ref.softmax_by_source(np.zeros((4, 1)), src, 1)
    assert np.allclose(a, 0.25)


def test_numpy_attention_handles_a_single_neighbour():
    a = ref.softmax_by_source(np.array([[7.0]]), np.array([0]), 1)
    assert float(a[0, 0]) == pytest.approx(1.0)


@needs_torch
def test_torch_attention_matches_numpy_reference():
    """Same check as CGCNN's: two implementations, written separately."""
    import torch as T

    from src.models.architectures import GATv2Conv

    rng = np.random.default_rng(3)
    src = np.array([0, 0, 0, 1, 1, 2])
    scores = rng.normal(scale=3.0, size=(6, 4))

    got = GATv2Conv._softmax_by_source(
        T.from_numpy(scores), T.from_numpy(src), 3).numpy()
    want = ref.softmax_by_source(scores, src, 3)

    assert np.allclose(got, want, atol=1e-12), f"max diff {np.max(np.abs(got - want)):.2e}"
