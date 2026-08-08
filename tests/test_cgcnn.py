"""CGCNN tests.

A hand-written graph convolution has nothing to check itself against, and "it
runs and the loss goes down" is entirely compatible with a layer computing the
wrong function. So the model is verified two ways:

1. **Against an independent implementation.** ``cgcnn_reference.py`` is the same
   equations in NumPy, written from the maths rather than from the PyTorch code.
   The two must agree elementwise on random weights and random graphs.
2. **Against physics.** Relabelling atoms, or replicating the unit cell, must not
   change an intensive prediction. Those are properties the architecture is
   supposed to have, and they fail loudly if pooling or edge handling breaks.

The NumPy half runs everywhere. The PyTorch half is skipped where torch is not
installed, so CI stays fast and this file still guards the maths.

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

from src.models import cgcnn_reference as ref  # noqa: E402

def _has_torch() -> bool:
    try:
        __import__("torch")
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures: a small two-crystal batch
# ---------------------------------------------------------------------------

def toy_graphs(seed=0):
    rng = np.random.default_rng(seed)
    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)          # TiO3 | TiO
    batch_index = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    node_ptr = np.array([0, 4, 6], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    u = ref.gaussian_expand(rng.uniform(1.5, 7.5, size=len(src)))
    return z, src, dst, u, node_ptr, batch_index


def numpy_params(d=8, n_conv=3, h=16, n_e=41, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "embedding": rng.normal(scale=0.3, size=(100, d)),
        "conv_W": [rng.normal(scale=0.2, size=(2 * d + n_e, 2 * d)) for _ in range(n_conv)],
        "conv_b": [rng.normal(scale=0.1, size=2 * d) for _ in range(n_conv)],
        "fc_W": rng.normal(scale=0.2, size=(d, h)), "fc_b": rng.normal(scale=0.1, size=h),
        "out_W": rng.normal(scale=0.2, size=(h, 1)), "out_b": rng.normal(scale=0.1, size=1),
    }


# ---------------------------------------------------------------------------
# The NumPy reference is itself correct
# ---------------------------------------------------------------------------

def test_sigmoid_is_numerically_stable():
    """The naive form overflows around -750 and would poison a whole batch."""
    x = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    s = ref.sigmoid(x)
    assert np.isfinite(s).all()
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(1.0)
    assert s[2] == pytest.approx(0.5)


def test_softplus_matches_its_definition_and_stays_finite():
    x = np.array([-5.0, 0.0, 5.0])
    assert np.allclose(ref.softplus(x), np.log1p(np.exp(x)))
    assert np.isfinite(ref.softplus(np.array([1000.0]))).all()
    # Above the threshold torch switches to the identity; we must too.
    assert ref.softplus(np.array([1000.0]))[0] == pytest.approx(1000.0)


def test_duplicate_edges_accumulate():
    """Two periodic images of the same neighbour are two real bonds.

    Indexed assignment would keep only one of them and silently halve a
    coordination number. np.add.at / index_add_ accumulate.
    """
    d, n_e = 8, 41
    p = numpy_params(d=d)
    v = np.ones((2, d))
    once = ref.conv_layer(v, np.array([0]), np.array([1]),
                          np.zeros((1, n_e)), p["conv_W"][0], p["conv_b"][0])
    twice = ref.conv_layer(v, np.array([0, 0]), np.array([1, 1]),
                           np.zeros((2, n_e)), p["conv_W"][0], p["conv_b"][0])
    assert not np.allclose(once[0], twice[0]), "second edge was dropped"


def test_pool_is_a_mean_not_a_sum():
    v = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]])
    out = ref.pool(v, np.array([0, 2, 3]))
    assert np.allclose(out[0], [2.0, 3.0])
    assert np.allclose(out[1], [10.0, 10.0])


def test_reference_is_permutation_invariant():
    z, src, dst, u, node_ptr, _ = toy_graphs()
    p = numpy_params()
    base = ref.forward(z, src, dst, u, node_ptr, p)

    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    zp = np.concatenate([z[:4][perm], z[4:]])
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))
    got = ref.forward(zp, remap(src), remap(dst), u, node_ptr, p)

    assert np.allclose(base, got, atol=1e-12)


def test_reference_is_supercell_invariant():
    """A 2x replica is the same material, so an intensive property must not move."""
    p = numpy_params()
    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.full(4, 2.0))

    single = ref.forward(z1, s1, d1, u1, np.array([0, 3]), p)
    double = ref.forward(np.concatenate([z1, z1]),
                         np.concatenate([s1, s1 + 3]),
                         np.concatenate([d1, d1 + 3]),
                         np.concatenate([u1, u1]), np.array([0, 6]), p)
    assert np.allclose(single, double, atol=1e-12)


def test_reference_batching_matches_one_at_a_time():
    """Two graphs in one call must equal two separate calls."""
    z, src, dst, u, node_ptr, _ = toy_graphs()
    p = numpy_params()
    both = ref.forward(z, src, dst, u, node_ptr, p)

    a = ref.forward(z[:4], src[:6], dst[:6], u[:6], np.array([0, 4]), p)
    b = ref.forward(z[4:], src[6:] - 4, dst[6:] - 4, u[6:], np.array([0, 2]), p)
    assert np.allclose(both, np.concatenate([a, b]), atol=1e-12)


# ---------------------------------------------------------------------------
# PyTorch matches the reference
# ---------------------------------------------------------------------------

@needs_torch
def test_torch_matches_numpy_reference():
    """The load-bearing test of this phase.

    Same weights, same graph, two implementations written independently from the
    same equations. If they disagree, one of them is wrong, and no amount of
    watching a loss curve would have told us which.
    """
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    z, src, dst, u, node_ptr, batch_index = toy_graphs()
    model = CGCNN(CGCNNConfig(atom_fea_len=16, n_conv=3, h_fea_len=32,
                              use_batch_norm=False)).double().eval()

    batch = {
        "z": T.from_numpy(z), "src": T.from_numpy(src), "dst": T.from_numpy(dst),
        "u": T.from_numpy(u), "batch_index": T.from_numpy(batch_index), "n_graphs": 2,
    }
    with T.no_grad():
        got = model(batch).numpy()

    want = ref.forward(z, src, dst, u, node_ptr, model.export_numpy_params())
    assert np.allclose(got, want, atol=1e-9), f"max diff {np.max(np.abs(got - want)):.3e}"


@needs_torch
def test_torch_is_permutation_invariant():
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    z, src, dst, u, _, batch_index = toy_graphs()
    model = CGCNN(CGCNNConfig(atom_fea_len=16, n_conv=2,
                              use_batch_norm=False)).double().eval()
    base_batch = {"z": T.from_numpy(z), "src": T.from_numpy(src), "dst": T.from_numpy(dst),
                  "u": T.from_numpy(u), "batch_index": T.from_numpy(batch_index),
                  "n_graphs": 2}
    with T.no_grad():
        base = model(base_batch).numpy()

    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))
    with T.no_grad():
        got = model({**base_batch,
                     "z": T.from_numpy(np.concatenate([z[:4][perm], z[4:]])),
                     "src": T.from_numpy(remap(src)),
                     "dst": T.from_numpy(remap(dst))}).numpy()
    assert np.allclose(base, got, atol=1e-10)


@needs_torch
def test_torch_is_supercell_invariant():
    """Guards the mean readout. A sum would double the prediction."""
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    model = CGCNN(CGCNNConfig(atom_fea_len=16, n_conv=2,
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

    assert np.allclose(
        run(z1, s1, d1, u1),
        run(np.concatenate([z1, z1]), np.concatenate([s1, s1 + 3]),
            np.concatenate([d1, d1 + 3]), np.concatenate([u1, u1])),
        atol=1e-10,
    )


@needs_torch
def test_batched_equals_individual():
    """Batching must be an optimisation, not a change of function."""
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    z, src, dst, u, _, batch_index = toy_graphs()
    model = CGCNN(CGCNNConfig(atom_fea_len=16, n_conv=2,
                              use_batch_norm=False)).double().eval()

    def run(z, s, d, u, n):
        with T.no_grad():
            return model({"z": T.from_numpy(z), "src": T.from_numpy(s),
                          "dst": T.from_numpy(d), "u": T.from_numpy(u),
                          "batch_index": T.zeros(len(z), dtype=T.long),
                          "n_graphs": 1}).numpy()

    with T.no_grad():
        both = model({"z": T.from_numpy(z), "src": T.from_numpy(src),
                      "dst": T.from_numpy(dst), "u": T.from_numpy(u),
                      "batch_index": T.from_numpy(batch_index), "n_graphs": 2}).numpy()

    a = run(z[:4], src[:6], dst[:6], u[:6], 1)
    b = run(z[4:], src[6:] - 4, dst[6:] - 4, u[6:], 1)
    assert np.allclose(both, np.concatenate([a, b]), atol=1e-10)


@needs_torch
def test_gradients_reach_every_parameter():
    """A parameter with no gradient is dead weight, and silently so."""
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    z, src, dst, u, _, batch_index = toy_graphs()
    model = CGCNN(CGCNNConfig(atom_fea_len=16, n_conv=2, use_batch_norm=False)).double()
    out = model({"z": T.from_numpy(z), "src": T.from_numpy(src), "dst": T.from_numpy(dst),
                 "u": T.from_numpy(u), "batch_index": T.from_numpy(batch_index),
                 "n_graphs": 2})
    out.sum().backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or T.all(p.grad == 0))]
    # Only the embedding rows for elements absent from this toy batch may be zero.
    dead = [n for n in dead if "embedding" not in n]
    assert not dead, f"no gradient reaches: {dead}"


@needs_torch
def test_model_can_overfit_a_tiny_set():
    """If it cannot memorise 8 graphs it cannot learn 80,000.

    The cheapest possible check that the whole loop -- forward, loss, backward,
    step -- is actually connected.
    """
    import torch as T

    from src.models.cgcnn import CGCNN, CGCNNConfig

    rng = np.random.default_rng(0)
    model = CGCNN(CGCNNConfig(atom_fea_len=32, n_conv=2, use_batch_norm=False))
    opt = T.optim.Adam(model.parameters(), lr=0.02)

    z, src, dst, u, _, batch_index = toy_graphs()
    batch = {"z": T.from_numpy(z), "src": T.from_numpy(src), "dst": T.from_numpy(dst),
             "u": T.from_numpy(u.astype(np.float32)),
             "batch_index": T.from_numpy(batch_index), "n_graphs": 2}
    y = T.tensor(rng.normal(size=2).astype(np.float32))

    first = float(T.nn.functional.l1_loss(model(batch), y))
    for _ in range(200):
        opt.zero_grad()
        loss = T.nn.functional.l1_loss(model(batch), y)
        loss.backward()
        opt.step()

    assert float(loss) < first * 0.1, f"loss barely moved: {first:.4f} -> {float(loss):.4f}"


@needs_torch
def test_parameter_count_is_in_the_documented_range():
    """docs/cgcnn_math.md quotes ~81k for the default configuration."""
    from src.models.cgcnn import CGCNN, CGCNNConfig

    n = CGCNN(CGCNNConfig()).n_parameters()
    assert 60_000 < n < 110_000, f"{n:,} parameters, docs say ~81,000"
