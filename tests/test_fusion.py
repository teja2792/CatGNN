"""Tests for Phase 5 -- feeding tabulated chemistry into the graph network.

The periodic-table tests run everywhere, because a misaligned element table is
the failure mode that would look most like a boring scientific result. If row 17
held argon's properties instead of chlorine's, every atom in the dataset would
get its neighbour's chemistry, the model would train perfectly happily, and the
conclusion would be "adding element properties does not help" -- which is a
much harder bug to notice than a crash.

So the table is checked against chemistry that is true independently of this
code: halogens must resemble halogens, alkali metals must resemble alkali metals,
and a halogen must not resemble an alkali metal.

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

from src.features.element_features import element_feature_table  # noqa: E402


def _has_torch() -> bool:
    try:
        __import__("torch")
        return True
    except ImportError:
        return False


needs_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# The element table
# ---------------------------------------------------------------------------

def test_table_is_indexed_by_atomic_number_not_row_order():
    """The bug this whole file exists to catch.

    property_matrix() is indexed by alphabetical symbol; the graphs are indexed
    by atomic number. Confusing the two puts actinium's properties on hydrogen.
    """
    from src.features.descriptors import element_table as raw

    table, known, names = element_feature_table()
    elements, props = raw()
    z_col = names.index("Z")

    # Standardisation is affine and monotone, so ordering by Z must survive it.
    zs = [1, 6, 8, 26, 47, 79]
    got = [table[z][z_col] for z in zs]
    assert got == sorted(got), "the Z column is not monotone in the row index"

    for symbol in ("H", "O", "Fe", "Au", "U"):
        z = int(elements[symbol]["Z"])
        assert known[z], f"{symbol} (Z={z}) is missing from the table"


def test_table_has_no_holes_and_is_standardised():
    table, known, names = element_feature_table()

    assert table.shape == (101, len(names))
    assert not np.isnan(table).any()
    assert not np.isinf(table).any()
    assert known.sum() >= 95, f"only {known.sum()} elements covered"

    assert np.allclose(table[known].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(table[known].std(axis=0), 1.0, atol=1e-5)


def test_unknown_elements_get_the_average_element_not_a_random_vector():
    """The entire point. An unknown element must be neutral, not arbitrary."""
    table, known, _ = element_feature_table()
    for z in np.where(~known)[0]:
        assert np.all(table[z] == 0.0)


def test_chemically_similar_elements_have_similar_vectors():
    """Checked against chemistry, not against this code's own output."""
    table, _, _ = element_feature_table()
    Cl, Br, I, Na, K, Cs, Fe = 17, 35, 53, 11, 19, 55, 26

    # Within a group
    assert cos(table[Cl], table[Br]) > 0.8
    assert cos(table[Na], table[K]) > 0.8

    # Across very different groups
    assert cos(table[Cl], table[Na]) < 0.3
    assert cos(table[Cl], table[Fe]) < 0.5

    # Group trends are ordered: Cl is more like Br than like I
    assert cos(table[Cl], table[Br]) > cos(table[Cl], table[I])
    assert cos(table[Na], table[K]) > cos(table[Na], table[Cs])


def test_a_held_out_element_still_has_real_chemistry():
    """The Phase 5 hypothesis in one assertion.

    Selenium is the element the Phase 3 model would have had no vector for. Here
    it must still land near its group neighbour sulfur, because nothing about
    this featurisation depends on having trained on selenium.
    """
    table, _, _ = element_feature_table()
    S, Se, Te, Cu = 16, 34, 52, 29

    assert cos(table[Se], table[S]) > 0.75
    assert cos(table[Se], table[Te]) > 0.75
    assert cos(table[Se], table[Cu]) < cos(table[Se], table[S])


# ---------------------------------------------------------------------------
# The featuriser and the fused network
# ---------------------------------------------------------------------------

def toy_batch(torch, n_comp=198, with_comp=False):
    rng = np.random.default_rng(0)
    from src.models import cgcnn_reference as ref

    z = np.array([22, 8, 8, 8, 22, 8], dtype=np.int64)
    bidx = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    src = np.array([0, 0, 1, 2, 3, 1, 4, 5], dtype=np.int64)
    dst = np.array([1, 2, 0, 0, 1, 3, 5, 4], dtype=np.int64)
    u = ref.gaussian_expand(rng.uniform(1.5, 7.5, size=len(src)))
    batch = {"z": torch.from_numpy(z), "src": torch.from_numpy(src),
             "dst": torch.from_numpy(dst), "u": torch.from_numpy(u),
             "batch_index": torch.from_numpy(bidx), "n_graphs": 2}
    if with_comp:
        batch["comp"] = torch.from_numpy(rng.normal(size=(2, n_comp)))
    return batch, z, src, dst, u


@needs_torch
@pytest.mark.parametrize("mode", ["learned", "properties", "both"])
def test_featuriser_shape_and_finiteness(mode):
    import torch as T

    from src.models.fusion import AtomFeaturiser

    f = AtomFeaturiser(mode, d=16).double()
    v = f(T.tensor([1, 8, 22, 47]))
    assert v.shape == (4, 16)
    assert T.isfinite(v).all()


@needs_torch
def test_property_featuriser_is_deterministic_for_the_same_element():
    """Two oxygens anywhere in the batch must start identical."""
    import torch as T

    from src.models.fusion import AtomFeaturiser

    f = AtomFeaturiser("properties", d=16).double().eval()
    v = f(T.tensor([8, 22, 8]))
    assert T.allclose(v[0], v[2])


@needs_torch
def test_property_featuriser_generalises_where_learned_cannot():
    """The mechanism behind the Phase 5 hypothesis, asserted directly.

    Take an element the model never trained on. Under 'learned' its vector is
    whatever random initialisation produced, so it bears no relation to its
    chemical neighbours. Under 'properties' it must still sit close to them.
    """
    import torch as T

    from src.models.fusion import AtomFeaturiser

    S, Se, Cu = 16, 34, 29
    z = T.tensor([S, Se, Cu])

    prop = AtomFeaturiser("properties", d=32).double().eval()(z).detach().numpy()
    learned = AtomFeaturiser("learned", d=32).double().eval()(z).detach().numpy()

    assert cos(prop[1], prop[0]) > 0.5, "Se and S should be close under properties"
    # A free embedding at initialisation knows nothing; Se is no closer to its
    # own group than to a transition metal.
    assert abs(cos(learned[1], learned[0])) < 0.5


@needs_torch
def test_featuriser_rejects_an_unknown_mode():
    from src.models.fusion import AtomFeaturiser

    with pytest.raises(ValueError, match="mode must be one of"):
        AtomFeaturiser("magic", d=8)


@needs_torch
def test_unknown_elements_are_reported():
    import torch as T

    from src.models.fusion import AtomFeaturiser

    f = AtomFeaturiser("properties", d=8)
    assert f.unknown_elements(T.tensor([8, 22, 47])).numel() == 0
    assert 0 in f.unknown_elements(T.tensor([0, 8])).tolist()


@needs_torch
@pytest.mark.parametrize("backbone", ["cgcnn", "mpnn", "gatv2"])
@pytest.mark.parametrize("mode", ["learned", "properties", "both"])
def test_fused_forward(backbone, mode):
    import torch as T

    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(backbone=backbone, atom_features=mode,
                              atom_fea_len=16, n_conv=2, h_fea_len=32,
                              use_batch_norm=False)).double().eval()
    batch, *_ = toy_batch(T)
    with T.no_grad():
        out = m(batch)
    assert out.shape == (2,)
    assert T.isfinite(out).all()


@needs_torch
@pytest.mark.parametrize("mode", ["properties", "both"])
def test_fused_permutation_invariance(mode):
    import torch as T

    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(atom_features=mode, atom_fea_len=16, n_conv=2,
                              h_fea_len=32, use_batch_norm=False)).double().eval()
    batch, z, src, dst, u = toy_batch(T)
    with T.no_grad():
        base = m(batch).numpy()

    perm = np.array([2, 0, 3, 1])
    inv = {int(o): i for i, o in enumerate(perm)}
    remap = np.vectorize(lambda v: inv[int(v)] if v < 4 else int(v))
    with T.no_grad():
        got = m({**batch,
                 "z": T.from_numpy(np.concatenate([z[:4][perm], z[4:]])),
                 "src": T.from_numpy(remap(src)),
                 "dst": T.from_numpy(remap(dst))}).numpy()

    assert np.allclose(base, got, atol=1e-9)


@needs_torch
@pytest.mark.parametrize("mode", ["properties", "both"])
def test_fused_supercell_invariance(mode):
    """Band gap is intensive; doubling the cell must not change it."""
    import torch as T

    from src.models import cgcnn_reference as ref
    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(atom_features=mode, atom_fea_len=16, n_conv=2,
                              h_fea_len=32, use_batch_norm=False)).double().eval()

    z1 = np.array([22, 8, 8], dtype=np.int64)
    s1 = np.array([0, 0, 1, 2], dtype=np.int64)
    d1 = np.array([1, 2, 0, 0], dtype=np.int64)
    u1 = ref.gaussian_expand(np.full(4, 2.0))

    def run(z, s, d, u):
        with T.no_grad():
            return m({"z": T.from_numpy(z), "src": T.from_numpy(s),
                      "dst": T.from_numpy(d), "u": T.from_numpy(u),
                      "batch_index": T.zeros(len(z), dtype=T.long),
                      "n_graphs": 1}).numpy()

    single = run(z1, s1, d1, u1)
    double = run(np.concatenate([z1, z1]), np.concatenate([s1, s1 + 3]),
                 np.concatenate([d1, d1 + 3]), np.concatenate([u1, u1]))
    assert np.allclose(single, double, atol=1e-9)


@needs_torch
def test_composition_branch_changes_the_prediction():
    """A descriptor branch that is wired up but ignored is a silent failure."""
    import torch as T

    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(use_composition=True, n_composition=12,
                              atom_fea_len=16, n_conv=2, h_fea_len=32,
                              use_batch_norm=False)).double().eval()
    batch, *_ = toy_batch(T, n_comp=12, with_comp=True)
    with T.no_grad():
        a = m(batch).numpy()
        b = m({**batch, "comp": batch["comp"] + 3.0}).numpy()

    assert not np.allclose(a, b), "the composition vector is being ignored"


@needs_torch
def test_composition_branch_fails_loudly_when_the_batch_lacks_it():
    import torch as T

    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(use_composition=True, n_composition=12,
                              atom_fea_len=16, n_conv=2,
                              use_batch_norm=False)).double().eval()
    batch, *_ = toy_batch(T)
    with pytest.raises(KeyError, match="attach_composition"):
        m(batch)


@needs_torch
@pytest.mark.parametrize("mode", ["learned", "properties", "both"])
def test_gradients_reach_every_trainable_parameter(mode):
    import torch as T

    from src.models.fusion import FusedGNN, FusionConfig

    m = FusedGNN(FusionConfig(atom_features=mode, atom_fea_len=16, n_conv=2,
                              h_fea_len=32, use_batch_norm=False)).double()
    batch, *_ = toy_batch(T)
    m(batch).sum().backward()

    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or T.all(p.grad == 0))
            and "embedding" not in n]
    assert not dead, f"no gradient reaches {dead}"


@needs_torch
def test_the_periodic_table_is_a_buffer_not_a_parameter():
    """If the table were learnable, an unseen element's row would drift to
    whatever minimises training loss -- reintroducing the exact failure this
    phase exists to fix, and doing it invisibly."""
    from src.models.fusion import AtomFeaturiser

    f = AtomFeaturiser("properties", d=8)
    assert "table" in dict(f.named_buffers())
    assert "table" not in dict(f.named_parameters())


@needs_torch
def test_property_mode_has_no_per_element_free_parameters():
    """'properties' must contain no embedding at all, or the comparison is
    meaningless -- it would be testing 'properties AND memorisation'."""
    from src.models.fusion import AtomFeaturiser

    f = AtomFeaturiser("properties", d=8)
    assert not hasattr(f, "embedding")
    assert all("embedding" not in n for n, _ in f.named_parameters())
