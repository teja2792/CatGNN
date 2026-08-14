"""Eight numbers describing a binding site, and why they are the real baseline.

WHAT THIS IS FOR
----------------
Phase 7 has been comparing the graph network against "predict the mean binding
energy of this surface". That baseline is weak by construction -- on a
surface-disjoint split it degenerates to the global mean, because a surface never
seen has no mean to borrow. Beating it proves almost nothing.

These descriptors are the baseline it should have been compared against. They are
what a catalysis chemist would write down after looking at a structure for a
minute: what kind of site is it, how coordinated are the atoms forming the bond,
how far away is the molecule, what is it made of. Nothing learned, nothing fitted,
eight numbers per row.

Ridge regression on them reaches R2 = 0.568 under grouped cross-validation by
surface. The 81,000-parameter graph network reaches 0.21 +- 0.05 on the same
split. A linear model on eight hand-written features beats the network by a
factor of nearly three.

That is the single most useful measurement in this phase. It rules out the
explanations that would otherwise be reached for -- the signal is not too weak,
the sample is not too small, the split is not too hard -- because a linear model
extracts the signal from the same rows under the same split. What is left is that
the network is not extracting it, which is a statement about the network.

WHY THESE EIGHT
---------------
Chosen from surface-science practice before any of them was fitted:

    n_site_atoms      1 = atop, 2 = bridge, 3+ = hollow. The classic adsorption
                      site taxonomy, and the strongest single feature here.
    mean/min CN       coordination of the surface atoms doing the binding. The
                      generalised-coordination-number model (Calle-Vallejo et al.,
                      Science 350, 185, 2015) predicts binding from little else.
    height            how far the molecule sits above the surface plane. Long
                      means weak.
    mean/min bond     the adsorbate-surface contact distances themselves.
    mean Z of site    what element it is actually touching. On these nitrides the
                      CO often sits over nitrogen rather than the metal, and that
                      is a different bond.
    CN of adsorbate   how many things the CO itself is in contact with.

They are correlated and that is fine -- ridge handles it, and the point is not a
parsimonious model but a fair floor for the network to clear.

WHAT THIS DOES NOT SAY
----------------------
It does not say graph networks cannot do this. It says THIS graph network, at this
size, on this many rows, does not. A linear baseline this strong usually means the
network is under-regularised, over-parameterised for the sample, or trained with
settings carried over from a different problem -- all three are true here and are
being fixed in turn.
"""

from __future__ import annotations

import numpy as np

# The contact radius that defines the site. Matches SITE_RADIUS in
# scripts/build_slab_graphs.py; at 2.6 A this found no surface atom for 19% of
# slabs and the descriptors below were silently degenerate for them.
SITE_RADIUS = 3.5

FEATURE_NAMES = [
    "n_site_atoms",        # atop / bridge / hollow
    "site_cn_mean",        # coordination of the binding surface atoms
    "site_cn_min",
    "height",              # adsorbate above the surface plane
    "bond_mean",           # adsorbate-surface contact distances
    "bond_min",
    "site_z_mean",         # which element is being bound
    "adsorbate_cn",        # contacts made by the molecule itself
]


def describe_site(z, is_adsorbate, height, coordination, src, dst, dist,
                  radius: float = SITE_RADIUS) -> np.ndarray:
    """The eight descriptors for one slab.

    Every value is computed from the graph the network also sees, so this is a
    fair comparison: the baseline is not given information the network lacks.
    """
    a = np.asarray(is_adsorbate, dtype=bool)
    cross = a[src] & ~a[dst]                     # adsorbate -> surface contacts
    near = cross & (dist <= radius)
    site_atoms = np.unique(dst[near])
    bonds = dist[near]

    n = len(site_atoms)
    cn_site = coordination[site_atoms] if n else np.array([0.0])
    z_site = z[site_atoms].astype(np.float64) if n else np.array([0.0])
    # A slab with no surface contact inside the radius is degenerate; the
    # fallbacks below are deliberately out-of-range so such rows are visible as
    # outliers rather than blending in with real sites.
    h = (float(height[a].min() - height[~a].max()) if (~a).any() else 0.0)

    return np.array([
        float(n),
        float(cn_site.mean()),
        float(cn_site.min()),
        h,
        float(bonds.mean()) if bonds.size else 2 * radius,
        float(bonds.min()) if bonds.size else 2 * radius,
        float(z_site.mean()),
        float(coordination[a].mean()) if a.any() else 0.0,
    ], dtype=np.float64)


def descriptor_matrix(store, radius: float = SITE_RADIUS) -> np.ndarray:
    """(n_rows, 8) for every graph in a SlabStore."""
    out = np.zeros((len(store.material_id), len(FEATURE_NAMES)))
    for i in range(len(store.material_id)):
        n0, n1 = store.node_ptr[i], store.node_ptr[i + 1]
        e0, e1 = store.edge_ptr[i], store.edge_ptr[i + 1]
        out[i] = describe_site(
            store.z[n0:n1], store.is_adsorbate[n0:n1], store.height[n0:n1],
            store.coordination[n0:n1],
            store.src[e0:e1].astype(np.int64) - 0,
            store.dst[e0:e1].astype(np.int64) - 0,
            store.dist[e0:e1], radius)
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    """Standardise on TRAIN only, then closed-form ridge. Returns a predictor.

    The scaler is fitted on the training rows alone for the same reason the
    network's target normaliser is: statistics of the test set must not reach the
    model, however harmlessly.
    """
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.c_[(X - mu) / sd, np.ones(len(X))]
    # The intercept column is not penalised, which is why the last diagonal entry
    # is zeroed: shrinking it would bias every prediction toward zero rather than
    # toward the training mean.
    P = np.eye(A.shape[1])
    P[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + lam * P, A.T @ y)

    def predict(Z: np.ndarray) -> np.ndarray:
        return np.c_[(Z - mu) / sd, np.ones(len(Z))] @ w

    return predict, w
