# CatGNN

**Does knowing a material's *structure* predict its properties better than knowing its *chemistry* — and can the two be combined?**

Two materials can share a chemical formula and behave nothing alike. Rutile and
anatase are both TiO₂; one is a pigment, the other is the workhorse photocatalyst.
Any model that only sees "TiO₂" is blind to that difference by construction.

This repository builds models that see the actual crystal — atoms, and the contacts
between them, as a graph — and tests them head to head against models that only see
the chemical formula. Then it feeds the chemistry *into* the structural model to find
out whether they combine, and where.

> **Status: Phase 0 of 10 — the question, the data sourcing, and the hardware budget
> are settled. No models trained yet.** This repo is being built in the open, and the
> roadmap below shows honestly what exists and what does not. Everything on this page
> is reproducible today from the data in this repository.

---

## Reading this without a machine learning background

Everything above **"Under the hood"** is written for a chemical or materials engineer.
Here is the whole vocabulary you need:

| Term | What it means |
|---|---|
| **Graph** | A bonding diagram written down so a computer can read it. Dots and lines, nothing more exotic. |
| **Node** | One atom. |
| **Edge** | One neighbour contact — two atoms close enough to interact. |
| **Message passing** | Each atom updates its own description using its neighbours', over and over. After three rounds an atom's description reflects its full coordination environment — the same information you use when you say "octahedral Ti". |
| **Embedding** | A learned numerical fingerprint of an atom or a whole material. Similar materials get similar fingerprints. |
| **Descriptor** | A chemical property you already know and can look up: electronegativity, ionic radius, oxidation state, coordination number. |
| **Training / test split** | The model learns from one set of materials and is graded on a different set it has never seen. Grading it on what it memorised would tell you nothing. |
| **MAE** | Mean absolute error — the typical size of the model's mistake, in the property's own units. "MAE 0.3 eV" means predictions are off by about 0.3 eV on average. |
| **DFT** | Density functional theory, the quantum calculation that produced most of the numbers here. Calculated, not measured. |
| **GNN** | Graph neural network — a model that learns from a graph. |

---

## The problem, in one figure

![One formula, many structures, many band gaps](results/figures/fig1_polymorph_problem.png)

Materials Project holds **44 different TiO₂ crystal structures** computed with the same
DFT settings. Their band gaps run from **0.00 to 3.42 eV**. Same formula, every one of
them.

Anatase comes out at 2.06 eV, rutile at 1.77 eV, brookite at 2.29 eV — and those are
just the three anyone has named.

A model that is handed only the formula "TiO₂" has no way to tell these apart. It
receives one input, so it must produce one output, for all 44. The best it can possibly
do is predict the middle of the pack — and the average distance from that middle to the
real values is an error it can **never** remove, no matter how good the model is or how
much data you give it.

**For TiO₂ that unavoidable error is 0.43 eV.**

For scale: a well-known structure-aware model (CGCNN) has a *total* band-gap error of
about 0.39 eV across all of Materials Project. So on this family, being blind to
structure costs more than a good model's entire error budget. The same pattern shows up
in every family we checked — 0.55 eV for Fe₂O₃, 0.82 eV for CeO₂.

**That gap is the reason this repository exists.** How much of it can a structure-aware
model actually recover? And on which properties? Nobody should assume the answer; it
needs measuring.

> **An honest caveat, since it matters.** This is one formula, and it motivates the
> question rather than answering it. The real answer needs baselines run across the
> whole dataset — that is Phase 2. If those disagree with this figure, this figure is
> what gets corrected.

---

## What a crystal graph actually is

![How a crystal becomes a graph](results/figures/fig2_crystal_to_graph.png)

Nothing exotic happens between panels 1 and 2. A graph is a bonding diagram in a form
a computer can read: atoms become nodes, neighbour contacts become edges, and each edge
carries its bond length.

Panel 3 is the only genuinely new idea. Each atom repeatedly rewrites its own
description using its neighbours' descriptions. After one round it knows its immediate
neighbours; after three, it carries its whole local coordination environment. That is
what "message passing" means, and it is why these models can tell rutile from anatase
when a formula cannot.

The structure in that figure is real rutile, built from published neutron-diffraction
parameters. The script recomputes the Ti–O bond lengths and gets **1.949 Å (×4
equatorial) and 1.980 Å (×2 apical)** against the measured 1.946 and 1.983 Å — and it
asserts that agreement on every run, so a typo in a lattice constant fails loudly
instead of quietly producing a wrong picture.

<details>
<summary><b>A bug worth reading about, if you ever build one of these</b></summary>

<br>

Rutile's unit cell contains only 4 oxygens, yet every Ti is octahedrally coordinated by
6. Those extra oxygens live in the *neighbouring repeat* of the crystal.

The textbook shortcut — the "minimum image convention", keeping only the closest copy
of each neighbouring atom — silently returns **Ti CN = 4** and destroys the octahedron.
Nothing downstream complains. The model just trains on subtly wrong chemistry and
returns plausible-looking numbers.

The fix is to count *every* periodic image inside the cutoff, not just the nearest.
This is now checked by a test that fails if coordination numbers drift
(`tests/test_phase0.py::test_rutile_coordination_numbers`).

</details>

---

## Where every number comes from

![Data provenance](results/figures/fig3_data_provenance.png)

**Nothing in this repository is invented.** Four of the five prediction targets are
downloaded directly from open databases. The fifth — catalytic activity — is computed
from real adsorption energies through a published thermodynamic equation, and is called
a *descriptor* everywhere it appears, never a rate.

| What | Where from | What kind of number |
|---|---|---|
| Band gap, formation energy, stability | Materials Project, JARVIS-DFT | Calculated by DFT |
| Band gap, ~4,600 materials | `matminer` `expt_gap` | **Measured in a lab** |
| Adsorption energy | Catalysis-Hub (SUNCAT), Open Catalyst | Calculated by DFT |
| Catalytic activity descriptor | Derived here from adsorption energies, via scaling relations | A thermodynamic proxy, **not a measured rate** |

Full licences, API endpoints, access dates and citations: [`SOURCES.md`](SOURCES.md).
The tiering rules: [`DATA_GROUNDING.md`](DATA_GROUNDING.md).

### What this repository does *not* do

- **It does not predict measured catalytic activity.** No dataset pairs crystal
  structures with measured turnover frequencies at the scale a neural network needs.
  What it produces is a theoretical activity descriptor computed from predicted
  adsorption energies. That is a legitimate screening quantity. It is not a rate you
  would measure in a reactor.
- **It inherits DFT's errors.** The models learn from DFT numbers, so wherever DFT is
  systematically wrong they will be confidently wrong in the same direction. The
  experimental band gaps let us *measure* that inheritance instead of asserting it.
  (A sibling repo, [`MPExplorer`](https://github.com/teja2792/MPExplorer), documents a
  ~75% DFT underestimate for Cu₂O specifically.)
- **"Stable" means stable in DFT.** Our own snapshot shows the catch: of 44 TiO₂
  polymorphs, DFT puts **anatase** on the convex hull, not rutile — while rutile is the
  ambient-stable phase in reality.
- **It was trained on a laptop.** Reduced scale, and it will not match published
  leaderboard numbers. The gap gets reported, with its reason.

Everything that could go wrong, written down before the results existed:
[`LIMITATIONS.md`](LIMITATIONS.md).

---

## What is built, and what is not

![Build plan and status](results/figures/fig4_roadmap.png)

The interesting phase is **5**. A neural network learns representations from structure;
a chemist already has decades of intuition encoded as descriptors. Do they combine?
Four fusion strategies get ablated, and the headline figure will be a data-efficiency
curve: **below some training-set size, known chemistry beats learned structure.** Where
that crossover sits is a practical answer to "which should I use?" for anyone with a
few hundred measurements rather than a few hundred thousand — which is the normal
situation in a catalysis lab.

Be warned that the answer may be unflattering to the neural networks. On several
standard benchmarks, plain composition descriptors with gradient boosting are
competitive with structural GNNs. If that reproduces here, it gets reported.

---

## Running it yourself

Phase 0 needs three packages. Nothing heavier is required to regenerate everything on
this page.

```bash
git clone https://github.com/teja2792/CatGNN
cd CatGNN
pip install "numpy>=1.24" "pandas>=2.0" "matplotlib>=3.7" "pytest>=7.4"

python scripts/benchmark_hardware.py    # measure YOUR machine, writes COMPUTE_BUDGET.md
python scripts/make_figures.py          # rebuild every figure above from source data
pytest -q                               # 14 correctness tests
```

No figure here is hand-drawn or hand-edited — each is generated from the data in
`data/reference/` or from published crystal structures, and CI fails if a committed
figure stops matching its script.

`benchmark_hardware.py` exists because quoting somebody else's GPU timings would be
useless. It times graph construction and message passing on *your* CPU and writes a
dataset-size budget to [`COMPUTE_BUDGET.md`](COMPUTE_BUDGET.md). Every scoping decision
in Phase 1 refers to that measured number rather than to a guess.

---

## Under the hood

> Everything below assumes a machine learning background. Nothing above it does.

### The experiment

| | |
|---|---|
| **Question** | Does structure-aware message passing beat composition descriptors, and does fusing them help? |
| **Primary targets** | Band gap and formation energy (Materials Project, GGA), then adsorption energy |
| **Baselines** | Ridge / Random Forest / gradient boosting on Magpie + hand-built chemical descriptors |
| **Architectures** | CGCNN (from scratch), MPNN, MEGNet, ALIGNN, GATv2 |
| **Splits** | Random, composition-disjoint, structure-similarity-disjoint, Matbench official folds |
| **Protocol** | Identical wall-clock budget per model, ≥3 seeds, error bars on everything |
| **Hardware** | Ryzen 5 laptop, CPU only |

Each architecture is present to test one specific thing, not for completeness: MPNN asks
whether generic message passing suffices; MEGNet asks whether an explicit global state
helps (and is the natural injection point for a descriptor vector); ALIGNN asks whether
bond angles matter; GATv2 provides genuine attention.

### On "attention"

CGCNN, MEGNet and ALIGNN have **no attention mechanism.** CGCNN uses a sigmoid *gate*
in its convolution — `σ(zW_f + b_f) ⊙ g(zW_s + b_s)` — with no softmax over neighbours
and no query/key. ALIGNN uses edge gating. MEGNet uses a global state vector. All three
are routinely mislabelled as attention in blog posts and portfolio repos.

Where this repo shows attention maps they come from GATv2, which actually has attention.
For the other models it shows gate activations and integrated-gradient attributions, and
labels them as such.

### Splits, and why the obvious one is wrong

Random splits on materials databases leak: near-duplicate polymorphs and same-composition
entries land on both sides, so the test set is not unseen and the reported error is
optimistic. All four schemes above are run and reported side by side. The random split
is expected to look best, and the size of that gap is itself a result.

### Where this sits in the current landscape

CGCNN (2018), MEGNet (2019) and ALIGNN (2021) are here because they are the clearest
crystal GNNs to implement and understand — not because they are current. The field has
moved to equivariant architectures (NequIP, MACE, SevenNet) and universal interatomic
potentials (M3GNet, CHGNet, MatterSim), and by 2025–26 the Matbench leaderboards are led
by foundation-model approaches. Those assume GPU compute this project deliberately does
not. Nothing here is a statement about what the best current method can do.

---

## Related repositories

Part of a portfolio applying ML to catalysis and materials research. CatGNN is the
**representation** layer: it learns a numerical fingerprint of a material that the
others can use.

| Repo | What it does | How it connects |
|---|---|---|
| [`MPExplorer`](https://github.com/teja2792/MPExplorer) | Live Materials Project explorer; documents three DFT failure modes | Supplies the data layer, and the snapshot behind Figure 1 |
| [`CatalystBO`](https://github.com/teja2792/CatalystBO) | Bayesian optimisation — decides which catalyst to make next | Phase 10: learned embeddings replace hand-built search-space features |
| [`ExplainableCatML`](https://github.com/teja2792/ExplainableCatML) | Do different explanation methods agree? | Phase 6 extends that question from tabular models to graphs |
| [`SpectraHub`](https://github.com/teja2792/SpectraHub) | XAS spectra → oxidation state, coordination number, bond length | Those become chemical descriptors in Phase 2 |
| [`CatalystML`](https://github.com/teja2792/CatalystML) | Property prediction from tabular features | The non-graph point of comparison |
| [`MieCatalystML`](https://github.com/teja2792/MieCatalystML) | Validated Mie scattering physics engine for Cu₂O | Downstream consumer of predicted optical properties |

## Licence

MIT (see [`LICENSE`](LICENSE)). Each dataset carries its own licence — see
[`SOURCES.md`](SOURCES.md).
