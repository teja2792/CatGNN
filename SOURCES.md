# Sources

Every dataset this repository uses or plans to use, with its licence, how it is
accessed, and what has actually been downloaded so far. If a number appears
anywhere in this repo and cannot be traced to a row in this file, that is a bug —
please open an issue.

**Status key:** ✅ in the repo now · ⬜ planned, not yet downloaded

---

## 1. Materials Project

| | |
|---|---|
| Status | ✅ small snapshot in repo, ⬜ full pull in Phase 1 |
| What | DFT-computed properties of inorganic crystals: band gap, formation energy, energy above hull, density, space group, elastic moduli |
| Scale | ~150,000 materials |
| Access | `mp-api` Python client. Requires a free API key from <https://next-gen.materialsproject.org> |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Cite | A. Jain et al., *Commingling of the Materials Genome Initiative*, APL Materials 1, 011002 (2013) |
| In this repo | `data/reference/mp_summary_snapshot.csv` |

**About the snapshot.** 77 rows covering TiO₂, Fe₂O₃, CeO₂ and Cu₂O, produced by
[`MPExplorer`](https://github.com/teja2792/MPExplorer) (a sibling repository) via the
live Materials Project API, and copied here unchanged so that Figure 1 reproduces
without an API key. Columns are exactly as returned by the API.

**Two things to know before using it:**

- It mixes DFT functionals. GGA, GGA+U and r2SCAN band gaps are **not comparable
  numbers**, and averaging across them would be a methodological error. Every
  analysis in this repo groups by `dft_run_type` first.
- DFT band gaps are systematically too small. `MPExplorer` documents three specific
  failure modes across these four materials, including a ~75% underestimate for Cu₂O.
  Models trained on these labels inherit that bias; Phase 1 measures how much.

---

## 2. JARVIS-DFT (NIST)

| | |
|---|---|
| Status | ⬜ Phase 1 |
| What | DFT properties for ~80,000 materials, including band gaps computed with the TBmBJ meta-GGA functional as well as with standard GGA |
| Access | `jarvis-tools` Python package. **No API key needed.** Also exposed through OPTIMADE at <https://jarvis.nist.gov/optimade/jarvisdft> |
| Licence | US Government work / NIST public data |
| Cite | K. Choudhary et al., *The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design*, npj Computational Materials 6, 173 (2020) |

**Why both this and Materials Project.** They calculate many of the same materials
with different settings and get different numbers. Training on one and testing on the
other is a cross-database generalisation test — a much more realistic measure of
whether a model transfers than a random split within a single database.

---

## 3. Experimental band gaps (`matminer` `expt_gap`)

| | |
|---|---|
| Status | ⬜ Phase 1 |
| What | ~4,600 **experimentally measured** band gaps |
| Access | `matminer.datasets.load_dataset("expt_gap")` — one line, no key |
| Licence | as distributed with matminer (BSD-style) |
| Cite | Y. Zhuo, A. Mansouri Tehrani, J. Brgoch, *Predicting the Band Gaps of Inorganic Solids by Machine Learning*, J. Phys. Chem. Lett. 9, 1668 (2018) |

The only labels in this repository that came from a laboratory rather than a computer.
Small, and worth far more than its size: it is what makes it possible to ask how much
of DFT's systematic error a model trained on DFT labels reproduces.

---

## 4. Catalysis-Hub (SUNCAT)

| | |
|---|---|
| Status | ⬜ Phase 7 (primary adsorption source) |
| What | >100,000 chemisorption and reaction energies on catalyst surfaces, with the surface geometries and calculation parameters |
| Access | GraphQL API at <http://api.catalysis-hub.org/graphql>. **No key needed.** Console at <https://www.catalysis-hub.org/graphQLConsole> |
| Licence | open, per site terms — confirm and record the exact terms at download time |
| Cite | K. Winther et al., *Catalysis-Hub.org, an open electronic structure database for surface reactions*, Scientific Data 6, 75 (2019) |

Chosen as the primary adsorption-energy source over OC20 because it is key-free,
laptop-sized, drawn from published studies, and comes with the calculation settings
attached.

---

## 5. Open Catalyst Project (OC20 / OC22)

| | |
|---|---|
| Status | ⬜ Phase 7, stretch goal |
| What | OC20: ~1.2M DFT relaxations of adsorbates on catalyst surfaces. OC22: oxide electrocatalysts specifically |
| Subset used | **IS2RE 10k** — an official small training split. The full IS2RE set is ~460k and the full OC20 required >200M compute hours to generate; neither is laptop territory |
| Access | `fairchem` package, <https://fair-chem.github.io> |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Cite | L. Chanussot et al., *The Open Catalyst 2020 (OC20) Dataset and Community Challenges*, ACS Catalysis 11, 6059 (2021); R. Tran et al., *The Open Catalyst 2022 (OC22) Dataset*, ACS Catalysis 13, 3066 (2023) |

---

## 6. Matbench

| | |
|---|---|
| Status | ⬜ Phase 1 |
| What | 13 standardised materials-property tasks with **official train/test folds** and a public leaderboard |
| Access | `matbench` package, <https://matbench.materialsproject.org> |
| Licence | MIT |
| Cite | A. Dunn et al., *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm*, npj Computational Materials 6, 138 (2020) |

Used so that at least one number in this repository is directly comparable to a public
leaderboard, rather than only to itself.

---

## 7. Rutile TiO₂ crystal structure

| | |
|---|---|
| Status | ✅ used in Figure 2 |
| What | Lattice parameters a = b = 4.5937 Å, c = 2.9587 Å, oxygen parameter u = 0.30478, space group P4₂/mnm |
| Source | C. J. Howard, T. M. Sabine, F. Dickson, *Structural and thermal parameters for rutile and anatase*, Acta Crystallographica B47, 462 (1991) — neutron diffraction |

Hard-coded in `src/figures/fig_crystal_to_graph.py`. The script recomputes the Ti–O
bond lengths from these parameters and asserts they match the published values
(1.946 Å ×4, 1.983 Å ×2) to within 0.01 Å, so a typo in the constants fails loudly
instead of silently producing a wrong figure.

---

## 8. Published model results quoted for comparison

Quoted as reference points in figures and text. No data from these is redistributed here.

| Model | Result quoted | Reference |
|---|---|---|
| CGCNN | MAE 0.388 eV (band gap), 0.039 eV/atom (formation energy) on Materials Project | T. Xie & J. C. Grossman, *Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, Phys. Rev. Lett. 120, 145301 (2018) |
| MEGNet | — | C. Chen et al., Chem. Mater. 31, 3564 (2019) |
| ALIGNN | — | K. Choudhary & B. DeCost, npj Comput. Mater. 7, 185 (2021) |
| MPNN | — | J. Gilmer et al., ICML (2017) |
| GATv2 | — | S. Brody, U. Alon, E. Yahav, ICLR (2022) |

---

## Recording rules

When any dataset is downloaded, the following go into `data/raw/<source>/manifest.json`
and are **never** overwritten by a later pull:

- the exact query or API call, verbatim
- the date of access
- the database version or release tag, if one is exposed
- row count and a checksum of the downloaded file
- the licence in force on that date

Raw downloads are `.gitignore`d — they are large and reproducible from the manifest.
The manifests themselves are committed.

## Band gap: DFT versus experiment

Used in figure 13 and in the README's "what the numbers mean" section, to set the
model's error against the error of the data it is trained on.

- Kim, S., Lee, M., Hong, C., Yoon, Y., An, H., Lee, D., Jeong, W., Yoo, D.,
  Kang, Y., Youn, Y. & Han, S. *A band-gap database for semiconducting inorganic
  materials calculated with hybrid functional.* **Scientific Data 7, 387 (2020)**.
  https://www.nature.com/articles/s41597-020-00723-8

  Quoted for its reported RMSE of existing **GGA-based databases against
  experiment: 0.75–1.05 eV**, against 0.36 eV for their hybrid-functional
  database. This repository trains on GGA / GGA+U / r2SCAN values from Materials
  Project, so that 0.75–1.05 eV is the error already present in the labels before
  any model touches them.

- Materials Project documentation, *Electronic Structure* methodology page, which
  states that band gaps in the database are underestimated relative to experiment.
  https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure

The reference band gaps for Ge, Si, GaAs, GaP, CdS, TiO2, ZnO and diamond used in
figure 13 are standard room-temperature **experimental** values from
semiconductor physics texts, and are labelled as experimental on the figure
precisely because the dataset holds DFT values, which are smaller.
