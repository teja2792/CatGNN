# Limitations

Written before the results exist, so that the caveats are not retrofitted around
whatever happens to come out well. Updated as each phase lands.

Data-side caveats live in [`DATA_GROUNDING.md`](DATA_GROUNDING.md). This file is
about the models, the protocol, and the hardware.

---

## 1. Everything here was trained on a laptop CPU

One Ryzen 5, no GPU. Consequences, all real:

- **Reduced dataset sizes.** Tens of thousands of structures, not the full ~150k.
- **Small models.** Hidden dimension 64, three convolution layers. Published results
  often use larger models trained far longer.
- **Fixed compute budget per model.** Every architecture gets the same wall-clock
  allowance. This makes the comparison *fair*, but it means no model is trained to
  convergence, and a model that converges slowly is penalised.
- **A slow architecture is penalised.** Under an equal-time budget, a model with an
  expensive convolution simply sees fewer epochs. This is why ALIGNN was dropped
  from the comparison entirely (§11) rather than reported with an asterisk.

**These numbers will not match published leaderboards, and are not meant to.** Where a
published number is quoted for scale, the gap and its cause are stated next to it.
The measured hardware budget is in [`COMPUTE_BUDGET.md`](COMPUTE_BUDGET.md).

## 2. The main question is comparative, not absolute

The repository asks *when does structure beat chemistry, and does combining them help?*
That is a question about **differences between conditions** measured under one protocol
— which is exactly the kind of question that survives reduced scale, because both sides
are handicapped equally.

It is not a claim about the best achievable accuracy on any of these targets. Anyone
wanting that should look at the Matbench leaderboards.

## 3. Splits, and why the obvious one is wrong

Random train/test splits on materials databases **leak**. Near-duplicate polymorphs
and same-composition entries end up on both sides, so the test set is not really unseen
and the reported error is optimistic.

Four split schemes are therefore run and reported side by side: **random**,
**formula-disjoint** (no test formula appears in training), **chemical-system
disjoint** (no test element *combination* appears in training), and
**element-disjoint** (some elements are held out of training altogether).

Measured, not assumed: 42.6% of a random test set shares a formula with something
in training. The random split looks best because it is the easiest, and that gap
is a result rather than an inconvenience. Any repository reporting only a random
split should be read with that in mind.

Matbench's official folds are *not* used. They would make these numbers comparable
to a public leaderboard, which would be a genuine benefit; they are not used
because the element-disjoint split — the one that produced the most useful result
here — is not among them.

## 4. Models are old by design, and that is a limitation

CGCNN (2018), MEGNet (2019) and GATv2 (2022) are here because they are the clearest
architectures to learn from and implement, not because they are current.

The field has moved to equivariant architectures (NequIP, MACE, SevenNet) and
universal interatomic potentials (M3GNet, CHGNet, MatterSim), and by 2025–26 the
Matbench leaderboards are led by foundation-model approaches. Those need GPU compute
this project deliberately does not assume. Nothing here should be read as a statement
about what the best current method can do.

## 5. "Attention" means something specific here

CGCNN and MEGNet have **no attention mechanism** — CGCNN uses a sigmoid gate scoring
each bond independently, MEGNet uses a global state vector. Both are routinely
mislabelled as attention. Only a softmax *across* an atom's neighbours produces
weights that sum to one and can be read as "the model looked here rather than
there"; this is asserted in a test rather than assumed.

Where this repo shows attention maps, they come from the GATv2 model, which actually
has attention. For the others it shows gate activations and integrated-gradient
attributions, and calls them that.

## 6. Interpretability results are not explanations of physics

Attribution methods explain *the model*, not the material. A model can attribute
importance to the chemically "right" properties while having learned the
relationship for the wrong reasons, and the attribution would look identical.

Phase 6 is reported against two controls — an untrained model, and one trained to
convergence on shuffled labels (R² = −0.04, so it genuinely learned nothing).
Rank correlation with the trained profile is +0.08 and +0.34, so neither
reproduces it. That is the strongest statement the design supports: **the ranking
describes this model rather than the attribution method**. It is not evidence
about band-gap physics.

Three specific caveats on the Phase 6 numbers:

- **Raw importance is partly input geometry.** The shuffled control also ranks
  electronegativity first. Only the control-corrected column ("what training
  changed") should be read as being about the model, and the README leads with
  that rather than the raw ranking.
- **The enrichment ratio has an arbitrary floor.** Dividing by a control value
  near zero would explode, so a floor of 0.02 is added to the denominator. That
  caps the largest achievable enrichment and makes the exact multiple — 13.8×
  for electron affinity — sensitive to the floor. The *ordering* is not.
- **One seed, one split, 2,000 crystals.** Attribution was run on the random
  split only, on a sample of the test set, from a single trained model. The
  headline ordering is stable enough to report; specific multiples are not.

My first version of this check also used invented thresholds and called an
ordinary result "borderline" — see §16.

## 7. The activity descriptor amplifies error

Rate depends exponentially on binding energy. A ±0.2 eV adsorption-energy error — an
unremarkable model error — becomes a large multiplicative error in the derived
activity. Quantified in Phase 7. Screening rankings built on these numbers are
trustworthy only to the resolution the propagated error supports.

## 8. Bulk crystals and catalyst surfaces are different problems

Phases 1–6 work on periodic bulk crystals. Phase 7 works on surface slabs with
adsorbates. Transferring between them is a real domain shift, measured explicitly
rather than assumed away.

The Phase 7 data has four problems of its own, all measured on the 3,554 CO rows
downloaded so far and all recorded before any model was fitted:

- **The target column mixes quantities.** Catalysis-Hub's `reactionEnergy` holds
  whatever each equation says: single-atom deposition, molecular chemisorption,
  and multi-species reactions with negative stoichiometric coefficients. Only
  rows matching exactly `A(g) + * -> A*` are used, decided in
  `src/data/adsorption.py`. **41% of pre-filtered rows were rejected** — on CO,
  the adsorbate that looked cleanest.
- **Eleven physically impossible values**, up to +32.0 eV for CO adsorption.
  0.31% of rows, and each would dominate a squared-error loss.
- **23 DFT functionals**, including `RPBE_-0.413VSHE` and `BEEF-vdW_-0.42VSHE` —
  electrochemical energies at an applied potential, which is not the same
  quantity as gas-phase chemisorption. Worse than the Materials Project
  functional mixing in §3.
- **63.4% of rows come from one publication**, 83% from two. One paper means one
  code, one functional, one set of surface conventions, so a random split lets a
  model score by recognising a calculation rather than its chemistry. The
  catalysis analogue of the 42.6% formula leakage, and it argues for a
  publication-disjoint split.

## 8b. Composition cannot solve the catalysis target, and that is measured

Grouping the CO data by what a composition model could know gives a hard ceiling:

| Known to the model | Best possible error | R² ceiling |
|---|---|---|
| Surface composition | 0.92 eV | 0.44 |
| Surface composition + facet | **0.81 eV** | **0.57** |
| Full slab formula + facet | 1.00 eV | 0.34 |

Against a spread of 1.23 eV. The missing 43% is **where on the surface the
molecule sits** — 88% of rows sit in repeated (surface, facet, adsorbate) groups
whose energies span up to 3.25 eV, and the `sites` column records the site only
as an opaque index (`site1` … `site47`) that carries no usable information and
does not transfer between surfaces.

This inverts the band-gap result rather than repeating it. There, structure was
worth 2–3% over composition and the phase would have survived without it. Here
composition **cannot get there at all**, so the geometries are mandatory and a
composition-only Phase 7 would be a demonstration of the ceiling rather than a
model worth having.

## 9. Single-formula illustrations are illustrations

Figure 1's TiO₂ result (44 polymorphs, a 0.43 eV composition-only error floor) is
computed from real data and is real. It is also **one formula**. It motivates the
question; it does not answer it. The answer requires Phase 2's baselines across the
full dataset, and if those disagree with the illustration, the illustration is what
gets corrected.

## 10. The architecture comparison rests on one seed per model

Each of the four architectures in §4 of the README was trained **once**. The
reported uncertainty comes from resampling the 4,308 test materials, which
captures "would this gap survive a different test set" but *not* "would it survive
a different random initialisation". Seed-to-seed variation in a network this size
is typically comparable to the smaller gaps reported.

Concretely, this means:

- The **null result is the robust part.** CGCNN and MPNN differ by 0.001 eV with
  a 95% range of [−0.013, +0.010] eV. Adding seed variance would widen that
  interval, which can only strengthen "indistinguishable" — it cannot turn a null
  into a difference.
- The **ranking of the losers is the fragile part.** GATv2 at 0.439 and MEGNet at
  0.475 are separated by 0.037 eV. That gap would probably survive reseeding, but
  it has not been shown to.

Five seeds per architecture is the correct experiment and costs roughly twelve
hours on this laptop. It is not run, and the results are labelled accordingly
rather than presented as if it had been.

## 11. MEGNet is confounded in both directions, and cannot be cleanly ranked

MEGNet carries **132,673 parameters against MPNN's 60,417** — an advantage — and
is slow enough per epoch that it completed **37 passes through the data against
MPNN's 55** inside the same 35 minutes — a disadvantage. Its validation error was
still falling when the budget expired.

So "MEGNet is the worst of the four" is not an architecture result. It is a result
about MEGNet *under a fixed wall-clock budget on a CPU*, which is a legitimate and
practically relevant question but a different one. A reader who wants "is a global
state a good idea" should treat the MEGNet number as unresolved.

This is the same objection that kept **ALIGNN** out of the comparison entirely —
its line-graph convolution is several times more expensive per epoch, so under a
shared clock it would report an architecture verdict that is really a speed
verdict. MEGNet was included because its overhead is mild enough to be worth the
caveat; ALIGNN's is not.

## 12. Phase 5 gives the network knowledge the earlier comparison withheld

Phase 5 replaces the network's learned per-element vector with tabulated element
properties. It is worth being explicit that this **changes what the comparison
means**, and in a direction that is fairer rather than less fair.

Up to Phase 4 the descriptor baselines had the periodic table and the graph
networks did not. The descriptors read electronegativity and ionic radius out of
a published table; the networks had to infer an equivalent from training data
alone. So "structure beats chemistry by 20%" was measured with the structural
model handicapped, and "the network collapses on unseen elements" was partly a
statement about that handicap.

What Phase 5 does **not** do is leak anything. The element table is generated from
pymatgen and committed to the repository; it contains no target values, no
information about any specific material, and nothing derived from the training
labels. An element's electronegativity is the same number whether or not the
dataset contains a single compound of it.

What it **does** do is make the graph networks and the descriptor baselines share
an input that only the baselines had before. A reader comparing the Phase 3 and
Phase 5 numbers should read the difference as "what the periodic table is worth
to a graph network", not as a like-for-like architecture improvement.

## 13. In the element split, validation and test hold out DIFFERENT elements

The element-disjoint split holds out ten elements — Ac, Ga, He, Ho, Kr, N, Nd,
Pt, Si, Sm — and divides them between validation and test so that neither shares
chemistry with the other. That is the right design for the question being asked,
but it has a consequence worth stating.

**Validation error is not an unbiased estimate of test error here.** They are
measured on different chemistry. In the Phase 5 runs, validation MAE sat around
0.93–1.12 eV while test MAE came out at 0.66–0.84 eV for the same models. The
val/test gap is a property of which elements landed where, not of the models.

Two things follow:

- Comparisons **between variants** remain valid, because every variant uses the
  identical validation and test sets. All the reported differences are of that
  kind.
- **Early stopping is noisier than usual.** "Best epoch" is chosen on one set of
  unfamiliar elements and reported on another. `cgcnn_both_comp` had its best
  validation score at epoch 0 and stopped after 16 epochs on that basis; a
  different val/test division of the same ten elements could plausibly have
  stopped it elsewhere.

Not a bug in the split — holding elements out is the entire point — but it means
the Phase 5 numbers carry more run-to-run variance than the random-split numbers
do, on top of the single-seed caveat in §10.

## 14. A stated mechanism was refuted by its own diagnostic

The README first explained Phase 5's `both` result as shortcut learning: given a
free per-element code and real element properties, the model would lean on the
code because it fits training data more easily, leaving the chemistry pathway
underdeveloped.

`scripts/diagnose_fusion.py` measured it and found the reverse — the tabulated
properties carry **62%** of the between-element signal, the learned codes 38%. The
model did **not** prefer the shortcut, and the explanation was wrong.

The corrected mechanism — that a pathway does not need to dominate to do damage,
because for a held-out element the learned row was never trained — was then tested
by ablation: neutralise only those rows in the already-trained model and re-score.
It recovers **53%** of the penalty (0.836 → 0.744 eV against 0.663 eV for
`properties`).

So the third explanation is **confirmed but partial**. Roughly half the cost of
keeping the learned code is untrained rows; the other 0.081 eV is not accounted
for. The obvious untested suspect is capacity — `both` carries 14,720 more
parameters than `properties` — and a size-matched control has not been run. The
README says half, not all.

Two process notes, both uncomfortable and both worth keeping:

- The *first* diagnostic was also badly designed. It tested whether held-out
  embedding rows were "still at their random initialisation" by comparing row
  **lengths** — 8.06 against 8.02, no difference. But a vector rotates without
  changing length, and a 64-dimensional unit-normal row starts near √64 = 8
  regardless. The statistic could not have detected the effect it was looking
  for. Replaced with a test of whether the table encodes chemistry.
- Both the refuted mechanism and the useless test are recorded in the script's
  docstring rather than deleted, because a diagnostic that has only ever agreed
  with the hypothesis it was written for is not evidence of anything.

## 15. Only the winning Phase 5 variant was run on all four splits

`properties` has now been run on all four splits. The other two fusion variants —
`both` and `both_comp` — were run on the **element-disjoint split only**, because
that is the split their comparison is about and each additional run costs 35
minutes.

So the four-split table in README §5 is complete for the three approaches it
shows, and the variant comparison beneath it is a single-split result. It would be
reasonable to expect `both` to trail `properties` on the easier splits too, by a
smaller margin; that has not been measured and the figure does not imply it.

## 16. A sanity check with a guessed threshold is not a sanity check

The Phase 6 control originally compared importance profiles by cosine similarity
with hand-picked thresholds: above 0.9 suspicious, below 0.7 fine. It reported
the shuffled-label control at **0.785 — "borderline"**, which reads as a
half-failed sanity check.

It was nothing of the sort. Importance profiles are **non-negative by
construction**, and two *unrelated* non-negative vectors of length 31 score a
cosine of **0.753** on average (0.654–0.840 for 90% of draws). Both observed
values sit inside that band. The thresholds had been chosen on the intuition that
unrelated vectors score near zero — true for signed vectors, false here — so the
check was mis-calibrated in the direction that manufactures an alarming number.

Fixed by computing the null by sampling rather than assuming it, and by reporting
Spearman rank correlation alongside, whose null is zero regardless of sign.

Recorded because the failure mode is general and easy to repeat: a threshold that
feels right, on a statistic whose null distribution was never checked, in a test
whose entire purpose is to be harder to pass than the thing it guards.

## 17. The catalysis geometry set is 400 rows from one paper with one functional

Geometry does not come back in pages. The GraphQL endpoint returns **one row per
request** when `InputFile` is asked for, against a published budget of 500 requests
a day with automatic suspension. All 3,554 clean CO rows would be eight days of
budget. The bulk route was tried and is closed: CatHub's documented public Postgres
credentials are rejected (`password authentication failed for user "apiuser"`).

So the geometry set is **400 rows — 11% of the clean CO rows**, and three things
follow that limit what any result on it can claim.

**PBE only.** The full table mixes 23 DFT functionals, including potential-referenced
electrochemical ones like `BEEF-vdW_-0.73VSHE`. Those are not the same quantity as a
gas-phase PBE binding energy, and pooling them puts a systematic offset into the
labels that no architecture can remove. Restricting to PBE removes the confound by
construction. The cost is that nothing here transfers to other functionals without
being re-checked.

**One publication.** The PBE subset is 2,243 of 2,252 rows from `YohannesCombined2023`,
so the sample is single-source. This **rules out the publication-disjoint split** that
§8b argued for — the catalysis analogue of the element split that mattered most in
Phase 5. The available generalisation test is surface-disjoint: hold out whole
(surface, facet) groups. That is a real test, and it is a weaker one, because two
studies disagreeing about the same surface is a failure mode this sample cannot see.

**40 surfaces × 10 sites, chosen rather than sampled.** A uniform random 400 would
have given roughly one site per surface, and site variation is exactly the 43% of the
variance that composition cannot reach — the sample would have contained none of the
signal it was bought for. Taking the 11 largest groups instead gives plenty of site
variation and a held-out set of two or three surfaces, which cannot support a
generalisation claim. 40×10 is the compromise, with groups drawn from across the
size range rather than the top so the set is not just the most-studied surfaces.
The selection is in `src/data/geometry_sample.py`, deterministic and tested.

Measured on the real table: median within-surface energy spread **1.45 eV**, against
a composition-only floor of 0.806 eV. The signal is present. Whether the model finds
it is the experiment.

## 18. Single author, no independent replication

Everything here was built and checked by one person. The mitigations — assertions
that fail loudly, figures regenerated from source data rather than hand-edited, CI on
every commit — reduce the chance of a silent error but do not replace a second pair
of eyes.

If you find a mistake, please open an issue. Errors found after publication get fixed
in place *and* recorded, in the same spirit as
[`MPExplorer`](https://github.com/teja2792/MPExplorer), which documents a citation
error the author made and later caught.
