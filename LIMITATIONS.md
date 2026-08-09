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
importance to the chemically "right" atoms while having learned the relationship for
the wrong reasons.

Phase 6 therefore includes a randomisation control: retrain on shuffled labels and
re-run the attributions. If they still look plausible, they were never explaining
anything. Interpretability results are reported alongside that control, and any case
where methods disagreed is reported rather than quietly dropped.

## 7. The activity descriptor amplifies error

Rate depends exponentially on binding energy. A ±0.2 eV adsorption-energy error — an
unremarkable model error — becomes a large multiplicative error in the derived
activity. Quantified in Phase 7. Screening rankings built on these numbers are
trustworthy only to the resolution the propagated error supports.

## 8. Bulk crystals and catalyst surfaces are different problems

Phases 1–6 work on periodic bulk crystals. Phase 7 works on surface slabs with
adsorbates. Transferring between them is a real domain shift, measured explicitly
rather than assumed away.

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

## 13. Single author, no independent replication

Everything here was built and checked by one person. The mitigations — assertions
that fail loudly, figures regenerated from source data rather than hand-edited, CI on
every commit — reduce the chance of a silent error but do not replace a second pair
of eyes.

If you find a mistake, please open an issue. Errors found after publication get fixed
in place *and* recorded, in the same spirit as
[`MPExplorer`](https://github.com/teja2792/MPExplorer), which documents a citation
error the author made and later caught.
