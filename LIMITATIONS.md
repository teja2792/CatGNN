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
- **ALIGNN is handicapped.** Its line-graph convolution costs roughly 3–8× CGCNN per
  epoch, so under an equal-time budget it sees far fewer epochs. Any ALIGNN result
  here is a lower bound on what ALIGNN can do.

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

Four split schemes are therefore run and reported side by side: random,
composition-disjoint, structure-similarity-disjoint, and Matbench's official folds.
Expect the random split to look best. That gap is a result, not an inconvenience —
and any repo reporting only a random split should be read with that in mind.

## 4. Models are old by design, and that is a limitation

CGCNN (2018), MEGNet (2019) and ALIGNN (2021) are here because they are the clearest
architectures to learn from and implement, not because they are current.

The field has moved to equivariant architectures (NequIP, MACE, SevenNet) and
universal interatomic potentials (M3GNet, CHGNet, MatterSim), and by 2025–26 the
Matbench leaderboards are led by foundation-model approaches. Those need GPU compute
this project deliberately does not assume. Nothing here should be read as a statement
about what the best current method can do.

## 5. "Attention" means something specific here

CGCNN, MEGNet and ALIGNN have **no attention mechanism** — CGCNN uses a sigmoid gate,
ALIGNN uses edge gating, MEGNet uses a global state vector. These are routinely
mislabelled as attention.

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

## 10. Single author, no independent replication

Everything here was built and checked by one person. The mitigations — assertions
that fail loudly, figures regenerated from source data rather than hand-edited, CI on
every commit — reduce the chance of a silent error but do not replace a second pair
of eyes.

If you find a mistake, please open an issue. Errors found after publication get fixed
in place *and* recorded, in the same spirit as
[`MPExplorer`](https://github.com/teja2792/MPExplorer), which documents a citation
error the author made and later caught.
