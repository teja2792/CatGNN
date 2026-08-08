# CGCNN, derived

What the convolution actually computes, why each piece is there, and what it can
and cannot represent. Written because "I ran CGCNN" and "I understand CGCNN" are
different claims, and only the second one survives a follow-up question.

Reference: T. Xie and J. C. Grossman, *Crystal Graph Convolutional Neural Networks
for an Accurate and Interpretable Prediction of Material Properties*,
Phys. Rev. Lett. **120**, 145301 (2018).

---

## 1. The object

A crystal becomes a graph `G = (V, E)`:

- **Node** `i` — one atom, carrying a feature vector `v_i ∈ ℝ^d`.
  Initialised by embedding the atomic number: chemistry enters here and nowhere else.
- **Edge** `(i, j)` — a neighbour contact within 8 Å, carrying `u_(i,j) ∈ ℝ^41`,
  the Gaussian expansion of the bond length.

Two atoms can be joined by **more than one edge**, through different periodic
images. That is not a quirk of the data structure; it is the physics. Rutile's
cell holds 4 oxygens and each Ti is 6-coordinate, so two of those bonds reach
into the neighbouring repeat of the crystal.

## 2. The convolution

For each edge, concatenate the two endpoints and the bond:

```
z_(i,j) = [ v_i ‖ v_j ‖ u_(i,j) ]           ∈ ℝ^(2d + 41)
```

Project once to twice the node width and split the result in half:

```
[ f_(i,j) ‖ c_(i,j) ] = z_(i,j) W + b        W ∈ ℝ^((2d+41) × 2d)
```

Then the update, which is the part worth staring at:

```
v_i ← softplus(  v_i  +  Σ_{j ∈ N(i)}  σ(f_(i,j))  ⊙  softplus(c_(i,j))  )
                 └─┬─┘  └──────────────────┬──────────────────────────┘
              residual              gated sum over neighbours
```

with `σ` the logistic sigmoid, `⊙` elementwise multiplication, and the sum
running over every edge incident to `i`.

### Why each piece

**`σ(f)` is a gate, not attention.** It is a per-edge, per-channel number in
`(0, 1)` that decides how much of this neighbour's message to let through. There
is no softmax over neighbours, so the gates do not compete: every edge can be
fully open at once. Attention normalises across neighbours and therefore forces a
budget; a gate does not. This distinction gets blurred constantly in write-ups of
CGCNN, and it matters for interpretation — gate values are *not* a distribution
over neighbours and should never be plotted as if they were.

**`softplus(c)` is the message content.** Smooth and non-negative. Non-negativity
matters: combined with the residual `v_i +`, node features can only grow through
the network, which keeps the aggregation from cancelling itself out when an atom
has many neighbours.

**The sum, not the mean.** Coordination number is information. An atom with 12
neighbours *should* receive more signal than one with 4, and a mean would throw
that away. The consequence is that the model must handle wildly different node
degrees, which is why the 12-neighbour cap in graph construction exists.

**The residual `v_i +`.** Three layers of this and gradients still reach the
embedding. Without it, deep stacks of message passing wash out into a constant —
the over-smoothing problem.

### Batch normalisation

The reference implementation applies BatchNorm twice: once to `[f ‖ c]` before
splitting, once to the summed neighbour term before the residual add. This repo
follows that, with one deliberate deviation: **BN is disabled in the reference
implementation used for testing**, because it makes the layer's output depend on
the rest of the batch, and a unit test of a graph convolution should not.

## 3. Readout

Message passing produces per-atom vectors. A crystal-level property needs one
vector per crystal:

```
v_crystal = (1/N) Σ_i v_i          →   MLP   →   scalar
```

**Mean, not sum, here** — the opposite choice from the aggregation above, and for
a reason. Formation energy per atom and band gap are *intensive*: doubling the
unit cell must not change them. A sum readout would double the prediction; a mean
makes supercell invariance automatic. If this repo ever predicts an extensive
quantity (total energy rather than energy per atom), this line has to change,
and `tests/test_cgcnn.py::test_supercell_invariance` is what would catch it.

## 4. What this architecture cannot represent

Worth knowing before attributing a failure to training rather than to the model:

- **Bond angles.** Edges carry a distance and nothing else. Two structures with
  identical bond lengths and different angles produce identical graphs and are
  therefore indistinguishable. This is the specific gap ALIGNN closes by running
  a second convolution over the line graph, and it is why ALIGNN is in the
  comparison at all.
- **Absolute orientation and position.** By construction, and correctly so.
- **Long-range electrostatics** beyond the 8 Å cutoff.
- **Anything beyond 3 hops** in a 3-layer network. An atom's representation is a
  function of its 3-hop neighbourhood; nothing further can influence it.

## 5. Parameter count

With `d = 64`, 3 convolutions, and 41 edge features:

| Component | Parameters |
|---|---|
| Atom embedding (100 elements → 64) | 6,400 |
| Convolution × 3, each `(2·64+41) × 128 + 128` | 3 × 21,760 = 65,280 |
| BatchNorm × 6 | 1,152 |
| Readout MLP (64 → 128 → 1) | 8,449 |
| **Total** | **≈ 81,000** |

Small by modern standards, and deliberately so: this has to train on a laptop
CPU under a fixed wall-clock budget, and the comparison in Phase 4 is only fair
if every architecture is similarly sized.

## 6. Implementation note

The layer is implemented **twice** in this repository:

- `src/models/cgcnn_reference.py` — NumPy, forward pass only, written directly
  from the equations above.
- `src/models/cgcnn.py` — PyTorch, what actually trains.

`tests/test_cgcnn.py` asserts the two agree to floating-point tolerance on random
inputs. Two independent implementations of the same equations disagreeing is a
much louder signal than one implementation quietly being wrong, and it is the
only practical way to check a hand-written layer against something other than
itself.
