"""Measure what this laptop can actually do, before deciding how big to go.

This repository is trained on one CPU. Rather than quoting somebody else's GPU
timings and hoping, we measure the two operations that dominate the run time --
building periodic neighbour lists, and one pass of gathering/scattering messages
over a batch of graphs -- and extrapolate to a full training run.

The output is written to COMPUTE_BUDGET.md and is the basis for every dataset-size
decision in Phase 1. Re-run it any time the machine or the environment changes.

Usage
-----
    python scripts/benchmark_hardware.py                # quick, ~1-2 minutes
    python scripts/benchmark_hardware.py --n 2000       # more structures, tighter estimate
    python scripts/benchmark_hardware.py --no-write     # print only, leave the doc alone

Deliberately depends on numpy only. It has to be runnable before torch or
pymatgen are installed, because its whole purpose is to tell you what to install
them *for*.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "COMPUTE_BUDGET.md"
JSON_OUT = REPO / "results" / "hardware_benchmark.json"

# Reference workload sizes, matching the CGCNN convention used later in the repo.
CUTOFF = 8.0          # angstrom
MAX_NEIGHBOURS = 12
HIDDEN_DIM = 64
N_CONV_LAYERS = 3
BATCH_SIZE = 64
EPOCHS_ASSUMED = 100


# ---------------------------------------------------------------------------
# Machine description
# ---------------------------------------------------------------------------

def cpu_name() -> str:
    """Best-effort human-readable CPU name across Windows, Linux and macOS."""
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10
            ).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
        elif sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def physical_cores() -> int | None:
    """Physical (not logical) core count, if we can work it out without psutil.

    This matters: PyTorch on CPU is usually *slower* when you give it more threads
    than physical cores, because the threads fight over the same cache and memory
    bandwidth. The recommendation printed at the end depends on this number.
    """
    try:
        import psutil  # optional
        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            txt = Path("/proc/cpuinfo").read_text()
            ids = {
                tuple(
                    l.split(":", 1)[1].strip()
                    for l in block.splitlines()
                    if l.startswith(("physical id", "core id"))
                )
                for block in txt.split("\n\n") if block.strip()
            }
            ids.discard(())
            if ids:
                return len(ids)
    except Exception:
        pass
    return None


def machine_info() -> dict:
    return {
        "cpu": cpu_name(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ---------------------------------------------------------------------------
# Synthetic crystals
#
# Random structures with realistic atom counts and densities. We are timing the
# *machine*, not the chemistry, so real structures are unnecessary here -- and
# requiring a Materials Project API key just to benchmark a laptop would be silly.
# ---------------------------------------------------------------------------

def random_structures(n: int, rng: np.random.Generator, sites=(4, 30)):
    out = []
    for _ in range(n):
        n_sites = int(rng.integers(sites[0], sites[1] + 1))
        a = float(rng.uniform(3.5, 9.0))
        lattice = np.diag([a, a * rng.uniform(0.9, 1.1), a * rng.uniform(0.7, 1.3)])
        frac = rng.random((n_sites, 3))
        out.append((frac, lattice))
    return out


def build_graph(frac, lattice, cutoff=CUTOFF, max_nbr=MAX_NEIGHBOURS):
    """Periodic neighbour list -- the expensive part of turning a crystal into a graph.

    Same approach used in Phase 1: search the 3x3x3 block of neighbouring cell
    images and keep every contact inside the cutoff, then truncate to the
    ``max_nbr`` closest. Counting all images, not just the nearest one, is what
    keeps coordination numbers correct in dense structures.
    """
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
    )
    cart_shifts = shifts @ lattice
    cart = frac @ lattice
    n = len(cart)

    edges = 0
    for i in range(n):
        d = cart[None, :, :] + cart_shifts[:, None, :] - cart[i]
        dist = np.linalg.norm(d, axis=2).ravel()
        dist = dist[dist > 1e-8]
        near = np.sort(dist[dist <= cutoff])[:max_nbr]
        edges += len(near)
    return n, edges


def time_graph_building(structs) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    nodes = edges = 0
    for frac, lattice in structs:
        n, e = build_graph(frac, lattice)
        nodes += n
        edges += e
    return time.perf_counter() - t0, nodes, edges


def time_message_passing(nodes: int, edges: int, repeats: int = 20) -> float:
    """Time the gather/scatter that dominates a graph convolution.

    A message-passing layer is, at heart: read a feature vector for each edge's two
    endpoints, combine them, then sum the results back onto the nodes. That
    irregular memory access -- not the matrix multiply -- is what makes GNNs slow
    on CPU, so it is the right thing to measure.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((nodes, HIDDEN_DIM)).astype(np.float32)
    src = rng.integers(0, nodes, edges)
    dst = rng.integers(0, nodes, edges)
    w = rng.standard_normal((2 * HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(repeats):
        for _layer in range(N_CONV_LAYERS):
            msg = np.concatenate([x[src], x[dst]], axis=1) @ w
            msg = np.maximum(msg, 0.0)
            agg = np.zeros_like(x)
            np.add.at(agg, dst, msg)
            x = np.tanh(x + agg)
    return (time.perf_counter() - t0) / repeats


# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def run(n_structures: int) -> dict:
    rng = np.random.default_rng(42)
    print(f"Benchmarking on {n_structures} synthetic structures "
          f"(cutoff {CUTOFF} A, max {MAX_NEIGHBOURS} neighbours)...\n")

    structs = random_structures(n_structures, rng)

    print("  [1/2] periodic neighbour lists ...", end="", flush=True)
    graph_time, nodes, edges = time_graph_building(structs)
    print(f" {graph_time:.2f} s")

    print("  [2/2] message passing (forward pass) ...", end="", flush=True)
    per_batch_nodes = int(nodes / n_structures * BATCH_SIZE)
    per_batch_edges = int(edges / n_structures * BATCH_SIZE)
    mp_time = time_message_passing(per_batch_nodes, per_batch_edges)
    print(f" {mp_time * 1000:.1f} ms per batch of {BATCH_SIZE}\n")

    graph_per_struct = graph_time / n_structures
    # Backward pass costs roughly 2x the forward pass; the 1.35 covers optimiser
    # step, data loading and Python overhead. Deliberately conservative -- better
    # to be pleasantly surprised than to plan a run that never finishes.
    train_step = mp_time * 3.0 * 1.35

    info = machine_info()
    est = {}
    for size in (5_000, 10_000, 30_000, 50_000):
        steps_per_epoch = max(1, size // BATCH_SIZE)
        est[size] = {
            "graph_build_once_s": graph_per_struct * size,
            "epoch_s": train_step * steps_per_epoch,
            "train_100ep_s": train_step * steps_per_epoch * EPOCHS_ASSUMED,
        }

    return {
        "machine": info,
        "settings": {
            "cutoff_angstrom": CUTOFF,
            "max_neighbours": MAX_NEIGHBOURS,
            "hidden_dim": HIDDEN_DIM,
            "conv_layers": N_CONV_LAYERS,
            "batch_size": BATCH_SIZE,
            "epochs_assumed": EPOCHS_ASSUMED,
            "n_structures_benchmarked": n_structures,
        },
        "measured": {
            "graph_build_s_per_structure": graph_per_struct,
            "mean_nodes_per_structure": nodes / n_structures,
            "mean_edges_per_structure": edges / n_structures,
            "forward_s_per_batch": mp_time,
            "est_train_step_s": train_step,
        },
        "estimates": est,
    }


def render(res: dict) -> str:
    m, s, meas, est = res["machine"], res["settings"], res["measured"], res["estimates"]
    cores = m["physical_cores"] or "unknown"

    rows = "\n".join(
        f"| {size:,} | {fmt_duration(v['graph_build_once_s'])} | "
        f"{fmt_duration(v['epoch_s'])} | {fmt_duration(v['train_100ep_s'])} |"
        for size, v in est.items()
    )

    # A container or VM often exposes only 1-2 cores. Numbers measured there are
    # pessimistic by several times and should not be presented as the laptop's.
    provisional = ""
    if isinstance(m["physical_cores"], int) and m["physical_cores"] <= 2:
        provisional = (
            "\n> ⚠️ **These numbers are provisional.** They were measured on a machine "
            f"reporting only {m['physical_cores']} physical core(s), which usually means a "
            "container or VM rather than the real laptop. Re-run "
            "`python scripts/benchmark_hardware.py` on the target machine before using this "
            "table to size a dataset — expect it to be several times faster.\n"
        )

    return f"""# Compute budget

**Auto-generated by `scripts/benchmark_hardware.py`. Do not edit by hand — re-run the script.**

Last measured: {m['timestamp_utc']}
{provisional}

## The machine everything in this repo was trained on

| | |
|---|---|
| CPU | {m['cpu']} |
| Physical cores | {cores} |
| Logical cores | {m['logical_cores']} |
| GPU | none — this repository is CPU-only by design |
| OS | {m['platform']} |
| Python | {m['python']} |

## What was measured

Two operations dominate the run time of a crystal graph neural network, and both
were timed on {s['n_structures_benchmarked']:,} structures with realistic atom counts:

1. **Building periodic neighbour lists** — turning a crystal into a graph.
   Done once and cached, so it is a fixed setup cost, not a per-epoch cost.
2. **Message passing** — the gather/scatter that moves information between
   neighbouring atoms. This is the per-step cost, and on a CPU it is bound by
   irregular memory access rather than arithmetic.

| Measurement | Value |
|---|---|
| Graph construction | {meas['graph_build_s_per_structure'] * 1000:.1f} ms per structure |
| Mean atoms per structure | {meas['mean_nodes_per_structure']:.1f} |
| Mean edges per structure | {meas['mean_edges_per_structure']:.1f} |
| Forward pass, batch of {s['batch_size']} | {meas['forward_s_per_batch'] * 1000:.1f} ms |
| Estimated full training step | {meas['est_train_step_s'] * 1000:.1f} ms |

Model shape assumed: hidden dimension {s['hidden_dim']}, {s['conv_layers']} convolution
layers, {s['cutoff_angstrom']} Å cutoff, at most {s['max_neighbours']} neighbours per atom —
the CGCNN convention.

## What that means for dataset size

| Structures | Graph build (once) | One epoch | {s['epochs_assumed']} epochs |
|---|---|---|---|
{rows}

**How to read this table.** The last column is one model, one target, one split, one
seed. A fair comparison needs several of those, so multiply accordingly before
choosing a dataset size. The estimates are deliberately conservative and assume a
backward pass costing about twice the forward pass.

They are also a *floor*: ALIGNN's line-graph convolution costs roughly 3–8× CGCNN,
and is the reason ALIGNN is run on a subset at reduced depth rather than at full size.

## Settings this implies

- `torch.set_num_threads({cores if isinstance(cores, int) else 6})` — physical cores, not
  logical. More threads than physical cores usually makes CPU training *slower*.
- Cache built graphs to `data/cache/`. Rebuilding them inside the training loop is
  the most common way CPU training becomes unusable.
- Filter to `n_sites <= 30`. A handful of large cells will otherwise dominate every
  epoch for no scientific gain.
- Fix an identical wall-clock budget per model and report it, so that architecture
  comparisons are matched rather than accidental.

## Honesty note

Every result in this repository is produced under this budget. They are therefore
**reduced-scale** and will not match published leaderboard numbers trained on GPUs
with the full dataset. Where a comparison to a published number is made, the gap and
the reason for it are stated alongside it. See `LIMITATIONS.md`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="number of structures to benchmark")
    ap.add_argument("--no-write", action="store_true", help="print only; do not update COMPUTE_BUDGET.md")
    args = ap.parse_args()

    res = run(args.n)
    m, est = res["machine"], res["estimates"]

    print("=" * 66)
    print(f"  {m['cpu']}")
    print(f"  {m['physical_cores'] or '?'} physical cores / {m['logical_cores']} logical")
    print("=" * 66)
    for size, v in est.items():
        print(f"  {size:>6,} structures | graphs {fmt_duration(v['graph_build_once_s']):>7}"
              f" | epoch {fmt_duration(v['epoch_s']):>7}"
              f" | 100 epochs {fmt_duration(v['train_100ep_s']):>7}")
    print("=" * 66)

    if not args.no_write:
        DOC.write_text(render(res), encoding="utf-8")
        JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {DOC.relative_to(REPO)} and {JSON_OUT.relative_to(REPO)}")
    else:
        print("\n(--no-write: nothing written)")


if __name__ == "__main__":
    main()
