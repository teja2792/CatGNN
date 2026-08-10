"""Choosing WHICH 400 rows to fetch geometries for.

Geometry costs one request per row. The budget is 450 a day. So the sample is not
a sampling detail to be handled with `random.sample` — it is the experiment, and
choosing it badly makes the phase unable to answer its own question no matter how
good the model is.

WHAT THE SAMPLE HAS TO SUPPORT
------------------------------
The inspection measured a ceiling: a model that knows only the surface and facet
explains 57% of the variance in CO binding and gets stuck at 0.806 eV. The other
43% is WHERE on the surface the molecule sits. That is the whole reason to fetch
geometry, so the sample must contain the thing being tested:

    many sites on the SAME surface   -> the model can be asked to tell them apart
    many DIFFERENT surfaces          -> the model can be asked to generalise

These pull in opposite directions on a fixed budget, and the two obvious samples
both fail:

    400 surfaces x 1 site    no site variation at all. The graph model is handed
                             exactly the information composition already had, and
                             the ceiling cannot be beaten even in principle.
    11 surfaces x 38 sites   (the largest groups) plenty of site variation, but a
                             held-out set of 2-3 surfaces, which cannot support a
                             claim about generalisation.

So: 40 surfaces x 10 sites. Enough surfaces that ~8 can be held out entirely, and
enough sites on each that within-surface variation is present in training and in
the test. Measured on the real table, the median within-surface energy spread of
such a sample is 1.65 eV — the signal is there to be found.

ONE FUNCTIONAL, NOT TWENTY-THREE
--------------------------------
The full CO table mixes 23 DFT functionals, including potential-referenced
electrochemical ones (`BEEF-vdW_-0.73VSHE`). Those are not the same quantity as a
gas-phase PBE binding energy, and averaging them into one target column would put
a systematic offset into the labels that no architecture can undo.

Restricting to PBE removes that confound by construction and costs little: 2,252
of the 3,543 clean CO rows are PBE. The price is stated rather than hidden — the
PBE subset is 2,243 rows from one publication, so this sample cannot support a
publication-disjoint split. It buys a clean target at the cost of an external
validity claim, and that trade is recorded in LIMITATIONS.

WHY GROUPS ARE SPREAD ACROSS THE SIZE RANGE
-------------------------------------------
Taking the 40 largest groups would sample the most-studied surfaces, which are
studied because they are interesting — an availability bias straight into the
training set. `numpy.linspace` over the size-sorted list takes groups from across
the range instead, so the sample spans ordinary surfaces as well as popular ones.

Everything here is deterministic. The same table gives the same 400 rows, which
is what makes a partially-completed download resumable without a stored plan.
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict

# Values outside this window are not adsorption energies. The inspection found 11
# rows up to +32.02 eV for CO, which is roughly three times the C=O bond strength
# and therefore a failed calculation or a mis-parsed equation, not a weak bond.
ENERGY_MIN, ENERGY_MAX = -12.0, 6.0

# One functional, so the target column means one thing. See the module docstring.
FUNCTIONAL = "PBE"


def decode_reaction_id(node_id: str) -> int | None:
    """GraphQL exposes ids as base64 of "Reaction:<int>"; the filter wants <int>.

    `reactions(id: ...)` takes the integer, but every id the API hands back is the
    opaque global form. Without this the fetcher cannot ask for a specific row,
    which is the only way geometry comes back one row at a time.

    Returns None rather than raising: an id that does not decode is one row to
    skip, not a reason to abandon a download that has already been paid for.
    """
    if not isinstance(node_id, str):
        return None
    try:
        raw = base64.b64decode(node_id, validate=True).decode("ascii")
    except Exception:                       # noqa: BLE001 - any malformed id
        return None
    prefix, _, number = raw.partition(":")
    if prefix != "Reaction" or not number.isdigit():
        return None
    return int(number)


def group_key(row: dict) -> tuple:
    """A surface is a (composition, facet) pair. Sites vary within one."""
    return (row.get("surfaceComposition"), row.get("facet"))


def eligible_rows(rows, adsorbate: str = "CO", functional: str = FUNCTIONAL):
    """Rows that could be in the sample at all, before any group logic.

    Three filters, each for a stated reason:
      - the adsorbate, because binding energies of different species are
        different quantities and the model is being asked about one;
      - the energy window, because impossible values are failed calculations;
      - the functional, because mixing them puts an offset into the labels.
    """
    out = []
    for r in rows:
        if r.get("adsorbate") != adsorbate:
            continue
        e = r.get("reactionEnergy")
        if not isinstance(e, (int, float)) or not ENERGY_MIN <= e <= ENERGY_MAX:
            continue
        if functional is not None and r.get("dftFunctional") != functional:
            continue
        out.append(r)
    return out


def select(rows, n_surfaces: int = 40, sites_per_surface: int = 10,
           adsorbate: str = "CO", functional: str = FUNCTIONAL) -> list[dict]:
    """The sample. Deterministic, so a resumed download needs no stored plan.

    Groups are size-sorted, then picked at evenly spaced ranks rather than from
    the top, so the sample is not just the most-studied surfaces. Within a group
    rows are ordered by id, which is arbitrary with respect to energy and
    therefore does not select for strong or weak binding.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in eligible_rows(rows, adsorbate, functional):
        groups[group_key(r)].append(r)

    big = [k for k, v in groups.items() if len(v) >= sites_per_surface]
    # Sort by size then key: ties in size must not depend on dict ordering, or
    # two runs of the same code could disagree about which rows are in the study.
    big.sort(key=lambda k: (-len(groups[k]), str(k)))
    if not big:
        return []

    if n_surfaces >= len(big):
        picked = big
    else:
        step = (len(big) - 1) / (n_surfaces - 1) if n_surfaces > 1 else 0
        picked = [big[round(i * step)] for i in range(n_surfaces)]

    out = []
    for k in picked:
        out.extend(sorted(groups[k], key=lambda r: str(r.get("id")))[:sites_per_surface])
    return out


def describe(sample) -> dict:
    """What the sample is, for the manifest and for the limitations section.

    Reported whether or not it is flattering. The publication and functional
    counts here are the ones that decide which splits are possible, so they
    belong in the record rather than in a memory of the run.
    """
    from collections import Counter

    groups: dict[tuple, list] = defaultdict(list)
    for r in sample:
        groups[group_key(r)].append(r)

    spreads = []
    for v in groups.values():
        e = [x["reactionEnergy"] for x in v]
        spreads.append(max(e) - min(e))
    spreads.sort()

    def median(xs):
        if not xs:
            return 0.0
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    return {
        "rows": len(sample),
        "surfaces": len(groups),
        "sites_per_surface_min": min((len(v) for v in groups.values()), default=0),
        "median_within_surface_spread_eV": round(median(spreads), 3),
        "publications": dict(Counter(r.get("pubId") for r in sample)),
        "functionals": dict(Counter(r.get("dftFunctional") for r in sample)),
        "adsorbates": dict(Counter(r.get("adsorbate") for r in sample)),
    }


def load_rows(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
