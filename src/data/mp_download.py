"""Bulk download of Materials Project crystals and properties.

Differs from the per-formula lookup in the sibling repo
[`MPExplorer`](https://github.com/teja2792/MPExplorer): this pulls tens of
thousands of structures across all chemistries, and it keeps the **atomic
positions**, because a graph neural network needs the crystal itself and not
just a table of properties.

Design notes, all driven by the fact that this runs on a laptop over a home
connection:

* **Resumable.** Progress is written in chunks. Re-running after an interruption
  skips what already landed. A 30k-structure download that has to start over
  because the wifi dropped is a download that never finishes.
* **Compact storage.** Full pymatgen structure dictionaries carry a lot we do not
  need. We keep the lattice matrix, fractional coordinates and species -- enough
  to rebuild the graph exactly -- which is roughly a third of the size.
* **Every pull writes a manifest.** Query, field list, filters, date, row count,
  checksum, and a fingerprint of the API key. Without that, a downloaded dataset
  is an orphan a year later.
* **The DFT functional is downloaded, not assumed.** GGA, GGA+U and r2SCAN band
  gaps are different quantities. Anything that averages across them is wrong, and
  you cannot group by a column you never fetched.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import (
    RAW,
    MAX_SITES,
    MIN_SITES,
    get_mp_api_key,
    key_fingerprint,
)

DEST = RAW / "materials_project"

# Summary fields. Anything not listed here is not downloaded, so additions mean
# a re-download -- think before trimming.
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "nsites",
    "nelements",
    "volume",
    "density",
    "symmetry",
    "structure",                    # the crystal itself -- the point of this repo
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "is_metal",
    "theoretical",                  # True = never experimentally observed
    "deprecated",
]

CHUNK_SIZE = 2000  # materials per output file


# ---------------------------------------------------------------------------
# Structure compaction
# ---------------------------------------------------------------------------

def compact_structure(structure: Any) -> dict | None:
    """Reduce a pymatgen Structure to what graph construction actually needs.

    Returns lattice (3x3 Cartesian matrix, angstrom), fractional coordinates, and
    the element symbol at each site. Disordered sites -- where one crystallographic
    position holds a statistical mixture of elements -- are returned as None and
    skipped upstream, because "70% Fe, 30% Co on this site" is not a thing a graph
    node can represent without a modelling decision this repo has not made.
    """
    try:
        lattice = structure.lattice.matrix.tolist()
        frac, species = [], []
        for site in structure:
            if not site.is_ordered:
                return None
            frac.append([float(c) for c in site.frac_coords])
            species.append(str(site.specie.symbol))
        return {"lattice": lattice, "frac_coords": frac, "species": species}
    except Exception:
        return None


def flatten_doc(doc: Any) -> dict | None:
    """One Materials Project document -> one flat JSON-serialisable row."""

    def g(name, default=None):
        v = getattr(doc, name, default)
        return default if v is None else v

    struct = compact_structure(g("structure"))
    if struct is None:
        return None  # disordered or unreadable

    sym = g("symmetry")
    row = {
        "material_id": str(g("material_id")),
        "formula_pretty": g("formula_pretty"),
        "chemsys": g("chemsys"),
        "elements": [str(e) for e in (g("elements") or [])],
        "nsites": g("nsites"),
        "nelements": g("nelements"),
        "volume": g("volume"),
        "density": g("density"),
        "spacegroup_number": getattr(sym, "number", None) if sym else None,
        "spacegroup_symbol": getattr(sym, "symbol", None) if sym else None,
        "crystal_system": str(getattr(sym, "crystal_system", "")) or None if sym else None,
        "band_gap": g("band_gap"),
        "formation_energy_per_atom": g("formation_energy_per_atom"),
        "energy_above_hull": g("energy_above_hull"),
        "is_stable": g("is_stable"),
        "is_metal": g("is_metal"),
        "theoretical": g("theoretical"),
        "deprecated": g("deprecated"),
        "structure": struct,
    }
    return row


# ---------------------------------------------------------------------------
# Functional (energy_type) lookup
# ---------------------------------------------------------------------------

# Materials Project's "blessed" mixed scheme. Empirically -- see
# results/functional_diagnosis.json -- there is exactly one thermo record with this
# thermo_type per material, and it is the record the summary endpoint is built from.
MIXED_THERMO_TYPE = "GGA_GGA+U_R2SCAN"

# Last-resort ordering, used only when neither the formation-energy match nor the
# mixed-scheme record resolves a material. Reaching this is recorded, not hidden.
FUNCTIONAL_PREFERENCE = ("GGA+U", "GGA", "R2SCAN", "r2SCAN")

# Formation energies agree to well under this when they come from the same record.
FE_MATCH_TOL = 1e-4


def _rank(energy_type: str) -> int:
    for i, name in enumerate(FUNCTIONAL_PREFERENCE):
        if energy_type.upper() == name.upper():
            return i
    return len(FUNCTIONAL_PREFERENCE)


def fetch_energy_types(mpr, rows: list[dict], batch: int = 1000) -> dict[str, dict]:
    """Which DFT functional produced each material's summary properties.

    A band gap without its functional is not a usable number, and pooling a GGA
    gap with an r2SCAN gap is a methodological error that produces perfectly
    plausible-looking nonsense. So this has to be right, not approximately right.

    Two earlier attempts were both wrong, and the way they were wrong is worth
    recording:

    1. **Keep whichever thermo record arrives first.** Materials Project returns
       ~2.2 thermo records per material, so this made the recorded functional a
       function of network ordering. A coin flip, not a decision.

    2. **Prefer GGA over r2SCAN by a fixed rule.** Better, but still wrong.
       `scripts/diagnose_functional.py` checked the rule against evidence and
       refuted it: the summary endpoint agreed with GGA for 86% of TiO2 entries
       and with **r2SCAN for the other 14%**. A fixed preference would have
       mislabelled roughly one material in seven -- invisibly, since a mislabelled
       functional produces no error, just a wrong grouping downstream.

    What the diagnosis showed instead is that the functional is *identifiable per
    material*, not guessable in general:

    * `formation_energy_per_atom` appears in both endpoints. Whichever thermo
      record reproduces the summary value **is** the record summary was built
      from. That is direct evidence, material by material.
    * Independently, there is exactly one thermo record per material carrying
      `thermo_type == "GGA_GGA+U_R2SCAN"` -- Materials Project's blessed mixed
      scheme -- and it is that record. This works even when the summary formation
      energy is missing.

    So we resolve by evidence first, structure second, preference only as a
    last resort, and we record which route was taken for every material in
    `energy_type_resolution`. A dataset that cannot tell you how confident it is
    in its own labels is a dataset that will mislead you eventually.
    """
    want = {r["material_id"] for r in rows}
    summary_fe = {
        r["material_id"]: r.get("formation_energy_per_atom")
        for r in rows
        if r.get("formation_energy_per_atom") is not None
    }

    records: dict[str, list[dict]] = {}
    ids = sorted(want)
    for i in range(0, len(ids), batch):
        block = ids[i : i + batch]
        try:
            docs = mpr.materials.thermo.search(
                material_ids=block,
                fields=["material_id", "energy_type", "thermo_type",
                        "formation_energy_per_atom"],
            )
            for d in docs:
                mid = str(getattr(d, "material_id", ""))
                et = getattr(d, "energy_type", None)
                if not mid or not et:
                    continue
                records.setdefault(mid, []).append({
                    "energy_type": str(et),
                    "thermo_type": str(getattr(d, "thermo_type", "") or ""),
                    "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
                })
        except Exception as exc:  # noqa: BLE001
            print(f"    ! thermo lookup failed for batch {i // batch}: {exc}")

    out: dict[str, dict] = {}
    for mid, recs in records.items():
        available = sorted({r["energy_type"] for r in recs})
        chosen = resolution = None

        # 1. Direct evidence: which record does the summary formation energy match?
        target = summary_fe.get(mid)
        if target is not None:
            hits = {
                r["energy_type"]
                for r in recs
                if r["formation_energy_per_atom"] is not None
                and abs(float(r["formation_energy_per_atom"]) - float(target)) < FE_MATCH_TOL
            }
            if len(hits) == 1:
                chosen, resolution = hits.pop(), "matched_formation_energy"
            elif len(hits) > 1:
                # Several records agree with summary and with each other on energy
                # but disagree on functional -- genuinely unresolvable this way.
                chosen, resolution = sorted(hits, key=_rank)[0], "ambiguous_energy_match"

        # 2. Structural evidence: the blessed mixed-scheme record.
        if chosen is None:
            mixed = [r for r in recs if r["thermo_type"] == MIXED_THERMO_TYPE]
            if len(mixed) == 1:
                chosen, resolution = mixed[0]["energy_type"], "mixed_thermo_type"
            elif mixed:
                chosen = sorted({r["energy_type"] for r in mixed}, key=_rank)[0]
                resolution = "mixed_thermo_type_multiple"

        # 3. Last resort. Recorded loudly so it can be counted and excluded.
        if chosen is None:
            chosen, resolution = sorted(available, key=_rank)[0], "fallback_preference"

        out[mid] = {
            "energy_type": chosen,
            "energy_types_available": available,
            "energy_type_ambiguous": len(available) > 1,
            "energy_type_resolution": resolution,
        }
    return out


# ---------------------------------------------------------------------------
# Chunk IO
# ---------------------------------------------------------------------------

def chunk_path(index: int) -> Path:
    return DEST / f"mp_chunk_{index:04d}.jsonl.gz"


def write_chunk(rows: list[dict], index: int) -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    path = chunk_path(index)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    tmp.replace(path)  # atomic: a killed process never leaves a half-written chunk
    return path


def read_chunks() -> Iterator[dict]:
    for path in sorted(DEST.glob("mp_chunk_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def existing_ids() -> set[str]:
    """material_ids already on disk, so a resumed run does not refetch them."""
    ids = set()
    for path in sorted(DEST.glob("mp_chunk_*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ids.add(json.loads(line)["material_id"])
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            print(f"  ! {path.name} unreadable ({exc}); delete it and re-run to refetch")
    return ids


def next_chunk_index() -> int:
    existing = sorted(DEST.glob("mp_chunk_*.jsonl.gz"))
    return 0 if not existing else int(existing[-1].stem.split("_")[2].split(".")[0]) + 1


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(rows_written: int, query: dict, api_key: str, elapsed: float) -> Path:
    files = sorted(DEST.glob("mp_chunk_*.jsonl.gz"))
    h = hashlib.sha256()
    total_bytes = 0
    for p in files:
        data = p.read_bytes()
        h.update(data)
        total_bytes += len(data)

    manifest = {
        "source": "Materials Project",
        "endpoint": "mp-api / MPRester.materials.summary.search",
        "url": "https://next-gen.materialsproject.org",
        "licence": "CC BY 4.0",
        "cite": "A. Jain et al., APL Materials 1, 011002 (2013)",
        "access_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "access_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "api_key_fingerprint": key_fingerprint(api_key),
        "query": query,
        "fields": SUMMARY_FIELDS,
        "rows": rows_written,
        "chunk_files": [p.name for p in files],
        "total_bytes": total_bytes,
        "sha256_of_all_chunks": h.hexdigest(),
        "download_seconds": round(elapsed, 1),
        "notes": [
            "Structures stored compactly: lattice matrix, fractional coordinates, species.",
            "Disordered (partially occupied) sites are excluded -- a graph node cannot "
            "represent a statistical mixture without a modelling decision not made here.",
            "energy_type (GGA / GGA+U / r2SCAN) fetched separately from the thermo endpoint. "
            "Band gaps from different functionals are different quantities and must never "
            "be pooled.",
        ],
    }
    path = DEST / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(api_key: str, n: int = 3) -> None:
    """Fetch a handful of materials and print exactly what came back.

    Run this before a full download. It verifies the key, the network, and the
    installed mp-api version's response shape in about ten seconds, instead of
    finding out an hour into a 30,000-structure pull.
    """
    from mp_api.client import MPRester

    print("Probing Materials Project...\n")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula="TiO2", fields=SUMMARY_FIELDS, num_chunks=1, chunk_size=n
        )
        docs = list(docs)[:n]
        print(f"  returned {len(docs)} documents\n")
        if not docs:
            print("  ! No documents returned. Check the key and the network.")
            return

        d = docs[0]
        print("  fields present on the first document:")
        for f in SUMMARY_FIELDS:
            v = getattr(d, f, "<<MISSING>>")
            if f == "structure":
                shown = f"<Structure, {len(v)} sites>" if v is not None and v != "<<MISSING>>" else v
            elif f == "symmetry":
                shown = f"number={getattr(v, 'number', '?')} symbol={getattr(v, 'symbol', '?')}"
            else:
                shown = repr(v)[:70]
            flag = "  " if v != "<<MISSING>>" else "!!"
            print(f"    {flag} {f:<28} {shown}")

        row = flatten_doc(d)
        print("\n  flattened row:")
        if row is None:
            print("    ! flatten_doc returned None (disordered structure?)")
        else:
            for k, v in row.items():
                shown = (
                    f"lattice {len(v['lattice'])}x3, {len(v['species'])} sites, "
                    f"species={v['species'][:6]}"
                    if k == "structure" else repr(v)[:70]
                )
                print(f"      {k:<28} {shown}")

        probe_rows = [flatten_doc(x) for x in docs]
        probe_rows = [r for r in probe_rows if r]
        et = fetch_energy_types(mpr, probe_rows)
        print("\n  functional resolution:")
        if not et:
            print("    ! returned nothing -- investigate before downloading")
        for mid, info in et.items():
            print(f"    {mid:<14} {info['energy_type']:<8} "
                  f"available={info['energy_types_available']} "
                  f"via {info['energy_type_resolution']}")

    print("\n  Probe finished. If every field above is present, the full download will work.")


# ---------------------------------------------------------------------------
# Full download
# ---------------------------------------------------------------------------

OPTIONAL_SEARCH_KWARGS = ("deprecated", "theoretical", "num_sites")


def _search_tolerantly(mpr, kwargs: dict):
    """Run a summary search, dropping filter arguments this mp-api version rejects.

    mp-api's search signature moves between releases -- filters get renamed,
    added, or pushed behind a different endpoint. Rather than pin an exact
    version (which rots) or let a one-word mismatch abort a long download, we
    retry without the offending optional filter and say loudly which one was
    dropped, so the manifest and the console never disagree about what was
    actually queried.

    `fields` is never dropped: without it we would silently download the wrong
    columns, which is far worse than a crash.
    """
    kwargs = dict(kwargs)
    dropped: list[str] = []

    for _ in range(len(OPTIONAL_SEARCH_KWARGS) + 1):
        try:
            return mpr.materials.summary.search(**kwargs)
        except TypeError as exc:
            msg = str(exc)
            victim = next(
                (k for k in OPTIONAL_SEARCH_KWARGS if k in kwargs and k in msg), None
            )
            if victim is None:
                raise
            kwargs.pop(victim)
            dropped.append(victim)
            print(f"    ! this mp-api version rejected '{victim}' -- retrying without it")

    if dropped:
        print(f"    ! filters not applied server-side: {dropped}; filtering locally instead")
    return mpr.materials.summary.search(**kwargs)


def download(
    api_key: str,
    max_materials: int | None = None,
    max_sites: int = MAX_SITES,
    min_sites: int = MIN_SITES,
    exclude_theoretical: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    from mp_api.client import MPRester

    query = {
        "num_sites": [min_sites, max_sites],
        "exclude_theoretical": exclude_theoretical,
        "max_materials": max_materials,
        "deprecated": False,
    }

    already = existing_ids()
    if already:
        print(f"  resuming: {len(already):,} materials already on disk\n")

    t0 = time.perf_counter()
    buffer: list[dict] = []
    index = next_chunk_index()
    written = skipped_disordered = skipped_dupe = skipped_filtered = 0

    print(f"  querying materials with {min_sites} <= nsites <= {max_sites} ...")
    with MPRester(api_key) as mpr:
        search_kwargs = dict(
            num_sites=(min_sites, max_sites),
            fields=SUMMARY_FIELDS,
            deprecated=False,
        )
        if exclude_theoretical:
            search_kwargs["theoretical"] = False

        docs = _search_tolerantly(mpr, search_kwargs)
        total = len(docs)
        print(f"  {total:,} materials match. Downloading...\n")

        for i, doc in enumerate(docs):
            mid = str(getattr(doc, "material_id", ""))
            if mid in already:
                skipped_dupe += 1
                continue

            # Enforce the filters locally as well as server-side. If a filter was
            # dropped by _search_tolerantly, this is what still makes the dataset
            # match its manifest -- a manifest that describes a filter nobody
            # applied is worse than no manifest.
            ns = getattr(doc, "nsites", None)
            if ns is not None and not (min_sites <= ns <= max_sites):
                skipped_filtered += 1
                continue
            if getattr(doc, "deprecated", False):
                skipped_filtered += 1
                continue
            if exclude_theoretical and getattr(doc, "theoretical", False):
                skipped_filtered += 1
                continue

            row = flatten_doc(doc)
            if row is None:
                skipped_disordered += 1
                continue

            buffer.append(row)

            if len(buffer) >= chunk_size:
                _flush(mpr, buffer, index)
                written += len(buffer)
                print(f"    chunk {index:04d}: {written:,} written "
                      f"({i + 1:,}/{total:,} seen, {time.perf_counter() - t0:.0f}s)")
                buffer, index = [], index + 1

            if max_materials and written + len(buffer) >= max_materials:
                print(f"    reached --max-materials {max_materials:,}, stopping")
                break

        if buffer:
            _flush(mpr, buffer, index)
            written += len(buffer)
            print(f"    chunk {index:04d}: {written:,} written (final)")

    elapsed = time.perf_counter() - t0
    manifest = write_manifest(written + len(already), query, api_key, elapsed)

    summary = {
        "written_this_run": written,
        "already_present": len(already),
        "total_on_disk": written + len(already),
        "skipped_disordered": skipped_disordered,
        "skipped_filtered_locally": skipped_filtered,
        "skipped_already_present": skipped_dupe,
        "seconds": round(elapsed, 1),
        "manifest": str(manifest),
    }
    return summary


def _flush(mpr, buffer: list[dict], index: int) -> None:
    """Attach the DFT functional(s) to a buffer of rows, then write it out."""
    et = fetch_energy_types(mpr, buffer)
    for r in buffer:
        info = et.get(r["material_id"]) or {}
        r["energy_type"] = info.get("energy_type")
        r["energy_types_available"] = info.get("energy_types_available")
        r["energy_type_ambiguous"] = info.get("energy_type_ambiguous")
        r["energy_type_resolution"] = info.get("energy_type_resolution", "not_found")
    write_chunk(buffer, index)
