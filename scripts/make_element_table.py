"""Generate the element property table the descriptors are built from.

Run once. Writes ``data/reference/element_properties.json``, which is committed,
so nothing downstream needs pymatgen just to compute a descriptor. Re-run only if
you want to add a property or refresh against a newer pymatgen.

    python scripts/make_element_table.py

Why a committed table rather than calling pymatgen at feature time: descriptors
should be reproducible from the repository alone. If the numbers can change when
someone upgrades a library, then two runs of the "same" experiment are not
comparable, and there is nothing in the results to tell you that happened. The
table records the pymatgen version it came from.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import REFERENCE  # noqa: E402

# Scalar properties pulled straight off pymatgen's Element.
# Chosen to span the axes a chemist would actually reach for: size, electron
# affinity/electronegativity, position in the table, and bulk thermal behaviour.
SCALARS = [
    "Z",
    "atomic_mass",
    "X",                        # Pauling electronegativity
    "atomic_radius",
    "atomic_radius_calculated",
    "average_ionic_radius",
    "row",                      # period
    "group",
    "mendeleev_no",
    "melting_point",
    "boiling_point",
    "molar_volume",
    "thermal_conductivity",
    "electron_affinity",
    "max_oxidation_state",
    "min_oxidation_state",
]

FLAGS = [
    "is_metal",
    "is_transition_metal",
    "is_metalloid",
    "is_alkali",
    "is_alkaline",
    "is_chalcogen",
    "is_halogen",
    "is_lanthanoid",
    "is_actinoid",
    "is_noble_gas",
]


def valence_counts(element) -> dict[str, float]:
    """Electrons in the outermost s, p, d and f sub-shells.

    Magpie's most useful block. Bonding is largely a story about valence
    electrons, so giving the model these directly saves it from having to
    rediscover the periodic table from atomic number alone.
    """
    out = {"n_s": 0.0, "n_p": 0.0, "n_d": 0.0, "n_f": 0.0}
    try:
        shells = element.full_electronic_structure
        if not shells:
            return out
        max_n = max(n for n, _, _ in shells)
        for n, orbital, count in shells:
            key = f"n_{orbital}"
            if key not in out:
                continue
            # Outermost shell for s and p; for d and f the shell below counts too,
            # which is what makes a transition metal a transition metal.
            if orbital == "s" and n == max_n:
                out[key] += count
            elif orbital == "p" and n == max_n:
                out[key] += count
            elif orbital == "d" and n >= max_n - 1:
                out[key] += count
            elif orbital == "f" and n >= max_n - 2:
                out[key] += count
    except Exception:
        pass
    out["n_valence"] = sum(out.values())
    return out


def _pymatgen_version() -> str:
    """pymatgen stopped exposing __version__ at top level; ask the metadata."""
    from importlib.metadata import PackageNotFoundError, version

    for name in ("pymatgen", "pymatgen-core"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    warnings.filterwarnings("ignore")
    try:
        from pymatgen.core import Element
    except ImportError:
        print("pymatgen is required to regenerate the table:\n\n    pip install pymatgen\n")
        sys.exit(1)

    table: dict[str, dict] = {}
    missing: dict[str, int] = {}

    for z in range(1, 104):  # H through Lr
        try:
            el = Element.from_Z(z)
        except Exception:
            continue

        rec: dict[str, float | None] = {}
        for p in SCALARS:
            try:
                v = getattr(el, p, None)
                # pymatgen returns FloatWithUnit for several of these.
                rec[p] = float(v) if v is not None else None
            except Exception:
                rec[p] = None
            if rec[p] is None:
                missing[p] = missing.get(p, 0) + 1

        for f in FLAGS:
            try:
                rec[f] = float(bool(getattr(el, f)))
            except Exception:
                rec[f] = 0.0

        rec.update(valence_counts(el))
        table[el.symbol] = rec

    props = sorted({k for rec in table.values() for k in rec})
    blob = {
        "source": "pymatgen.core.Element",
        "pymatgen_version": _pymatgen_version(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_elements": len(table),
        "properties": props,
        "note": (
            "Committed so descriptors are reproducible from this repository alone, "
            "without depending on which pymatgen version happens to be installed. "
            "Missing values are null and are imputed at feature time by the "
            "training-set median -- see src/features/descriptors.py."
        ),
        "missing_counts": missing,
        "elements": table,
    }

    REFERENCE.mkdir(parents=True, exist_ok=True)
    out = REFERENCE / "element_properties.json"
    out.write_text(json.dumps(blob, indent=1), encoding="utf-8")

    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(table)} elements, {len(props)} properties each")
    print(f"  pymatgen {blob['pymatgen_version']}")
    if missing:
        print("\n  properties with gaps (imputed at feature time, never guessed here):")
        for p, n in sorted(missing.items(), key=lambda kv: -kv[1]):
            print(f"    {p:<28} missing for {n} elements")


if __name__ == "__main__":
    main()
