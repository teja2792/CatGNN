"""Phase 1 tests -- the data layer, tested without a network or an API key.

Everything here runs against fake Materials Project documents built to mimic the
shape mp-api returns. That keeps CI offline and fast, and it means the parsing
logic is tested against the awkward cases (disordered sites, missing fields,
interrupted downloads) that a live download would rarely produce on demand.

Run:  pytest -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import MissingAPIKey, get_mp_api_key, key_fingerprint  # noqa: E402
from src.data import mp_download  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes that quack like pymatgen / mp-api objects
# ---------------------------------------------------------------------------

class FakeSite:
    def __init__(self, symbol, frac, ordered=True):
        self.is_ordered = ordered
        self.frac_coords = np.array(frac, dtype=float)
        self.specie = types.SimpleNamespace(symbol=symbol)


class FakeStructure:
    def __init__(self, lattice, sites):
        self.lattice = types.SimpleNamespace(matrix=np.array(lattice, dtype=float))
        self._sites = sites

    def __iter__(self):
        return iter(self._sites)

    def __len__(self):
        return len(self._sites)


def rutile_doc(material_id="mp-2657", **overrides):
    """A stand-in for one Materials Project summary document (rutile TiO2)."""
    sym = types.SimpleNamespace(number=136, symbol="P4_2/mnm", crystal_system="Tetragonal")
    doc = dict(
        material_id=material_id,
        formula_pretty="TiO2",
        chemsys="O-Ti",
        elements=["Ti", "O"],
        nsites=6,
        nelements=2,
        volume=62.43,
        density=4.236,
        symmetry=sym,
        structure=FakeStructure(
            [[4.5937, 0, 0], [0, 4.5937, 0], [0, 0, 2.9587]],
            [
                FakeSite("Ti", [0.0, 0.0, 0.0]),
                FakeSite("Ti", [0.5, 0.5, 0.5]),
                FakeSite("O", [0.30478, 0.30478, 0.0]),
                FakeSite("O", [0.69522, 0.69522, 0.0]),
                FakeSite("O", [0.80478, 0.19522, 0.5]),
                FakeSite("O", [0.19522, 0.80478, 0.5]),
            ],
        ),
        band_gap=1.7719,
        formation_energy_per_atom=-3.4644,
        energy_above_hull=0.0435,
        is_stable=False,
        is_metal=False,
        theoretical=False,
        deprecated=False,
    )
    doc.update(overrides)
    return types.SimpleNamespace(**doc)


@pytest.fixture
def tmp_dest(tmp_path, monkeypatch):
    """Point the downloader's output directory at a throwaway folder."""
    monkeypatch.setattr(mp_download, "DEST", tmp_path / "materials_project")
    return tmp_path / "materials_project"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_flatten_doc_keeps_everything_needed_to_rebuild_the_crystal():
    row = mp_download.flatten_doc(rutile_doc())
    assert row is not None

    s = row["structure"]
    assert np.array(s["lattice"]).shape == (3, 3)
    assert len(s["frac_coords"]) == len(s["species"]) == row["nsites"] == 6
    assert s["species"] == ["Ti", "Ti", "O", "O", "O", "O"]
    assert row["spacegroup_symbol"] == "P4_2/mnm"
    assert row["spacegroup_number"] == 136


def test_flattened_row_is_json_serialisable():
    """Numpy types leaking through would break chunk writing at download time."""
    row = mp_download.flatten_doc(rutile_doc())
    json.dumps(row)  # raises TypeError on numpy scalars


def test_disordered_sites_are_rejected_not_averaged():
    """A site holding a statistical mixture of elements is not a graph node.

    Silently taking the majority element, or averaging, would fabricate a
    structure that was never calculated. Rejecting is the honest option.
    """
    doc = rutile_doc(
        structure=FakeStructure(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            [FakeSite("Fe", [0, 0, 0], ordered=False), FakeSite("O", [0.5, 0.5, 0.5])],
        )
    )
    assert mp_download.flatten_doc(doc) is None


def test_missing_optional_fields_do_not_crash():
    """Elastic and some electronic properties are absent for many entries."""
    row = mp_download.flatten_doc(rutile_doc(band_gap=None, energy_above_hull=None))
    assert row is not None
    assert row["band_gap"] is None and row["energy_above_hull"] is None


def test_missing_symmetry_block_is_tolerated():
    row = mp_download.flatten_doc(rutile_doc(symmetry=None))
    assert row is not None and row["spacegroup_number"] is None


# ---------------------------------------------------------------------------
# Chunked, resumable storage
# ---------------------------------------------------------------------------

def test_chunk_round_trip(tmp_dest):
    rows = [mp_download.flatten_doc(rutile_doc(f"mp-{i}")) for i in range(5)]
    mp_download.write_chunk(rows, 0)
    back = list(mp_download.read_chunks())
    assert len(back) == 5
    assert back[0]["structure"]["species"] == rows[0]["structure"]["species"]


def test_resume_skips_what_is_already_downloaded(tmp_dest):
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-1"))], 0)
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-2"))], 1)

    assert mp_download.existing_ids() == {"mp-1", "mp-2"}
    assert mp_download.next_chunk_index() == 2


def test_chunk_writes_are_atomic(tmp_dest):
    """A killed download must never leave a half-written chunk behind.

    write_chunk writes to a .tmp file and renames it, so the only files matching
    the chunk glob are complete ones.
    """
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-1"))], 0)
    assert not list(tmp_dest.glob("*.tmp"))
    assert len(list(tmp_dest.glob("mp_chunk_*.jsonl.gz"))) == 1


def test_corrupt_chunk_is_reported_not_silently_skipped(tmp_dest, capsys):
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-1"))], 0)
    (tmp_dest / "mp_chunk_0001.jsonl.gz").write_bytes(b"not gzip at all")

    ids = mp_download.existing_ids()
    assert "mp-1" in ids
    assert "unreadable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Manifest and credentials
# ---------------------------------------------------------------------------

def test_manifest_records_provenance_and_never_the_key(tmp_dest):
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-1"))], 0)
    secret = "SUPERSECRETKEY123456"
    path = mp_download.write_manifest(1, {"num_sites": [1, 30]}, secret, 12.3)

    text = path.read_text(encoding="utf-8")
    assert secret not in text, "API key leaked into the manifest"

    m = json.loads(text)
    for field in ("source", "licence", "cite", "access_date", "query", "fields",
                  "rows", "sha256_of_all_chunks", "api_key_fingerprint"):
        assert field in m, f"manifest missing {field}"
    assert m["api_key_fingerprint"] == key_fingerprint(secret)


def test_manifest_checksum_changes_when_data_changes(tmp_dest):
    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-1"))], 0)
    first = json.loads(mp_download.write_manifest(1, {}, "k", 1).read_text())

    mp_download.write_chunk([mp_download.flatten_doc(rutile_doc("mp-2"))], 1)
    second = json.loads(mp_download.write_manifest(2, {}, "k", 1).read_text())

    assert first["sha256_of_all_chunks"] != second["sha256_of_all_chunks"]


def test_key_fingerprint_is_short_and_not_reversible():
    fp = key_fingerprint("hunter2")
    assert len(fp) == 12
    assert "hunter2" not in fp
    assert fp == key_fingerprint("hunter2")
    assert fp != key_fingerprint("hunter3")


def test_missing_key_gives_actionable_instructions(monkeypatch, tmp_path):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr("src.config.REPO", tmp_path)  # no .env in a temp dir

    with pytest.raises(MissingAPIKey) as exc:
        get_mp_api_key(None)

    msg = str(exc.value)
    assert "setx MP_API_KEY" in msg and ".env" in msg


def test_dotenv_is_read(monkeypatch, tmp_path):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    (tmp_path / ".env").write_text('MP_API_KEY="from_dotenv"\n', encoding="utf-8")
    monkeypatch.setattr("src.config.REPO", tmp_path)

    assert get_mp_api_key(None) == "from_dotenv"


def test_env_var_beats_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("MP_API_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.setattr("src.config.REPO", tmp_path)
    monkeypatch.setenv("MP_API_KEY", "from_env")

    assert get_mp_api_key(None) == "from_env"


def test_dotenv_is_gitignored():
    """The single most important line in .gitignore for this phase."""
    patterns = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "\n.env" in patterns or patterns.startswith(".env")


# ---------------------------------------------------------------------------
# Functional (energy_type) handling
#
# Materials Project returns MORE THAN ONE thermo record per material -- the probe
# showed 6 records for 3 materials. An earlier version kept whichever arrived
# first, making the recorded functional depend on network ordering. These tests
# pin down the fix.
# ---------------------------------------------------------------------------

class FakeMPR:
    """Stands in for MPRester, returning thermo records in a chosen order."""

    def __init__(self, records):
        self._records = records
        self.materials = types.SimpleNamespace(thermo=types.SimpleNamespace(search=self._search))

    def _search(self, material_ids=None, fields=None):
        return [
            types.SimpleNamespace(**r)
            for r in self._records
            if r["material_id"] in set(material_ids or [])
        ]


def test_all_functionals_are_kept_not_just_the_first():
    mpr = FakeMPR([
        {"material_id": "mp-1", "energy_type": "r2SCAN", "thermo_type": "R2SCAN"},
        {"material_id": "mp-1", "energy_type": "GGA", "thermo_type": "GGA_GGA+U"},
    ])
    info = mp_download.fetch_energy_types(mpr, ["mp-1"])["mp-1"]

    assert info["energy_types_available"] == ["GGA", "r2SCAN"]
    assert info["energy_type_ambiguous"] is True


def test_primary_functional_does_not_depend_on_api_ordering():
    """The bug this replaced: whichever record arrived first won.

    Same two records, opposite order, must give the same answer -- otherwise the
    dataset's labels are a function of network timing.
    """
    a = [
        {"material_id": "mp-1", "energy_type": "r2SCAN", "thermo_type": "R2SCAN"},
        {"material_id": "mp-1", "energy_type": "GGA", "thermo_type": "GGA_GGA+U"},
    ]
    first = mp_download.fetch_energy_types(FakeMPR(a), ["mp-1"])["mp-1"]
    second = mp_download.fetch_energy_types(FakeMPR(list(reversed(a))), ["mp-1"])["mp-1"]

    assert first == second
    assert first["energy_type"] == "GGA"  # per FUNCTIONAL_PREFERENCE


def test_single_functional_is_not_flagged_ambiguous():
    mpr = FakeMPR([{"material_id": "mp-2", "energy_type": "GGA+U", "thermo_type": "GGA_GGA+U"}])
    info = mp_download.fetch_energy_types(mpr, ["mp-2"])["mp-2"]

    assert info["energy_type"] == "GGA+U"
    assert info["energy_type_ambiguous"] is False


def test_functional_preference_order_is_respected():
    mpr = FakeMPR([
        {"material_id": "mp-3", "energy_type": "r2SCAN", "thermo_type": "R2SCAN"},
        {"material_id": "mp-3", "energy_type": "GGA+U", "thermo_type": "GGA_GGA+U"},
        {"material_id": "mp-3", "energy_type": "GGA", "thermo_type": "GGA_GGA+U"},
    ])
    assert mp_download.fetch_energy_types(mpr, ["mp-3"])["mp-3"]["energy_type"] == "GGA+U"


def test_unknown_functional_sorts_last_but_is_kept():
    """A functional we have never seen must not silently outrank a known one."""
    mpr = FakeMPR([
        {"material_id": "mp-4", "energy_type": "SOMETHING_NEW", "thermo_type": "?"},
        {"material_id": "mp-4", "energy_type": "GGA", "thermo_type": "GGA_GGA+U"},
    ])
    info = mp_download.fetch_energy_types(mpr, ["mp-4"])["mp-4"]

    assert info["energy_type"] == "GGA"
    assert "SOMETHING_NEW" in info["energy_types_available"]


def test_thermo_failure_does_not_abort_the_download():
    """A dropped thermo call must lose the functional, not the structures."""
    class Broken:
        def __init__(self):
            self.materials = types.SimpleNamespace(
                thermo=types.SimpleNamespace(search=self._boom))

        def _boom(self, **kwargs):
            raise RuntimeError("gateway timeout")

    assert mp_download.fetch_energy_types(Broken(), ["mp-1"]) == {}
