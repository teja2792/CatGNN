"""Paths, constants and credential handling.

One place for everything that would otherwise get hard-coded in six scripts and
then drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]

DATA = REPO / "data"
RAW = DATA / "raw"            # downloads, gitignored, reproducible from manifests
CACHE = DATA / "cache"        # built graphs, gitignored, rebuildable
REFERENCE = DATA / "reference"  # small committed snapshots used by figures

RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"

for _d in (RAW, CACHE, REFERENCE, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Graph construction
#
# The CGCNN convention (Xie & Grossman 2018). Note that an 8 A cutoff is far
# wider than a bond: it deliberately includes contacts nobody would call bonded,
# and the model learns how much each is worth. An edge here means "close enough
# to matter", not "chemically bonded".
# ---------------------------------------------------------------------------

CUTOFF_ANGSTROM = 8.0
MAX_NEIGHBOURS = 12

# Cells bigger than this are dropped. A handful of 100-atom structures would
# dominate every epoch on a laptop CPU for no scientific gain. Recorded in the
# manifest so the filter is never invisible.
MAX_SITES = 30
MIN_SITES = 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

class MissingAPIKey(RuntimeError):
    pass


def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser. Avoids a dependency for one small job."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_mp_api_key(explicit: str | None = None) -> str:
    """Materials Project API key, from the environment or a local .env file.

    Resolution order:
      1. the value passed in
      2. the MP_API_KEY environment variable
      3. MP_API_KEY in a .env file at the repo root

    `.env` is gitignored. The key is never written to any file this repo creates,
    never printed, and never recorded in a manifest -- manifests store a short
    fingerprint instead, so you can tell *which* key produced a download without
    the download revealing the key.
    """
    key = explicit or os.environ.get("MP_API_KEY") or _read_dotenv(REPO / ".env").get("MP_API_KEY")
    if not key:
        raise MissingAPIKey(
            "No Materials Project API key found.\n\n"
            "Get one free at https://next-gen.materialsproject.org/api, then either:\n\n"
            "  PowerShell (persists across terminals):\n"
            '    setx MP_API_KEY "your_key_here"\n'
            "    # then open a NEW terminal\n\n"
            "  or create a .env file in the repo root (already gitignored):\n"
            "    MP_API_KEY=your_key_here\n"
        )
    return key.strip()


def get_catalysis_hub_key(explicit: str | None = None) -> str:
    """Catalysis-Hub API key. A DIFFERENT key from the Materials Project one.

    Same resolution order and the same discipline as get_mp_api_key: environment
    first, then a gitignored .env, never written to a file this repo creates,
    never printed, only ever fingerprinted into a manifest.

    Worth recording why this function exists at all. The 2019 Scientific Data
    paper describes an open API and this repository's first draft said "openly
    accessible without a key". The probe returned HTTP 401 with a message
    pointing at an auth endpoint, so that claim is now wrong. Schema
    introspection still works unauthenticated; only data queries do not.

    That is a small example of a general problem: a documented fact about a live
    service has a shelf life, and the only way to know it has expired is to make
    the request.
    """
    key = (explicit
           or os.environ.get("CATALYSIS_HUB_API_KEY")
           or _read_dotenv(REPO / ".env").get("CATALYSIS_HUB_API_KEY"))
    if not key:
        raise MissingAPIKey(
            "No Catalysis-Hub API key found.\n\n"
            "This is NOT your Materials Project key -- it is a separate one.\n"
            "Get it at https://api.catalysis-hub.org/auth/login, then either:\n\n"
            "  PowerShell (persists across terminals):\n"
            '    setx CATALYSIS_HUB_API_KEY "your_key_here"\n'
            "    # then open a NEW terminal\n\n"
            "  or add a line to the .env file in the repo root (gitignored):\n"
            "    CATALYSIS_HUB_API_KEY=your_key_here\n"
        )
    return key.strip()


def key_fingerprint(key: str) -> str:
    """Short, non-reversible identifier for a key, safe to record in a manifest."""
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()[:12]
