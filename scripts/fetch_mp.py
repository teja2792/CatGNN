"""Download crystal structures and properties from the Materials Project.

Run the probe first. It takes about ten seconds and catches a wrong key, a
blocked network, or an mp-api version whose response shape has changed -- all of
which are much cheaper to find out now than an hour into a full download.

    python scripts/fetch_mp.py --probe

Then, once the probe looks right:

    python scripts/fetch_mp.py --max-materials 2000     # a small first pull
    python scripts/fetch_mp.py                          # everything under the site cap

The download is resumable. If it is interrupted, run the same command again and
it picks up where it stopped.

Requires an API key. Either set MP_API_KEY in your environment:

    setx MP_API_KEY "your_key_here"      # PowerShell, then open a NEW terminal

or put it in a .env file at the repo root (already gitignored):

    MP_API_KEY=your_key_here

The key is never written to any file this script creates. Manifests record a
short one-way fingerprint instead, so you can tell which key produced a download
without the download exposing the key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import MAX_SITES, MIN_SITES, MissingAPIKey, get_mp_api_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--probe", action="store_true",
                    help="fetch 3 materials, print the response shape, exit")
    ap.add_argument("--max-materials", type=int, default=None,
                    help="stop after this many (useful for a first pull)")
    ap.add_argument("--max-sites", type=int, default=MAX_SITES,
                    help=f"drop cells larger than this (default {MAX_SITES})")
    ap.add_argument("--min-sites", type=int, default=MIN_SITES)
    ap.add_argument("--exclude-theoretical", action="store_true",
                    help="keep only materials that have been experimentally observed")
    ap.add_argument("--api-key", default=None,
                    help="override; prefer MP_API_KEY in the environment")
    args = ap.parse_args()

    try:
        key = get_mp_api_key(args.api_key)
    except MissingAPIKey as exc:
        print(exc)
        sys.exit(1)

    try:
        __import__("mp_api.client")
    except ImportError:
        print("mp-api is not installed.\n\n    pip install mp-api\n")
        sys.exit(1)

    from src.data import mp_download

    if args.probe:
        try:
            mp_download.probe(key)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  PROBE FAILED: {type(exc).__name__}: {exc}\n")
            print("  Most likely causes, in order:")
            print("    1. the API key is wrong or was regenerated")
            print("    2. no network access, or a proxy blocking api.materialsproject.org")
            print("    3. mp-api is out of date:  pip install -U mp-api")
            sys.exit(1)
        return

    print("\nMaterials Project bulk download")
    print(f"  cell size filter : {args.min_sites} <= nsites <= {args.max_sites}")
    print(f"  limit            : {args.max_materials or 'none'}")
    print(f"  theoretical      : {'excluded' if args.exclude_theoretical else 'included'}\n")

    try:
        summary = mp_download.download(
            api_key=key,
            max_materials=args.max_materials,
            max_sites=args.max_sites,
            min_sites=args.min_sites,
            exclude_theoretical=args.exclude_theoretical,
        )
    except KeyboardInterrupt:
        print("\n  interrupted -- chunks already written are intact.")
        print("  Re-run the same command to resume.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
        print("  Chunks already written are intact; re-run to resume.")
        sys.exit(1)

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("=" * 62)
    print("\nNext:  python scripts/inspect_mp.py")


if __name__ == "__main__":
    main()
