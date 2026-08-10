"""Phase 7 -- adsorption energies from Catalysis-Hub.

    python scripts/fetch_catalysis_hub.py --probe     # 20 s, look before building
    python scripts/fetch_catalysis_hub.py             # the actual download

WHY THIS REPOSITORY NEEDS THIS PHASE
-------------------------------------
It is called CatGNN, it lives in a folder called CatalysisAI, and everything in
it so far is bulk-crystal band gap. That is a real mismatch between the name and
the contents, and it is the first thing a reader from the field would notice.

Band gap was the right target to build the machinery on -- 100k labelled
materials, a clean question, a property with textbook physics to check the
attributions against. But the question the portfolio is actually about is
catalysis, where the useful quantity is how strongly a molecule binds to a
surface, and where the Sabatier principle says the best catalyst binds
intermediates neither too weakly nor too strongly. Getting that binding energy
right is the whole game.

WHY CATALYSIS-HUB RATHER THAN OC20
-----------------------------------
OC20 is the obvious alternative and is far larger -- millions of relaxations.
It is also hundreds of gigabytes, which rules it out on a laptop for the same
reason a GPU-scale model does. Catalysis-Hub holds tens of thousands of DFT
surface reaction energies with structures, is openly accessible without a key,
and fits. The size difference is a limitation, not a preference, and it is
recorded as one.

WHY THIS FILE STARTS WITH A PROBE
----------------------------------
Phase 1 taught this the hard way. Writing a full downloader against an API whose
response shape you have only read about produces code that fails in the middle of
a long download, or worse, succeeds and returns something subtly different from
what you assumed. So: introspect the schema, pull three records, print everything
raw, and only then write the pipeline against what actually came back.

The probe also answers questions that decide the design and cannot be guessed:
how many reactions are single-adsorbate chemisorptions rather than multi-step
reactions, which DFT functionals are mixed together, and whether the structures
come back attached or need a second query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

API = "https://api.catalysis-hub.org/graphql"
RAW = REPO / "data" / "raw"


def post(query: str, timeout: int = 60) -> dict:
    """One GraphQL request. urllib only, so there is no new dependency."""
    import urllib.error
    import urllib.request

    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "CatGNN/0.1 (github.com/teja2792/CatGNN)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"\nHTTP {e.code} from {API}\n{e.read().decode('utf-8')[:600]}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\nCannot reach {API}: {e.reason}\n"
              "Check the network, or whether the service has moved.\n")
        sys.exit(1)


def probe() -> None:
    """Look at the API before writing anything that depends on its shape."""
    print(f"\n{'=' * 76}\n  Probing {API}\n{'=' * 76}")

    # 1. What fields does a reaction actually have? Asking beats assuming: the
    #    field names in the 2019 paper may not be the field names today.
    print("\n[1] Schema introspection — what a reaction record contains\n")
    intro = post("""
    { __type(name: "Reaction") { fields { name type { name kind
        ofType { name kind } } } } }
    """)
    t = (intro.get("data") or {}).get("__type")
    if not t:
        print("  introspection returned nothing usable:")
        print("  " + json.dumps(intro)[:800])
        print("\n  The type may be named differently. Trying the root query…\n")
        roots = post('{ __schema { queryType { fields { name } } } }')
        print("  " + json.dumps(roots)[:1200])
        return

    fields = t["fields"]
    print(f"  {len(fields)} fields:\n")
    for f in fields:
        ty = f["type"]
        name = ty.get("name") or (ty.get("ofType") or {}).get("name") or ty["kind"]
        print(f"    {f['name']:<28}{name}")

    # 2. Three real records, printed raw. The point is to see what the values
    #    look like, not to confirm they exist.
    print("\n[2] Three real records\n")
    have = {f["name"] for f in fields}
    wanted = ["Equation", "reactants", "products", "reactionEnergy",
              "activationEnergy", "surfaceComposition", "facet", "sites",
              "coverages", "chemicalComposition", "dftCode", "dftFunctional",
              "pubId", "id"]
    ask = [w for w in wanted if w in have]
    skipped = [w for w in wanted if w not in have]
    if skipped:
        print(f"  (not offered by the API, so not requested: {skipped})\n")

    sample = post("{ reactions(first: 3) { totalCount edges { node { "
                  + " ".join(ask) + " } } } }")
    print("  " + json.dumps(sample, indent=2)[:2600])

    data = (sample.get("data") or {}).get("reactions") or {}
    total = data.get("totalCount")
    if total:
        print(f"\n  totalCount: {total:,} reactions in the database")

    # 3. Are the structures attached, or is a second query needed? This decides
    #    whether Phase 7 can build graphs at all.
    print("\n[3] Are atomic structures reachable from a reaction?\n")
    if "systems" in have:
        sys_probe = post("""
        { reactions(first: 1) { edges { node { Equation systems {
            Formula energy InputFile(format: "json") } } } } }
        """)
        blob = json.dumps(sys_probe)
        print(f"  systems field present. Response length {len(blob):,} chars.")
        print("  " + blob[:1200])
    else:
        print("  no 'systems' field on Reaction — structures need a separate query.")
        print("  Checking the root schema for a structures/systems entry point…")
        roots = post('{ __schema { queryType { fields { name } } } }')
        names = [f["name"] for f in
                 (((roots.get("data") or {}).get("__schema") or {})
                  .get("queryType") or {}).get("fields", [])]
        print(f"  root query fields: {names}")

    print(f"\n{'=' * 76}")
    print("  Paste this output back. The download script gets written against")
    print("  what is actually here, not against what the 2019 paper described.")
    print(f"{'=' * 76}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="inspect the API and stop (do this first)")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    print("\nThe downloader is not written yet — deliberately.\n"
          "Run --probe first and paste the output, so the pipeline is built\n"
          "against the API's real response shape rather than an assumed one.\n"
          "That order is why Phase 1's download survived 102,957 records.\n")


if __name__ == "__main__":
    main()
