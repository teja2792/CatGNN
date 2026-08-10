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
surface reaction energies with structures, and fits. The size difference is a
limitation, not a preference, and it is recorded as one.

IT NEEDS A KEY NOW, WHICH THE PROBE DISCOVERED
-----------------------------------------------
The first version of this file said "openly accessible without a key", on the
authority of the 2019 Scientific Data paper. The probe came back HTTP 401 with a
pointer to an auth endpoint. Schema introspection still works unauthenticated;
data queries do not.

That claim was wrong for exactly as long as it took to make one request, which is
the argument for probing before building in miniature. The key is a Catalysis-Hub
key and has nothing to do with the Materials Project one -- see
config.get_catalysis_hub_key.

THE RATE LIMITS SHAPE THE WHOLE DESIGN
---------------------------------------
    10 requests per minute
    500 requests per day -- accounts exceeding this are AUTOMATICALLY SUSPENDED
    a per-request row cap, unpublished, discovered by the probe
    cursor pagination via first / after

Materials Project let this repository pull 102,957 records by brute force. Here
that is impossible, and the failure mode is far worse than a slow download: going
over the daily cap is not a retryable error, it ends the account.

So the download is built around a budget rather than around throughput:

  * every request goes through src/data/rate_limit.RateLimiter, which keeps a
    ROLLING 24-HOUR LEDGER ON DISK. An in-memory counter would protect one
    process; the server counts every process ever run with this key.
  * the ledger stops at 90% of the published cap, because it cannot see requests
    made from the web console or anywhere else.
  * downloads are RESUMABLE by cursor. When the budget runs out the run stops
    cleanly and the next one continues, rather than starting over and spending
    the next day's budget re-fetching what it already had.
  * rows per request are maximised, since the scarce resource is requests and
    not rows.

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
# Outside data/raw so that clearing the data cache cannot reset the ledger
# and hand the account a fresh 450 requests it has not got.
BUDGET_FILE = REPO / "data" / "cache" / "catalysis_hub_budget.json"


def post(query: str, timeout: int = 60, key: str | None = None) -> dict:
    """One GraphQL request. urllib only, so there is no new dependency.

    `key` is optional because introspection does not need one. Data queries do,
    and the caller decides -- that way the probe can still report the schema on a
    machine with no key configured, which is more useful than refusing to run.
    """
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json",
               "User-Agent": "CatGNN/0.1 (github.com/teja2792/CatGNN)"}
    if key:
        headers["X-API-Key"] = key

    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8")[:600]
        print(f"\nHTTP {e.code} from {API}\n{detail}\n")
        if e.code in (401, 403):
            print("This endpoint needs a Catalysis-Hub key, which is NOT your\n"
                  "Materials Project key. Get one at\n"
                  "  https://api.catalysis-hub.org/auth/login\n"
                  "then:\n"
                  '  setx CATALYSIS_HUB_API_KEY "your_key_here"   (open a NEW terminal)\n'
                  "or add CATALYSIS_HUB_API_KEY=... to the gitignored .env file.\n\n"
                  "Do not paste the key into a chat window or a commit.\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\nCannot reach {API}: {e.reason}\n"
              "Check the network, or whether the service has moved.\n")
        sys.exit(1)


def probe() -> None:
    """Look at the API before writing anything that depends on its shape.

    Costs about five requests out of a 450/day budget, which is the cheapest
    possible insurance against spending the whole budget on a wrong assumption.
    """
    from src.config import MissingAPIKey, get_catalysis_hub_key, key_fingerprint
    from src.data.rate_limit import DailyBudgetExhausted, RateLimiter

    print(f"\n{'=' * 76}\n  Probing {API}\n{'=' * 76}")

    try:
        key = get_catalysis_hub_key()
        print(f"\n  key found, fingerprint {key_fingerprint(key)} "
              "(the key itself is never printed or written anywhere)")
    except MissingAPIKey as e:
        key = None
        print("\n  " + "!" * 70)
        print("  NO KEY VISIBLE TO THIS PROCESS.")
        print("  If you just ran setx, that is the reason: setx only affects")
        print("  processes started AFTERWARDS. Close this terminal, open a new")
        print("  one, and run again. The key is fine; this shell cannot see it.")
        print("  " + "!" * 70)
        print(f"\n{e}")

    limiter = RateLimiter(BUDGET_FILE)
    print(f"  {limiter.report()}")

    def ask(query: str) -> dict:
        limiter.acquire()
        return post(query, key=key)

    # 1. What fields does a reaction actually have?
    print("\n[1] Schema introspection — what a reaction record contains\n")
    intro = ask("""
    { __type(name: "Reaction") { fields { name type { name kind
        ofType { name kind } } } } }
    """)
    t = (intro.get("data") or {}).get("__type")
    if not t:
        print("  introspection returned nothing usable:")
        print("  " + json.dumps(intro)[:800])
        return

    fields = t["fields"]
    have = {f["name"] for f in fields}
    print(f"  {len(fields)} fields:\n")
    for f in fields:
        ty = f["type"]
        name = ty.get("name") or (ty.get("ofType") or {}).get("name") or ty["kind"]
        print(f"    {f['name']:<28}{name}")

    if key is None:
        print("\n  Everything below needs a key. Configure one and re-run.\n")
        return

    # 2. Three real records, printed raw.
    print("\n[2] Three real records\n")
    wanted = ["Equation", "reactants", "products", "reactionEnergy",
              "activationEnergy", "surfaceComposition", "facet", "sites",
              "coverages", "chemicalComposition", "dftCode", "dftFunctional",
              "pubId", "id"]
    fld = [w for w in wanted if w in have]
    if [w for w in wanted if w not in have]:
        print(f"  (not offered, so not requested: {[w for w in wanted if w not in have]})\n")

    try:
        sample = ask("{ reactions(first: 3) { totalCount edges { node { "
                     + " ".join(fld) + " } } } }")
    except DailyBudgetExhausted as e:
        print(f"\n{e}\n")
        return
    print("  " + json.dumps(sample, indent=2)[:2600])

    total = ((sample.get("data") or {}).get("reactions") or {}).get("totalCount")
    if total:
        print(f"\n  totalCount: {total:,} reactions")

    # 3. The per-request row cap. Unpublished, and it decides how many requests
    #    the whole download needs -- which is the binding constraint, not time.
    print("\n[3] How many rows will it actually return at once?\n")
    for want in (200, 1000):
        try:
            r = ask("{ reactions(first: %d) { edges { node { id } } "
                    "pageInfo { hasNextPage endCursor } } }" % want)
        except DailyBudgetExhausted as e:
            print(f"  {e}")
            break
        node = (r.get("data") or {}).get("reactions")
        if not node:
            print(f"  asked {want:>5} → error: {json.dumps(r)[:300]}")
            continue
        got = len(node.get("edges", []))
        page = node.get("pageInfo") or {}
        print(f"  asked {want:>5} → got {got:>5} rows   hasNextPage="
              f"{page.get('hasNextPage')}   cursor={str(page.get('endCursor'))[:24]}")
        if got < want:
            print(f"\n  → the cap is {got} rows per request.")
            if total:
                need = -(-total // max(got, 1))
                print(f"    {total:,} reactions / {got} = {need:,} requests, "
                      f"against a budget of ~450/day.")
                print(f"    That is {need / 450:.1f} days for metadata alone, "
                      "before any structures.")
            break

    # 4. Are structures attached, and what do they cost?
    print("\n[4] Are atomic structures reachable from a reaction?\n")
    if "systems" in have:
        try:
            sp = ask("""
            { reactions(first: 1) { edges { node { Equation systems {
                Formula energy InputFile(format: "json") } } } } }
            """)
            blob = json.dumps(sp)
            print(f"  systems field present. One reaction's structures = "
                  f"{len(blob):,} chars.")
            print("  " + blob[:1000])
        except DailyBudgetExhausted as e:
            print(f"  {e}")
    else:
        print("  no 'systems' field — structures need a separate query.")

    print(f"\n  {limiter.report()}")
    print(f"\n{'=' * 76}")
    print("  Paste this back. The row cap and the structure cost together decide")
    print("  what Phase 7 can actually cover inside the request budget.")
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

    print("\nThe downloader is not written yet — deliberately.\n\n"
          "Run --probe first and paste the output. Two numbers from it decide the\n"
          "whole design and cannot be guessed: the per-request row cap, and what a\n"
          "structure costs to fetch. With 500 requests a day and suspension for\n"
          "exceeding it, those set what Phase 7 can cover.\n")


if __name__ == "__main__":
    main()
