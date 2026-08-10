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

WHAT THE PROBE FOUND, AND WHY IT CHANGED THE PLAN
--------------------------------------------------
    158,304 reactions
    200 rows per request  (asked for 1000, got 200)
    -> 792 requests to page the whole table = 1.8 days of a 450/day budget
    one reaction's structures = 5.3 kB, so all of them would be ~840 MB

TARGET HETEROGENEITY -- the finding that matters most. `reactionEnergy` is NOT a
single comparable quantity. The three sampled records were:

    Rh(g) + * -> Rh*                                    -6.54 eV
    Au(g) + * -> Au*                                    -2.78 eV
    3.0CH4(g) + H2O(g) - 2.0H2(g) + * -> 3.0CH3* + HO*  +9.44 eV

The first two are single metal atoms depositing on a surface. The third is three
methyls and a hydroxyl formed together at a stated coverage, with stoichiometric
coefficients including a NEGATIVE one. Its +9.44 eV is large because four species
are formed at once, not because the binding is weak.

Training on this column as-is would fit a model to a target whose physical
meaning changes from row to row -- a worse version of the DFT-functional
ambiguity Phase 1 hit with Materials Project, and one that would produce a
plausible-looking number that means nothing.

So Phase 7 has to filter to a comparable subset before it trains on anything.
The natural one is single-adsorbate chemisorption: exactly one product, with
coefficient 1, of the form A(g) + * -> A*. Whether metal-atom deposition belongs
in the same target as molecular adsorption is a further question -- physically
they are different processes, and mixing them is the same mistake one level down.

SERVER-SIDE FILTERING EXISTS, AND IT IS NOT ENOUGH ON ITS OWN
--------------------------------------------------------------
`reactions` takes 20 filterable arguments, so only the wanted rows need
fetching -- the 792-request estimate above applies only to a blind full scan.

But `products: "~COstar"` is a substring match, and among its 6,518 hits was

    CHO* -> hfH2(g) + CO*     -0.53 eV

a dehydrogenation step, not a CO adsorption. Filtering on products says nothing
about reactants, so the target heterogeneity described above survives the filter
one level down. Narrowing on both sides gets close; the rest is rejected locally
after parsing, which is cheap once the rows are in hand.

Two clean rows from the same query, as a sanity check that the data is real:

    CO(g) + * -> CO*   on Ru3Ga   -2.06 eV     (ruthenium grips CO)
    CO(g) + * -> CO*   on Ag3Au   -0.05 eV     (silver barely touches it)

A 2 eV spread across alloys is exactly the variation a binding-energy model is
wanted for, and it runs the right way round.

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

# Stated up front so the cost is known before anything is spent: one
# introspection, one sample, two row-cap probes, one structure probe.
MAX_PROBE_REQUESTS = 5


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


def show_budget() -> None:
    """Report the ledger without making a request.

    Exists so "how many have I used?" never has to be answered by trying one and
    seeing what happens, which is the sort of thing that costs an account.
    """
    from src.data.rate_limit import RateLimiter

    limiter = RateLimiter(BUDGET_FILE)
    print(f"\n  {limiter.report()}")
    print(f"  ledger: {BUDGET_FILE}")
    wait = limiter.seconds_until_free()
    if wait > 0:
        print(f"  budget exhausted; frees up in {wait / 3600:.1f} hours")
    print("\n  Published limits: 10/minute, 500/day with automatic suspension.")
    print("  This tool self-limits to 90% of the daily cap and waits out the")
    print("  per-minute one. Requests made from the website are NOT counted here.\n")


# The standard intermediates of heterogeneous catalysis. These are the species
# Norskov-style scaling relations are built on, and the ones a binding-energy
# model is actually wanted for. Chosen on chemistry, before counting how many
# rows each has, so the dataset is not defined by whatever happened to be
# plentiful.
ADSORBATES = ["CO", "H", "O", "OH", "N", "C", "CH3", "NO", "S", "OOH"]


def plan() -> None:
    """How many CLEAN single-adsorbate rows exist, per adsorbate?

    The filter probe showed server-side filtering works, and immediately showed
    its limit: products="~COstar" also returned

        CHO* -> hfH2(g) + CO*     -0.53 eV

    which is a dehydrogenation step, not a CO adsorption. A substring match on
    products says nothing about the reactants, so the target heterogeneity
    survives the filter one level down.

    Narrowing on BOTH sides -- reactants contain the gas-phase species, products
    contain the adsorbed one -- gets much closer, and whatever slips through is
    cheap to reject locally once the rows are in hand. This counts what each
    adsorbate would actually yield, so the dataset can be scoped before a single
    row is downloaded.

    One request per adsorbate.
    """
    from src.config import get_catalysis_hub_key
    from src.data.rate_limit import DailyBudgetExhausted, RateLimiter

    key = get_catalysis_hub_key()
    limiter = RateLimiter(BUDGET_FILE)
    print(f"\n{'=' * 76}\n  How much clean data is there, per adsorbate?\n{'=' * 76}")
    print(f"\n  {limiter.report()}   (this uses up to {len(ADSORBATES)})")
    print(f"\n  {'adsorbate':<12}{'products~':>12}{'+ reactants~':>14}   example equation")
    print("  " + "-" * 74)

    counts = {}
    for ads in ADSORBATES:
        try:
            limiter.acquire()
        except DailyBudgetExhausted as e:
            print(f"\n{e}\n")
            break

        r = post('{ reactions(first: 2, products: "~%sstar", '
                 'reactants: "~%sgas") { totalCount edges { node { Equation '
                 'reactionEnergy surfaceComposition } } } }' % (ads, ads), key=key)
        node = (r.get("data") or {}).get("reactions")
        if not node:
            print(f"  {ads:<12}{'error':>12}   {json.dumps(r)[:120]}")
            continue

        n = node.get("totalCount", 0)
        edges = node.get("edges", [])
        eg = edges[0]["node"]["Equation"] if edges else "—"
        counts[ads] = n
        print(f"  {ads:<12}{'':>12}{n:>14}   {eg[:44]}")

    if counts:
        total = sum(counts.values())
        pages = sum(-(-n // 200) for n in counts.values())
        print(f"\n  {total:,} rows across {len(counts)} adsorbates")
        print(f"  = {pages} requests at 200 rows each, against a 450/day budget")
        print(f"  ({pages / 450:.0%} of one day — the whole table would have been 792)")
        print("\n  Structures are a separate cost: ~5.3 kB per reaction, and")
        print("  including them will reduce rows per request. Measured next.")

    print(f"\n  {limiter.report()}\n")


def probe_filters() -> None:
    """Can the server filter, or must 158,304 rows be pulled to find the useful ones?

    This is the question the first probe raised and could not answer, and it is
    worth two requests because it changes the budget by an order of magnitude.

    The first probe established: 200 rows per request, 158,304 reactions, so 792
    requests -- 1.8 days of a 450/day budget -- to page the whole table. That is
    survivable but wasteful, because most of those rows are not usable as a
    single, comparable target (see the docstring's TARGET HETEROGENEITY note).

    If `reactions` accepts server-side filters, only the wanted rows need
    fetching and the cost collapses. If it does not, the download has to page
    everything and filter locally, and Phase 7 has to be planned across two days.

    Introspecting the ARGUMENTS of the reactions field answers it. The first probe
    introspected the Reaction *type* -- its fields -- which says what comes back,
    not what can be asked for.
    """
    from src.config import get_catalysis_hub_key
    from src.data.rate_limit import RateLimiter

    key = get_catalysis_hub_key()
    limiter = RateLimiter(BUDGET_FILE)
    print(f"\n{'=' * 76}\n  Can the server filter?\n{'=' * 76}")
    print(f"\n  {limiter.report()}   (this probe uses at most 2)")

    limiter.acquire()
    r = post("""
    { __schema { queryType { fields { name args {
        name defaultValue type { name kind ofType { name kind } } } } } } }
    """, key=key)

    fields = (((r.get("data") or {}).get("__schema") or {})
              .get("queryType") or {}).get("fields", [])
    target = next((f for f in fields if f["name"] == "reactions"), None)
    if not target:
        print("  no 'reactions' field on the root query — schema has changed:")
        print("  " + json.dumps([f["name"] for f in fields])[:400])
        return

    args = target.get("args", [])
    print(f"\n  reactions(...) accepts {len(args)} arguments:\n")
    pagination = {"first", "last", "before", "after", "offset"}
    filters = []
    for a in sorted(args, key=lambda x: x["name"]):
        ty = a["type"]
        tn = ty.get("name") or (ty.get("ofType") or {}).get("name") or ty["kind"]
        kind = "pagination" if a["name"] in pagination else "FILTER"
        if kind == "FILTER":
            filters.append(a["name"])
        print(f"    {a['name']:<28}{str(tn):<14}{kind}")

    if not filters:
        print("\n  → no server-side filtering. The download must page all 792\n"
              "    requests and filter locally, across two days of budget.")
        return

    print(f"\n  → {len(filters)} filterable arguments. The download can ask for\n"
          "    only the rows it needs instead of paging the whole table.")

    # Try the one that matters: can we ask for single-adsorbate reactions?
    if "products" in filters:
        print("\n  Testing a products filter (single CO adsorption):\n")
        limiter.acquire()
        t = post('{ reactions(first: 3, products: "~COstar") { totalCount '
                 'edges { node { Equation reactionEnergy surfaceComposition '
                 'facet } } } }', key=key)
        print("  " + json.dumps(t, indent=2)[:1400])

    print(f"\n  {limiter.report()}\n")


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
    print(f"  this probe will make at most {MAX_PROBE_REQUESTS} requests, each one")
    print("  checked against the ledger before it is sent")

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
    ap.add_argument("--budget", action="store_true",
                    help="show the request ledger and exit, spending nothing")
    ap.add_argument("--probe-filters", action="store_true",
                    help="can the server filter? decides the whole request budget")
    ap.add_argument("--plan", action="store_true",
                    help="count the clean single-adsorbate rows per adsorbate")
    args = ap.parse_args()

    if args.budget:
        show_budget()
        return

    if args.plan:
        plan()
        return

    if args.probe_filters:
        probe_filters()
        return

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
