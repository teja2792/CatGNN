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

THE PRE-FILTER IS CASE-INSENSITIVE SUBSTRING MATCHING, AND IT LIES
-------------------------------------------------------------------
Counting rows per adsorbate with `products: "~<A>star"` gave, among others,
13,364 for H and 34,709 for O. Both are mostly pollution:

    ~Hstar    matched  Rhstar          rhodium, returned when asked for hydrogen
    ~Nstar    matched  Znstar          zinc, returned when asked for nitrogen
    ~OHstar   matched  CH3CH2OHstar    ethanol, returned when asked for hydroxyl
    ~OOHstar  matched  COOHstar        carboxyl
    ~Ostar    matched  HOstar, COstar, CH3Ostar, ...

Only CO, C and NO came back clean. The counts looked entirely plausible, which is
the dangerous part -- scoping a dataset from them yields a large, well-formed
table of the wrong thing.

So the server filter narrows the download and decides nothing. Membership is
decided locally in src/data/adsorption.py, by exact match on the parsed JSON, and
the adsorbate label is read from the ROW rather than from the query that fetched
it. The contaminated rows above are kept as regression tests.

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

GEOMETRY IS NOT AVAILABLE THROUGH THIS API AT ANY REASONABLE COST
------------------------------------------------------------------
Measured, not estimated:

    metadata only                              asked 200  ->  got 200   22 kB
    + systems (Formula, energy)                asked 200  ->  got 200   46 kB
    + systems with InputFile (the geometry)    asked  20  ->  got   1    7 kB

Asking for the geometry caps the response at ONE ROW per request. So the 3,554 CO
rows already downloaded would cost 3,554 requests -- eight days of a 450/day
budget, for a single adsorbate.

That is a wall, not a slow path, and it means the GraphQL API is the wrong tool
for the part of Phase 7 that matters. The inspection showed geometry carries 43%
of the variance in CO binding; an API that serves it one row at a time cannot
supply it.

THE BULK ROUTE
--------------
Catalysis-Hub publishes a Python package, CatHub, whose CathubSQL class talks to
the database directly rather than through GraphQL:

    from cathub.cathubsql import CathubSQL
    db = CathubSQL()
    df = db.get_dataframe(pub_id="YohannesCombined2023", include_atoms=True)

That returns reaction energies AND atoms together, per publication, in one
operation. Since 63.4% of the CO rows come from one publication and 83% from two,
two such calls would cover most of the dataset -- against 3,554 API requests for
the same thing.

TESTED, AND IT IS ALSO CLOSED:

    psycopg2.OperationalError: connection to server at
    "catalysishub.cx2awgo40dih.us-west-2.rds.amazonaws.com", port 5432 failed:
    FATAL: password authentication failed for user "apiuser"

CatHub ships hardcoded public read-only Postgres credentials, and they no longer
authenticate. That is consistent with the API key requirement appearing on the
GraphQL side: direct database access looks to have been closed at the same time.
The package installs and imports fine, so the failure only shows up at connect
time -- which is why it was worth five minutes to test rather than assume.

So BOTH routes to geometry are shut:

    GraphQL + InputFile     1 row per request  ->  8 days for one adsorbate
    CathubSQL direct        credentials rejected

This is a genuine external constraint rather than a thing to engineer around, and
it bounds what Phase 7 can be. What it does NOT bound is the finding already in
hand: the ceiling measurement needs no structures at all, and it is the strongest
statement this repository has made. Composition tops out at R2 = 0.57 on CO
binding because 43% of the variance is geometry -- which is now demonstrated
twice over, once by the variance decomposition and once by the fact that the
geometry is hard to get.

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

OUT_DIR = RAW / "catalysis_hub"
ROWS_FILE = OUT_DIR / "adsorption.jsonl"
MANIFEST = OUT_DIR / "manifest.json"
# Alongside the budget ledger: both describe progress, not data, and
# neither should be lost by clearing data/raw.
STATE_FILE = REPO / "data" / "cache" / "catalysis_hub_state.json"


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
from src.data.adsorption import ADSORBATES  # noqa: E402


def probe_structures() -> None:
    """What does asking for geometries cost, in requests?

    This is now the question the phase turns on. The inspection put a ceiling of
    R2 = 0.57 on any model that knows only the surface and facet: 43% of the
    variance in CO binding is WHERE on the surface the molecule sits, and the
    `sites` column records that only as an opaque index (site1 ... site47) that
    carries nothing usable and does not transfer between surfaces.

    So unlike band gap -- where structure was worth 2-3% over composition and the
    phase would have survived without it -- here geometry is the target's dominant
    variable. Without it there is no Phase 7 worth doing, only a composition
    baseline stuck at 0.81 eV.

    Three requests: one page without structures, one with, one larger with. The
    comparison gives the real rows-per-request and therefore the real cost of the
    3,554 rows already downloaded.
    """
    from src.config import get_catalysis_hub_key
    from src.data.rate_limit import RateLimiter

    key = get_catalysis_hub_key()
    limiter = RateLimiter(BUDGET_FILE)
    print(f"\n{'=' * 76}\n  What does fetching geometries cost?\n{'=' * 76}")
    print(f"\n  {limiter.report()}   (this probe uses 3)")

    print(f"\n  {'query':<44}{'asked':>7}{'got':>6}{'kB':>9}")
    print("  " + "-" * 66)

    trials = [
        ("metadata only", 200, "id Equation reactionEnergy"),
        ("+ systems (Formula, energy)", 200,
         "id reactionEnergy systems { Formula energy }"),
        ("+ systems with InputFile (the geometry)", 20,
         'id reactionEnergy systems { Formula energy InputFile(format: "json") }'),
    ]
    results = []
    for label, want, fields in trials:
        limiter.acquire()
        r = post('{ reactions(first: %d, reactants: "~COgas", products: "~COstar") '
                 '{ edges { node { %s } } } }' % (want, fields), key=key)
        node = (r.get("data") or {}).get("reactions")
        if not node:
            print(f"  {label:<44}{want:>7}{'error':>6}")
            print("    " + json.dumps(r)[:300])
            continue
        got = len(node.get("edges", []))
        kb = len(json.dumps(r)) / 1024
        print(f"  {label:<44}{want:>7}{got:>6}{kb:>9.1f}")
        results.append((label, want, got, kb))

    if len(results) == 3:
        _, _, got_geom, kb_geom = results[2]
        if got_geom:
            per_row = kb_geom / got_geom
            need = -(-3554 // got_geom)
            print(f"\n  {per_row:.1f} kB per row with geometry")
            print(f"  {3554:,} CO rows / {got_geom} per request = {need} requests")
            print(f"  = {need / 450:.0%} of one day's budget, and "
                  f"{3554 * per_row / 1024:.0f} MB on disk")
            if need <= limiter.remaining():
                print("\n  → affordable today. Geometry is the 43% of the variance")
                print("    that composition cannot reach, so this is the request")
                print("    the rest of the phase depends on.")
            else:
                print(f"\n  → needs {need} requests, {limiter.remaining()} left today.")
                print("    The download is resumable, so this is a two-session job")
                print("    rather than a problem.")

    print(f"\n  {limiter.report()}\n")


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
    ap.add_argument("--adsorbates", nargs="*", default=None,
                    help=f"which to fetch (default: all of {' '.join(ADSORBATES)})")
    ap.add_argument("--page", type=int, default=200,
                    help="rows per request; the server caps this at 200")
    ap.add_argument("--probe-structures", action="store_true",
                    help="measure what fetching geometries costs (3 requests)")
    ap.add_argument("--geometries", action="store_true",
                    help="fetch slab geometries for the designed sample "
                         "(1 request per row; resumable; --dry-run first)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the sample and its cost without spending a request")
    ap.add_argument("--surfaces", type=int, default=40,
                    help="how many surfaces in the geometry sample")
    ap.add_argument("--sites", type=int, default=10,
                    help="how many rows to take from each qualifying surface")
    ap.add_argument("--min-sites", type=int, default=None,
                    help="how many sites a surface needs to QUALIFY (default: "
                         "same as --sites). Hold this fixed while raising "
                         "--sites to widen the sample without stranding rows.")
    ap.add_argument("--allow-strand", action="store_true",
                    help="proceed even if the sample abandons rows already paid for")
    args = ap.parse_args()

    if args.budget:
        show_budget()
        return

    if args.geometries:
        geometries(n_surfaces=args.surfaces, sites_per_surface=args.sites,
                   dry_run=args.dry_run, min_sites=args.min_sites,
                   allow_strand=args.allow_strand)
        return

    if args.probe_structures:
        probe_structures()
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

    download(adsorbates=args.adsorbates or ADSORBATES, page=args.page)


GEOM_FILE = OUT_DIR / "geometries.jsonl"
GEOM_MANIFEST = OUT_DIR / "geometries_manifest.json"
GEOM_FAILED = REPO / "data" / "cache" / "catalysis_hub_failed.json"


def post_or_none(query: str, key: str, attempts: int = 3, limiter=None):
    """One request, retried on a SERVER fault, returning None if it will not work.

    Exists because `post()` calls sys.exit on any HTTP error, and that turned one
    transient HTTP 500 into the loss of a 60-minute download that had already
    spent budget. A row-at-a-time fetch makes ~600 requests over an hour; the
    probability of at least one 5xx in that window is not small, and aborting is
    the wrong response to it.

    Distinguishes the two cases that matter:
      * 5xx and network errors are the SERVER's problem and usually transient,
        so they are retried with backoff.
      * 4xx means the request itself is wrong -- a bad key, a row that does not
        exist -- and retrying it would burn budget to get the same answer, so it
        returns immediately.

    Every attempt is a request and is charged to the ledger BEFORE it is made,
    because the server counts attempts, not successes.
    """
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json",
               "User-Agent": "CatGNN/0.1 (github.com/teja2792/CatGNN)",
               "X-API-Key": key}
    body = json.dumps({"query": query}).encode("utf-8")

    for attempt in range(attempts):
        if limiter is not None:
            limiter.acquire()          # may raise DailyBudgetExhausted; correct
        try:
            req = urllib.request.Request(API, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return None            # our fault; retrying cannot help
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = str(e)
        if attempt < attempts - 1:
            import time as _t
            _t.sleep(5.0 * (attempt + 1) ** 2)
        else:
            print(f"      giving up on this row after {attempts} attempts ({last})")
    return None


def geometries(n_surfaces: int = 40, sites_per_surface: int = 10,
               dry_run: bool = False, min_sites: int | None = None,
               allow_strand: bool = False) -> None:
    """Fetch slab geometries, one request per row, for a deliberately chosen sample.

    This is the expensive mode and the one the phase turns on. The metadata
    download got 200 rows a request; geometry comes back one row at a time, so
    every row costs a request against a 450/day budget. 3,554 rows would be eight
    days. 400 rows is one.

    Which 400 is a scientific decision, not a sampling detail, and it is made in
    src/data/geometry_sample.py where it can be tested without spending anything.
    The short version: 40 surfaces x 10 sites, PBE only. Enough surfaces to hold
    some out, enough sites on each that within-surface variation — the 43% of the
    variance composition cannot reach — is present to be learned and to be tested.

    Resumable by row id. A run stopped by the daily budget re-derives the same
    sample tomorrow (the selection is deterministic), skips what is already on
    disk, and continues. Nothing is re-fetched, because re-fetching is not free.
    """
    from src.config import get_catalysis_hub_key, key_fingerprint
    from src.data.geometry_sample import (
        FUNCTIONAL, decode_reaction_id, describe, load_rows, select)
    from src.data.rate_limit import DailyBudgetExhausted, RateLimiter

    if not ROWS_FILE.exists():
        print(f"\n  {ROWS_FILE.relative_to(REPO)} does not exist.\n"
              "  Run the metadata download first — the sample is chosen from it.\n")
        sys.exit(1)

    rows = load_rows(ROWS_FILE)
    sample = select(rows, n_surfaces=n_surfaces, sites_per_surface=sites_per_surface,
                    min_sites=min_sites)
    desc = describe(sample)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    have = set()
    if GEOM_FILE.exists():
        with GEOM_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    have.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [r for r in sample if r["id"] not in have]
    # Rows already paid for that this sample no longer wants. Loud, because the
    # selector picks surfaces at evenly spaced ranks: changing --surfaces
    # reshuffles WHICH ranks are picked, so a wider sample is not automatically a
    # superset of a narrower one. Stranding 140 rows is a third of a budget day
    # spent on data that leaves the study, and nothing else would report it.
    stranded = have - {r["id"] for r in sample}
    limiter = RateLimiter(BUDGET_FILE)

    print(f"\n{'=' * 76}\n  Catalysis-Hub: slab geometries for the chosen sample"
          f"\n{'=' * 76}")
    print(f"\n  The sample ({len(rows):,} clean rows on disk -> {desc['rows']} chosen)")
    print(f"    {desc['surfaces']} surfaces x >={desc['sites_per_surface_min']} sites"
          f"   adsorbate {'/'.join(desc['adsorbates'])}   functional {FUNCTIONAL}")
    print(f"    median within-surface energy spread "
          f"{desc['median_within_surface_spread_eV']:.2f} eV")
    print("      ^ this is the signal being bought. A model that only knows the")
    print("        surface predicts one number for all 10 sites and cannot get")
    print("        inside this spread at all. The exact ceiling depends on the")
    print("        sample and is measured by scripts/build_slab_graphs.py --")
    print("        1.189 eV on the first 400 rows, against 0.806 eV on the full")
    print("        table, because a site-rich sample is HARDER for composition.")

    print("\n  What the sample is NOT (state this, do not discover it later)")
    for label, field in (("publications", "publications"), ("functionals", "functionals")):
        items = desc[field]
        print(f"    {label:<14}{len(items)}: "
              f"{', '.join(f'{k} ({v})' for k, v in items.items())}")
    if len(desc["publications"]) == 1:
        print("    -> one publication, so NO publication-disjoint split is possible")
        print("       here. The surface-disjoint split is, and is the honest test.")

    if stranded:
        print(f"\n  REFUSING TO RUN: {len(stranded)} rows already on disk are not in")
        print("    this sample. They cost one request each against a 450/day cap,")
        print("    and this selection would abandon them.")
        print("\n    Two causes, both fixable without spending anything:")
        print("      * --surfaces changed. Surfaces are picked at evenly spaced")
        print("        ranks, so a wider sample is NOT automatically a superset.")
        print("      * --sites raised without --min-sites. Raising --sites also")
        print("        raises the qualifying bar and drops smaller groups.")
        print("\n    To widen safely, keep --min-sites at the value already used")
        print(f"    ({sites_per_surface if min_sites is None else min_sites} here) and raise --sites:")
        print(f"      --surfaces {n_surfaces} --min-sites 10 --sites {sites_per_surface + 5}")
        print("\n    Override with --allow-strand only if abandoning them is intended.")
        if not allow_strand:
            sys.exit(1)
        print("\n    --allow-strand given; continuing anyway.")

    print(f"\n  Cost\n    {len(sample)} rows, {len(have)} already fetched, "
          f"{len(todo)} to go, 1 request each")
    print(f"    {limiter.report()}")
    if len(todo) > limiter.remaining():
        print(f"    -> {len(todo)} needed, {limiter.remaining()} left today. "
              "This will stop on the budget and resume.")

    if dry_run:
        print("\n  --dry-run: nothing fetched, no request spent.\n")
        return
    if not todo:
        print("\n  Sample complete. Nothing to fetch.\n")
        _write_geometry_manifest(desc, len(have), n_surfaces, sites_per_surface)
        return

    key = get_catalysis_hub_key()
    print(f"\n  key {key_fingerprint(key)}")
    print(f"\n  {'done':>6}{'kept':>7}{'no geom':>9}{'elapsed':>10}   note")
    print("  " + "-" * 52)

    import time
    t0 = time.time()
    done = kept = nogeom = 0
    stopped = None
    consecutive_failures = 0
    failed: list[str] = []

    with GEOM_FILE.open("a", encoding="utf-8") as out:
        for r in todo:
            rid = decode_reaction_id(r["id"])
            if rid is None:
                nogeom += 1
                continue

            q = ('{ reactions(id: %d) { edges { node { id reactionEnergy '
                 'systems { Formula energy InputFile(format: "json") } } } } }' % rid)
            try:
                resp = post_or_none(q, key, limiter=limiter)
            except DailyBudgetExhausted as e:
                stopped = e
                break
            done += 1

            if resp is None:
                # The row is skipped, not fatal, and recorded so a later run can
                # try it again. One bad row must not end a download that has
                # already spent an hour of budget.
                failed.append(r["id"])
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\n  10 consecutive failures -- the API looks down.")
                    print("  Stopping so the rest of the budget is not spent on")
                    print("  errors. Re-run later; progress so far is on disk.")
                    stopped = "api_down"
                    break
                continue
            consecutive_failures = 0

            node = (resp.get("data") or {}).get("reactions")
            edges = (node or {}).get("edges") or []
            if not edges:
                nogeom += 1
            else:
                systems = edges[0]["node"].get("systems") or []
                # Keep the row even with no systems, so a resumed run does not
                # pay for it again to rediscover that it has no geometry.
                if not systems:
                    nogeom += 1
                else:
                    kept += 1
                out.write(json.dumps({
                    "id": r["id"], "reaction_id": rid,
                    "reactionEnergy": r["reactionEnergy"],
                    "surfaceComposition": r.get("surfaceComposition"),
                    "facet": r.get("facet"), "sites": r.get("sites"),
                    "dftFunctional": r.get("dftFunctional"),
                    "pubId": r.get("pubId"), "adsorbate": r.get("adsorbate"),
                    "systems": systems,
                }) + "\n")

            if done % 25 == 0:
                out.flush()
                el = time.time() - t0
                left = (len(todo) - done) * el / done
                print(f"  {done:>6}{kept:>7}{nogeom:>9}{el / 60:>9.1f}m"
                      f"   ~{left / 60:.0f}m left")

    el = time.time() - t0
    print(f"  {done:>6}{kept:>7}{nogeom:>9}{el / 60:>9.1f}m   "
          f"{'budget reached' if stopped else 'sample complete'}")

    if failed:
        GEOM_FAILED.write_text(json.dumps(sorted(set(failed)), indent=2),
                               encoding="utf-8")
        print(f"\n  {len(failed)} rows failed and were skipped; ids recorded in")
        print(f"  {GEOM_FAILED.relative_to(REPO)}. Re-running retries them.")

    total = len(have) + kept
    print(f"\n  {total} rows with geometry in {GEOM_FILE.relative_to(REPO)}")
    print(f"  {limiter.report()}")
    _write_geometry_manifest(desc, total, n_surfaces, sites_per_surface)

    if stopped == "api_down":
        print("\n  Stopped because the API was failing, not because the budget\n"
              "  ran out. Re-run when it recovers.\n")
    elif stopped:
        print(f"\n{stopped}\n")
        print("  Re-run tomorrow. The sample is deterministic and rows already\n"
              "  on disk are skipped, so nothing is paid for twice.\n")
    else:
        print("\n  Done. Next: build slab graphs. Note MAX_SITES=30 is a bulk-cell\n"
              "  cap — a slab plus adsorbate exceeds it routinely, and silently\n"
              "  dropping the large ones would bias the set toward small surfaces.\n")


def _write_geometry_manifest(desc, total, n_surfaces, sites_per_surface) -> None:
    """The record of what was sampled and, more importantly, what it excludes."""
    GEOM_MANIFEST.write_text(json.dumps({
        "source": API,
        "rows_with_geometry": total,
        "design": {
            "surfaces": n_surfaces, "sites_per_surface": sites_per_surface,
            "rule": "src/data/geometry_sample.select — size-sorted groups picked "
                    "at evenly spaced ranks, not the largest, so the sample is "
                    "not only the most-studied surfaces",
            "why": "1 request per row against a 450/day budget. 40x10 keeps "
                   "enough surfaces to hold some out AND enough sites on each "
                   "to contain the within-surface variation being tested.",
        },
        "observed": desc,
        "limitations": [
            "PBE only. Removes the 23-functional confound by construction and "
            "costs the ability to say anything about other functionals.",
            "One publication, so no publication-disjoint split is possible. The "
            "surface-disjoint split is the honest generalisation test here.",
            "400 of 3,554 clean CO rows. A sample size, stated, not hidden.",
        ],
    }, indent=2), encoding="utf-8")


def download(adsorbates, page: int = 200) -> None:
    """Page the wanted rows, keep only the clean ones, and survive being stopped.

    Resumable by design rather than as a nicety. The budget is 450 requests a day
    and a full pass may need more, so a run that could not continue where it left
    off would spend the next day re-fetching what it already had.

    State is per-adsorbate: a cursor, whether it finished, and running counts.
    Rows are appended to JSONL as they arrive, so an interrupted run keeps
    everything it had already paid for.
    """
    from src.config import get_catalysis_hub_key, key_fingerprint
    from src.data.adsorption import (is_metal_atom_adsorption, which_adsorbate)
    from src.data.rate_limit import DailyBudgetExhausted, RateLimiter

    key = get_catalysis_hub_key()
    limiter = RateLimiter(BUDGET_FILE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    cursors = state.setdefault("cursors", {})
    finished = set(state.setdefault("finished", []))
    counts = state.setdefault("counts", {})

    # Rows already on disk, so a resumed run cannot duplicate them. Reading the
    # file is cheaper than re-fetching and is the only source of truth.
    seen_ids = set()
    if ROWS_FILE.exists():
        with ROWS_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"\n{'=' * 76}\n  Catalysis-Hub: single-adsorbate chemisorption energies"
          f"\n{'=' * 76}")
    print(f"\n  key {key_fingerprint(key)}   {limiter.report()}")
    print(f"  {len(seen_ids):,} rows already on disk"
          f"   finished: {sorted(finished) or 'none'}")
    print(f"\n  {'adsorbate':<10}{'fetched':>9}{'kept':>8}{'rejected':>10}"
          f"{'requests':>10}   state")
    print("  " + "-" * 62)

    fields = ("id Equation reactants products reactionEnergy surfaceComposition "
              "chemicalComposition facet sites coverages dftCode dftFunctional "
              "pubId")
    stopped_early = False

    with ROWS_FILE.open("a", encoding="utf-8") as out:
        for ads in adsorbates:
            if ads in finished:
                print(f"  {ads:<10}{'':>9}{counts.get(ads, 0):>8}{'':>10}{'':>10}"
                      f"   already complete")
                continue

            fetched = kept = rejected = requests = 0
            cursor = cursors.get(ads)

            while True:
                try:
                    limiter.acquire()
                except DailyBudgetExhausted as e:
                    stopped_early = True
                    print(f"  {ads:<10}{fetched:>9}{kept:>8}{rejected:>10}"
                          f"{requests:>10}   budget reached")
                    print(f"\n{e}\n")
                    break

                after = f', after: "{cursor}"' if cursor else ""
                q = ('{ reactions(first: %d%s, reactants: "~%sgas", '
                     'products: "~%sstar") { pageInfo { hasNextPage endCursor } '
                     'edges { node { %s } } } }'
                     % (page, after, ads, ads, fields))
                requests += 1
                node = (post(q, key=key).get("data") or {}).get("reactions")
                if not node:
                    print(f"  {ads:<10}{fetched:>9}{kept:>8}{rejected:>10}"
                          f"{requests:>10}   API returned nothing")
                    break

                for edge in node.get("edges", []):
                    row = edge["node"]
                    fetched += 1

                    # The label comes from the ROW. A query asking for "H" returns
                    # rhodium depositions, and trusting the query would file them
                    # under hydrogen.
                    found = which_adsorbate(row.get("reactants"), row.get("products"))
                    if found is None or row.get("id") in seen_ids:
                        rejected += 1
                        continue

                    row["adsorbate"] = found
                    row["is_metal_atom_adsorption"] = is_metal_atom_adsorption(found)
                    row["fetched_under"] = ads
                    out.write(json.dumps(row) + "\n")
                    seen_ids.add(row["id"])
                    kept += 1

                info = node.get("pageInfo") or {}
                cursor = info.get("endCursor")
                cursors[ads] = cursor
                if not info.get("hasNextPage"):
                    finished.add(ads)
                    break

            out.flush()
            counts[ads] = counts.get(ads, 0) + kept
            state["cursors"], state["finished"] = cursors, sorted(finished)
            state["counts"] = counts
            STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

            if not stopped_early:
                print(f"  {ads:<10}{fetched:>9}{kept:>8}{rejected:>10}"
                      f"{requests:>10}   "
                      f"{'complete' if ads in finished else 'partial'}")
            else:
                break

    total = len(seen_ids)
    print(f"\n  {total:,} clean single-adsorbate rows in "
          f"{ROWS_FILE.relative_to(REPO)}")
    print(f"  {limiter.report()}")

    MANIFEST.write_text(json.dumps({
        "source": API,
        "key_fingerprint": key_fingerprint(key),
        "rows": total,
        "adsorbates_requested": list(adsorbates),
        "finished": sorted(finished),
        "selection": "src/data/adsorption.is_single_adsorbate — exact match on "
                     "parsed JSON, NOT the server's substring filter",
        "note": "The ~ filter is a case-insensitive substring match and returns "
                "rhodium for hydrogen, zinc for nitrogen, ethanol for hydroxyl. "
                "It narrows the download; it decides nothing.",
    }, indent=2), encoding="utf-8")

    if stopped_early:
        print("\n  Stopped on the daily budget. Re-run tomorrow — it resumes "
              "from the saved cursors.\n")
    else:
        print("\n  Done. Next: measure what fraction have usable structures.\n")


if __name__ == "__main__":
    main()
