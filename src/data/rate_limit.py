"""A request budget that survives restarts, because the penalty here is a ban.

Catalysis-Hub publishes two limits: **10 requests per minute**, and **accounts
exceeding 500 requests per day are automatically suspended**.

Those two need very different treatment, and conflating them is how accounts get
banned.

Going over ten in a minute earns an HTTP 429. That is a retryable error: sleep,
try again, nothing is lost. Going over five hundred in a day suspends the
account. That is *not* retryable -- no amount of careful error handling gets the
account back, and the download is over.

So this module is deliberately paranoid about the second one:

  IT PERSISTS.  An in-memory counter protects a single process. The daily limit
                is enforced by the server across every process you ever run, so
                the ledger has to live on disk. A crashed run that restarts with
                a fresh counter is exactly how the limit gets exceeded.

  IT USES A ROLLING WINDOW.  The published limit says "per day" without saying
                when the day starts. A rolling 24-hour window is conservative
                under every interpretation: if they reset at midnight UTC, this
                is stricter than needed; if they use a rolling window, it is
                exactly right. Guessing midnight and being wrong is a ban.

                DO NOT "OPTIMISE" THIS INTO CALENDAR-DAY ACCOUNTING. The tempting
                move is to notice that local midnight is minutes away, switch to
                counting per calendar day, and collect a second 450 immediately.
                It requires believing (a) that their reset is calendar-based
                rather than rolling, and (b) that it follows the client's
                timezone rather than the server's. Neither has been tested here,
                and both have to hold. The upside is saving a day; the downside
                is an account that cannot be recovered by retrying. A rolling
                window costs at most one day of waiting and is correct either
                way.

  IT KEEPS HEADROOM.  It stops at 90% by default. The ledger cannot see requests
                made by anything other than this code -- the web console, a
                notebook, a colleague sharing the key -- so budgeting to exactly
                500 assumes information it does not have.

  IT RECORDS BEFORE IT SENDS.  A request is written to the ledger *before* it
                goes out, not after. If the process dies mid-request the server
                still counted it, and a ledger that only records successes would
                undercount precisely when things are going wrong.

The clock is injectable so the tests can run in microseconds instead of minutes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class DailyBudgetExhausted(RuntimeError):
    """Raised instead of sending a request that could get the account suspended."""


class RateLimiter:
    """Enforces a per-minute rate and a persistent rolling daily cap."""

    def __init__(self, state_path: Path, per_minute: int = 10,
                 per_day: int = 500, headroom: float = 0.9,
                 clock=time.time, sleeper=time.sleep):
        self.state_path = Path(state_path)
        self.per_minute = per_minute
        self.per_day = per_day
        self.budget = int(per_day * headroom)
        self._clock = clock
        self._sleep = sleeper
        self._stamps: list[float] = self._load()

    # -- ledger ----------------------------------------------------------

    def _load(self) -> list[float]:
        if not self.state_path.exists():
            return []
        try:
            blob = json.loads(self.state_path.read_text(encoding="utf-8"))
            return [float(t) for t in blob.get("requests", [])]
        except (json.JSONDecodeError, ValueError, TypeError):
            # A corrupt ledger must not read as "no requests made". Refusing to
            # guess is the safe failure: worst case the user waits a day.
            raise DailyBudgetExhausted(
                f"{self.state_path} is unreadable, so the number of requests "
                "already made today is unknown. Delete it only if you are "
                "certain no requests have been made in the last 24 hours."
            )

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"requests": self._stamps,
                        "note": "rolling 24h ledger; see src/data/rate_limit.py"}),
            encoding="utf-8")

    def _prune(self) -> None:
        cutoff = self._clock() - 86_400
        self._stamps = [t for t in self._stamps if t > cutoff]

    # -- accounting ------------------------------------------------------

    def used_today(self) -> int:
        self._prune()
        return len(self._stamps)

    def remaining(self) -> int:
        return max(0, self.budget - self.used_today())

    def seconds_until_free(self) -> float:
        """How long until the daily budget allows one more request."""
        self._prune()
        if len(self._stamps) < self.budget:
            return 0.0
        return max(0.0, self._stamps[-self.budget] + 86_400 - self._clock())

    # -- the one method callers use --------------------------------------

    def acquire(self) -> None:
        """Block until a request is safe. Raise rather than exceed the day."""
        self._prune()

        if len(self._stamps) >= self.budget:
            wait = self.seconds_until_free()
            raise DailyBudgetExhausted(
                f"{len(self._stamps)} requests in the last 24 hours, against a "
                f"self-imposed budget of {self.budget} "
                f"(the published limit is {self.per_day}/day, and exceeding it "
                f"suspends the account).\n"
                f"The budget frees up in {wait / 3600:.1f} hours.\n"
                f"Downloads are resumable — re-run then and it will continue "
                f"from where it stopped.")

        # Per-minute: wait until the oldest of the last `per_minute` requests is
        # more than a minute old.
        if len(self._stamps) >= self.per_minute:
            oldest = self._stamps[-self.per_minute]
            wait = oldest + 60.0 - self._clock()
            if wait > 0:
                self._sleep(wait)

        # Recorded BEFORE the request goes out. See the module docstring.
        self._stamps.append(self._clock())
        self._save()

    def report(self) -> str:
        used = self.used_today()
        return (f"{used}/{self.budget} requests used in the last 24 h "
                f"({self.remaining()} left; published cap {self.per_day}/day)")
