"""Tests for the request budget.

These matter more than most tests in this repository. Every other failure here
costs a re-run; getting this wrong costs the Catalysis-Hub account, which is
suspended automatically and not by a process that can be appealed to with a
retry loop.

The clock is injected, so a 24-hour rolling window is tested in microseconds.

Run:  pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.rate_limit import DailyBudgetExhausted, RateLimiter  # noqa: E402


class FakeClock:
    """A clock the test drives, and a sleep that jumps it forward."""

    def __init__(self, t=1_000_000.0):
        self.t = t
        self.slept = 0.0

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.slept += seconds
        self.t += seconds


def limiter(tmp_path, clock, **kw):
    return RateLimiter(tmp_path / "budget.json", clock=clock.time,
                       sleeper=clock.sleep, **kw)


def test_headroom_is_kept_below_the_published_cap():
    """Budgeting to exactly the cap assumes nothing else uses the key."""
    rl = RateLimiter(Path("/tmp/never-written"), per_day=500, headroom=0.9)
    assert rl.budget == 450


def test_per_minute_limit_forces_a_wait(tmp_path):
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=10, per_day=500)

    for _ in range(10):
        rl.acquire()
    assert clock.slept == 0.0, "the first ten should not have waited"

    rl.acquire()
    assert clock.slept == pytest.approx(60.0), "the eleventh must wait a minute"


def test_daily_budget_raises_rather_than_exceeding(tmp_path):
    """The important one. Over the daily cap the account is suspended, so this
    must refuse rather than sleep-and-retry."""
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=1000, per_day=10, headroom=1.0)

    for _ in range(10):
        rl.acquire()

    with pytest.raises(DailyBudgetExhausted, match="suspends the account"):
        rl.acquire()


def test_the_ledger_survives_a_restart(tmp_path):
    """An in-memory counter protects one process. The server counts them all."""
    clock = FakeClock()
    first = limiter(tmp_path, clock, per_minute=1000, per_day=10, headroom=1.0)
    for _ in range(10):
        first.acquire()

    # A brand-new object, as if the process had crashed and been restarted.
    second = limiter(tmp_path, clock, per_minute=1000, per_day=10, headroom=1.0)
    assert second.used_today() == 10
    with pytest.raises(DailyBudgetExhausted):
        second.acquire()


def test_budget_frees_up_after_24_hours(tmp_path):
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=1000, per_day=4, headroom=1.0)
    for _ in range(4):
        rl.acquire()
    with pytest.raises(DailyBudgetExhausted):
        rl.acquire()

    clock.t += 86_401
    assert rl.used_today() == 0
    rl.acquire()


def test_rolling_window_not_calendar_day(tmp_path):
    """Requests expire individually, 24 h after each was made."""
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=1000, per_day=3, headroom=1.0)
    rl.acquire()
    clock.t += 40_000
    rl.acquire()
    rl.acquire()
    assert rl.used_today() == 3

    clock.t += 50_000          # the first is now >24 h old, the others are not
    assert rl.used_today() == 2


def test_a_corrupt_ledger_refuses_rather_than_assuming_zero(tmp_path):
    """Reading a damaged ledger as 'no requests yet' is the dangerous default."""
    path = tmp_path / "budget.json"
    path.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(DailyBudgetExhausted, match="unreadable"):
        RateLimiter(path)


def test_the_request_is_recorded_before_it_is_sent(tmp_path):
    """If the process dies mid-request the server still counted it."""
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=1000, per_day=10)
    rl.acquire()

    on_disk = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert len(on_disk["requests"]) == 1


def test_seconds_until_free_is_reported(tmp_path):
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_minute=1000, per_day=2, headroom=1.0)
    rl.acquire()
    rl.acquire()
    assert rl.seconds_until_free() == pytest.approx(86_400.0)
    clock.t += 86_000
    assert rl.seconds_until_free() == pytest.approx(400.0)


def test_report_mentions_the_published_cap(tmp_path):
    clock = FakeClock()
    rl = limiter(tmp_path, clock, per_day=500)
    rl.acquire()
    text = rl.report()
    assert "1/450" in text and "500/day" in text
