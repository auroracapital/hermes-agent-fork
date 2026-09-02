"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48






def test_repeated_crashes_truncates_huge_tracebacks():
    """Full Python tracebacks can be tens of KB. The title stays one
    line (≤160 chars); the detail caps at 500 chars + ellipsis so the
    card doesn't explode visually."""
    huge = "Traceback (most recent call last):\n" + ("  File\n" * 500)
    task = _task(status="ready")
    runs = [
        _run(outcome="crashed", run_id=1, error=huge),
        _run(outcome="crashed", run_id=2, error=huge),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    d = diags[0]
    # Title only the first line, capped.
    assert "\n" not in d.title
    assert len(d.title) < 250
    # Detail contains the snippet with ellipsis.
    assert d.detail.endswith("…") or len(d.detail) < 700


# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"


# ---------------------------------------------------------------------------
# stranded_in_review
#
# Regression cover for the live incident on 2026-08-22 (card t_1165020d):
# request_review(reviewer="reviewer") wrote a non-existent profile onto
# tasks.assignee, the review dispatcher skipped the rows as non-spawnable
# every tick (a deliberately quiet bucket), and two finished healify cards
# sat ungraded in 'review' for ~4h with no signal on any surface.
# ---------------------------------------------------------------------------


def test_stranded_in_review_is_critical_when_reviewer_is_not_a_profile(
    monkeypatch,
):
    """The headline regression: an unroutable reviewer is critical
    immediately, not after 6x the threshold, because no amount of waiting
    can ever produce a worker for it."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "verifier"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_diagnostics._assignee_has_run_history",
        lambda name: False,
    )
    now = 100_000
    task = _task(status="review", assignee="reviewer", claim_lock=None)
    # 45 min: past the 30 min threshold but well under the 6x (3h) mark
    # that would make a ROUTABLE reviewer critical on age alone.
    events = [_event("review_requested", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_review"]
    assert len(stranded) == 1
    assert stranded[0].severity == "critical"
    assert stranded[0].data["assignee"] == "reviewer"
    assert stranded[0].data["assignee_is_profile"] is False
    # The operator must be told WHY it can never be claimed.
    assert "not an existing hermes profile" in stranded[0].detail.lower()


def test_stranded_in_review_real_profile_is_only_a_warning(monkeypatch):
    """A real reviewer profile that is merely slow is a warning at the
    same age — this is what keeps the rule from crying wolf on a busy
    but correctly-wired board."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "verifier"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_diagnostics._assignee_has_run_history",
        lambda name: False,
    )
    now = 100_000
    task = _task(status="review", assignee="verifier", claim_lock=None)
    events = [_event("review_requested", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_review"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["assignee_is_profile"] is True


def test_stranded_in_review_silent_under_threshold_and_when_claimed(
    monkeypatch,
):
    """No false positives: a fresh review, and a review under a live
    reviewer claim, must both stay silent even with a bad assignee."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "verifier"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_diagnostics._assignee_has_run_history",
        lambda name: False,
    )
    now = 100_000
    fresh = _task(status="review", assignee="reviewer", claim_lock=None)
    diags = kd.compute_task_diagnostics(
        fresh, [_event("review_requested", ts=now - 5 * 60)], [], now=now
    )
    assert [d for d in diags if d.kind == "stranded_in_review"] == []

    claimed = _task(
        status="review", assignee="reviewer", claim_lock="host:123"
    )
    diags = kd.compute_task_diagnostics(
        claimed, [_event("review_requested", ts=now - 45 * 60)], [], now=now
    )
    assert [d for d in diags if d.kind == "stranded_in_review"] == []


def test_stranded_in_review_ignores_non_review_status(monkeypatch):
    """A ready task with the same stale timestamps belongs to
    stranded_in_ready, not this rule. Guards against double-flagging."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "verifier"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_diagnostics._assignee_has_run_history",
        lambda name: False,
    )
    now = 100_000
    task = _task(status="ready", assignee="reviewer", claim_lock=None)
    events = [_event("review_requested", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert [d for d in diags if d.kind == "stranded_in_review"] == []


def test_stranded_in_review_is_registered():
    """The rule must be wired into the registry and the kind legend, or
    it computes nothing and the UI cannot render it."""
    assert kd._rule_stranded_in_review in kd._RULES
    assert "stranded_in_review" in kd.DIAGNOSTIC_KINDS


def test_stranded_in_review_human_lane_with_run_history_is_only_a_warning(
    monkeypatch,
):
    """A human pull-lane (e.g. 'sam') has no Hermes profile but has
    real run history. Age-based flag still fires, but it is NOT the
    'unclaimable' critical that would fire for a typo like 'reviewer'."""
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "verifier"
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_diagnostics._assignee_has_run_history",
        lambda name: name == "sam",
    )
    now = 100_000
    task = _task(status="review", assignee="sam", claim_lock=None)
    events = [_event("review_requested", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_review"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["assignee_is_profile"] is False
    assert stranded[0].data["assignee_has_lane_history"] is True
    assert "unclaimable" not in stranded[0].title.lower()
