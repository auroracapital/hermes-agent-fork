"""Tests for process wait timeout-result clarity (not-an-error semantics)."""

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return ProcessRegistry()


def _spawn_sleeper(registry, notify=False):
    session = registry.spawn_local("sleep 30", cwd="/tmp", task_id="t-waitclar")
    session.notify_on_complete = notify
    return session.id


class TestWaitTimeoutClarity:
    def test_wait_timeout_marks_process_running(self, registry):
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "timeout"
            assert r["process_running"] is True
            assert "not an error" in r["timeout_note"]
            assert "Uptime" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_suggests_notify_when_unset(self, registry):
        sid = _spawn_sleeper(registry, notify=False)
        try:
            r = registry.wait(sid, timeout=1)
            assert "notify_on_complete=true" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_defers_to_notify_when_set(self, registry):
        sid = _spawn_sleeper(registry, notify=True)
        try:
            r = registry.wait(sid, timeout=1)
            assert "you will be notified on exit" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_clamped_wait_keeps_clamp_note_and_running_semantics(self, registry, monkeypatch):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert "clamped" in r["timeout_note"]
            assert "not an error" in r["timeout_note"]
            assert r["process_running"] is True
        finally:
            registry.kill_process(sid)

    def test_exited_process_unaffected(self, registry):
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=10)
        assert r["status"] == "exited"
        assert "process_running" not in r


class TestWaitCeilingIsIndependentOfForegroundDefault:
    """The wait cap must not collapse onto the FOREGROUND default.

    Regression: `max_timeout = default_timeout` made the ceiling for a
    background wait equal to the foreground default. A job started with
    background=true specifically to outlive that default then could not be
    awaited past it, and every clamp surfaced as status="timeout" on a run that
    was perfectly healthy. Observed on a 313s test suite against the 180s
    default: the suite could never be watched to completion in one call.

    Contract: an EXPLICIT TERMINAL_TIMEOUT still wins in both directions, so an
    operator who deliberately lowers it keeps that ceiling. Only the unset case
    is lifted to PROCESS_WAIT_MAX_TIMEOUT.

    These assert the clamp DECISION via timeout_note rather than blocking for
    real seconds — the process is killed immediately after, so the tests stay
    fast.
    """

    def test_unset_terminal_timeout_allows_a_wait_past_the_foreground_default(
        self, registry, monkeypatch
    ):
        # Pre-fix: ceiling is 180, so 400 clamps. Post-fix: ceiling is 600.
        # Uses a short-lived process so the assertion is about the CLAMP DECISION,
        # not about actually blocking for 400 real seconds.
        monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=400)
        assert r["status"] == "exited"
        assert "clamped" not in (r.get("timeout_note") or "")

    def test_an_explicit_terminal_timeout_still_lowers_the_ceiling(
        self, registry, monkeypatch
    ):
        """Operator intent is preserved — this is what the raised cap must not eat."""
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert "clamped" in r["timeout_note"]
            assert "1s" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_ceiling_still_clamps_beyond_the_wait_max(self, registry, monkeypatch):
        """The cap is raised, not removed — an absurd wait is still bounded."""
        from tools.process_registry import PROCESS_WAIT_MAX_TIMEOUT

        monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=PROCESS_WAIT_MAX_TIMEOUT + 1000)
        # The process exits immediately; what matters is that the request was
        # clamped to the ceiling rather than honoured verbatim.
        assert r["status"] == "exited"
        assert str(PROCESS_WAIT_MAX_TIMEOUT) in (r.get("timeout_note") or "")

    def test_env_override_is_honoured(self, monkeypatch):
        """PROCESS_WAIT_MAX_TIMEOUT is operator-tunable like FOREGROUND_MAX_TIMEOUT."""
        import importlib

        import tools.process_registry as pr

        monkeypatch.setenv("PROCESS_WAIT_MAX_TIMEOUT", "1234")
        reloaded = importlib.reload(pr)
        try:
            assert reloaded.PROCESS_WAIT_MAX_TIMEOUT == 1234
        finally:
            monkeypatch.delenv("PROCESS_WAIT_MAX_TIMEOUT", raising=False)
            importlib.reload(pr)

    def test_junk_env_falls_back_to_the_default(self, monkeypatch):
        import importlib

        import tools.process_registry as pr

        for junk in ("not-a-number", "0", "-5"):
            monkeypatch.setenv("PROCESS_WAIT_MAX_TIMEOUT", junk)
            reloaded = importlib.reload(pr)
            assert reloaded.PROCESS_WAIT_MAX_TIMEOUT == 600, junk
        monkeypatch.delenv("PROCESS_WAIT_MAX_TIMEOUT", raising=False)
        importlib.reload(pr)
