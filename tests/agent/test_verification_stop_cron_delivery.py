"""Regression: an unattended cron run must never have its report replaced by
the verify-on-stop loop's own closing sentence.

Incident (2026-08-20, job f98f9fcf2561 "Bol seller account wind-down"): the job
produced a real report as its final answer. Because the run had touched a temp
file, verify-on-stop fired, discarded that report to an interim message, ran one
more internal turn, and cron delivered THAT turn's text --
"Temporary diagnostic script removed. No workspace code remains changed, so no
test suite was applicable." -- to Sam as if it were the Bol findings. Verified
in state.db: session cron_f98f9fcf2561_20260820_100024 holds both assistant
messages, the report (id 1283149) and the delivered sentence (id 1283156).

The bug is surface-specific, not a cross-job mixup: on cron the turn's
``final_response`` IS the delivered artifact, so swapping it corrupts the
delivery. Sam's live config has ``agent.verify_on_stop: true``, so the gate must
hold against an explicit opt-in and against the env override, both of which are
statements about interactive coding sessions.
"""

import pytest

from agent.verification_stop import verify_on_stop_enabled


@pytest.fixture
def clear_verify_env(monkeypatch):
    for var in (
        "HERMES_VERIFY_ON_STOP",
        "HERMES_PLATFORM",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_cron_platform_disables_verify_on_stop_despite_explicit_config(clear_verify_env):
    """The exact live shape: agent.verify_on_stop true + platform 'cron'."""
    config = {"agent": {"verify_on_stop": True}}
    # Control: on an interactive surface the same config still opts IN, so this
    # test cannot pass by disabling the feature everywhere.
    assert verify_on_stop_enabled(config, platform="cli") is True
    # The regression itself.
    assert verify_on_stop_enabled(config, platform="cron") is False


def test_cron_platform_beats_env_override(clear_verify_env):
    """HERMES_VERIFY_ON_STOP=1 must not re-enable the swap on a scheduled run."""
    clear_verify_env.setenv("HERMES_VERIFY_ON_STOP", "1")
    assert verify_on_stop_enabled({"agent": {}}, platform="cli") is True
    assert verify_on_stop_enabled({"agent": {}}, platform="cron") is False


def test_cron_session_env_also_gates(clear_verify_env):
    """Secondary net: a cron identity arriving via env, not the platform arg."""
    clear_verify_env.setenv("HERMES_PLATFORM", "cron")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": True}}) is False


def test_non_cron_surfaces_are_untouched(clear_verify_env):
    """Negative control: the gate narrows nothing outside cron."""
    config = {"agent": {"verify_on_stop": True}}
    for surface in ("cli", "tui", "desktop", "telegram", "kanban", ""):
        assert verify_on_stop_enabled(config, platform=surface) is True, surface


def test_production_call_site_passes_the_agent_platform():
    """The loop must actually forward agent.platform, or the gate is dead code.

    Guards the wiring, not the policy: a fix that lives only in
    verification_stop.py and is never reached from conversation_loop.py would
    pass every test above while the incident still reproduces.
    """
    import inspect
    import re

    from agent import conversation_loop

    src = inspect.getsource(conversation_loop)
    call = re.search(
        r"verify_on_stop_enabled\(\s*platform=getattr\(\s*agent,\s*[\"']platform[\"']",
        src,
    )
    assert call, "conversation_loop must call verify_on_stop_enabled(platform=agent.platform)"
