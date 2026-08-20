"""Regression: an interrupted one-shot must NEVER be silently deleted.

Real incident (2026-08-20, Sam's Mac). Cron job ``f98f9fcf2561`` was a one-shot
carrying a real financial deadline (export a Bol seller account before the
account closed). Its dispatch was claimed at 10:03:30 by ``sfmbp.local:3395``;
the scheduler process died before ``mark_job_run`` wrote ``last_run_at``. At
10:34:27 the recovery guard DELETED the record, leaving only a markdown file in
``cron/output/f98f9fcf2561/`` that nobody reads. The job vanished from
``cron/jobs.json`` with no delivery to the origin channel — the deadline was
lost.

Contract these tests pin, for BOTH reaper sites (``claim_dispatch`` and
``get_due_jobs``):

1. The job record is NOT removed from the store.
2. It cannot re-fire (``enabled`` False, ``next_run_at`` None, not returned
   as due).
3. A loud report is delivered to the job's origin channel, naming the job id,
   name, and schedule.

These tests deliberately patch only ``cron.scheduler._deliver_result`` — a
symbol that exists both before and after the fix — so the pre-fix run fails on
the real contract (the job was deleted / nothing was delivered) rather than on
a missing helper.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.jobs import claim_dispatch, get_due_jobs, load_jobs, save_jobs


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp Hermes home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _wedged_oneshot(run_at):
    """The incident shape: dispatch claimed, run never completed."""
    return {
        "id": "f98f9fcf2561",
        "name": "Bol seller account export before closure",
        "prompt": "Export the Bol seller account data before the account closes.",
        "enabled": True,
        "schedule": {"kind": "once", "run_at": run_at},
        # claim_dispatch() already consumed the dispatch...
        "repeat": {"times": 1, "completed": 1},
        # ...but mark_job_run() never ran, so last_run_at was never written and
        # next_run_at was never cleared — the record still looks due, which is
        # exactly what dragged it into the due-scan reaper.
        "last_run_at": None,
        "next_run_at": run_at,
        # Older than the run-claim TTL (1800s), so the claim reads as stale
        # rather than an in-flight run.
        "run_claim": {"at": run_at, "by": "sfmbp.local:3395"},
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "12345"},
    }


def _past():
    return (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()


class TestInterruptedOneShotIsNotSilentlyDeleted:
    def test_claim_dispatch_does_not_delete_the_job(self, tmp_cron_dir):
        run_at = _past()
        save_jobs([_wedged_oneshot(run_at)])

        with patch("cron.scheduler._deliver_result", return_value=None):
            assert claim_dispatch("f98f9fcf2561") is False

        remaining = load_jobs()
        assert len(remaining) == 1, (
            "interrupted one-shot was DELETED by claim_dispatch — silent data "
            "loss (the f98f9fcf2561 incident)"
        )
        job = remaining[0]
        assert job["id"] == "f98f9fcf2561"
        # Disarmed: retained but cannot re-fire.
        assert job["enabled"] is False
        assert job["next_run_at"] is None
        assert job["state"] == "interrupted"

    def test_get_due_jobs_does_not_delete_the_job(self, tmp_cron_dir):
        save_jobs([_wedged_oneshot(_past())])

        with patch("cron.scheduler._deliver_result", return_value=None):
            due = get_due_jobs()

        assert due == [], "an interrupted one-shot must not be re-dispatched"
        remaining = load_jobs()
        assert len(remaining) == 1, (
            "interrupted one-shot was DELETED by the due-scan reaper — silent "
            "data loss (the f98f9fcf2561 incident)"
        )
        job = remaining[0]
        assert job["enabled"] is False
        assert job["next_run_at"] is None
        assert job["state"] == "interrupted"

    def test_removal_is_reported_to_the_origin_channel(self, tmp_cron_dir):
        """The report must reach the job's origin chat, not only a log file,
        and must name the job id, name, and schedule."""
        run_at = _past()
        save_jobs([_wedged_oneshot(run_at)])

        with patch("cron.scheduler._deliver_result", return_value=None) as deliver:
            assert claim_dispatch("f98f9fcf2561") is False

        assert deliver.call_count == 1, (
            "an interrupted one-shot was handled with NO delivery to its "
            "origin channel — silent loss"
        )
        delivered_job, delivered_text = deliver.call_args.args[:2]
        assert delivered_job["origin"] == {"platform": "telegram", "chat_id": "12345"}
        assert "f98f9fcf2561" in delivered_text
        assert "Bol seller account export before closure" in delivered_text
        assert "once at" in delivered_text and run_at in delivered_text

    def test_report_keeps_the_run_claim_forensics(self, tmp_cron_dir):
        """Disarming clears run_claim on the STORED record, but the report is
        built from a pre-mutation snapshot — otherwise the report degrades to
        'unknown by unknown' and loses the who/when that made the original
        incident diagnosable."""
        run_at = _past()
        save_jobs([_wedged_oneshot(run_at)])

        with patch("cron.scheduler._deliver_result", return_value=None) as deliver:
            assert claim_dispatch("f98f9fcf2561") is False

        text = deliver.call_args.args[1]
        assert "sfmbp.local:3395" in text
        assert run_at in text
        assert "unknown by unknown" not in text
        # ...while the persisted record is properly disarmed.
        assert load_jobs()[0]["run_claim"] is None

    def test_local_delivery_job_is_escalated_to_origin(self, tmp_cron_dir):
        """deliver=local must not mean 'lose the job in silence'."""
        job = _wedged_oneshot(_past())
        job["deliver"] = "local"
        save_jobs([job])

        with patch("cron.scheduler._deliver_result", return_value=None) as deliver:
            assert claim_dispatch("f98f9fcf2561") is False

        assert deliver.call_count == 1
        assert deliver.call_args.args[0]["deliver"] == "origin"
        assert len(load_jobs()) == 1

    def test_normally_completed_oneshot_still_retires_quietly(self, tmp_cron_dir):
        """A one-shot that DID run is a completion, not an interruption — it
        must retire as `completed` and must not emit the loud report."""
        run_at = _past()
        job = _wedged_oneshot(run_at)
        job["last_run_at"] = run_at  # the run completed normally
        save_jobs([job])

        with patch("cron.scheduler._deliver_result", return_value=None) as deliver:
            assert claim_dispatch("f98f9fcf2561") is False

        remaining = load_jobs()
        assert len(remaining) == 1
        assert remaining[0]["state"] == "completed"
        deliver.assert_not_called()
