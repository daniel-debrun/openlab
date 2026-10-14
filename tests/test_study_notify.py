"""Notifications: a dry run writes the message to the outbox instead of sending it."""

import json

from openlab.study.notify import deliver


def test_a_dry_run_alert_goes_to_the_outbox_not_the_mail(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_ALERT_DRY_RUN", "1")
    monkeypatch.setenv("OPENLAB_SMTP_HOST", "smtp.invalid")     # configured, but a dry run never sends
    monkeypatch.setenv("OPENLAB_ALERT_TO", "person@example.com")
    outbox = tmp_path / "outbox.jsonl"
    said = deliver("study s", [{"alert": r"\mathrm{x} > 3", "interpretation": "x is 5"}], outbox)
    assert "dry run" in said
    rec = json.loads(outbox.read_text())
    assert rec["to"] == ["person@example.com"] and "1 fired in study s" in rec["subject"] and "x is 5" in rec["body"]

