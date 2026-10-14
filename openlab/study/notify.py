"""Notifications: `deliver` sends one message for a run's fired alerts (a study forecast whose gate failed, a
monitoring pass that found something), emailed over SMTP when these environment variables are set, and otherwise
written to the outbox file it is given:

    OPENLAB_SMTP_HOST, OPENLAB_SMTP_PORT (587; 465 is SSL), OPENLAB_SMTP_USER, OPENLAB_SMTP_PASSWORD,
    OPENLAB_ALERT_TO (comma-separated), OPENLAB_ALERT_FROM (default: the user)

``OPENLAB_ALERT_DRY_RUN=1`` never sends, even when SMTP is set: the message goes to the outbox.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import clock


def deliver(doc: str, fired: list, outbox: Path, subject: str = "") -> str:
    """One message for a run's fired alerts: email when SMTP is configured, else a line in ``outbox``."""
    subject = subject or f"openlab alert: {len(fired)} fired in {doc}"
    body = "\n".join(f"{a['alert']}\n    {a['interpretation']}" for a in fired)
    env = os.environ
    to = [t.strip() for t in env.get("OPENLAB_ALERT_TO", "").split(",") if t.strip()]
    why = "dry run" if env.get("OPENLAB_ALERT_DRY_RUN") == "1" else "SMTP not set (OPENLAB_SMTP_HOST, OPENLAB_ALERT_TO)"
    if env.get("OPENLAB_SMTP_HOST") and to and env.get("OPENLAB_ALERT_DRY_RUN") != "1":
        import smtplib
        from email.message import EmailMessage
        m = EmailMessage()
        m["Subject"], m["To"] = subject, ", ".join(to)
        m["From"] = env.get("OPENLAB_ALERT_FROM") or env.get("OPENLAB_SMTP_USER") or "openlab@localhost"
        m.set_content(body)
        port = int(env.get("OPENLAB_SMTP_PORT", "587"))
        try:
            with (smtplib.SMTP_SSL if port == 465 else smtplib.SMTP)(env["OPENLAB_SMTP_HOST"], port, timeout=30) as s:
                if port != 465:
                    s.starttls()
                if env.get("OPENLAB_SMTP_USER"):
                    s.login(env["OPENLAB_SMTP_USER"], env.get("OPENLAB_SMTP_PASSWORD", ""))
                s.send_message(m)
            return f"emailed to {', '.join(to)}"
        except (OSError, smtplib.SMTPException) as e:     # the alert is kept in the outbox, not lost
            why = f"SMTP failed: {type(e).__name__}: {e}"[:200]
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": clock.now(), "to": to, "subject": subject,
                             "body": body}) + "\n")
    return f"not emailed ({why}): written to {outbox}"


def desktop(subject: str, body: str) -> str:
    """A desktop notification (notify-send); says why when none was shown."""
    exe = shutil.which("notify-send")
    if not exe:
        return "no desktop notification: notify-send is not installed"
    try:
        subprocess.run([exe, "--app-name=openlab", subject, body[:1000]], capture_output=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as e:
        return f"no desktop notification: {type(e).__name__}"
    return "shown on the desktop"


#: where a notification goes besides `deliver` (email, else the outbox): nowhere else, or the desktop too
NOTIFY = {"outbox": None, "desktop": desktop}
