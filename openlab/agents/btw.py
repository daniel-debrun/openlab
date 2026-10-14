"""A side question to a lead, answered now (the `/btw` of Claude Code, for a headless lead).

The lead keeps working; openlab forks its Claude Code session (`claude -p --resume <session> --fork-session`) with no
tools, no MCP servers and no project hooks, asks the question there and records the answer. The fork reads the lead's
context as it stands (what it did, what it knows, what it plans), answers in seconds, and is not saved: nothing goes
back into the lead's own session, so the lead neither sees the question nor changes course. To redirect a lead, tell
it (Store.tell with ask=True: it reads the message at its next tool call and answers with agent_reply).

A Codex lead has no session openlab can fork: tell it instead. The fork's cost is the person's CLI spend like the
lead's own; it is kept apart, in the lead's spent.btw_usd, so the lead's budget is not charged for the person's
questions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from .store import Store

TIMEOUT = 180.0             # seconds a side answer may take
ANSWER_MAX = 3000           # characters of an answer kept on the record
FRAME = ("(A side question from the person, asked in openlab show while you keep working in your main session. Answer "
         "it briefly from what you know and did so far. You have no tools here, and this exchange does not go back "
         "into your main session.)\n\n")


class BtwError(Exception):
    def __init__(self, error: str, meaning: str = "", try_: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.try_ = error, meaning, try_


def command(a: dict, question: str) -> list:
    exe = os.environ.get("OPENLAB_CLAUDE_BIN") or shutil.which("claude") or "claude"
    cmd = [exe, "-p", FRAME + question, "--resume", a["session_id"], "--fork-session", "--no-session-persistence",
           "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers": {}}', "--setting-sources", "user",
           "--output-format", "json"]
    return cmd + (["--model", a["model"]] if a.get("model") else [])


def ask(store: Store, id: str, question: str, by: str) -> dict:
    """Ask lead `id` a side question; returns {seq, answer, usd, seconds}. The question and the answer are events
    (`btw`, then a `reply` with btw true), so they join the lead's conversation in openlab show."""
    a = store.get(id)
    if a is None:
        raise BtwError(f"no agent {id!r}")
    if not question.strip() or len(question) > ANSWER_MAX:
        raise BtwError("a side question is 1 to 3000 characters")
    if a["cli"] != "claude" or not a["session_id"]:
        raise BtwError(f"{id} has no Claude Code session to ask", "a side question forks the lead's Claude session; "
                       "a Codex lead has none", "tell it instead: it reads the message at its next tool call")
    seq = store.event(id, "btw", summary=f"{by}: {question}", payload={"text": question, "by": by}, exact=True)
    t0 = time.monotonic()
    try:
        r = subprocess.run(command(a, question), cwd=a["cwd"] or str(store.root), capture_output=True, text=True,
                           timeout=TIMEOUT, stdin=subprocess.DEVNULL,
                           env={**os.environ, "OPENLAB_BTW": "1", "OPENLAB_WORKSPACE": str(store.root)})
    except (OSError, subprocess.TimeoutExpired) as e:
        store.event(id, "btw_error", summary=f"side question {seq}: {type(e).__name__}")
        raise BtwError("the side question got no answer", f"{type(e).__name__}: {e}"[:300],
                       "try again, or tell the lead instead") from None
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        out = {}
    answer = str(out.get("result") or "").strip()
    if r.returncode != 0 or out.get("is_error") or not answer:
        store.event(id, "btw_error", summary=f"side question {seq}: exit {r.returncode}")
        raise BtwError("the side question got no answer", (answer or r.stderr or r.stdout or "")[-300:],
                       "try again, or tell the lead instead")
    if len(answer) > ANSWER_MAX:
        answer = answer[:ANSWER_MAX].rstrip() + " …"
    usd = float(out.get("total_cost_usd") or 0)
    store.add_spent(id, btw_usd=round(usd, 6))
    store.event(id, "reply", summary=answer, payload={"text": answer, "to": seq, "btw": True, "usd": round(usd, 6)},
                exact=True)
    return {"seq": seq, "answer": answer, "usd": round(usd, 6), "seconds": round(time.monotonic() - t0, 1)}
