"""`python -m openlab.agents.supervisor ROOT AGENT`: run one spawned agent's CLI and watch it to the end.

Started detached by openlab.agents.spawn (its own session, so it outlives the command that spawned it). It runs the
CLI in its own process group, writes every stdout line to `.openlab/agents/<id>/transcript.jsonl`, turns the stream
into control-store events (what the agent said, each tool call, tool errors and openlab refusals, cost and tokens),
and holds the agent to its budget ($, wall minutes not counting paused time, turns): at 80% of a limit the agent is
told to ask for more rather than end with work in progress; at the limit it is paused, not ended, and openlab files a
budget request for it in the person's inbox (openlab.agents.inbox.hold), which raises the limit and resumes it when
approved. It acts on a person's control: pause = SIGSTOP to the group, resume = SIGCONT, stop = SIGTERM then
SIGKILL. The final status is done, failed or stopped, with the exit code.

The CLI is `claude` or `codex` on PATH, or $OPENLAB_CLAUDE_BIN / $OPENLAB_CODEX_BIN (tests use a fake).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .store import Store, short

POLL = 0.25                 # seconds between control and budget checks
GRACE = 5.0                 # seconds between SIGTERM and SIGKILL

WARN_AT = 0.8              # the share of a limit at which the agent is told to ask for more


def price(model: str) -> tuple:
    """(input, output) $ per million tokens (openlab.agents.pricing)."""
    from .pricing import price as p
    return p(model)[:2]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def refusal(text: str) -> Optional[str]:
    """The openlab refusal in a tool result's text (a dispatch envelope with ok false, or a study error), else
    None."""
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and obj.get("ok") is False and obj.get("error"):
        return short(f"{obj['error']}" + (f": {obj['meaning']}" if obj.get("meaning") else ""), 200)
    if isinstance(obj, dict) and set(obj) >= {"error", "meaning"}:
        return short(f"{obj['error']}: {obj['meaning']}", 200)
    for line in t.splitlines():
        if line.startswith("error: ") and ("openlab" in t or "meaning:" in t):
            return short(line[7:], 200)
    return None


def tool_line(name: str, inp) -> str:
    if not isinstance(inp, dict):
        return short(f"{name} {inp}", 160)
    for k in ("command", "file_path", "path", "pattern", "url", "query", "folder", "description", "prompt"):
        if inp.get(k):
            return short(f"{name} {inp[k]}", 160)
    return short(f"{name} {json.dumps(inp, default=str)}", 160)


class ClaudeStream:
    """Claude Code `--output-format stream-json --verbose` lines into events and spend."""

    def __init__(self, store: Store, agent: dict):
        self.store, self.id, self.model = store, agent["id"], agent.get("model") or ""
        self.usage: dict = {}               # message id -> (tokens_in, tokens_out, usd) already counted
        self.turn_ids: set = set()
        self.tools: dict = {}               # tool_use_id -> tool name
        self.tasks: dict = {}               # Agent/Task tool_use_id -> (subagent_type, description)
        self.children: dict = {}            # Agent/Task tool_use_id -> child agent id
        self.result: Optional[dict] = None

    def child(self, tool_use_id: str) -> str:
        """The sub-agent behind a parent_tool_use_id: the one a SubagentStart hook adopted, else registered here."""
        if tool_use_id in self.children:
            return self.children[tool_use_id]
        found = self.store.find(tool_use_id=tool_use_id)
        if found:
            cid = found[0]["id"]
        else:
            role, desc = self.tasks.get(tool_use_id, ("subagent", ""))
            cid = f"{self.id}.{len(self.children) + 1}"
            parent = self.store.get(self.id) or {}
            self.store.register(cid, "claude-subagent", name=cid, role=role or "subagent", parent=self.id,
                                cli="claude", study=parent.get("study", ""), question=desc,
                                cwd=parent.get("cwd", ""), session_id=parent.get("session_id", ""),
                                tool_use_id=tool_use_id, status="running")
            self.store.event(cid, "start", summary=f"sub-agent {role} of {self.id}: {desc}")
        self.children[tool_use_id] = cid
        return cid

    def feed(self, obj: dict) -> None:
        t = obj.get("type")
        if t == "system" and obj.get("subtype") == "init":
            self.model = obj.get("model") or self.model
            mcp = {s.get("name"): s.get("status") for s in obj.get("mcp_servers") or [] if isinstance(s, dict)}
            self.store.update(self.id, model=self.model)
            self.store.event(self.id, "start", summary=f"{self.model or 'claude'} started; openlab MCP "
                             f"{mcp.get('openlab', 'absent')}", payload={"session_id": obj.get("session_id"),
                                                                       "mcp": mcp, "cwd": obj.get("cwd")})
            if mcp.get("openlab") not in (None, "connected"):
                self.store.event(self.id, "warning", summary=f"the openlab MCP server is {mcp.get('openlab')}")
        elif t == "assistant":
            self._assistant(obj)
        elif t == "user":
            self._user(obj)
        elif t == "result":
            self._result(obj)

    def _assistant(self, obj: dict) -> None:
        msg = obj.get("message") or {}
        sub = obj.get("parent_tool_use_id")
        who = self.child(sub) if sub else self.id
        mid = msg.get("id") or f"anon-{len(self.usage)}"
        u = msg.get("usage") or {}
        if u:
            from .pricing import cost, tokens_in
            tin, tout = tokens_in(u), int(u.get("output_tokens") or 0)
            usd = cost(msg.get("model") or self.model, u)
            before = self.usage.get(mid, (0, 0, 0.0))
            delta = (tin - before[0], tout - before[1], usd - before[2])
            self.usage[mid] = (tin, tout, usd)
            if any(delta):
                self.store.add_spent(who, tokens_in=delta[0], tokens_out=delta[1])
                if self.result is None:
                    self.store.add_spent(self.id, usd=round(delta[2], 6))
                    self.store.set_spent(self.id, usd_estimated=True)
        if not sub and mid not in self.turn_ids:
            self.turn_ids.add(mid)
            self.store.add_spent(self.id, turns=1)
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                self.store.event(who, "say", summary=block["text"])
            elif block.get("type") == "tool_use":
                name, inp = block.get("name", "?"), block.get("input") or {}
                self.tools[block.get("id", "")] = name
                if name in ("Agent", "Task"):
                    self.tasks[block.get("id", "")] = (inp.get("subagent_type") or "subagent",
                                                       inp.get("description") or "")
                self.store.add_spent(who, tool_calls=1)
                self.store.event(who, "tool", tool=name, summary=tool_line(name, inp),
                                 payload={"input": inp, "id": block.get("id")})

    def _user(self, obj: dict) -> None:
        msg = obj.get("message") or {}
        sub = obj.get("parent_tool_use_id")
        who = self.child(sub) if sub else self.id
        for block in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            name = self.tools.get(block.get("tool_use_id", ""), "?")
            text = _text(block.get("content"))
            if name in ("Agent", "Task") and block.get("tool_use_id") in self.children:
                cid = self.children[block["tool_use_id"]]
                self.store.set_status(cid, "done", only_from=("running", "idle", "waiting"))
                self.store.event(cid, "end", summary=text)
            why = refusal(text)
            if why:
                self.store.event(who, "refusal", tool=name, summary=f"{name}: {why}")
            elif block.get("is_error"):
                self.store.event(who, "tool_error", tool=name, summary=f"{name}: {text}")

    def _result(self, obj: dict) -> None:
        self.result = obj
        u = obj.get("usage") or {}
        cur = (self.store.get(self.id) or {}).get("spent", {})
        vals = {"usd_estimated": False}
        if obj.get("total_cost_usd") is not None:
            vals["usd"] = round(float(obj["total_cost_usd"]), 6)   # the session's total, a --resume'd one's too
        if u:
            tin = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                   "cache_read_input_tokens"))
            vals["tokens_in"] = max(tin, cur.get("tokens_in", 0))
            vals["tokens_out"] = max(int(u.get("output_tokens") or 0), cur.get("tokens_out", 0))
        if obj.get("num_turns") is not None:
            vals["turns"] = max(int(obj["num_turns"]), cur.get("turns", 0))
        self.store.set_spent(self.id, **vals)
        self.store.event(self.id, "result", summary=f"{obj.get('subtype', 'result')}"
                         + (f" ${obj['total_cost_usd']:.4f}" if obj.get("total_cost_usd") is not None else "")
                         + f": {obj.get('result') or ''}",
                         payload={k: obj.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd",
                                                          "usage", "session_id", "duration_ms")})

    @property
    def failed(self) -> bool:
        return bool(self.result and (self.result.get("is_error") or
                                     str(self.result.get("subtype", "success")) != "success"))


class CodexStream:
    """`codex exec --json` JSONL, read generically: newer `{"type": "item.completed", "item": {...}}` /
    `turn.completed` with usage, and older `{"msg": {"type": ...}}` events."""

    def __init__(self, store: Store, agent: dict):
        self.store, self.id, self.model = store, agent["id"], agent.get("model") or "gpt-5"
        self.result: Optional[dict] = None
        self.error = False

    def feed(self, obj: dict) -> None:
        t = obj.get("type") or ""
        item = obj.get("item") if isinstance(obj.get("item"), dict) else None
        if "msg" in obj and isinstance(obj["msg"], dict):            # older codex: {"id":..,"msg":{"type":..}}
            item, t = obj["msg"], "msg." + str(obj["msg"].get("type"))
        if t in ("thread.started", "session.created", "msg.session_configured"):
            self.store.event(self.id, "start", summary=f"codex started {obj.get('thread_id') or ''}".strip())
        elif t in ("turn.started", "msg.task_started"):
            self.store.add_spent(self.id, turns=1)
        elif t in ("item.completed", "msg.agent_message", "msg.exec_command_begin", "msg.mcp_tool_call_begin") and item:
            self._item(item)
        elif t in ("turn.completed", "msg.token_count"):
            u = obj.get("usage") or (item or {}).get("info", {}).get("last_token_usage") or (item or {}).get("usage") or {}
            if u:
                from .pricing import codex_cost
                tin, tout = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
                self.store.add_spent(self.id, tokens_in=tin, tokens_out=tout, usd=round(codex_cost(self.model, u), 6))
                self.store.set_spent(self.id, usd_estimated=True)
        elif t in ("turn.failed", "error", "msg.error", "msg.stream_error"):
            self.error = True
            err = obj.get("error") or obj.get("message") or (item or {}).get("message")
            self.store.event(self.id, "tool_error", summary=f"codex error: {_text(err.get('message') if isinstance(err, dict) else err)}")
        elif t in ("msg.task_complete",):
            self.result = item

    def _item(self, item: dict) -> None:
        kind = item.get("type") or item.get("item_type") or ""
        if kind in ("agent_message", "assistant_message"):
            text = item.get("text") or item.get("message") or ""
            if text.strip():
                self.store.event(self.id, "say", summary=text)
            self.result = item
        elif kind in ("command_execution", "exec_command_begin", "local_shell_call"):
            cmd = item.get("command")
            cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
            self.store.add_spent(self.id, tool_calls=1)
            self.store.event(self.id, "tool", tool="shell", summary=short(f"shell {cmd}", 160))
            out = item.get("aggregated_output") or ""
            if (why := refusal(out)):
                self.store.event(self.id, "refusal", tool="shell", summary=f"shell: {why}")
        elif kind in ("mcp_tool_call", "mcp_tool_call_begin"):
            inv = item.get("invocation") or item
            name = f"mcp__{inv.get('server', '?')}__{inv.get('tool', '?')}"
            self.store.add_spent(self.id, tool_calls=1)
            self.store.event(self.id, "tool", tool=name, summary=tool_line(name, inv.get("arguments") or {}))
            res = item.get("result")
            text = _text((res or {}).get("content")) if isinstance(res, dict) else _text(res)
            if (why := refusal(text)):
                self.store.event(self.id, "refusal", tool=name, summary=f"{name}: {why}")
        elif kind == "file_change":
            self.store.add_spent(self.id, tool_calls=1)
            self.store.event(self.id, "tool", tool="edit",
                             summary=short("edit " + ", ".join(c.get("path", "?") for c in item.get("changes") or []), 160))

    @property
    def failed(self) -> bool:
        return self.error


def command(root: Path, agent: dict, d: Path) -> list:
    launch = json.loads((d / "launch.json").read_text()) if (d / "launch.json").exists() else {}
    prompt, brief_text = (d / "prompt.txt").read_text(), (d / "brief.txt").read_text()
    if (d / "allocation.txt").is_file():                # where its work runs (openlab.nodes.alloc.brief)
        brief_text += "\n\n" + (d / "allocation.txt").read_text()
    if agent["cli"] == "codex":
        exe = os.environ.get("OPENLAB_CODEX_BIN") or shutil.which("codex") or "codex"
        env = {"OPENLAB_AGENT": agent["id"], "OPENLAB_WORKSPACE": str(root)}
        if (d / "token").is_file():
            env["OPENLAB_AGENT_TOKEN"] = (d / "token").read_text().strip()
        cmd = [exe, "exec", "--json", "--skip-git-repo-check",
               "-c", f"mcp_servers.openlab.command={json.dumps(sys.executable)}",
               "-c", 'mcp_servers.openlab.args=["-m","openlab.mcp_server"]',
               "-c", "mcp_servers.openlab.env={" + ",".join(f"{k}={json.dumps(v)}" for k, v in env.items()) + "}"]
        if agent.get("model"):
            cmd += ["-m", agent["model"]]
        return cmd + [brief_text + "\n\n" + prompt]
    exe = os.environ.get("OPENLAB_CLAUDE_BIN") or shutil.which("claude") or "claude"
    again = d / "relaunch.txt"                          # `openlab agent relaunch`: the same session, told why it goes on
    session = ["--resume", agent["session_id"]] if again.is_file() else ["--session-id", agent["session_id"]]
    cmd = [exe, "-p", again.read_text() if again.is_file() else prompt, "--output-format", "stream-json", "--verbose",
           *session,
           "--mcp-config", str(d / "mcp.json"), "--strict-mcp-config", "--settings", str(d / "settings.json"),
           "--permission-mode", launch.get("permission_mode") or "acceptEdits",
           "--append-system-prompt", brief_text]
    if launch.get("allowed_tools"):
        cmd += ["--allowedTools", ",".join(launch["allowed_tools"])]
    if launch.get("disallowed_tools"):                  # Agent,Task: delegation goes through openlab_agent_spawn
        cmd += ["--disallowedTools", ",".join(launch["disallowed_tools"])]
    # no --max-budget-usd / --max-turns: the CLI would end the agent at them, and openlab pauses it instead (Supervisor
    # ._budget), so the person can raise the limit and the work goes on
    if agent.get("model"):
        cmd += ["--model", agent["model"]]
    return cmd


class Supervisor:
    def __init__(self, root: Path, agent_id: str):
        self.root, self.id = Path(root), agent_id
        self.store = Store(self.root)
        self.agent = self.store.get(agent_id)
        if self.agent is None:
            raise SystemExit(f"no agent {agent_id}")
        self.dir = self.store.agent_dir(agent_id)
        self.proc: Optional[subprocess.Popen] = None
        self.stopping: Optional[str] = None
        self.paused_at: Optional[float] = None
        self.paused_total = 0.0
        self.warned: set = set()

    def _signal(self, sig) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass

    def _read(self, parser) -> None:
        with open(self.dir / "transcript.jsonl", "ab") as raw:
            for line in self.proc.stdout:
                raw.write(line)
                raw.flush()
                try:
                    obj = json.loads(line)
                except ValueError:
                    if line.strip():
                        self.store.event(self.id, "output", summary=line.decode("utf-8", "replace"))
                    continue
                if isinstance(obj, dict):
                    try:
                        parser.feed(obj)
                    except Exception as e:                          # noqa: BLE001 -- one odd line never kills it
                        self.store.event(self.id, "warning", summary=f"unparsed line: {type(e).__name__}: {e}")

    def stop(self, why: str) -> None:
        if self.stopping:
            return
        if self.proc is None:
            raise SystemExit(f"stopped before start: {why}")
        self.stopping = why
        self.store.event(self.id, "stopping", summary=why)
        self._signal(signal.SIGTERM)
        self._signal(signal.SIGCONT)                                # a stopped group handles SIGTERM only once continued
        deadline = time.monotonic() + GRACE
        while self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.proc.poll() is None:
            self._signal(signal.SIGKILL)

    def _control(self) -> None:
        a = self.store.get(self.id) or {}
        want = a.get("control", "")
        if want == "stop":
            self.stop(f"stopped by a person: {a.get('control_reason') or 'no reason given'}")
        elif want == "pause" and self.paused_at is None:
            self._signal(signal.SIGSTOP)
            self.paused_at = time.monotonic()
            self.store.set_status(self.id, "paused")
            self.store.event(self.id, "paused", summary=f"SIGSTOP: {a.get('control_reason') or ''}")
        elif want != "pause" and self.paused_at is not None:
            self._signal(signal.SIGCONT)
            self.paused_total += time.monotonic() - self.paused_at
            self.paused_at = None
            self.store.set_spent(self.id, paused_minutes=round(self.paused_total / 60, 3))
            self.store.set_status(self.id, "running")
            self.store.event(self.id, "resumed", summary="SIGCONT")

    def _budget(self, t0: float) -> None:
        if self.paused_at is not None:
            return
        a = self.store.get(self.id) or {}
        if a.get("control") == "pause":                             # held already: the next tick stops the group
            return
        b, s = a.get("budget") or {}, a.get("spent") or {}
        minutes = (time.monotonic() - t0 - self.paused_total) / 60
        est = "~" if s.get("usd_estimated") else ""
        used = {"minutes": (minutes, f"{minutes:.0f} of {float(b.get('minutes') or 0):g} minutes"),
                "usd": (s.get("usd") or 0, f"{est}${s.get('usd') or 0:.2f} of ${float(b.get('usd') or 0):g}"),
                "turns": (s.get("turns") or 0, f"{s.get('turns') or 0} of {int(b.get('turns') or 0)} turns")}
        for key, (n, text) in used.items():
            limit = float(b.get(key) or 0)
            if not limit:
                continue
            if n > limit if key == "turns" else n >= limit:
                from .inbox import hold
                hold(self.store, self.id, key, text)
                return
            if n >= WARN_AT * limit and (key, limit) not in self.warned:
                self.warned.add((key, limit))
                self.store.tell(self.id, f"you have used {text}. If your work needs more, ask now: "
                                f"openlab_agent_request(kind=\"budget\", args={{\"{key}\": <new limit>}}, reason). Do not "
                                "end with work in progress (a live run's result would go unreported): at the limit openlab pauses you "
                                "and asks the person for more.", "openlab")

    def _allocation(self, launch: dict) -> Optional[dict]:
        """The allocation this agent works in (its own, requested at spawn, or its parent's it shares), once it runs;
        waiting meanwhile (status waiting, the reason as an event when it changes). None or a row not running when
        it could not be had (refused, its Slurm job gone, stopped by a person while it waited)."""
        from ..nodes import alloc as A
        deadline, row = time.monotonic() + 60, None
        while row is None and time.monotonic() < deadline:
            row = A.of_agent(self.id, self.root)
            if row is None:
                time.sleep(0.5)
        if row is None:
            return None
        if row["state"] != "running":
            self.store.set_status(self.id, "waiting", only_from=("running",))
            self.store.event(self.id, "allocation_wait", summary=row["why"] or row["state"],
                             payload={"allocation": row["id"]})
            last = [row["why"]]

            def on_wait(r):
                if r["why"] != last[0]:
                    last[0] = r["why"]
                    self.store.event(self.id, "allocation_wait", summary=r["why"], payload={"allocation": r["id"]})

            def stopped() -> bool:
                a = self.store.get(self.id) or {}
                if a.get("control") == "stop" and not self.stopping:
                    self.stopping = f"stopped by a person: {a.get('control_reason') or 'no reason given'}"
                return bool(self.stopping)

            row = A.acquire(row["id"], poll=float(os.environ.get("OPENLAB_ALLOC_POLL") or 5), on_wait=on_wait,
                            stopped=stopped)
            self.store.set_status(self.id, "running", only_from=("waiting",))
        if row is not None and row["state"] == "running":
            self.store.event(self.id, "allocation_start", summary=A.describe(row),
                             payload={k: row[k] for k in ("id", "node", "backend", "host", "cores", "memory_gb", "gpus",
                                                          "gpu_list", "gpu_names", "ref", "partition")})
            (self.dir / "allocation.txt").write_text(A.brief(row))
        return row

    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: self.stop("the supervisor was terminated"))
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        launch = json.loads((self.dir / "launch.json").read_text()) if (self.dir / "launch.json").exists() else {}
        held = None
        if launch.get("resources") or launch.get("share"):
            try:
                held = self._allocation(launch)
            except Exception as e:                                  # noqa: BLE001 -- ends the agent, said why
                held = {"state": "failed", "why": f"{type(e).__name__}: {e}"}
            if held is None or held["state"] != "running":
                return self._unplaced(held)
        self.held = held
        try:
            return self._run(held)
        finally:
            self._release()

    def _unplaced(self, row: Optional[dict]) -> int:
        """End an agent whose allocation could not be had, saying why."""
        status = "stopped" if self.stopping else "failed"
        why = self.stopping or (row or {}).get("why") or "no allocation was recorded for it"
        self.store.event(self.id, "end", summary=f"{status} before it started: no allocation ({why})",
                         payload={"exit_code": None, "status": status, "reason": why, "usd": 0.0})
        self.store.set_status(self.id, status)
        self.store.notify("end", self.id, f"{self.id} {status}: it never got its allocation ({why})")
        return 0

    def _release(self) -> None:
        """Let go of the allocation (once): before the final status, so a waiter sees it ended."""
        held, self.held = getattr(self, "held", None), None
        if held is None:
            return
        from ..nodes import alloc as A
        try:
            r = A.leave(held["id"], self.id)
        except Exception as e:                                      # noqa: BLE001 -- the run has ended either way
            print(f"supervisor: could not release {held['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return
        if r is not None and r["state"] == "ended":
            self.store.event(self.id, "allocation_end", summary=f"{r['id']} on {r['node']} ended: "
                             f"{r['usage'].get('hours', 0):.2f} h, {r['usage'].get('gpu_hours', 0):.2f} GPU-hours",
                             payload={"allocation": r["id"], "node": r["node"], **(r["usage"] or {})})

    def _run(self, held: Optional[dict]) -> int:
        a = self.agent
        cmd = command(self.root, a, self.dir)
        env = {**os.environ, "OPENLAB_AGENT": self.id, "OPENLAB_WORKSPACE": str(self.root),
               "OPENLAB_AGENT_TOKEN": os.environ.get("OPENLAB_AGENT_TOKEN") or self.store.token(self.id),
               # claude -p stops background tasks and exits 10 min after its last turn; a lead waiting on a long
               # run would end as done, its children orphaned. Openlab's wall-minute budget bounds the wait instead
               "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": os.environ.get("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS", "0")}
        if held is not None:
            from ..nodes import alloc as A
            aenv, prefix = A.launch(held)
            env.update(aenv)
            cmd = prefix + cmd
            here = Path(sys.executable).parent              # `openlab node exec` from the agent's shell
            if (here / "openlab").exists() and str(here) not in env.get("PATH", "").split(os.pathsep):
                env["PATH"] = f"{here}{os.pathsep}{env.get('PATH', '')}"
            mcp = self.dir / "mcp.json"
            if mcp.is_file():                               # the MCP server's runs are admitted within it too
                cfg = json.loads(mcp.read_text())
                for srv in (cfg.get("mcpServers") or {}).values():
                    srv.setdefault("env", {}).update(aenv)
                mcp.write_text(json.dumps(cfg, indent=1))
        (self.dir / "command.json").write_text(json.dumps(cmd))
        parser = CodexStream(self.store, a) if a["cli"] == "codex" else ClaudeStream(self.store, a)
        try:
            err = open(self.dir / "stderr.log", "ab")
            self.proc = subprocess.Popen(cmd, cwd=a["cwd"] or str(self.root), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=err, env=env, start_new_session=True)
        except OSError as e:
            self.store.update(self.id, exit_code=127)
            self.store.set_status(self.id, "failed")
            self.store.event(self.id, "end", summary=f"could not start {cmd[0]}: {e}")
            return 127
        self.store.update(self.id, pid=self.proc.pid, pgid=self.proc.pid, supervisor=os.getpid())
        self.store.event(self.id, "launch", summary=f"{Path(cmd[0]).name} pid {self.proc.pid}")
        reader = threading.Thread(target=self._read, args=(parser,), daemon=True)
        reader.start()
        t0 = time.monotonic()
        while self.proc.poll() is None:
            try:
                self._control()
                if not self.stopping:
                    self._budget(t0)
            except Exception as e:                                  # noqa: BLE001 -- keep supervising
                print(f"supervisor: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            time.sleep(POLL)
        reader.join(timeout=10)
        rc = self.proc.returncode
        status = "stopped" if self.stopping else ("done" if rc == 0 and not parser.failed else "failed")
        spent = (self.store.get(self.id) or {}).get("spent") or {}
        billed = not spent.get("usd_estimated") and getattr(parser, "result", None) is not None
        money = f"{'' if billed else '~'}${spent.get('usd') or 0:.4f} {'billed' if billed else 'estimated'}"
        self.store.event(self.id, "end", summary=f"{status} (exit {rc}), {money}"
                         + (f": {self.stopping}" if self.stopping else ""),
                         payload={"exit_code": rc, "status": status, "reason": self.stopping or "",
                                  "usd": spent.get("usd") or 0.0, "usd_billed": billed})
        for c in self.store.children(self.id):                       # its Claude sub-agents end with it
            if c["kind"] == "claude-subagent":
                self.store.set_status(c["id"], status, only_from=("running", "idle", "waiting", "paused"))
        self.store.update(self.id, exit_code=rc)
        self._release()
        self.store.set_status(self.id, status)                      # last: a waiter sees the whole story
        a = self.store.get(self.id) or {}
        pending = len(self.store.inbox("pending", agent=self.id))
        self.store.notify("end", self.id, f"{self.id} ({a.get('role') or 'agent'}) {status}, {money}"
                          + (f"; {pending} inbox item(s) wait for you" if pending else ""))
        try:                                                        # [harness] reflect: a reflector reads the run
            from ..harness.reflect import after
            r = after(self.root, self.id)
            if r:
                self.store.event(self.id, "note", summary=f"reflector {r['reflector']} reads this run")
        except Exception as e:                                      # noqa: BLE001 -- the run has ended either way
            print(f"supervisor: no reflection: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m openlab.agents.supervisor ROOT AGENT", file=sys.stderr)
        return 2
    return Supervisor(Path(argv[0]), argv[1]).run()


if __name__ == "__main__":
    sys.exit(main())
