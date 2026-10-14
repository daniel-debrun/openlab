"""Comments on the research's files: a person's notes, and the agents' replies.

A person comments on any file of the research in `openlab show`'s Workspace: on text they selected (the comment
keeps the quoted text and where it stood) or on the whole file (a PDF, an image, a data file). A comment is kept by
its study (the folder with a study.toml it is in, or "." for the research's own files) and its file within it. An
agent lists the open comments (`agent_comments`) and answers each (`agent_comment_reply`), resolving it once it is
dealt with; the person can reply and reopen. A person's comment is also told to the live lead of its study, else to
every live lead (Store.tell, delivered at their next tool call), with the file's path, the quote and the comment; a
lead may hand it to an agent under it. With no lead live it waits, open, for the next one.

Every act is a line appended to `.openlab/comments.jsonl` and never rewritten:

    {"op": "new", "id": 3, "study": "red-good", "file": "paper/sections/results.tex", "quote": "...", "start": 120,
     "end": 168, "text": "...", "by": "person:dana", "at": "..."}
    {"op": "reply", "id": 3, "text": "...", "by": "lead-3c3162", "at": "..."}
    {"op": "resolve" | "reopen", "id": 3, "text": "", "by": "...", "at": "..."}

and noted as a `comment` event on the hash-chained control store, whose chain covers the line's sha. Appends hold an
fcntl lock on `comments.lock`, so two writers never number two comments alike. A line that does not parse is skipped.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .store import Store, canonical, now

TEXT_MAX = 4000          # characters of a comment or a reply
QUOTE_MAX = 1000         # characters of the quoted text a comment keeps
OPS = ("new", "reply", "resolve", "reopen")
LIVE = ("running", "idle", "waiting", "paused")


class CommentError(ValueError):
    def __init__(self, error: str, try_: str = ""):
        super().__init__(error)
        self.error, self.try_ = error, try_


def _path(store: Store) -> Path:
    return store.dir / "comments.jsonl"


@contextmanager
def _locked(store: Store):
    with open(store.dir / "comments.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _lines(store: Store) -> list:
    p = _path(store)
    if not p.is_file():
        return []
    out = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("op") in OPS and isinstance(d.get("id"), int):
            out.append(d)
    return out


def _text(x, what: str, most: int = TEXT_MAX) -> str:
    x = str(x or "").strip()
    if not x:
        raise CommentError(f"the {what} is empty")
    if len(x) > most:
        raise CommentError(f"a {what} is at most {most} characters (this one has {len(x)})", "say it shorter")
    return x


def _append(store: Store, line: dict, summary: str) -> dict:
    text = canonical(line)
    with open(_path(store), "a", encoding="utf-8") as f:
        f.write(text + "\n")
    store.event(line["by"], "comment", summary=summary,
                payload={"id": line["id"], "op": line["op"], "sha": hashlib.sha256(text.encode()).hexdigest()})
    return line


def threads(store: Store, study: Optional[str] = None) -> list:
    """Every comment (of `study`), oldest first, folded: {id, study, file, quote, start, end, text, by, at,
    replies: [{text, by, at, op}], resolved, resolved_by}."""
    out: dict = {}
    for d in _lines(store):
        if d["op"] == "new":
            if d["id"] in out:
                continue
            out[d["id"]] = {k: d.get(k) for k in ("id", "study", "file", "quote", "start", "end", "text", "by", "at")}
            out[d["id"]].update(replies=[], resolved=False, resolved_by="")
            continue
        t = out.get(d["id"])
        if t is None:
            continue
        if d["op"] in ("resolve", "reopen"):
            t["resolved"], t["resolved_by"] = d["op"] == "resolve", d["by"] if d["op"] == "resolve" else ""
        if d.get("text") or d["op"] != "reply":
            t["replies"].append({k: d.get(k) for k in ("op", "text", "by", "at")})
    return [t for t in out.values() if study is None or t["study"] == study]


def _tell(store: Store, study: str, text: str, by: str) -> list:
    """Tell the study's live leads (and a live lead with no study yet), else every live lead of the research:
    delivered at their next tool call. The agents under a lead hear from it, not from the person."""
    told = []
    if len(text) > 2900:
        text = text[:2850] + "… (agent_comments has it whole)"
    live = [a for a in store.agents() if a.get("status") in LIVE and store.is_lead(a)]
    for a in [a for a in live if a.get("study") == study or not a.get("study")] or live:
        try:
            store.tell(a["id"], text, by)
            told.append(a["id"])
        except Exception:                                               # noqa: BLE001 -- the comment stands
            pass
    return told


def where(t: dict) -> str:
    """A comment's file as a path from the research's folder."""
    study, file = str(t.get("study") or "."), str(t.get("file") or "")
    return file if study in (".", "") else f"{study}/{file}"


def comment(store: Store, study: str, file: str, quote: str, start: int, end: int, text: str, by: str) -> dict:
    """A person's new comment on `quote` (characters start..end of `file`), or on the whole file with no quote; told
    to the study's live lead (else every live lead)."""
    text, quote = _text(text, "comment"), str(quote or "")[:QUOTE_MAX]
    if not str(file or "").strip():
        raise CommentError("a comment names the file it is on", "file: paper/sections/<name>.tex")
    with _locked(store):
        n = max([d["id"] for d in _lines(store) if d["op"] == "new"] or [0]) + 1
        line = {"op": "new", "id": n, "study": str(study), "file": str(file), "quote": quote,
                "start": max(0, int(start or 0)), "end": max(0, int(end or 0)), "text": text, "by": by, "at": now()}
        _append(store, line, f"comment #{n} on {file}: {text}")
    path = where(line)
    told = _tell(store, study, f"the person commented (comment #{n}) on {path}"
                 + (f", on this text:\n> " + quote.replace("\n", "\n> ") if quote else " (the whole file)")
                 + f"\nTheir comment: {text}\nDeal with it yourself or hand it to an agent under you; answer it with "
                 f"agent_comment_reply({n}, text, resolve=true once it is dealt with).", by)
    return {**_one(store, n), "told": told}


def _one(store: Store, id: int) -> dict:
    for t in threads(store):
        if t["id"] == id:
            return t
    raise CommentError(f"no comment #{id}", "agent_comments lists them")


def reply(store: Store, id: int, text: str, by: str, resolve: Optional[bool] = None, person: bool = False) -> dict:
    """A reply to comment #id; `resolve` True resolves it, False reopens it (None leaves it). A person's reply is
    told to the study's working agents."""
    id = int(id)
    with _locked(store):
        t = _one(store, id)
        if text and str(text).strip():
            _append(store, {"op": "reply", "id": id, "text": _text(text, "reply"), "by": by, "at": now()},
                    f"reply to comment #{id}: {str(text).strip()}")
        elif resolve is None:
            raise CommentError("the reply is empty", "send text, or resolve=true")
        if resolve is not None and bool(resolve) != t["resolved"]:
            op = "resolve" if resolve else "reopen"
            _append(store, {"op": op, "id": id, "text": "", "by": by, "at": now()}, f"comment #{id} {op}d")
    if person and (text or resolve is False):
        _tell(store, t["study"], f"the person {'replied to' if text else 'reopened'} comment #{id} on {where(t)}"
              + (f": {str(text).strip()}" if text else "") + "\nagent_comments shows the thread.", by)
    return _one(store, id)


def locate(text: str, t: dict) -> Optional[tuple]:
    """(start, end) of a comment's quote in the file's text now: where it was, else the occurrence nearest to it;
    None when the quote is no longer in the text."""
    q, s = t.get("quote") or "", int(t.get("start") or 0)
    if not q:
        return None
    if text[s:s + len(q)] == q:
        return s, s + len(q)
    best, i = None, text.find(q)
    while i >= 0:
        if best is None or abs(i - s) < abs(best - s):
            best = i
        i = text.find(q, i + 1)
    return (best, best + len(q)) if best is not None else None
