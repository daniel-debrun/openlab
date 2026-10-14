r"""The workspace's own tools: its files (ws_*), its notes, what the agents did and what it cost, and the tool
list itself. The study verbs are in study_tools.py, the paper tools in papers.py, the agents' in agent_tools.py."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from .registry import TOOLS, ToolContext, ToolError, tool


# --- workspace ----------------------------------------------------------------------------------------
@tool("workspace")
def ws_list(ctx: ToolContext, path: str = "", hidden: bool = False) -> dict:
    """List a folder of the workspace: folders, then files (`hidden` shows dot-names too)."""
    base = ctx.path(path)
    if not base.is_dir():
        raise ToolError("not a folder", f"{path!r} is not a folder", "list its parent")
    rows = []
    for p in sorted(base.iterdir(), key=lambda q: (q.is_file(), q.name.lower())):
        if p.name.startswith(".") and (not hidden or p.name == ".git"):
            continue
        try:
            ctx.path(ctx.rel(p))                                    # the vault, if it sits in the workspace, is not listed
        except ToolError:
            continue
        rows.append({"path": ctx.rel(p), "name": p.name, "kind": "folder" if p.is_dir() else "file",
                     "bytes": p.stat().st_size if p.is_file() else None})
    return {"path": ctx.rel(base) if base != ctx.root else "", "entries": rows}


WS_READ_CHARS = 200_000      # characters of a file a reply carries (see ws_read)


@tool("workspace")
def ws_read(ctx: ToolContext, path: str, chars: int = WS_READ_CHARS, offset: int = 0) -> dict:
    """Read a text file of the workspace: `chars` characters from `offset`, with `total_chars`, `truncated`
    and `chars_withheld` when the file is longer (`chars=-1` reads it whole).

    A workspace holds the data as well as the text, so `ws_read` of a csv is `ws_read` of however many
    megabytes that csv is -- a reply nobody can receive, where a frame at least had `data_preview` to fall
    back on. The window is generous (200k characters) and it is a window rather than a refusal so that a
    long file can still be walked with `offset`.

    Read a truncated file, then write it back, and the tail is gone -- which is why the marker is in the
    reply and not only in this docstring, and why `chars=-1` exists: an editor reads whole or not at all."""
    p = ctx.path(path)
    if not p.is_file():
        raise ToolError("no such file", f"{path!r} does not exist", "ws_list shows what does")
    text = p.read_text(encoding="utf-8")
    at = max(0, offset)
    if chars < 0 or at + chars >= len(text):
        return ({"path": ctx.rel(p), "text": text[at:], "total_chars": len(text), "offset": at}
                if at else {"path": ctx.rel(p), "text": text})
    return {"path": ctx.rel(p), "text": text[at:at + chars], "total_chars": len(text), "offset": at,
            "truncated": True, "chars_withheld": len(text) - at - chars,
            "read_with": f"ws_read(path, offset={at + chars}), or chars=-1 for the whole file"}


@tool("workspace", mutates=True)
def ws_write(ctx: ToolContext, path: str, text: str) -> dict:
    """Write a text file of the workspace (created with its folders if missing)."""
    p = ctx.path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return {"path": ctx.rel(p), "bytes": len(text.encode())}


@tool("workspace", destructive=True)
def ws_delete(ctx: ToolContext, path: str) -> dict:
    """Delete a file of the workspace. A study's record and the vault stay: deleting a file must not reset a
    study's count of trials or its holdout looks."""
    p = ctx.path(path)
    if not p.is_file():
        raise ToolError("no such file", f"{path!r} does not exist", "ws_list shows what does")
    p.unlink()
    return {"deleted": path}


@tool("workspace", mutates=True)
def ws_move(ctx: ToolContext, path: str, to: str) -> dict:
    """Move or rename a file. A study's record and the vault stay where they are -- a rename must not reset a
    trial count or reopen a holdout."""
    src, dst = ctx.path(path), ctx.path(to)
    if not src.is_file():
        raise ToolError("no such file", f"{path!r} does not exist", "ws_list shows what does")
    if dst.exists():
        raise ToolError("exists", f"{to!r} already exists", "pick another name, or delete it first")
    old = ctx.rel(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {"moved": old, "to": ctx.rel(dst),
            "note": "the vault stays under the old name: it is a record, not state"}


# --- notes: where an idea goes ---------------------------------------------------------------------------
# Ideas arrive while you are reading a chart, and anything that makes you leave the chart to record them
# loses them. So: one file, one append, no ceremony.
#
# NOT UNDER ``.data``. This is the only new state in the workspace that is the user's own writing rather than
# something derived from a run, so it is a markdown file in the root -- greppable, committable, diffable, and
# editable in any editor, with `[x]` for done so the state is legible in plain text. A scratchpad you can only
# read through the app is a scratchpad nobody trusts with anything they care about.
NOTES_FILE = "NOTES.md"
NOTES_LAST = 50            # notes per reply when the caller does not say
NOTES_MAX = 200            # ...and the most one reply carries, whatever `last` asked for
NOTE_CHARS = 1_000         # of one note, written or read back: a note is a line, not a document

#: Written once, when the file is created, so that someone who opens `NOTES.md` before they open the app knows
#: what the two boxes mean and that the file is theirs to edit.
NOTES_HEADER = """# Notes

One line per idea, oldest first. `- [ ]` is open and `- [x]` is done -- edit, reorder or delete any line by
hand; anything that is not a `- [ ]` line is left alone.

"""

#: A note, as a line of markdown. Everything after the box is optional, and a line that has only words is
#: still a note -- see `_note_parse`, which is written for the file a person edited rather than the one this
#: module wrote.
_NOTE_LINE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*(.*)$")
_NOTE_AT = re.compile(r"^(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\s*(.*)$")
#: What separates the head (date, document, tag) from the words. An em dash is what `notes_add` writes; the
#: other two are what a person types when they add a line by hand.
_NOTE_SEP = re.compile(r"\s+(?:—|–|--)\s+")
#: Between the fields of the head. A middle dot, because it appears in no path and in no tag.
_NOTE_DOT = "·"


def _note_head(head: str):
    """``(at, doc, tag)`` of a note's head, or ``None`` when that is not what it is.

    The head is a date, then ``· document``, then ``· #tag``, and nothing else -- so anything else is
    somebody's sentence that happened to contain a dash, and returning None here is what hands it back whole
    instead of filing half of it under a document it never mentioned."""
    rest = head.strip()
    m = _NOTE_AT.match(rest)
    at, rest = (m.group(1), m.group(2).strip()) if m else ("", rest)
    doc = tag = ""
    if rest:
        if not rest.startswith(_NOTE_DOT):
            return None
        for seg in [x.strip() for x in rest.lstrip(_NOTE_DOT).split(_NOTE_DOT)]:
            if seg.startswith("#") and not tag:
                tag = seg[1:].strip()
            elif seg and not seg.startswith("#") and not doc:
                doc = seg
            else:
                return None
    return at, doc, tag


def _note_parse(line: str, number: int) -> Optional[dict]:
    """One line of `NOTES.md` as a note, or ``None`` for a line that is not one.

    Written for a file that has been edited by hand, because that is the point of it being a file:

    * no checkbox, no note -- a heading, a blank line or a paragraph of prose is left alone, so the notes can
      live in a document somebody actually writes in;
    * the head is read only when the line has a separator AND the head parses as a head; otherwise every word
      after the box is the note, which is what a line somebody typed themselves looks like;
    * a date at the front with no separator is still read as the date, since it cannot be mistaken for prose;
    * nothing is ever dropped for being malformed. The worst case is a note whose date or document did not
      survive being edited, which is a note you can still read.

    ``number`` is the 1-based line, so the one affordance this file needs but these tools do not provide --
    marking a note done, which is `ws_write` on a markdown file, because that is what it is -- can address a
    single line instead of guessing at its text."""
    m = _NOTE_LINE.match(line)
    if m is None:
        return None
    done, rest = m.group(1) in "xX", m.group(2).strip()
    parts = _NOTE_SEP.split(rest, maxsplit=1)
    fields = _note_head(parts[0]) if len(parts) == 2 else None
    if fields is None:
        m2 = _NOTE_AT.match(rest)
        at, doc, tag = (m2.group(1) if m2 else ""), "", ""
        text = (m2.group(2) if m2 else rest).strip()
    else:
        at, doc, tag = fields
        text = parts[1].strip()
    note = {"at": at, "doc": doc, "tag": tag, "text": text[:NOTE_CHARS], "done": done, "line": number}
    if len(text) > NOTE_CHARS:                      # a hand-written essay on one line: bounded like any reply
        note["truncated"] = True
    return note


def _note_line(at: str, doc: str, tag: str, text: str, done: bool = False) -> str:
    """A note as the line that goes in the file: what `_note_parse` reads, and what a person would type."""
    head = " ".join([at] + ([f"{_NOTE_DOT} {doc}"] if doc else []) + ([f"{_NOTE_DOT} #{tag}"] if tag else []))
    return f"- [{'x' if done else ' '}] {head} — {text}"


def _one_line(s: str) -> str:
    """Whitespace collapsed: a note is one line of a file, so a pasted paragraph must not become half a note
    and a stray line of prose that the next reader cannot tell was ever part of it."""
    return " ".join((s or "").split())


@tool("workspace", mutates=True)
def notes_add(ctx: ToolContext, text: str, doc: str = "", tag: str = "") -> dict:
    """Append a dated line to the workspace's scratchpad, with the document you were looking at. An idea you
    have to stop and file somewhere is an idea you lose; this is one call, it reads nothing first, and it never
    rewrites what is already there.

    `ws_write` nearly does this, and is the wrong tool for one specific reason: it writes whole files, so
    appending a line means `ws_read` then `ws_write` -- and `ws_read` is a window, so a scratchpad longer than
    the window loses its tail to every note added after that. This opens the file in append mode and writes
    one line, which the operating system will not interleave with anybody else's.

    The store is `NOTES.md` in the workspace root, not a database and not something under `.data`: it is your
    own writing, so it is a file you can grep, commit, diff and edit by hand. `notes_list` reads it back;
    marking a note done, rewording it or deleting it is `ws_write` on a markdown file, because that is all it
    is.

        - [ ] 2026-09-25 16:44 · churn.openlab · #next — try a deeper tree; the depth-3 gain may be noise, not signal

    `doc` is the document you were looking at, recorded as you sent it -- a note about a document you go on to
    delete is still a note. `tag` is one word, with or without its `#`."""
    said = _one_line(text)
    if not said:
        raise ToolError("nothing to write", "a note with no words in it is not a note",
                        "notes_add(text=\"what you were about to forget\")")
    if len(said) > NOTE_CHARS:
        raise ToolError("too long for a note", f"a note is one line, at most {NOTE_CHARS} characters",
                        "shorten it, or write the long version into a workspace file and note the file")
    where, label = _one_line(doc), _one_line(tag).lstrip("#").strip()
    for bad, what in ((where, "doc"), (label, "tag")):
        if _NOTE_DOT in bad or _NOTE_SEP.search(f" {bad} "):
            raise ToolError("not writable as a note", f"{what} carries a character the line format uses "
                            f"({_NOTE_DOT}, or a dash between spaces)", "leave it out, or rename it")
    if " " in label:
        raise ToolError("a tag is one word", f"{label!r} is several", "use one word, or no tag")
    at = time.strftime("%Y-%m-%d %H:%M")
    line = _note_line(at, where, label, said)
    p = ctx.path(NOTES_FILE)
    # Created exclusively, never written over: two notes added at the same moment must not have one of them
    # truncate the file the other had just appended to.
    if not p.exists():
        try:
            import os
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(NOTES_HEADER)
        except FileExistsError:
            pass
    # A file whose last line was left without its newline (an editor, a `printf`) would otherwise take this
    # note onto the end of it, and two notes on one line is one note lost.
    joined = ""
    try:
        with p.open("rb") as f:
            f.seek(-1, 2)
            joined = "" if f.read(1) == b"\n" else "\n"
    except OSError:
        pass
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{joined}{line}\n")                 # ONE write, in append mode: the whole point of this tool
    return {"path": ctx.rel(p), "at": at, "doc": where, "tag": label, "text": said, "markdown": line}


@tool("workspace")
def notes_list(ctx: ToolContext, last: int = NOTES_LAST, doc: str = "", tag: str = "",
               open_only: bool = False) -> dict:
    """The scratchpad, newest first: what you wrote down, when, which document you were in, and whether it is
    still open.

    `doc` and `tag` narrow it, `open_only` drops what is done. `open` and `done` count the whole file before
    any of that, so a filter can be offered without a second call, and `notes_withheld` counts the notes that
    did not fit -- this reply is bounded like every other one here.

    `line` is where the note is in `NOTES.md`, which is what an edit addresses: this pair of tools appends and
    reads, and everything else (done, reworded, deleted) is `ws_write` on the markdown, so nothing about a
    scratchpad is locked inside the app. `openable` says the document is still there to open."""
    p = ctx.path(NOTES_FILE)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):            # no scratchpad yet: an empty one, not an error
        lines = []
    notes = [n for n in (_note_parse(l, i + 1) for i, l in enumerate(lines)) if n]
    counts = {"open": sum(1 for n in notes if not n["done"]), "done": sum(1 for n in notes if n["done"])}
    kept = [n for n in notes
            if (not doc or n["doc"] == doc) and (not tag or n["tag"] == tag.lstrip("#"))
            and (not open_only or not n["done"])]
    kept.reverse()                                   # newest first: the end of the file is what anyone reads
    shown = kept[:NOTES_MAX if last < 0 else max(0, min(last, NOTES_MAX))]
    seen: dict = {}
    for n in shown:
        where = n["doc"]
        if where and where not in seen:
            try:                                     # jailed, like every other path here: a hand-edited line
                seen[where] = ctx.path(where).is_file()          # may say anything at all
            except ToolError:
                seen[where] = False
        n["openable"] = bool(where) and seen.get(where, False)
    reply = {"path": ctx.rel(p) if p.exists() else NOTES_FILE, "notes": shown, "total": len(notes), **counts}
    if len(kept) > len(shown):
        reply["notes_withheld"] = len(kept) - len(shown)
    return reply


@tool("meta")
def usage(ctx: ToolContext, since: str = "", by: str = "tool", doc: str = "") -> dict:
    """What the analysis cost, call by call, summed by `tool` or by `doc`: calls, errors, wall seconds and the
    session CPU each spent (`since` a date or timestamp). The agent's own tokens are its CLI's to report."""
    from .usage import summary
    return summary(ctx.root, since, by, doc)


@tool("meta")
def agent_activity(ctx: ToolContext, last: int = 50) -> dict:
    """What the agents did here, newest first (from the control store): each tool call and refusal, with the
    agent that made it, and how many requests wait in the person's inbox."""
    from ..agents.store import Store, root_of
    n, root = max(1, min(last, 500)), root_of(ctx.root)
    if not (root / ".openlab" / "control.db").exists():
        return {"events": [], "waiting": 0, "refused": 0}
    st = Store(root)
    kinds = ("tool", "tool_error", "refusal", "bypass_attempt", "blocked", "request", "decision")
    evs = st.events(n=n, kinds=kinds)
    out = [{"at": e["at"], "agent": e["agent"], "kind": e["kind"], "tool": e["tool"] or e["kind"],
            "ok": e["kind"] in ("tool", "request", "decision"),
            **({"error": e["summary"]} if e["kind"] not in ("tool", "request", "decision") else {})}
           for e in reversed(evs)]
    return {"events": out, "waiting": len(st.inbox("pending")),
            "refused": sum(1 for e in out if e["kind"] in ("refusal", "bypass_attempt", "blocked"))}


def toolset_hash() -> str:
    """A short hash of every tool's name, section and input schema: two processes with the same hash offer the
    same tools."""
    import hashlib
    import json
    recs = sorted((t.name, t.section, json.dumps(t.schema, sort_keys=True, default=str)) for t in TOOLS.values())
    return hashlib.sha256(json.dumps(recs).encode()).hexdigest()[:12]


def version_info() -> dict:
    """The openlab version, the git commit it runs from (when it is a checkout) and the tool-set hash."""
    import subprocess
    from .. import __version__
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"version": __version__, "commit": commit, "toolset": toolset_hash()}


@tool("meta")
def key_info(ctx: ToolContext) -> dict:
    """The signing key's id and public half: what to publish so readers can check your signed files."""
    from ..study.signing import key_info as _key_info
    return _key_info()


DATASET_WALK_LIMIT = 4_000   # files walked looking for studies; a scratchpad is not a filesystem
#: folders a walk never enters
SKIP_DIRS = frozenset({"node_modules", "__pycache__", "site-packages", "dist", "build", "venv", "env",
                       "target", ".git"})


def _walk(ctx: ToolContext, path: str = "", suffixes=None):
    """Every file under `path` whose suffix is in `suffixes`, skipping dot-folders, vendor folders and the
    vault. Bounded, because a workspace that happens to contain a node_modules must not turn "list the
    datasets" into a filesystem crawl.

    `os.scandir` with a suffix filter rather than `iterdir` + `ctx.path` per entry, and that is not a
    micro-optimisation. The obvious version asked `ctx.path` whether each file was inside the vault, which
    resolves a path and re-reads the vault's location every time: 350 ms to find the fourteen studies in a
    workspace of a thousand files, paid on every page turn of a dataset. The vault is a set of DIRECTORIES,
    so it is resolved once and the walk is pruned at the folder instead."""
    import os
    base = ctx.path(path)                                        # jails the root, and refuses the vault, once
    if not base.is_dir():
        raise ToolError("not a folder", f"{path!r} is not a folder", "list its parent")
    from ..study.vault import closed_dirs
    closed = set()
    for v in closed_dirs(ctx.root):
        try:
            closed.add(os.path.realpath(v))
        except OSError:
            pass
    want = None if suffixes is None else {x.lower() for x in suffixes}
    out, stack, seen = [], [base], 0
    while stack and seen < DATASET_WALK_LIMIT:
        try:
            with os.scandir(stack.pop()) as it:                  # closed explicitly: a walk of a thousand
                entries = sorted(it, key=lambda e: e.name.lower())   # folders must not leak a handle each
        except OSError:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            seen += 1
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name in SKIP_DIRS or e.name.endswith(".egg-info"):
                        continue
                    if os.path.realpath(e.path) in closed:
                        continue
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    q = Path(e.path)
                    if want is None or q.suffix.lower() in want:
                        out.append(q)
            except OSError:                                      # a file that vanished mid-walk is not an error
                continue
    return out, seen >= DATASET_WALK_LIMIT


@tool("meta")
def tools(ctx: ToolContext, section: str = "") -> dict:
    """Every tool with its section, description and input schema: what the UI, the CLI and the agent can all do.
    `openlab` carries the version, git commit and tool-set hash of the process that answered."""
    return {"tools": [t.to_record() for t in TOOLS.values() if not section or t.section == section],
            "openlab": version_info()}


from . import papers  # noqa: E402,F401  -- registers the paper tools
from . import study_tools  # noqa: E402,F401  -- registers the study verbs
