"""The study record: append-only, hash-chained nodes and edges in SQLite, plus a content store for blobs.

Each node's hash covers its seq, kind, payload, links, actor, session, time and the previous node's hash, so
editing any row or link breaks the chain from that row on, and `verify` names it. There is no update or delete.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .clock import now
from . import personkey, schema
from .errors import StudyError

KINDS = schema.KINDS

ACTORS = ("agent", "person", "openlab")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
    payload TEXT NOT NULL, actor TEXT NOT NULL, session TEXT NOT NULL, at TEXT NOT NULL, prev TEXT NOT NULL,
    hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS edges(src TEXT NOT NULL, rel TEXT NOT NULL, dst TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
""" + "".join(f"CREATE INDEX IF NOT EXISTS nodes_{f} ON nodes(kind, json_extract(payload, '$.{f}'));\n"
              for f in schema.IDS)              # a source's ids: its duplicate check is a lookup, not a scan
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash(seq: int, kind: str, payload: str, links: list, actor: str, session: str, at: str, prev: str) -> str:
    text = f"{seq}|{kind}|{payload}|{canonical(links)}|{actor}|{session}|{at}|{prev}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Node:
    seq: int
    id: str
    kind: str
    payload: dict
    actor: str
    session: str
    at: str
    prev: str
    hash: str

    def body(self) -> "schema.Body":
        return schema.body(self.kind, self.payload)


def _node(row) -> Node:
    return Node(row["seq"], row["id"], row["kind"], json.loads(row["payload"]), row["actor"], row["session"],
                row["at"], row["prev"], row["hash"])


class Record:
    def __init__(self, study_dir: Path):
        self.folder = Path(study_dir)
        self.dir = self.folder / ".openlab"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "record.db"
        with closing(self._con()) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        return con

    def add(self, kind, payload: dict | None = None, *, actor: str = "agent", session: str = "",
            links: Iterable[tuple[str, str]] = (), once: bool = False) -> Node:
        """Append a node: `kind` is a schema Body, or a kind name with its payload. The schema's links and
        invariants are checked first; a node that breaks one is refused. `once`: if a node of this kind already
        exists, return the first one instead (atomic).

        A person node (actor "person") is signed here by the person key unlocked for this act
        (openlab.study.personkey.signing, made by openlab.agents.person.act after the passphrase); without one it is
        refused, and so is a key the study does not pin."""
        if isinstance(payload, dict) and personkey.SIG in payload:
            raise StudyError("`sig` is written by openlab, never by a caller",
                             "a person node is signed when the person acts, with their passphrase")
        body = schema.body(kind, payload)
        kind = body.KIND
        signer = None
        if actor == "person":                       # a person's act is signed, or it is not a person's act
            signer = personkey.current()
            if signer is None:
                raise StudyError(f"a person's {kind} must be signed with the person key",
                                 "this process holds no unlocked person key, so it is not the person acting: an "
                                 "agent or a script cannot write a person's act",
                                 "the person runs the act in their own terminal, where openlab asks for their "
                                 "passphrase (once, first: `openlab person init`)")
            if kind == "person_key" and body.fingerprint != signer.fingerprint:
                raise StudyError("a person_key is signed by the key it pins")
            session = session or signer.label
            if kind != "person_key":
                self._pin(signer)
        if not session:                             # every node names the agent that wrote it
            from ..agents.identity import current, ensure_agent_node
            session = current()
            if kind != "agent":
                ensure_agent_node(self, session)
        if actor not in ACTORS:
            raise StudyError(f"unknown actor `{actor}`", f"actors: {', '.join(ACTORS)}")
        text, links = canonical(dataclasses.asdict(body)), [[rel, dst] for rel, dst in links]
        with closing(self._con()) as con:
            con.execute("BEGIN IMMEDIATE")           # one writer at a time, so two processes cannot fork the chain
            try:
                if once:
                    row = con.execute("SELECT * FROM nodes WHERE kind=? ORDER BY seq LIMIT 1", (kind,)).fetchone()
                    if row is not None:
                        con.execute("ROLLBACK")
                        return _node(row)
                targets = []
                for rel, dst in links:
                    row = con.execute("SELECT * FROM nodes WHERE id=?", (dst,)).fetchone()
                    if row is None:
                        raise StudyError(f"link to unknown node `{dst}`", "a node can only link to earlier nodes")
                    targets.append((rel, _node(row)))
                schema.check(self, body, actor, targets)
                last = con.execute("SELECT seq, hash FROM nodes ORDER BY seq DESC LIMIT 1").fetchone()
                seq, prev = (last["seq"] + 1, last["hash"]) if last else (1, "")
                if signer is not None:
                    if kind != "person_key":                # the pin, inside the write lock
                        pin = con.execute("SELECT payload FROM nodes WHERE kind='person_key' ORDER BY seq DESC "
                                          "LIMIT 1").fetchone()
                        pinned = json.loads(pin["payload"])["fingerprint"] if pin else None
                        if pinned != signer.fingerprint:
                            raise StudyError(f"this study pins the person key {pinned}",
                                             f"the key unlocked is {signer.fingerprint}")
                    text = canonical({**json.loads(text), personkey.SIG: signer.sig(kind, text, links, session, prev)})
                at = now()
                h = _hash(seq, kind, text, links, actor, session, at, prev)
                node_id = f"{kind}:{h[:12]}"
                con.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",
                            (seq, node_id, kind, text, actor, session, at, prev, h))
                con.executemany("INSERT INTO edges VALUES (?,?,?)", [(node_id, rel, dst) for rel, dst in links])
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        return Node(seq, node_id, kind, json.loads(text), actor, session, at, prev, h)

    def _pin(self, signer: "personkey.Signer") -> None:
        """Before a study's first person node, a `person_key` node pinning the signer's key (checked against the
        workspace's pin when there is one); when the study pins an older key, the rotation that key endorsed."""
        try:                                        # the workspace first: its pin, and openlab.toml against its chain
            from ..agents.person import PinRefused, check_workspace
        except ImportError:                         # pragma: no cover -- a record read without the agents package
            check_workspace = None
        if check_workspace is not None:
            try:
                check_workspace(self.folder, signer)
            except PinRefused as e:
                raise StudyError(e.error, e.meaning + "; a study takes the workspace's key", e.fix) from None
        pins = self.find("person_key")
        if pins and pins[-1].payload["fingerprint"] == signer.fingerprint:
            return
        plan = self.find("plan")
        sealed = (plan[0].payload.get("person_key") or {}) if plan else {}
        if not pins and (not sealed or sealed["fingerprint"] == signer.fingerprint):
            self.add(schema.PersonKey(fingerprint=signer.fingerprint, public=signer.public, handle=signer.handle),
                     actor="person")
            return
        last = pins[-1].payload["fingerprint"] if pins else sealed["fingerprint"]
        hop = next((e for e in signer.endorsements if e.get("old") == last
                    and e.get("new_public") == signer.public), None)
        if hop is None:
            raise StudyError(f"this study {'pins' if pins else 'was sealed under'} the person key {last}",
                             f"the key unlocked is {signer.fingerprint}, and the pinned key never endorsed it",
                             "unlock the pinned key, or rotate from it (`openlab person rotate`)")
        self.add(schema.PersonKey(fingerprint=signer.fingerprint, public=signer.public, handle=signer.handle,
                                  rotates=last, endorsement=hop["sig"]), actor="person")

    def rows(self, upto: int | None = None) -> tuple[list, dict]:
        """The raw rows (oldest first, up to seq `upto`) and {node id: [[rel, dst], ...]}, for checking."""
        with closing(self._con()) as con:
            edges: dict = {}
            for src, rel, dst in con.execute("SELECT src, rel, dst FROM edges ORDER BY rowid"):
                edges.setdefault(src, []).append([rel, dst])
            q = "SELECT * FROM nodes" + (" WHERE seq<=?" if upto is not None else "") + " ORDER BY seq"
            rows = [dict(r) for r in con.execute(q, (upto,) if upto is not None else ())]
        return rows, edges

    def get(self, id: str) -> Node:
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM nodes WHERE id=?", (id,)).fetchone()
        if row is None:
            raise StudyError(f"no node `{id}` in the record")
        return _node(row)

    def find(self, kind: str | None = None, **where) -> list[Node]:
        """The nodes of `kind` whose payload has each `where` value, oldest first. A scalar is matched in SQL (by
        index where there is one); a list or table after loading."""
        want = json.loads(canonical(where))                 # compare as stored: tuples are lists, dates strings
        cond, args, rest = ["kind=?"] if kind else [], [kind] if kind else [], {}
        for k, v in want.items():
            if isinstance(v, (str, int, float)) and _FIELD.fullmatch(k):
                cond.append(f"json_extract(payload, '$.{k}')=?")
                args.append(v)
            else:
                rest[k] = v
        sql = "SELECT * FROM nodes" + (" WHERE " + " AND ".join(cond) if cond else "") + " ORDER BY seq"
        with closing(self._con()) as con:
            nodes = [_node(r) for r in con.execute(sql, args)]
        return [n for n in nodes if all(k in n.payload and n.payload[k] == v for k, v in rest.items())]

    def last(self, kind: str, **where) -> Node | None:
        found = self.find(kind, **where)
        return found[-1] if found else None

    def kinds(self) -> dict[str, int]:
        """{kind: number of nodes} over the record."""
        with closing(self._con()) as con:
            return dict(con.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY kind").fetchall())

    def links(self, id: str, rel: str | None = None, incoming: bool = False) -> list[tuple[str, str, str]]:
        col = "dst" if incoming else "src"
        sql = f"SELECT src, rel, dst FROM edges WHERE {col}=?" + (" AND rel=?" if rel else "") + " ORDER BY rowid"
        with closing(self._con()) as con:
            return [tuple(r) for r in con.execute(sql, (id, rel) if rel else (id,))]

    def _walk(self, id: str, incoming: bool, rels: tuple = ()) -> list[Node]:
        self.get(id)
        seen, todo = set(), [id]
        while todo:
            for src, rel, dst in self.links(todo.pop(), incoming=incoming):
                nxt = src if incoming else dst
                if nxt != id and nxt not in seen and (not rels or rel in rels):
                    seen.add(nxt)
                    todo.append(nxt)
        return sorted((self.get(i) for i in seen), key=lambda n: n.seq)

    def why(self, id: str) -> list[Node]:
        """Everything `id` depends on: its links, followed transitively, oldest first."""
        return self._walk(id, incoming=False)

    def impact(self, id: str, rels: tuple = ()) -> list[Node]:
        """Everything that depends on `id`, oldest first; with `rels`, through those relations only."""
        return self._walk(id, incoming=True, rels=rels)

    def verify(self, expected_key: "str | list | None" = None) -> dict:
        """Recompute every hash and prev link (`broken_at` is the first seq that does not match), and check every
        person node's signature against the key the study pinned (openlab.study.personkey.audit; `expected_key`: the
        fingerprints pinned elsewhere, e.g. the workspace's pin chain). `ok` fails on a broken chain, a forged or unpinned
        signature, a person-only kind written by another actor, or an unsigned person node under an agent's session;
        unsigned person nodes from before signatures under a person's terminal are `legacy` warnings."""
        prev, n, broken = "", 0, None
        rows, edges = self.rows()
        for r in rows:
            n += 1
            h = _hash(r["seq"], r["kind"], r["payload"], edges.get(r["id"], []), r["actor"], r["session"],
                      r["at"], r["prev"])
            if broken is None and (r["seq"] != n or r["prev"] != prev or r["hash"] != h
                                   or r["id"] != f"{r['kind']}:{h[:12]}"):
                broken = r["seq"]
            prev = r["hash"]
        sig = personkey.audit(rows, edges, expected_key)
        signatures = {"ok": sig["ok"] and not sig["agent_written"], "signed": sig["signed"], "pinned": sig["pinned"],
                      "invalid": sig["invalid"] + sig["agent_written"],
                      "legacy": [x for x in sig["legacy"]
                                 if x["seq"] not in {y["seq"] for y in sig["agent_written"]}]}
        return {"ok": broken is None and signatures["ok"], "nodes": n, "broken_at": broken, "head": prev,
                "signatures": signatures}

    def head(self) -> str:
        with closing(self._con()) as con:
            row = con.execute("SELECT hash FROM nodes ORDER BY seq DESC LIMIT 1").fetchone()
        return row["hash"] if row else ""

    def put_blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        path = self.dir / "blobs" / sha
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_name(f".{sha}.{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return sha

    def blob(self, sha: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise StudyError(f"`{sha}` is not a sha256")
        try:
            data = (self.dir / "blobs" / sha).read_bytes()
        except FileNotFoundError:
            raise StudyError(f"no blob {sha[:12]} in the content store") from None
        if hashlib.sha256(data).hexdigest() != sha:
            raise StudyError(f"blob {sha[:12]} was changed", "its content no longer matches its name")
        return data
