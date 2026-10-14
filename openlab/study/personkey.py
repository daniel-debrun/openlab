"""The person's signature on the record: every node whose actor is `person` carries `sig`, an Ed25519 signature by
the person key over the node's kind, payload (without `sig`), links, session and the hash of the node before it,
so a signed node cannot be moved, copied into another record or edited without breaking its signature.

A study pins the person key: its first person node is preceded by a `person_key` node (the public key, signed by
itself); a later key is accepted only through a `person_key` rotation endorsed by the pinned key. `verify` checks
every person node against the key pinned at that point of the chain.

This module only signs and checks. The key file, the passphrase and the prompt live in openlab.agents.person: a
`Signer` exists only after the person typed their passphrase, and Record.add signs with the one `signing()` made
active. Nothing here reads an environment variable.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

SIG = "sig"                                 # the reserved payload key of a signed node
ROTATE = "openlab-person-rotate"             # the domain of a rotation endorsement


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def fingerprint(public: bytes | str) -> str:
    """`ed25519:` and the first 16 bytes of the public key's sha256, hex in groups of 4."""
    raw = unb64(public) if isinstance(public, str) else public
    h = hashlib.sha256(raw).hexdigest()[:32]
    return "ed25519:" + ":".join(h[i:i + 4] for i in range(0, 32, 4))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def message(kind: str, payload_text: str, links: list, session: str, prev: str) -> bytes:
    """What a person node's signature covers. `payload_text` is the canonical payload without `sig`."""
    return f"openlab-node|{kind}|{payload_text}|{canonical(links)}|{session}|{prev}".encode("utf-8")


def rotation_message(old_fp: str, new_public: str) -> bytes:
    return f"{ROTATE}|{old_fp}|{new_public}".encode("utf-8")


def verify(public: str, msg: bytes, sig: str) -> bool:
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
        VerifyKey(unb64(public)).verify(msg, unb64(sig))
        return True
    except (BadSignatureError, ValueError, TypeError):
        return False
    except Exception:                                               # noqa: BLE001 -- a malformed key never verifies
        return False


@dataclass
class Signer:
    """The person's unlocked key, for one act. `endorsements`: the rotation chain from older keys to this one,
    oldest first ({old, old_public, new_public, sig}), so a study pinned to an older key accepts this one."""
    secret: object                                  # nacl.signing.SigningKey
    handle: str
    endorsements: list = field(default_factory=list)
    summary: str = ""                               # what the person was shown when they typed the passphrase

    @property
    def public(self) -> str:
        return b64(bytes(self.secret.verify_key))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.public)

    @property
    def label(self) -> str:
        return f"person:{self.handle}"

    def sign(self, msg: bytes) -> str:
        return b64(self.secret.sign(msg).signature)

    def sig(self, kind: str, payload_text: str, links: list, session: str, prev: str) -> dict:
        return {"key": self.fingerprint, "by": self.label,
                "value": self.sign(message(kind, payload_text, links, session, prev))}


_SIGNER: contextvars.ContextVar = contextvars.ContextVar("openlab_person_signer", default=None)


@contextmanager
def signing(signer: Optional[Signer]):
    """Person nodes written in this block are signed by `signer` (a no-op for None)."""
    if signer is None:
        yield None
        return
    token = _SIGNER.set(signer)
    try:
        yield signer
    finally:
        _SIGNER.reset(token)


def current() -> Optional[Signer]:
    return _SIGNER.get()


def activate(signer: Signer) -> None:
    """Make `signer` active for the rest of this context (the compatibility path of person.confirm)."""
    _SIGNER.set(signer)


def split(payload: dict) -> tuple[dict, Optional[dict]]:
    """(payload without `sig`, sig or None)."""
    if not isinstance(payload, dict) or SIG not in payload:
        return payload, None
    rest = {k: v for k, v in payload.items() if k != SIG}
    return rest, payload[SIG]


# --- checking a record --------------------------------------------------------------------------------------------
def check_rotation(prev_fp: str, prev_public: str, new_public: str, endorsement: str) -> bool:
    return bool(endorsement) and verify(prev_public, rotation_message(prev_fp, new_public), endorsement)


def audit(rows: list, edges: dict, expected=None) -> dict:
    """Check every person node of a record. `rows`: the record's rows in seq order (dicts with seq, id, kind,
    payload (text), actor, session, prev). `edges`: {node id: [[rel, dst], ...]}. `expected`: the fingerprints pinned
    elsewhere (the workspace's pin chain, or one fingerprint); every key the study pins must be one of them, so a
    study signed by a key the workspace never pinned fails.

    Returns {ok, signed, pinned, legacy: [...], invalid: [...], agent_written: [...]}: `legacy` are unsigned person
    nodes before the study pinned a key (records from before signatures; warnings); `invalid` are unsigned person
    nodes after the pin, bad signatures, signatures by an unpinned key, and person-only kinds written by another
    actor; `agent_written` are legacy person nodes whose session names an agent (an agent's act recorded as a
    person's)."""
    from .schema import PERSON_ONLY
    allowed = [expected] if isinstance(expected, str) else list(expected or [])
    pinned: Optional[tuple[str, str]] = None        # (fingerprint, public)
    first_pin: Optional[str] = None
    sealed: dict = {}                               # the key the plan was sealed under (schema.Plan.person_key)
    out = {"signed": 0, "legacy": [], "invalid": [], "agent_written": [], "pinned": None}
    for r in rows:
        payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
        body, sig = split(payload)
        where = {"seq": r["seq"], "id": r["id"], "kind": r["kind"], "session": r["session"]}
        if r["kind"] == "plan" and isinstance(body, dict):
            sealed = body.get("person_key") or {}
        if r["actor"] != "person":
            if r["kind"] in PERSON_ONLY or sig is not None:
                out["invalid"].append({**where, "why": f"a {r['kind']} written by {r['actor']} "
                                                       f"({r['session'] or 'no session'}), not a signed person act"
                                       if r["kind"] in PERSON_ONLY else "a signature on a non-person node"})
            continue
        if sig is None:
            if pinned is None:
                out["legacy"].append({**where, "why": "unsigned (legacy)"})
                s = r["session"] or ""
                if s and not s.startswith(("terminal:", "person:")):
                    out["agent_written"].append({**where, "why": f"a person's {r['kind']} recorded under agent "
                                                                 f"session {s}"})
            else:
                out["invalid"].append({**where, "why": "an unsigned person node after the study pinned a person key"})
            continue
        if not isinstance(sig, dict) or not all(isinstance(sig.get(k), str) for k in ("key", "value")):
            out["invalid"].append({**where, "why": "a malformed signature"})
            continue
        if r["kind"] == "person_key":
            fp, pub = body.get("fingerprint"), body.get("public", "")
            try:
                ok_fp = fingerprint(pub) == fp
            except Exception:                                       # noqa: BLE001
                ok_fp = False
            if not ok_fp:
                out["invalid"].append({**where, "why": "a person_key whose fingerprint is not its key's"})
                continue
            if pinned is None:
                if allowed and fp not in allowed:
                    out["invalid"].append({**where, "why": f"the study pins {fp}, but the workspace pins "
                                                           f"{' -> '.join(allowed)}"})
                if sealed and fp != sealed.get("fingerprint") and not (
                        body.get("rotates") == sealed.get("fingerprint")
                        and check_rotation(sealed["fingerprint"], sealed.get("public", ""), pub,
                                           body.get("endorsement", ""))):
                    out["invalid"].append({**where, "why": f"the study was sealed under {sealed.get('fingerprint')}, "
                                                           f"but its first pin is {fp}, not a rotation that key signed"})
                first_pin = first_pin or fp
            elif not check_rotation(pinned[0], pinned[1], pub, body.get("endorsement", "")) or \
                    body.get("rotates") != pinned[0]:
                out["invalid"].append({**where, "why": f"a key change to {fp} not endorsed by the pinned {pinned[0]}"})
                continue
            elif allowed and fp not in allowed:
                out["invalid"].append({**where, "why": f"the study rotates to {fp}, which the workspace never pinned "
                                                       f"({' -> '.join(allowed)})"})
            key = (fp, pub)
        else:
            if pinned is None:
                out["invalid"].append({**where, "why": "a signed person node before any person_key pin"})
                continue
            key = pinned
        msg = message(r["kind"], canonical(body), edges.get(r["id"], []), r["session"], r["prev"])
        if sig["key"] != key[0]:
            out["invalid"].append({**where, "why": f"signed by {sig['key']}, but the study pins {key[0]}"})
        elif not verify(key[1], msg, sig["value"]):
            out["invalid"].append({**where, "why": "the signature does not match the node (edited, moved or forged)"})
        else:
            out["signed"] += 1
            if r["kind"] == "person_key":
                pinned = key
    out["pinned"] = pinned[0] if pinned else None
    out["first_pin"] = first_pin
    out["ok"] = not out["invalid"]
    return out


def node_signed(rec, node) -> bool:
    """Whether one node is a person node whose signature checks against the key the study pinned before it."""
    if node.actor != "person" or SIG not in (node.payload or {}):
        return False
    rows, edges = rec.rows(upto=node.seq)
    res = audit(rows, edges)
    return not any(x["seq"] == node.seq for x in res["invalid"] + res["legacy"])
