"""The person key: the secret an agent's shell cannot reach, so a person's act is the person's.

`openlab person init` makes an Ed25519 key pair and stores the private key encrypted with the person's passphrase
(argon2id, then a SecretBox) at `~/.config/openlab/person.key` (0600), the public key beside it (`person.pub`).
Every person act goes through `act(summary, details)`: it shows exactly what will be signed, asks for the
passphrase on the terminal (/dev/tty, three tries), and makes the unlocked key active for the act, so every person
node the act writes is signed (openlab.study.record.Record.add). The passphrase IS the confirmation.

What this holds against, honestly: an agent on the same OS account can read the encrypted key file, but cannot sign
without the passphrase. It can still fake a terminal, run any verb, and write the record's SQLite directly; none of
that yields a valid person signature, and `openlab study verify` reports what it did. Each study pins the key in its
first person node; the workspace pins it in openlab.toml ([person] fingerprint) and as a hash-chained `person_pin`
event in its control store. A pinned workspace takes another key only through a rotation the pinned key signed
(`pin_workspace`, `check_workspace`), whatever HOME says, and an openlab.toml that disagrees with the chain refuses
person acts. Its remaining move is to delete the control store and rebuild the workspace and every study under a
key of its own, which shows only against a fingerprint kept elsewhere: keep it somewhere an agent cannot write (a
commit, a note) and compare it with `openlab person show`.

Nothing here reads an environment variable for the passphrase or the key: tests replace `_ask` (the prompt) only.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ..study import personkey
from ..study.personkey import Signer, b64, unb64

TRIES = 3
#: argon2id cost of the passphrase (tests may lower it; the file records the cost it was made with)
KDF = {"opslimit": None, "memlimit": None}


class NotAPerson(Exception):
    pass


class BadPassphrase(NotAPerson):
    pass


def config_dir() -> Path:
    return Path.home() / ".config" / "openlab"


def key_path() -> Path:
    return config_dir() / "person.key"


def pub_path() -> Path:
    return config_dir() / "person.pub"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the terminal ------------------------------------------------------------------------------------------------
def _tty_out(text: str) -> None:
    try:
        with open("/dev/tty", "w") as f:
            f.write(text + "\n")
    except OSError:
        print(text, file=sys.stderr)


def _ask(prompt: str) -> str:
    """One secret from the person at the terminal (/dev/tty, not echoed). Tests replace this function."""
    try:
        with open("/dev/tty"):
            pass
    except OSError:
        raise NotAPerson("a person act asks for the person's passphrase on a terminal, and this process has none") \
            from None
    return getpass.getpass(prompt)


def _line(prompt: str, default: str = "") -> str:
    """One visible line from the terminal (the handle)."""
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(prompt)
            tty.flush()
            return tty.readline().strip() or default
    except OSError:
        return default


# --- the key file ------------------------------------------------------------------------------------------------
def _kdf(passphrase: str, salt: bytes, opslimit: int, memlimit: int) -> bytes:
    from nacl import pwhash, secret
    return pwhash.argon2id.kdf(secret.SecretBox.KEY_SIZE, passphrase.encode("utf-8"), salt,
                               opslimit=opslimit, memlimit=memlimit)


def _seal(seed: bytes, passphrase: str) -> dict:
    from nacl import pwhash, secret, utils
    ops = KDF["opslimit"] or pwhash.argon2id.OPSLIMIT_MODERATE
    mem = KDF["memlimit"] or pwhash.argon2id.MEMLIMIT_MODERATE
    salt = utils.random(pwhash.argon2id.SALTBYTES)
    box = secret.SecretBox(_kdf(passphrase, salt, ops, mem)).encrypt(seed)
    return {"kdf": "argon2id", "opslimit": ops, "memlimit": mem, "salt": b64(salt), "box": b64(bytes(box))}


def _write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _write_pub(data: dict) -> None:
    pub = {k: data[k] for k in ("handle", "public", "fingerprint", "created")}
    pub_path().write_text(json.dumps(pub, indent=1) + "\n", encoding="utf-8")


def exists() -> bool:
    return key_path().is_file()


def info() -> Optional[dict]:
    """The public part of the person key (handle, public, fingerprint), or None when there is none."""
    for p in (pub_path(), key_path()):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return {k: d.get(k) for k in ("handle", "public", "fingerprint", "created")}
        except (OSError, ValueError):
            continue
    return None


def handle() -> Optional[str]:
    i = info()
    return i["handle"] if i else None


def create(handle_: str, passphrase: str) -> dict:
    """Make and store a new person key (refused when one exists: rotate it instead). Returns its public part."""
    from nacl.signing import SigningKey
    if exists():
        raise NotAPerson(f"a person key exists at {key_path()}: rotate it with `openlab person rotate`")
    if len(passphrase) < 8:
        raise NotAPerson("a passphrase has at least 8 characters")
    handle_ = (handle_ or "").strip() or getpass.getuser()
    sk = SigningKey.generate()
    public = b64(bytes(sk.verify_key))
    data = {"version": 1, "handle": handle_, "public": public, "fingerprint": personkey.fingerprint(public),
            "created": now(), "endorsements": [], **_seal(bytes(sk), passphrase)}
    _write_private(key_path(), data)
    _write_pub(data)
    return info()


def unlock(passphrase: str, summary: str = "") -> Signer:
    """The person key, decrypted with `passphrase`; BadPassphrase when it does not open."""
    from nacl import exceptions, secret
    from nacl.signing import SigningKey
    try:
        d = json.loads(key_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotAPerson("no person key here: run `openlab person init` first, in your own terminal") from None
    except (OSError, ValueError) as e:
        raise NotAPerson(f"the person key at {key_path()} cannot be read: {e}") from None
    try:
        key = _kdf(passphrase, unb64(d["salt"]), int(d["opslimit"]), int(d["memlimit"]))
        seed = secret.SecretBox(key).decrypt(unb64(d["box"]))
    except (exceptions.CryptoError, ValueError, KeyError):
        raise BadPassphrase("that passphrase does not open the person key; nothing was done") from None
    signer = Signer(SigningKey(seed), d["handle"], list(d.get("endorsements") or []), summary)
    if signer.public != d["public"]:
        raise NotAPerson("the person key file is inconsistent (its public key is not its private key's)")
    return signer


def unlock_interactive(summary: str, details: Iterable[str] = ()) -> Signer:
    """Show what will be signed, then ask for the passphrase (three tries). Run by `openlab person act HOST` on
    another machine (OPENLAB_SIGNER=stdio), the key is on the person's machine: it is asked there instead
    (openlab.agents.remote)."""
    from . import remote
    if remote.stdio():
        return remote.hello(summary, details)
    if not exists():
        raise NotAPerson(f"{summary or 'this'} is a person's act, signed with the person key, and there is none "
                         "here: run `openlab person init` first, in your own terminal")
    lines = [f"openlab person act: {summary}", *[f"  {d}" for d in details if d]]
    _tty_out("\n".join(lines))
    for i in range(TRIES):
        pw = _ask("passphrase to sign (empty cancels): ")
        if not pw:
            raise NotAPerson("cancelled; nothing was done")
        try:
            return unlock(pw, summary)
        except BadPassphrase:
            if i < TRIES - 1:
                _tty_out("that passphrase does not open the person key; try again")
    raise BadPassphrase(f"{TRIES} wrong passphrases; nothing was done")


@contextmanager
def act(summary: str, details: Iterable[str] = ()):
    """A person's act: shows `summary` and `details`, asks for the passphrase, and yields the unlocked Signer;
    every person node written inside the block is signed by it. Raises NotAPerson without a key, a terminal or the
    passphrase."""
    signer = unlock_interactive(summary, details)
    with personkey.signing(signer):
        yield signer


@contextmanager
def with_passphrase(passphrase: str, summary: str = ""):
    """A person's act whose passphrase arrived another way (the web app's request body): the server proves the
    person by opening the key with it."""
    signer = unlock(passphrase, summary)
    with personkey.signing(signer):
        yield signer


def confirm(what: str, detail: str = "", answer: str = "y") -> str:
    """The one confirmation of every person act, for callers written against the old typed confirmation: shows
    `what` and each line of `detail` (what will be signed), asks for the passphrase, makes the key active for the
    rest of this context (so the act's person nodes are signed), and returns the person's label
    (`person:<handle>`). Raises NotAPerson. New code may use the `act` context manager instead."""
    signer = unlock_interactive(what, detail.splitlines() if detail else [])
    personkey.activate(signer)
    return signer.label


def rotate(old: str, new: str) -> dict:
    """A new key, endorsed by the old one (the old passphrase opens it); studies pinned to the old key take the new
    one through a `person_key` rotation node at the next person act."""
    from nacl.signing import SigningKey
    signer = unlock(old)
    if len(new) < 8:
        raise NotAPerson("a passphrase has at least 8 characters")
    sk = SigningKey.generate()
    public = b64(bytes(sk.verify_key))
    hop = {"old": signer.fingerprint, "old_public": signer.public, "new_public": public,
           "sig": signer.sign(personkey.rotation_message(signer.fingerprint, public)), "at": now()}
    data = {"version": 1, "handle": signer.handle, "public": public, "fingerprint": personkey.fingerprint(public),
            "created": now(), "endorsements": [*signer.endorsements, hop], **_seal(bytes(sk), new)}
    _write_private(key_path(), data)
    _write_pub(data)
    return {**info(), "rotated_from": signer.fingerprint}


# --- pins --------------------------------------------------------------------------------------------------------
def _toml_pin(root: Path) -> Optional[str]:
    p = root / "openlab.toml"
    if not p.is_file():
        return None
    try:
        try:
            import tomllib
        except ImportError:                                         # Python 3.10
            import tomli as tomllib
        return ((tomllib.loads(p.read_text(encoding="utf-8")).get("person") or {}).get("fingerprint") or None)
    except Exception:                                               # noqa: BLE001
        return None


class PinRefused(NotAPerson):
    """A pin the workspace refuses, with what it means and how to go on (the shape of a study refusal)."""

    def __init__(self, error: str, meaning: str = "", fix: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.fix = error, meaning, fix


def _fp_of(public: str) -> Optional[str]:
    try:
        return personkey.fingerprint(public) if public else None
    except Exception:                                               # noqa: BLE001 -- a malformed key is no key
        return None


def endorsed(from_fp: str, from_public: Optional[str], hops: list, to_public: str) -> bool:
    """Whether `hops` (rotation endorsements {old, old_public, new_public, sig}, oldest first) lead, each signed by
    the key before it, from the key `from_fp` to `to_public`. Hops before `from_fp` (older history) are skipped. A
    claimed endorsement is checked, never trusted: a key file's list is the agent's to write."""
    at, at_pub = from_fp, from_public
    for h in hops or []:
        if not isinstance(h, dict) or h.get("old") != at:
            continue
        old_pub = at_pub or h.get("old_public") or ""
        new_pub = h.get("new_public") or ""
        if _fp_of(old_pub) != at or not personkey.check_rotation(at, old_pub, new_pub, h.get("sig") or ""):
            return False
        at, at_pub = _fp_of(new_pub), new_pub
        if at_pub == to_public:
            return True
    return False


def _pin_events(root: Path) -> list:
    from .store import Store
    return Store(root).events(None, n=1_000_000, kinds=("person_pin",))


def pin_state(folder) -> dict:
    """The workspace's pin, checked. The control store's `person_pin` events are a chain: the first pins a key, and
    each later pin of another key must carry the rotation endorsed by the key pinned before it (anything else is
    ignored and listed in `invalid`). openlab.toml `[person] fingerprint` must equal where that chain ends; when it
    does not (the file was edited), `conflict` says so and person acts are refused.

    Returns {root, pin, chain: [fingerprints, oldest first], first, toml, invalid: [...], conflict, broken_at}.
    `pin` is the chain's end (or openlab.toml's fingerprint when the control store holds no pin). Never creates a
    control store."""
    from .store import Store, root_of
    root = root_of(folder)
    toml_file = (root / "openlab.toml").is_file()
    toml = _toml_pin(root)
    chain: list = []
    invalid: list = []
    broken, last_seq = None, 0
    cur = cur_pub = None
    if (root / ".openlab" / "control.db").is_file():
        try:
            events = _pin_events(root)
            broken = Store(root).verify()["broken_at"]
        except Exception:                                           # noqa: BLE001 -- unreadable: no chain
            events = []
        for ev in events:
            pl = ev.get("payload") or {}
            fp, pub = pl.get("fingerprint"), pl.get("public") or ""
            last_seq = ev["seq"]
            if not fp or _fp_of(pub) != fp:
                invalid.append({"seq": ev["seq"], "fingerprint": fp, "why": "a pin whose fingerprint is not its key's"})
                continue
            if cur is None:
                cur, cur_pub = fp, pub
                chain.append(fp)
            elif fp == cur:
                continue
            elif pl.get("rotates") == cur and endorsed(cur, cur_pub, pl.get("hops") or [], pub):
                cur, cur_pub = fp, pub
                chain.append(fp)
            else:
                invalid.append({"seq": ev["seq"], "fingerprint": fp,
                                "why": f"a pin of {fp} that the pinned key {cur} never endorsed (ignored)"})
    conflict = ""
    if chain and toml_file and toml != cur:
        conflict = (f"openlab.toml [person] fingerprint is {toml or 'missing'}, but the workspace's pin chain in the "
                    f"control store (first pin {chain[0]}) ends at {cur}: openlab.toml was edited")
    elif chain and broken is not None and broken <= last_seq:
        conflict = f"the control store's hash chain is broken at event {broken}, at or before a pin: it was edited"
    return {"root": str(root), "pin": cur if chain else toml, "chain": chain or ([toml] if toml else []),
            "first": chain[0] if chain else toml, "toml": toml, "invalid": invalid, "conflict": conflict,
            "broken_at": broken}


def workspace_pin(folder) -> Optional[str]:
    """The person key the workspace pins (pin_state's `pin`): the end of the control store's pin chain, else
    openlab.toml [person] fingerprint. None when neither is set. Never creates a control store."""
    return pin_state(folder)["pin"]


def check_workspace(folder, signer: Signer) -> None:
    """Refuse a person act (PinRefused) when openlab.toml disagrees with the control store's pin chain, or the
    workspace pins another key than `signer`'s. A workspace with no pin takes any key (the study pins it at its
    first person node)."""
    s = pin_state(folder)
    if s["conflict"]:
        raise PinRefused("this workspace's person-key pin was tampered with", s["conflict"],
                         f'put fingerprint = "{s["pin"]}" back under [person] in openlab.toml (by hand, in your own '
                         "editor), or run `openlab person pin` with the pinned key; `openlab doctor` and `openlab study "
                         "verify` report it until then")
    if s["pin"] and s["pin"] != signer.fingerprint:
        rotated = endorsed(s["pin"], None, signer.endorsements, signer.public)
        raise PinRefused(f"this workspace pins the person key {s['pin']}",
                         f"the key unlocked is {signer.fingerprint}" + (
                             ", a rotation the pinned key endorsed, but the workspace still pins the old one"
                             if rotated else ", and the pinned key never endorsed it"),
                         "run `openlab person pin` here to move the workspace to your rotated key" if rotated else
                         "unlock the pinned key (compare fingerprints with `openlab person show`); a workspace takes "
                         "another key only through `openlab person rotate`, which needs the pinned key's passphrase")


def pinned_public(root) -> Optional[str]:
    """The public key the workspace's pin chain ends at (the control store's person_pin events), else the public
    key of the person key under HOME; None when there is neither. What a signed direction is checked against."""
    s = pin_state(root)
    if s["chain"] and not s["conflict"]:
        try:
            for ev in reversed(_pin_events(Path(s["root"]))):
                pl = ev.get("payload") or {}
                if pl.get("fingerprint") == s["pin"]:
                    return pl.get("public")
        except Exception:                                           # noqa: BLE001
            pass
    i = info()
    return i["public"] if i else None


def tell_message(agent: str, text: str) -> bytes:
    return f"openlab-tell|{agent}|{text}".encode("utf-8")


def pin_workspace(root, signer: Signer) -> dict:
    """Pin the key in the workspace: a `person_pin` event in the control store and `[person] fingerprint` in
    openlab.toml (when the workspace has one). The first pin takes any key; after that only the pinned key itself
    (which also repairs an edited openlab.toml) or a rotation the pinned key endorsed, whatever HOME says; else
    PinRefused. Written by the person, in their terminal; agents cannot edit openlab.toml (the hooks)."""
    from .store import Store, root_of
    root = root_of(root)
    s = pin_state(root)
    cur = s["pin"]
    payload = {"fingerprint": signer.fingerprint, "public": signer.public, "handle": signer.handle,
               "root": str(root), "sig": signer.sign(f"openlab-workspace-pin|{root}|{signer.fingerprint}".encode())}
    if cur and cur != signer.fingerprint:
        if not endorsed(cur, None, signer.endorsements, signer.public):
            raise PinRefused(f"this workspace pins the person key {cur}",
                             f"the key unlocked is {signer.fingerprint}, and the pinned key never endorsed it: a "
                             "pinned workspace takes another key only through a rotation signed by the key it pins, "
                             "so a key made under another HOME cannot replace it",
                             "rotate from the pinned key with `openlab person rotate` (it asks for the pinned key's "
                             "passphrase, then re-pins this workspace); a lost pinned key means a new workspace")
        hops, at = [], cur
        for h in signer.endorsements:                       # the hops from the pinned key to this one
            if isinstance(h, dict) and h.get("old") == at:
                hops.append(h)
                at = _fp_of(h.get("new_public") or "")
        payload.update({"rotates": cur, "hops": hops})
    if not (s["chain"] and s["chain"][-1] == signer.fingerprint and (root / ".openlab" / "control.db").is_file()
            and _pin_events(root)):
        Store(root).event("workspace", "person_pin", summary=f"person key {signer.fingerprint} ({signer.label})"
                          + (f", rotated from {cur}" if payload.get("rotates") else ""), payload=payload)
    toml = root / "openlab.toml"
    if toml.is_file():
        text = toml.read_text(encoding="utf-8")
        line = f'fingerprint = "{signer.fingerprint}"'
        if "[person]" in text:
            import re
            body = re.sub(r'(?m)^fingerprint\s*=\s*"[^"]*"\s*$', line, text) if re.search(
                r'(?m)^fingerprint\s*=', text) else text.replace("[person]", f"[person]\n{line}", 1)
        else:
            body = text.rstrip("\n") + f"\n\n[person]\n# the person key that signs person acts (openlab person show)\n{line}\n"
        toml.write_text(body, encoding="utf-8")
    return {"workspace": str(root), "fingerprint": signer.fingerprint, "toml": toml.is_file(),
            "rotated_from": payload.get("rotates")}


def label() -> Optional[str]:
    """`person:<handle>` when this account has a person key, else None (identity's terminal label)."""
    h = handle()
    return f"person:{h}" if h else None
