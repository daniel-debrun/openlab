"""Signed files: a data file a program downstream acts on (a study forecast, a reproduction bundle, a paper) gets
``<file>.sig.json`` beside it, an Ed25519 signature over the canonical JSON of its manifest and the file's sha256.

The public key travels in the manifest; whether that key is the signer's is a question of where you got its id.
The private key sits outside the workspace under the user's data dir (``$OPENLAB_SIGNING_KEY`` overrides), made on
the first signature. ``openlab verify FILE.sig.json`` checks one offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


# --- the key ------------------------------------------------------------------------------------------------
def _key_path() -> Path:
    if os.environ.get("OPENLAB_SIGNING_KEY"):
        return Path(os.environ["OPENLAB_SIGNING_KEY"]).expanduser()
    home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return home / "openlab" / "keys" / "signing.key"


def key_info() -> dict:
    """The signing key's public half and id, for publishing -- never the private half. No key yet: says so."""
    from nacl.encoding import HexEncoder
    from nacl.signing import SigningKey
    p = _key_path()
    if not p.exists():
        return {"exists": False, "path": str(p), "note": "no signing key yet: one is made on the first signature"}
    pub = SigningKey(bytes.fromhex(p.read_text().strip())).verify_key.encode(HexEncoder).decode()
    return {"exists": True, "path": str(p), "public_key": pub, "key_id": key_id(pub),
            "note": "publish the key id where readers of your reports can find it; verify compares against it"}


def signing_key():
    from nacl.signing import SigningKey
    p = _key_path()
    if p.exists():
        return SigningKey(bytes.fromhex(p.read_text().strip()))
    key = SigningKey.generate()
    p.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    p.write_text(key.encode().hex())
    os.chmod(p, 0o600)
    return key


def key_id(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]


# --- signed outputs: a file a program downstream acts on ---------------------------------------------
SIG_SUFFIX = ".sig.json"


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_file(path: Path, manifest: dict) -> Path:
    """Sign a data file for a downstream program: ``<path>.sig.json`` holds ``manifest`` plus the file's sha256,
    signed with the same Ed25519 key as the reports. The signature covers the canonical JSON of the manifest
    (keys sorted, no spaces), so any consumer can check it in a few lines without openlab."""
    from nacl.encoding import HexEncoder
    key = signing_key()
    pub = key.verify_key.encode(HexEncoder).decode()
    body = {**manifest, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "algorithm": "ed25519",
            "public_key": pub, "key_id": key_id(pub)}
    sig = key.sign(_canonical(body), encoder=HexEncoder).signature.decode()
    out = path.with_name(path.name + SIG_SUFFIX)
    out.write_text(json.dumps({**body, "signature": sig}, indent=1), encoding="utf-8")
    return out


def verify_file(sig_path: Path) -> dict:
    """Check a ``.sig.json``: the signature over its manifest, and the data file beside it against its sha256."""
    from nacl.encoding import HexEncoder
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey
    meta = json.loads(Path(sig_path).read_text(encoding="utf-8"))
    sig = meta.pop("signature", "")
    who = {k: meta.get(k) for k in ("file", "frame", "notebook", "study", "t", "gate", "exported_at", "expires_at",
                                    "monitor", "key_id")}
    try:
        VerifyKey(meta["public_key"], encoder=HexEncoder).verify(_canonical(meta), bytes.fromhex(sig))
    except (BadSignatureError, ValueError, KeyError):
        return {"ok": False, "why": "the signature does not match the manifest", **who}
    data = Path(sig_path).with_name(Path(sig_path).name[:-len(SIG_SUFFIX)])
    try:
        got = hashlib.sha256(data.read_bytes()).hexdigest()
    except OSError:
        return {"ok": False, "why": f"the signed file {data.name} is missing", **who}
    if got != meta.get("sha256"):
        return {"ok": False, "why": f"{data.name} was changed after it was signed", **who}
    return {"ok": True, **who}
