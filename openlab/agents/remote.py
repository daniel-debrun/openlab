"""Person acts on a machine that does not hold the person key: the record lives on the box, the key stays with the
person.

`openlab person act HOST -- ARGS` (the person's machine) runs `openlab ARGS` on HOST over SSH with OPENLAB_SIGNER=stdio.
There, the act's signer is a `RemoteSigner`: each signature it needs is a request on the SSH channel (a marked line
on stdout), answered on stdin by the person's machine, which:

- shows what the act says it is and asks for the passphrase there, once (the key is unlocked on the person's
  machine only);
- signs only messages of the purposes a person act signs (a record node, a direction to an agent, a resume, an
  inbox decision, a workspace pin), shows each one as it signs it, and never a key rotation (that stays local);
- signs at most MAX_SIGNATURES messages for one act, and nothing once the command has ended.

Everything else the remote command prints passes through to the person's terminal. Whoever controls the box
controls what it asks to sign; what the person's machine shows is what it signs.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
from typing import Iterable, Optional

from ..study import personkey

MARK = "\x1eopenlab-sign "                   # a request line on the remote's stdout
ENV = "OPENLAB_SIGNER"
MAX_SIGNATURES = 64
ALLOWED = {"openlab-node": "a record node", "openlab-tell": "a direction to an agent", "openlab-resume": "a resume",
           "openlab-inbox": "an inbox decision", "openlab-workspace-pin": "the workspace's pin"}


def stdio() -> bool:
    return os.environ.get(ENV) == "stdio"


# --- the box's side ----------------------------------------------------------------------------------------------
def _ask(req: dict) -> dict:
    """One request to the person's machine and its answer."""
    out = sys.__stdout__
    out.write(MARK + json.dumps(req) + "\n")
    out.flush()
    line = sys.__stdin__.readline()
    if not line:
        raise personkey_error("the person's machine closed the channel; nothing more is signed")
    ans = json.loads(line)
    if ans.get("error"):
        raise personkey_error(ans["error"])
    return ans


def personkey_error(text: str) -> Exception:
    from .person import NotAPerson
    return NotAPerson(text)


class RemoteSigner(personkey.Signer):
    """A Signer whose key is on the person's machine: `sign` asks it over the channel."""

    def __init__(self, public: str, handle: str, endorsements: list, summary: str = ""):
        super().__init__(secret=None, handle=handle, endorsements=list(endorsements or []), summary=summary)
        self._public = public

    @property
    def public(self) -> str:
        return self._public

    def sign(self, msg: bytes) -> str:
        return _ask({"op": "sign", "msg": base64.b64encode(msg).decode("ascii")})["value"]


def hello(summary: str, details: Iterable[str] = ()) -> RemoteSigner:
    """The act introduces itself; the person's machine unlocks the key and answers with its public half."""
    ans = _ask({"op": "hello", "summary": summary, "details": [d for d in details if d]})
    signer = RemoteSigner(ans["public"], ans["handle"], ans.get("endorsements") or [], summary)
    if personkey.fingerprint(signer.public) != ans.get("fingerprint"):
        raise personkey_error("the person's machine sent a public key whose fingerprint is not its own")
    return signer


# --- the person's side -------------------------------------------------------------------------------------------
def describe(msg: bytes) -> Optional[str]:
    """What a message signs, in one line; None when it is not a purpose a remote person act may sign."""
    text = msg.decode("utf-8", errors="replace")
    domain = text.split("|", 1)[0]
    if domain not in ALLOWED:
        return None
    parts = text.split("|")
    if domain == "openlab-node" and len(parts) >= 3:
        payload = parts[2] if len(parts) == 6 else "|".join(parts[2:-3])
        return f"{ALLOWED[domain]}: {parts[1]} {payload[:160]}"
    return f"{ALLOWED[domain]}: {'|'.join(parts[1:])[:200]}"


def serve(proc: subprocess.Popen, host: str, out=None, unlock=None) -> int:
    """Answer the remote act's requests until it ends; pass its other output through. `unlock(summary, details)`
    returns the local Signer (default: the passphrase on this terminal)."""
    out = out or sys.stdout
    if unlock is None:
        from .person import unlock_interactive as unlock
    signer, signed = None, 0
    for line in proc.stdout:
        if not line.startswith(MARK):
            out.write(line)
            out.flush()
            continue
        try:
            req = json.loads(line[len(MARK):])
            if req.get("op") == "hello":
                if signer is None:
                    signer = unlock(f"on {host}: {req.get('summary', '')}", req.get("details") or [])
                ans = {"public": signer.public, "handle": signer.handle, "fingerprint": signer.fingerprint,
                       "endorsements": signer.endorsements}
            elif req.get("op") == "sign":
                msg = base64.b64decode(req["msg"])
                what = describe(msg)
                if signer is None:
                    ans = {"error": "no act was introduced, so nothing is signed"}
                elif what is None:
                    ans = {"error": "the person's machine signs only what a person act signs (a record node, a "
                                    "direction, a resume, an inbox decision, a workspace pin); key changes stay local"}
                elif signed >= MAX_SIGNATURES:
                    ans = {"error": f"more than {MAX_SIGNATURES} signatures for one act; nothing more is signed"}
                else:
                    signed += 1
                    out.write(f"  signed on this machine: {what}\n")
                    out.flush()
                    ans = {"value": signer.sign(msg)}
            else:
                ans = {"error": f"unknown request {req.get('op')!r}"}
        except Exception as e:                                       # noqa: BLE001 -- answered, never raised there
            ans = {"error": f"{type(e).__name__}: {e}"}
        proc.stdin.write(json.dumps(ans) + "\n")
        proc.stdin.flush()
    proc.stdin.close()
    return proc.wait()


def act_on(host: str, argv: list, openlab: str = "openlab", cd: str = "", transport: Optional[list] = None,
           unlock=None) -> int:
    """Run `openlab ARGV` on HOST over SSH as a person act signed here. `transport`: the command that reaches the
    box (default `ssh -T HOST`; tests use a local shell)."""
    remote = f"{ENV}=stdio {shlex.quote(openlab)} " + " ".join(shlex.quote(a) for a in argv)
    if cd:
        remote = f"cd {shlex.quote(cd)} && {remote}"
    cmd = [*(transport if transport is not None else ["ssh", "-T", host]), remote]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    return serve(proc, host, unlock=unlock)
