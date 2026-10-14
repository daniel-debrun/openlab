import os

os.environ.setdefault("OPENLAB_MODULE_CACHE", "off")        # the suite proves every module for real
os.environ["OPENLAB_MIRROR"] = "off"                         # the suite's runs never reach anyone's W&B project
if "OPENLAB_BOX_DIR" not in os.environ:                      # the suite's child slots, apart from the person's runs
    import tempfile
    os.environ["OPENLAB_BOX_DIR"] = tempfile.mkdtemp(prefix="openlab-box-")

import sympy
from sympy.matrices import MatrixBase


def reeval(e):
    """Rebuild an expression so unevaluated Add/Mul/Pow collapse, but Integral, Sum, ...
    keep their structure. Deferred product/abs markers (resolved against the namespace in
    semantics) collapse to their scalar meaning here."""
    from openlab.engine.transform import Bars, Dot, Times, Star
    if isinstance(e, MatrixBase):
        return e.applyfunc(reeval)
    if not isinstance(e, sympy.Basic) or not e.args:
        return e
    args = [reeval(a) for a in e.args]
    if isinstance(e, (Dot, Times, Star)):
        return args[0] * args[1]
    if isinstance(e, Bars):
        return sympy.Abs(args[0])
    return e.func(*args)


def same(got, expected) -> bool:
    got = reeval(got)
    expected = reeval(sympy.sympify(expected))
    if got == expected:
        return True
    try:
        return bool(sympy.simplify(got - expected) == 0)
    except Exception:
        return False


# --- the person key: a fixture key whose passphrase the test types through a patched prompt ----------------------
import pytest

PASSPHRASE = "correct horse battery staple"


@pytest.fixture(autouse=True)
def _not_inside_an_agent(monkeypatch):
    """The suite may itself run inside an agent's shell (Claude Code, Codex): attribution reads these variables."""
    # and the workspace a session's hooks resolve before a hook's cwd: else a hook test writes the real workspace
    for k in ("OPENLAB_AGENT", "OPENLAB_AGENT_TOKEN", "OPENLAB_WORKSPACE", "CLAUDE_PROJECT_DIR", "CLAUDE_ENV_FILE",
              "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED"):
        monkeypatch.delenv(k, raising=False)
    # in-process, a person's terminal is read from sys.stdin only, so the suite labels the same whether it runs in
    # a terminal or under an agent (a pty test checks the real detection in a child process)
    import sys as _sys
    from openlab.agents import identity as _identity
    monkeypatch.setattr(_identity, "at_terminal", lambda: _sys.stdin is not None and _sys.stdin.isatty())


class PersonFixture:
    def __init__(self, info, monkeypatch):
        self.info, self.mp, self.typed = info, monkeypatch, []

    @property
    def signer(self):
        from openlab.agents import person
        return person.unlock(PASSPHRASE)

    def act(self):
        """Run the block as the person (their key unlocked, as after the passphrase)."""
        from openlab.study import personkey
        return personkey.signing(self.signer)

    def answers(self, *typed):
        """What the person types at the passphrase prompt, in order (default: the right passphrase, always)."""
        from openlab.agents import person
        seq = list(typed)
        self.typed = seq
        self.mp.setattr(person, "_ask", lambda prompt: seq.pop(0) if seq else PASSPHRASE)


@pytest.fixture
def person_home(tmp_path_factory, monkeypatch):
    """A home with no person key yet (and a cheap passphrase hash, for speed)."""
    from nacl import pwhash
    from openlab.agents import person
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setitem(person.KDF, "opslimit", pwhash.argon2id.OPSLIMIT_MIN)
    monkeypatch.setitem(person.KDF, "memlimit", pwhash.argon2id.MEMLIMIT_MIN)
    monkeypatch.setattr(person, "_tty_out", lambda text: None)
    return home


@pytest.fixture
def person_key(person_home, monkeypatch):
    from openlab.agents import person
    info = person.create("sam", PASSPHRASE)
    monkeypatch.setattr(person, "_ask", lambda prompt: PASSPHRASE)
    monkeypatch.setattr(person, "_tty_out", lambda text: None)
    return PersonFixture(info, monkeypatch)
