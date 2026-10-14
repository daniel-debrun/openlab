# Security

## Reporting a vulnerability

Report it privately through GitHub's security advisories: on
[github.com/daniel-debrun/openlab](https://github.com/daniel-debrun/openlab), open **Security**, then **Report a vulnerability**.
Do not open a public issue. Say what you did, what you expected and what happened, with the version
(`openlab --help` prints it) and a way to reproduce it if you have one.

Openlab is alpha: fixes go into the latest version only.

## In scope

- **The person-key gates.** A person act (a holdout look, an approval, a protected amendment, an inbox decision,
  resuming an agent) made, or accepted by `openlab study verify`, without the pinned person key's signature; a
  re-pin of a workspace without a rotation signed by the pinned key.
- **The record's integrity.** An edit of a study's record, or of the workspace's control store, that
  `openlab study verify` does not detect; a write that should be refused and is stored.
- **The hook gates.** A way for an agent to run a person-only verb, write into `.openlab/` or edit `openlab.toml` that
  the Claude Code hooks should deny and do not. The hooks are defence in depth, and the signature is the gate, so a
  bypass that also defeats a signature check matters most.
- **The local web server** (`openlab app`, `openlab show`). It binds 127.0.0.1 by default and every request needs the
  token it prints. Anything reachable without the token, from another origin, or from another machine when bound to
  127.0.0.1, is in scope.
- **The sandbox.** A study's `model.py`, run under bwrap, reaching the network, `.openlab/`, `data/` or the holdout rows.

## Out of scope

What the docs already state as limits: the record is tamper-evident, not tamper-proof, so someone who controls the
machine can rebuild a workspace under a key of their own; an agent on your account can read files the sandbox hides
from `model.py`; agent ids are attributed, not authenticated; a run where bwrap cannot start is recorded as
`unsandboxed`. See [docs/guards.md](docs/guards.md).
