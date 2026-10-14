# Install

Python 3.10 or later. Openlab 1.3.0 is alpha.

```
pip install "openlab-research[study]"
openlab doctor
```

The package is `openlab-research` on PyPI; the command and the Python package are `openlab`. From a checkout:

```
git clone https://github.com/daniel-debrun/openlab
pip install -e "./openlab[study]"
```

## Extras

| Install | What it adds |
|---|---|
| `pip install -e .` | the core: the study record, the agent control plane (`openlab agents`, `agent`, `watch`, `inbox`, `person`), `openlab-mcp` (the server your agents connect to), `openlab app` (the local web UI), and the checks behind derived claims and the paper's equations |
| `".[study]"` | scikit-learn (the templates' model), mlflow (the model store every run writes to) and matplotlib (paper figures). `openlab study run` needs it |
| `".[test]"` | pytest, hypothesis and httpx, for the test suite |
| `".[mirror]"` | wandb, for the [W&B mirror](mirror.md) of a study's runs |

The agent control plane, `openlab-mcp` and the web UI need no extra.

## The sandbox

On Linux, install bubblewrap (`apt install bubblewrap`, `dnf install bubblewrap`). A run then executes `model.py`
with no network, with `.openlab/` and `data/` masked, on a snapshot of your code. Where bwrap is missing or cannot
start (macOS, containers whose seccomp profile forbids user namespaces), the run still happens and is recorded as
`unsandboxed`. On a server or a rented container, see [On a server, a container or a GPU box](box.md). `[sandbox] threads` sets the
model's thread pools (default 1) and is recorded with each run.

## Claude Code and Codex

Openlab works with the coding agents you already use, under your own login. It needs no API key.

- **Claude Code.** In your workspace, `openlab init --claude` writes `.mcp.json` (the openlab tools),
  `.claude/settings.json` (the hooks), five role agents in `.claude/agents/` and `openlab.toml`. It merges into an
  existing settings file and keeps your own hooks and permissions; running it again replaces only openlab's
  entries. The hooks run `openlab hook EVENT` when `openlab` is on PATH, so the file can be committed for a lab;
  `$OPENLAB_HOOK_COMMAND` sets another launcher.
- **Codex.** `openlab init --codex` writes an `AGENTS.md` section with the contract and the MCP snippet to add to
  `~/.codex/config.toml`. Set `OPENLAB_AGENT=<name>` where you start `codex`, or spawn it with
  `openlab agent spawn --cli codex`.
- **Any MCP client.** `claude mcp add openlab -- openlab-mcp` (or your client's equivalent) gives the tools alone.
- **Spawned agents.** `openlab agent spawn` and `openlab propose execute` need `claude` (or `codex`) on PATH.
  `$OPENLAB_CLAUDE_BIN` and `$OPENLAB_CODEX_BIN` point at another binary.

## Your person key

`openlab person init` makes the key that signs your person acts (holdout looks, approvals, protected amendments,
inbox decisions, resuming a paused agent). It asks for a handle and a passphrase. `openlab person pin` pins its
fingerprint in the workspace's `openlab.toml`. See [Supervising agents](supervising-agents.md#the-person-key).

## The paper

`openlab study paper` builds the PDF with `latexmk` from a TeX distribution (TeX Live, for example). Everything else
works without one.

## `openlab doctor`

`openlab doctor` checks the workspace first (openlab.toml and its studies, the hooks, your person key, the agents
working here), then `claude` and `codex`, then the install (the extras, the sandbox, the MCP server). Each problem
comes with the one line that fixes it. `--json` gives the same rows to an agent. It never prints a key.

Data providers and snapshots: the snapshot root (`OPENLAB_SNAPSHOTS`) comes from the environment, else
`~/.config/openlab/research.env`, else `~/openlab-data/snapshots`. Keys for the public data providers go in that file
too, and doctor shows which are set.
