# Instruments

An instrument is a named, versioned program that measures or acts outside the sandbox: a broker's paper account, a
data collector, a scraper, a CLI the team relies on. Research code and agents use it only through openlab, so every
use is on the record and bound to one exact version.

Openlab's runs are sandboxed: no network, the record masked. That is right for a model and wrong for a program that
must call a broker or an API. An instrument is how such a program enters the research without becoming a hole in it.
It is registered once by a person, it never changes, and each run leaves a hashed entry that a study can pin and read.

## The guarantees

| Guarantee | What enforces it |
|---|---|
| Only a person registers a version | the `instrument` node is a person node, signed with the person key; an agent's attempt is refused by the record and the hooks, and it files `agent_request(kind="instrument")` with the exact registration instead |
| A version is what it runs with | the version is the first 12 hex of the sha256 of the manifest: entry argv, runtime, capabilities, secrets (names), fixed env, outputs, state, timeout, redaction, every snapshot file's sha256, the config's sha256 and every environment part's fingerprint |
| A version never changes | each run checks every snapshot file (none missing, changed or added), the config and every environment part again, and refuses on drift, naming what changed; new code or a new environment is a new registration |
| Runs execute the registered code | the code is copied into the content store and a read-only snapshot, and runs execute the snapshot, never the live folder |
| Every run is recorded | an `instrument_run` node: version, args, actor, start and end, exit code, the sha256 of the redacted stdout and stderr and of each declared output, all kept in the content store |
| A study uses only what it pins | `[instruments] alias = "NAME@VERSION"` is a protected field; a run with `--study` and a dataset from an instrument refuse any other version |
| Secrets stay out of the record | a secret is passed through by its variable's name; its value never enters the manifest or the node, and it is masked in the stored logs |
| A live order is the person's | a `live-order` instrument runs only as the person: a signed `instrument_authority` for those exact args is written before the run, and that one run links it |

The record is tamper-evident, not tamper-proof, as with a study's record: someone who controls the machine can
rebuild it. An environment is **fingerprinted, not copied**. A venv, node_modules or a system binary is checked
before every run, and a change refuses the run, but openlab cannot restore it. Rebuild the environment, or register
a new version.

## Register (a person's act)

```
openlab instrument register NAME --entry "CMD" [--source DIR] [--capabilities read,paper-order] \
    [--secrets ENV,ENV] [--env K=V] [--config FILE] [--outputs GLOB,GLOB] [--state] [--timeout S] \
    [--runtime python|node|binary|none] [--python PATH] [--node PATH] [--binary NAME] \
    [--env-file LOCKFILE] [--env-cmd "CMD"] [--include GLOB] [--exclude GLOB] [--redact REGEX] \
    [--no-version-probe] --reason "..."
```

It fingerprints everything and shows what the version will hold: its files, environment parts and capabilities. Then
it asks for your passphrase. Your passphrase signs the registration.

### Options

- **`--entry`** is the command as you would type it. It is stored as an argv, never as one runtime's field.
  - A word that names one of the source's files becomes `{snapshot}/<path>`. So does a path with a `/` that names one
    of its folders.
  - A path under the source that is not snapshotted, such as `.venv/bin/python`, is kept by its absolute path, as
    part of the environment.
  - A module after `-m` stays as written.
  - The placeholders `{snapshot}`, `{out}`, `{state}` and `{config}` can be used anywhere in it.
- **`--source`** is the code folder. In a git work tree, openlab snapshots the tracked files and the untracked ones
  that are not ignored; otherwise it takes every file.
  - Never snapshotted: venvs, `node_modules`, caches, `.git`, `state/`, `.env*`, `*.key`, `*.pem`, build output.
  - `--include` and `--exclude` adjust the selection.
  - Limits: 5000 files and 200 MB. Dependencies belong to the environment, not the snapshot.
- **Without `--source`**, the instrument is an external command, such as a system CLI. Its version is the
  executable's hash plus the environment fingerprint. It **cannot be copied, only verified**.

### The environment fingerprint

It is pluggable per runtime. The runtime comes from the entry, or from `--runtime`. Each part is hashed, and the
combination goes into the version.

| Part | What is hashed |
|---|---|
| python | the interpreter's path and real executable, its version and binary sha256, and for every installed distribution its version and the sha256 of each of its files; an editable install's source tree too. A reinstall with other code is a drift, even at the same version |
| node | the binary's path and sha256, `node --version`, and the tree of `node_modules` (linked into the snapshot, not copied) |
| binary | the resolved path, its sha256, and its `--version` output (off with `--no-version-probe`). A script's `#!` interpreter is fingerprinted as a binary |
| file (`--env-file`, repeatable) | any lockfile: `uv.lock`, `requirements.txt`, `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock` |
| cmd (`--env-cmd`, repeatable) | the stdout of a command, such as `uv pip freeze --python .venv/bin/python` or `mycli --list-plugins` |

### Capabilities

| Capability | Meaning |
|---|---|
| `read`, `write`, `network`, `paper-order` | declared, shown and recorded; agents may run the instrument |
| `live-order` | each run is a person's act, signed for that run |

## Run, verify, read

```
openlab instrument run NAME@VERSION [--study STUDY] [--note ...] [--reason ...] -- ARGS...
openlab instrument verify NAME@VERSION | --all
openlab instrument list | show REF [--verify] | runs [REF] [--study STUDY]
```

A run executes outside bwrap, in a per-run folder (`.openlab/instruments/runs/NAME@V/<time>/out`, its working
directory). It gets a minimal environment:

- `PATH`, `HOME` and the locale;
- the fixed `--env` variables;
- the declared secrets, taken from the caller's environment;
- `OPENLAB_INSTRUMENT` (its id), `OPENLAB_INSTRUMENT_VERSION`, `OPENLAB_INSTRUMENT_OUT`,
  `OPENLAB_INSTRUMENT_SNAPSHOT`, `OPENLAB_INSTRUMENT_STATE` (with `--state`: a folder kept across runs and
  versions), `OPENLAB_INSTRUMENT_CONFIG` and `OPENLAB_INSTRUMENT_CAPABILITIES`;
- for a python entry, `PYTHONPATH` set to the snapshot, so `python -m pkg` finds its package.

The files matching `--outputs`, written in the run folder, are hashed and kept. A version is one exact thing, so a
bare `NAME` is accepted only while it has one version.

Agents have the same verbs as tools: `instrument_list`, `instrument_show`, `instrument_verify`, `instrument_run` and
`instrument_runs`. Over MCP, call them as `openlab_tool(name="instrument_run", args={...})`.

## Studies pin versions, and read their outputs

```toml
[instruments]
paper = "ibkr_paper@3f2a9c0d1e4b"        # a protected field: only a person changes it

[data.positions]
instrument = "paper"                     # the alias
output = "positions.json"                # a file the run wrote: .parquet, .csv, .json, .jsonl
key = ["symbol"]
```

`openlab instrument run paper --study tender -- positions` runs the pinned version and records the run on the
workspace's instrument record and on the study's. `openlab study materialise tender positions` takes the latest
successful recorded run of the pinned version that wrote `positions.json`. It checks the file's sha256 and registers
it like any dataset, with the run's id in its source. Materialise never runs an instrument; you or an agent run it,
on the record. `known_at` applies as for any dataset, so put the observation time in a column.

## Examples, one per kind

```
# python -m, with the venv it runs in (its distributions are fingerprinted)
openlab instrument register ibkr_paper --source instruments/ibkr_paper \
    --entry ".venv/bin/python -m ibkr_paper" --capabilities read,paper-order --outputs "*.json,*.parquet" \
    --state --reason "the team's IBKR paper account"

# node, with its node_modules and lockfile
openlab instrument register prices --source tools/prices --entry "node fetch.js" \
    --env-file tools/prices/package-lock.json --secrets API_KEY --outputs "prices.csv" --reason "daily prices"

# a shell script (its #! interpreter is fingerprinted)
openlab instrument register collector --source tools/collector --entry "./collect.sh" --state \
    --outputs "rows.csv" --reason "the nightly collector"

# a bare binary on PATH: verified, never copied
openlab instrument register jq --entry "jq" --reason "the JSON filter the recipes call"
```

## Where it lives

`<workspace>/.openlab/instruments/`:

- `record.db`, the instrument record: append-only and hash-chained, with signed person nodes pinned to the
  workspace's key;
- `blobs/`, the content store for files, configs, logs and outputs;
- `snapshots/`, read-only, rebuilt from the content store when missing;
- `runs/`, each run's folder;
- `state/<name>/`.

`openlab doctor` reports the record and the snapshots. `openlab instrument verify --all` also re-checks every
environment.
