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
| A version is what it runs with | the version is the first 12 hex of the sha256 of the manifest: entry argv, runtime, capabilities, secrets (names), fixed env, outputs, state, timeout, redaction, every snapshot file's sha256, the config's sha256, every environment part's fingerprint and the usage's sha256 |
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
    [--no-version-probe] [--usage-arg ARG] [--usage-sub] [--usage-file FILE | --no-usage] --reason "..."
```

It fingerprints everything, probes the entry's usage, and shows what the version will hold: its files, environment
parts, capabilities and usage. Then it asks for your passphrase. Your passphrase signs the registration.

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

### The usage

An instrument is a command, and its `--help` is its contract: it tells an agent which subcommands and flags exist.
Register runs the entry with `--help` and keeps what it prints as part of the version.

- The probe runs **before** the passphrase prompt, and its output is part of what you read before you sign.
- It runs in a throwaway copy of what a run gets:
  - the snapshot, read-only;
  - an empty output folder and state folder;
  - the config;
  - the same minimal environment, with the declared secrets that are set.

  It has a 10 s timeout and makes no network assumptions. stdout and stderr are captured together, then the secrets'
  values and the `--redact` patterns are masked.
- The text goes into the content store. Its sha256 goes into the manifest, so it is part of the version hash.
- A probe that exits nonzero or times out does not refuse the registration. It is recorded as
  `usage: unavailable (exit 2)` or `usage: unavailable (timed out after 10s)`, and whatever it printed is kept.

| Option | Effect |
|---|---|
| `--usage-arg ARG` (repeatable) | what the probe adds to the entry, instead of `--help`: `help`, `-h` |
| `--usage-sub` | also probes each subcommand an argparse-style help lists as `{a,b,c}` (`ENTRY a --help`, ...): at most 40 subcommands and 64 KB in all. An option's choices (`--mode {x,y}`) are not subcommands |
| `--usage-file FILE` | keeps that file's text as the usage, without probing |
| `--no-usage` | records no usage. Use it for an entry that does not honour `--help`, because the probe runs the entry |

A version registered before usage capture keeps its version and hash. `show` says
`usage: not recorded (registered before usage capture)`. To give it a usage, register it again: that is a new
version, and studies that pin the old one need a person's pin amendment.

### Capabilities

| Capability | Meaning |
|---|---|
| `read`, `write`, `network`, `paper-order` | declared, shown and recorded; agents may run the instrument |
| `live-order` | each run is a person's act, signed for that run |

## Run, verify, read

```
openlab instrument run NAME@VERSION [--study STUDY] [--note ...] [--reason ...] -- ARGS...
openlab instrument verify NAME@VERSION | --all
openlab instrument list | show REF [--verify] [--usage] | runs [REF] [--study STUDY]
openlab instrument output RUN_ID [FILE]
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

The files matching `--outputs`, written in the run folder, are hashed and kept as written. A version is one exact
thing, so a bare `NAME` is accepted only while it has one version.

### What a run returns

Besides the run id and hashes, `run` returns what its caller needs to act on:

- the exit code;
- the stdout and stderr tails: the last 200 lines, at most 32 KB each, redacted as stored;
- each declared output's content, if it is text (UTF-8: JSON, CSV, ...) of at most 256 KB.
  - A larger output is withheld, shown with its size, its sha and `too large; materialise it in a study`.
  - A binary output (parquet) is withheld the same way, as `not text`.

Every returned text has the secrets' values and the `--redact` patterns masked. A text output that the redaction
changes also keeps its redacted copy in the content store, and that copy is what a caller is shown. The output itself
stays as written, because that is what a dataset reads.

```
$ openlab instrument run tcli@0abec38ad18e -- positions
instrument_run:235d66c4e071: tcli@0abec38ad18e exit 0, 2026-10-17T19:04:17.926+00:00 -> 2026-10-17T19:04:17.958+00:00
  outputs: big.txt (307200 B, def89517b7d1), blob.bin (1024 B, 785b0751fc2c), positions.json (67 B, 5a28252c3522)
  read it again: openlab instrument output instrument_run:235d66c4e071 [FILE]
--- stdout (1 lines) ---
2 positions
--- stderr (1 lines) ---
token [secret API_TOKEN]
--- positions.json (67 B, sha256 5a28252c3522) ---
[{"sym": "F", "qty": 99, "acct": "[redacted]", "tok": "[secret API_TOKEN]"}]
--- big.txt (307200 B, sha256 def89517b7d1): too large; materialise it in a study ---
--- blob.bin (1024 B, sha256 785b0751fc2c): not text; materialise it in a study ---
```

`openlab instrument output RUN_ID` prints a recorded run again, from the content store. It checks each sha first and
refuses a changed blob. `instrument output RUN_ID FILE` prints one declared output, or `stdout` / `stderr`, whole,
if it is text of at most 4 MB. It reads only the instrument record's runs and their blobs. Holdout data and the study
records stay masked as before.

A run recorded before this redacted copy existed is masked when it is read: with the `--redact` patterns, and with
the values of the declared secrets that are set in the reader's environment.

Agents have the same verbs as tools: `instrument_list`, `instrument_show` (with `usage_only`), `instrument_verify`,
`instrument_run`, `instrument_output` and `instrument_runs`. Over MCP, call them as
`openlab_tool(name="instrument_run", args={...})`.

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
- `blobs/`, the content store for files, configs, usages, logs and outputs (with their redacted copies);
- `snapshots/`, read-only, rebuilt from the content store when missing;
- `runs/`, each run's folder;
- `state/<name>/`.

`openlab doctor` reports the record and the snapshots. `openlab instrument verify --all` also re-checks every
environment.
