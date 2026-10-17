# Guarantees and refusals

Openlab's guarantees are structures, not conventions. Each record node kind has a typed schema
(`openlab/study/schema.py`) with declared links and rules that are checked when a node is written. A write that breaks
a rule is refused, never stored and flagged later. For you or your agent, a refusal is the structure doing its job:
it says what was refused (`error`), what that means (`meaning`), and what to do instead (`fix`, or `try` over MCP).

| Guarantee | Enforced by | The refusal you meet |
|---|---|---|
| The plan comes first | the first run seals `study.toml` (a `plan` node); every run compares the file with the plan plus its amendments | `study.toml was edited after the seal` |
| Protected fields change only by a person | amendments to the split, budget, sandbox, audit, primary metric, `known_at`, forecast gate, error budget or review coverage need a signed person amendment | `budget can only be amended by a person` |
| Features cannot see the future | a view serves only the rows known at `t` (`known_at`, vintages); the truncation audit replays sampled steps holding only the past | the audit is not `clean`, and a claim resting on that run is refused |
| No label leaks into training | a row trains a refit at `s` only if its label was known by `s`; segments are embargoed by the horizon | none: there is no way to ask for it |
| Research never loads holdout rows | the runner stages only the segment's rows; bwrap masks `.openlab/`, `data/` and `.git` and drops the network | a read of a masked path fails inside `model.py` |
| Concurrent runs stay honest | each run executes a snapshot of the code taken when it starts, and its audit replays that snapshot | none: an edit during a run does not touch it |
| Every run is a counted trial, with its reason | a run's key hashes its code, data, spec and the packages it ran; an identical key is cached; the first line of a run says which | the run past `budget.trials` is refused |
| A budget belongs to the question, not the folder | a copied record is refused; a new study on the same data and target must continue the first one's count or be declared a fresh question by a person | `demo2 asks the question of demo ... of the same data` |
| A study keeps the key it was sealed under | the plan records the person key the workspace pinned when the study was sealed; the study's first signature must be that key or a rotation it signed, even if the workspace's pin stores are later deleted | `this study was sealed under the person key ...` |
| A run's place in the search is what it says | `--op` and `--parent` are checked against the run's links before it starts | `an improve run needs its parent` |
| Runs never oversubscribe the machine | each sandboxed child holds one of the machine's per-core slots, read from the container's cgroup limits | none: a child waits for a free slot |
| A holdout whose labels the person keeps is scored only by the person | `split.labels = "person"`: the holdout is the rows with no target; its run records predictions and no result, and only a signed person result, carrying the labels' sha, can score it | `this holdout's labels are kept by the person: only the person's score is its result` |
| The final candidate is picked by a rule sealed first | `[selection] rule` is a person's field; a `selection` is derived from the record on write; with a rule, the look is refused unless its code is the selected run's | `the code here is not run:..., the run the selection rule picked` |
| A lesson says what its comparison shows | `lesson.direction` is checked against the paired interval of its basis | `the comparison does not show improves` |
| Code that fails its checks does not run | a run of code whose latest `module_check` is red is refused | `this code's module check is red` |
| Advice never sets a verdict | a `critique` carries findings, no verdict field; its findings are answered by checks | none: nothing reads it |
| Only a person looks at the holdout | `look` nodes are signed by the person key and numbered; a holdout run links a look with the same hashes | refused to an agent, which files an inbox request instead |
| Data has a recipe | a file dataset needs its recipe's sha; every read checks the data's sha again | refused at `materialise`, or on a changed file |
| Numbers are computed | a claim's verdict and level are computed from what it rests on; a derived value is computed from its inputs; "selected from N trials" is attached | `a claim's verdict and level are computed from what it rests on, never typed` |
| Testing spends a budget | `test` nodes chain the alpha budget (alpha-investing: never below 0) | the test past the budget is refused |
| Evidence is where it says | an extracted quote must be on the cited page, and its numbers written in it | refused at `extract` |
| Like is compared with like | results compare only when their protocol coordinates match | refused at `compare` |
| History is not silently repeated | a hypothesis sharing a tag with a dead end must link it and say what is different | refused at `hypothesis` |
| The paper says what the record says | bindings, the number and wording lints, citation quotes, the equation check, traceability | `study paper` lists every finding; only a person waives one |
| An instrument is what was registered | a version is the sha of its snapshot, entry, settings and environment fingerprint; registering it is a signed person node; each run re-checks every hash and is recorded with its logs' and outputs' hashes; a study runs and reads only the versions it pins (a protected field); a live-order run needs a signed authority for its exact args | `NAME@VERSION is not the version that was registered`, `STUDY does not pin NAME@VERSION`, `each run is a person's act` |
| The record is append-only | SQLite with a hash chain over each node, its links, actor and session; person nodes signed | `openlab study verify` names the first edited node or bad signature |

`python -m bench.harness --suite traps` plants traps of each kind and checks that the structure refuses or flags them.

## Trust levels, stated honestly

| What | What it means | What it does not mean |
|---|---|---|
| `recorded` | from research runs on your record: sealed, audited, counted, hash-chained | the chain is tamper-evident, not tamper-proof: whoever controls the machine can rebuild a whole record |
| `measured` | from a holdout run behind a look signed with your person key | the holdout rows live on the same machine; the signature proves you decided the look, not that no process on your account read the rows |
| person acts | signed by your passphrase-protected key, checked by `verify` against the workspace's pin, which only a rotation signed by the pinned key can change | an agent on your account can still delete the control store and rebuild the workspace and every study under its own key; compare the fingerprint (`openlab person show`) with one you keep elsewhere |
| agent attribution | each node names the agent that wrote it; a claimed sibling id is recorded as `unverified:` | recorded and cross-checked, not authenticated |
| `+reproduced` | a rerun from the record alone (or a signed bundle) matched | the same code and data ran again; it is not an independent reimplementation |

## The threat model

An agent running as your OS user can read the encrypted key file, fake a terminal, and write the record's SQLite
directly. It cannot make a signature without your passphrase, so a forged or unsigned person node fails `verify`,
and a claim on an unsigned look stays `recorded`, never `measured`.

It cannot re-pin your workspace to a key of its own. Once the workspace pins your key, `openlab person pin` takes
another key only as a rotation signed by the pinned key (`openlab person rotate`, which asks for your current
passphrase), whatever `HOME` says, and every person act by a key the workspace does not pin is refused. The pin is
kept twice, in `openlab.toml` and as a hash-chained event in the control store; a `openlab.toml` that disagrees with
the chain refuses person acts and shows as `TAMPERED` in `verify` and `doctor`.

What it can still do is start over: delete the control store, rewrite `openlab.toml` and rebuild every study's
record under a new key. A sealed study names the key it was sealed under, so its first signature must be that key
or a rotation it signed even after the pins are deleted; only a study not yet sealed takes whatever key the
workspace pins. `verify` then passes against the new pin, so the last check is yours: compare `openlab person show`
(or the `workspace pin:` line of `verify`) with a fingerprint you keep where no agent can write (a commit, a note).

The passphrase is typed on your terminal, and anything that can read your terminal can read it. The Claude Code
hooks are defence in depth (they deny `openlab person init|pin|rotate` and openlab under a changed `HOME`,
`XDG_CONFIG_HOME` or `OPENLAB_*` to agents); the signature is the gate. Without bwrap (or where it cannot start), a
run is recorded as `unsandboxed`.

## What these do not cover

The record is tamper-evident, not tamper-proof. A pinned workspace takes another person key only through a rotation
the pinned key signed, and a sealed study keeps the key it was sealed under, but someone who controls the machine can still delete the
control store and rebuild the workspace and every record under a key of their own, which shows as a fingerprint
other than the one you keep. An agent running as your OS user can read
the files the sandbox hides from `model.py`. Agent ids are attributed and cross-checked, not authenticated. Custody,
in which a third party holds the record's head, the holdout and the key, is not built. See
[Supervising agents](supervising-agents.md#what-this-holds-against) for the threat model.
