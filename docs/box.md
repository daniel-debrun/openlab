# On a server, a container or a rented GPU box

Openlab runs the same on a laptop, a lab server, a cloud VM or a rented container (vast.ai, RunPod, a Kubernetes
pod). Two things differ in a container: what the machine really lets you use, and whether a sandbox can start.
`openlab doctor` reports both.

```
$ openlab doctor
  ...
  FAIL  sandbox   bwrap is not installed: a study's runs would be recorded as unsandboxed
                  -> this container forbids user namespaces (its seccomp profile), so no sandbox can start here:
                     runs are recorded as unsandboxed; keep the vault and the record on another machine
  --    machine   23 cores, 120.8 GB (the container's limits; the host reports 96 cores, 251.6 GB)
                  at most 23 sandboxed children run at once on this machine (slots in /root/.openlab/box)
  --    GPUs      0, NVIDIA GeForce RTX 3090, 24576 MiB; 1, NVIDIA GeForce RTX 3090, 24576 MiB
```

## Cores and memory

Inside a container, `nproc` and `free` describe the host, not your share of it. Openlab reads the limits that bind:
the cgroup's CPU quota (v2 `cpu.max`, v1 `cpu.cfs_quota_us / cpu.cfs_period_us`), the CPUs the process may run on,
and the cgroup's memory limit minus what it uses (page cache it can reclaim counts as free). These numbers size:

- the runner's pool of sandboxed children (folds, seeds, truncation-audit replays),
- the memory gate an audit replay waits on before it starts,
- what `openlab replicate` tells its planning agent the machine can do.

`OPENLAB_AUDIT_WORKERS` still overrides the pool size.

## One slot per core, across every run

Each sandboxed child holds a slot while it runs: one of as many lock files as the machine has cores, under
`~/.openlab/box/slots`. Every openlab process on the machine shares them, so two studies, or two agents each starting
runs, together never run more children than there are cores. Without this, eight processes on four cores each run
at half speed or slower, and too many at once exhaust memory and get killed.

A slot is a file lock, so it goes with the process that held it: a killed run never leaves a slot taken.
`OPENLAB_BOX_SLOTS=N` sets the count (0 turns the gate off), and `OPENLAB_BOX_DIR` moves the lock directory, for
example to share one gate between users of the same machine.

## The queue

A run is admitted to the machine before it starts. It declares what it holds for its whole length:

```toml
[sandbox]
cores = 4          # default 1: the per-core slots share the rest; a declared number also caps the run's children
memory_gb = 24     # default 0
gpus = 1           # default 0
```

It starts when that fits beside the runs already running, first come first served. A younger run that fits may go
first when, by the running runs' expected ends (each study's measured pace), it ends before the oldest waiting run
could start, or leaves room for it then. While it waits, it says why on stderr and on the agent's timeline, so
`openlab watch` shows it queued, not stuck. The run records how long it waited (`box.waited_s`). A queued run whose
process is killed leaves the queue with it.

`openlab box` shows the machine as openlab uses it, and `openlab box queue` shows its running and waiting runs. The
person's schedule goes in `box.toml` beside the queue (`~/.openlab/box/box.toml`):

```toml
[box]
available_until = 2026-10-20T08:00:00   # no run is admitted that would end after it
keep_cores = 2                          # held back for your own work
keep_memory_gb = 8

[[reservation]]                         # a window you keep resources in
who = "sam"
start = 2026-10-12T14:00:00
end = 2026-10-12T18:00:00
gpus = 1
```

A run whose expected window crosses a reservation counts the reserved resources as taken. A run of unknown length
(a study's first) counts every reservation that has not ended. A run that would end after `available_until`, or
that needs more than the machine has, is refused with the fix: ask the person for another machine.
`OPENLAB_ADMISSION=off` turns admission off.

## Asking for another machine

Openlab never rents, pays for or stops a machine: you provide them, and openlab only knows what is there. When a run
needs more than this machine has, or would end after `available_until`, admission refuses it and says to ask you.
The agent files a `compute` request with what another machine would need:

```
openlab_agent_request(kind="compute", reason="the DBM needs two GPUs at once", args={"gpus": 2, "hours": 6})
```

Openlab attaches what this machine has, its queue and its declared availability, so the request's numbers are not
the agent's guess. In `openlab inbox` you approve it once you have added a machine (or raised `available_until`), or
deny it. Approving records your answer; nothing runs.

## Which machine ran a run

Every run records the machine it ran on, as `box: {host, cores, memory_gb}`, and its sandbox (`bwrap` or
`unsandboxed`). The machine is not part of the run's key: the same code, data, spec and packages on another machine
is the same trial, and is cached.

## Containers that forbid a sandbox

Docker's default seccomp profile, which most GPU rental containers use, refuses the user namespaces bwrap needs.
No install fixes that from inside the container. Runs still happen, and each is recorded as `unsandboxed`, which the
claims built on it carry.

In such a container every process runs as root and can read every file, including anything a sandbox would have
hidden. So keep what must stay secret off the box:

- **The vault** (held-out labels) and **your person key** stay on the machine you control. Take holdout looks there.
- **The record** is safest there too. A study whose runs happen on the box can be synced back and checked with
  `openlab study verify`, which recomputes the hash chain and every signature.
- Put on the box only what a run may see: the code and the research rows.

## Person acts from your own machine

When the box keeps the records, your person key still stays on your machine. Run the act there with
`openlab person act`:

```
openlab person act box --openlab /root/openlab/.venv/bin/openlab --cd /root/research -- person pin
openlab person act box --cd /root/research -- study holdout demo --reason "final model"
openlab person act box --cd /root/research -- inbox approve 3 --reason "x-only is final"
```

Your machine runs the command on the box over ssh. The act shows on your terminal what it is, and you type your
passphrase there. Each signature the act needs is asked over the same ssh channel and made on your machine, which
prints each one as it signs it. It signs only what a person act signs (a record node, a direction to an agent, a
resume, an inbox decision, the workspace's pin), at most 64 per act, and never a key rotation: `openlab person
rotate` stays on your machine. The key never reaches the box.

Whoever controls the box controls what it asks to sign. What your terminal shows is what you sign, so read it.

## Holdout labels that never reach the box

A static study can keep its holdout labels on your machine altogether. Its data on the box carries the holdout rows
with an empty target (a Kaggle-style test set), and study.toml says who holds the labels:

```toml
[split]
scheme = "stratified"
k = 5
labels = "person"       # the holdout is the rows whose target is empty; no split.fraction
labels_key = "id"       # the column that names a row in your labels file
```

Research runs never see those rows. A holdout look on the box (`openlab person act box -- study holdout STUDY
--reason ...`) predicts them and records the run with its predictions but no result: nothing there can score them.
Then, on your machine:

```
openlab study score box STUDY --labels labels.csv --cd /root/research --openlab /root/openlab/.venv/bin/openlab
```

It fetches the predictions over ssh, joins your labels on `labels_key`, computes the result here with openlab's own
metric code, and records it on the box as a person act signed here. The result carries your labels' sha, so a later
look must be scored against the same labels. The box checks what it can (the run, every predicted row scored, your
signature) and never holds the labels. Such a study uses openlab's built-in metrics, since a study's own metrics run
only in its sandbox.

## GPUs

`openlab doctor` lists what `nvidia-smi` sees. A study that needs GPUs says how many one run holds:

```toml
[sandbox]
gpus = 1
```

Each run then leases that many specific GPUs for its whole length, through one lock file per GPU beside the core
slots, so two runs never share one. It waits while they are taken. A run asking for more GPUs than the machine has
is refused before anything runs. The run's children see only the GPUs it leased (`CUDA_VISIBLE_DEVICES`), and
under bwrap the NVIDIA devices are bound into the sandbox for them. A run that leased none sees none, so it cannot
take another run's GPU. The run records the GPUs it held (`box.gpus`). Openlab only hands out GPUs its own process was
given: set `CUDA_VISIBLE_DEVICES` where you start openlab to keep some back.

## The web app over SSH

`openlab app` listens on 127.0.0.1:8321 (`OPENLAB_APP_PORT` changes it) and asks for the token it prints. From your
laptop:

```
ssh -L 8321:localhost:8321 you@box          # or LocalForward 8321 localhost:8321 in ~/.ssh/config
```

then open the printed URL on your laptop. Many rental images already use 8080 for Jupyter, so keep openlab on its
own port.

## How agents sign in on a box

Openlab holds no model key, and an agent on a box runs under whatever login the box has. Sign it in yourself:

- log in to `claude` (or `codex`) on the box, in your own ssh session, once; or
- set your own API key in the environment the agent starts in (`ANTHROPIC_API_KEY` for Claude Code; see Codex's
  authentication page for Codex).

Never copy `~/.claude` (or `~/.codex`) credentials from your machine onto a box. A rented container runs everything
as root, so whatever is on it can be read by every process there.

For [benchmark mode](benchmark.md) and other unattended runs, use an API key. Which way of signing in fits automated
use is set by the provider's terms, not by openlab: see
[Claude Code's legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance) and
[Codex authentication](https://developers.openai.com/codex/auth).

## What comes next

Sending runs to further boxes over SSH is not built yet.
