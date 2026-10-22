# Design note: worker agents with compute allocations

Status: phases 1, 3 and 4 implemented (2026-10-22), in `openlab/nodes/`; phase 2 (GPU-hour budgets, the nodes panel
in `openlab show`) is not. The open question of where a worker runs is settled: its process stays on the control
machine (option b below), and its work runs on the node through `openlab node exec`. "As built" at the end says what
differs from the proposal.

## The problem

A lead delegates work with `agent_spawn`, but every agent runs on the machine openlab runs on, and nothing says what a
worker may use. A lead that wants one worker training on a GPU, another sweeping on CPU cores and a third reading
papers cannot say so, and cannot use a second machine. Most work does not need its own machine: several workers
should share one, each holding its part, and a worker should go to another machine only when it does not fit.

The unit to schedule is therefore **a worker agent with an allocation**: the resources it holds for its life, on one
node, which its own runs and children share.

## What exists today

- `agent_spawn(prompt, role, study, question, budget_usd, minutes, cli, model, decide)`
  (`openlab/app/agent_tools.py`). A child's budget ($, wall minutes, turns) comes out of its parent's. No resources,
  no node.
- One machine's admission queue (`openlab/study/admission.py`). A run takes a ticket with its `Needs` (cores,
  memory_gb, gpus, expected minutes) and starts when it fits: first come first served, with backfill by expected
  ends. The person's schedule (`available_until`, `keep_cores`, `keep_memory_gb`, reservations) is in `box.toml`.
- Per-core slots and GPU leases (`openlab/study/machine.py`): lock files under the box dir. `lease_gpus(n)` holds
  whole GPUs, and openlab never hands out a GPU outside the `CUDA_VISIBLE_DEVICES` it was given.
- `agent_request(kind="compute", args={gpus, memory_gb, cores, hours})`: a request to the person for another machine.
- Remote machines only in benchmark mode (`openlab/benchmark/mle.py`, `Box`, over SSH).

## The model

An **allocation** is `{id, agent, node, cores, memory_gb, gpus: [indices], gpu_model, backend, backend_ref, requested,
started, ended}`.

- It is requested when a worker is spawned with `resources`, and held until the worker ends. It is freed when the
  worker's process is gone, as a ticket is today.
- The worker, its tool calls, its runs and its children all live inside it. The admission queue keeps working, with
  the allocation as its capacity instead of the machine's: `capacity()` returns the allocation's share.
- A child either shares its parent's allocation (the default) or gets its own, charged to the parent's compute budget
  as dollars are now.
- The lead chooses the shape and openlab chooses the placement. A lead may pin a node, but it never names a GPU index
  or a scheduler.

## The interface

```
agent_spawn(prompt, role, ..., resources={"cores": 4, "memory_gb": 24, "gpus": 1, "gpu_model": ""},
            node="", params={})
```

- `resources` absent: today's behaviour, on the control machine with no allocation. Everything that works now keeps
  working.
- `node`: optional, a node's name from the inventory. Refused if that node could never fit the request.
- `params`: structured settings for the worker (seeds, a config). They go into its `launch.json` and its brief as
  data, so the record holds them. They are task facts, so they are not part of `harness_sha`.
- It returns at once, with the allocation and its state: `running` (node, GPUs), or `queued` with the reason and an
  expected start (`"waiting for 1 GPU: node gpu2's GPUs are held until ~14:20"`). `agent_wait` already waits on agent
  ids; a queued worker counts as not ended.
- It is refused, not queued forever, when no node in the inventory could ever fit the request, or the request exceeds
  the parent's compute budget. The refusal names the largest node and `agent_request(kind="compute")`.

## Placement and enforcement

- Pack by default (best fit), so whole nodes stay free for large requests. Whole GPUs only at first.
- Each node keeps the person's schedule (`box.toml`: availability, kept resources, reservations).
- A worker sees only its share:
  - its GPUs through `CUDA_VISIBLE_DEVICES`;
  - its cores and memory through cgroups (`systemd-run --user --scope -p AllowedCPUs= -p MemoryMax=` on our own
    nodes, Slurm's cgroups under Slurm).
  - A node that cannot enforce limits is recorded as such. Its allocations are bookkeeping, and `openlab doctor` says
    so.

## Backends

openlab owns the agent-allocation binding, the record and the budgets. A backend only places and enforces:

```
capacity() -> nodes with free and total resources
request(alloc) -> backend_ref
status(backend_ref) -> queued(reason, expected_start) | running(node, gpus) | ended
start(backend_ref, argv, env) ; release(backend_ref)
```

1. **local**: today's admission queue and GPU leases, held for an agent's life instead of one run's. Useful as soon as
   it exists, on any multi-GPU machine.
2. **slurm**: one allocation is one job (`--cpus-per-task`, `--mem`, `--gres=gpu:N`, `-w` for a pinned node); the
   worker starts with `srun` inside it. `squeue --start` gives the reason and expected start. Agents never see Slurm.
3. **ssh nodes**: `[[node]]` entries in `box.toml` (host, and what it may offer), scheduled by openlab's own queue across
   them, enforced with cgroups on each node. For machines that cannot run a cluster daemon.

Nomad (one binary per node, packing, GPU device plugin) is a possible fourth. Kubernetes is too heavy for a handful of
machines. Ray enforces only GPU assignment, and dstack and SkyPilot place tasks rather than share nodes between
long-lived processes, so none of these is planned.

## The record and budgets

- Control store events: `allocation_request`, `allocation_start` (node, GPU indices and model, driver, backend_ref),
  `allocation_end` (GPU-hours, core-hours). Every run records its allocation, node and GPU model, since results can
  differ across hardware.
- Budgets gain `gpu_hours` beside $, minutes and turns. A child's comes out of its parent's. `[agents] cap_gpu_hours`
  bounds a proposal, and at the limit the supervisor holds the agent and asks the person, as for dollars.
- The inventory (`box.toml`, nodes) is the person's. An agent cannot add a node or raise a cap. `agent_request(kind=
  "compute")` stays the way to ask for something no node can give.

## Guardrails

- The hooks deny the backends' own commands to agents (`sbatch`, `srun`, `salloc`, `scancel`, `squeue` writes, `ssh`
  to a node, `nomad`, `kubectl`, `dstack`, `sky`). Otherwise a run could leave trial counting, the budget and the
  sandbox.
- A node never holds the holdout, the study record or the person key. Data reaches it through the study's
  materialised datasets; results come back hashed.
- A node where runs would be unsandboxed (no bwrap) needs the person's OK, once per node.
- Allocations, nodes and caps are code and person configuration, outside anything the harness optimizer can edit.

## Visibility

`openlab show` gains a nodes panel: each node's resources, its allocations and which agent holds each, plus the
queued requests with their reasons. An agent's row shows its node and GPUs. `openlab box` shows the same in a
terminal.

## Phases

1. `resources` on `agent_spawn` with the local backend: an allocation is an admission ticket held for the agent's life,
   GPUs pinned with `CUDA_VISIBLE_DEVICES`, cgroup limits where available.
2. GPU-hour budgets and the nodes panel.
3. The Slurm backend.
4. The SSH-node backend.

## Open questions

- **Where a worker runs on another node.**
  - (a) Its `claude` process runs on the node. Its hooks and MCP server then need the workspace there, minus the record
    and the keys, with record writes sent to the control machine over an authenticated channel.
  - (b) Its process stays on the control machine and only its commands execute on the node. The record and the keys
    never leave the control machine, but every shell call has to be forwarded.
  - (b) keeps the integrity model simpler; (a) is simpler for Claude Code. This decision gates phases 3 and 4.
- Fractional GPUs (MIG, MPS) and GPU-memory requests.
- Preemption: none at first. A worker holds its allocation until it ends, is stopped, or hits a limit.
- Moving data to nodes without a shared filesystem.
- Whether `params` should be validated against a role's declared settings.

## As built (2026-10-22)

The inventory is this machine plus box.toml's `[[node]]` entries; tested on two vast.ai containers (an RTX 3060 and an
RTX 5060 Ti, reached over SSH) and an Alliance Slurm cluster (L40S nodes, through its login node).

- **Where a worker runs: option (b).** None of the remote machines could hold option (a) safely: the containers have
  no Claude Code and no bwrap, and the cluster's compute nodes are reached only through Slurm from a login node shared
  by over a thousand users. So the agent's `claude` process, its hooks, the MCP server, the record and the keys stay
  on the control machine. Commands are not forwarded one by one: the worker runs `openlab node exec -- CMD`, which
  copies its folder to the node (rsync; never `.openlab/`, `.claude/`, `.mcp.json`, `openlab.toml`, `.env`, keys or
  venvs), runs CMD detached inside the allocation, follows its output for 90 s, and copies back what is newer there
  (never `.git`, at most 200 MB per file). `runs`, `logs`, `wait`, `stop` and `pull` reach a run again.
- **Enforcement.** Local: `taskset` for cores, a systemd user scope's `MemoryMax` for memory (cpuset is not delegated
  to user scopes), `CUDA_VISIBLE_DEVICES` for GPUs, and the supervisor holds the machine's GPU locks for the
  worker's life. SSH nodes: `taskset` and `CUDA_VISIBLE_DEVICES` per run, memory is bookkeeping only. Slurm: one
  allocation is one job (`sbatch --wrap 'sleep infinity'` with the worker's minutes plus 15 as `--time`), each command
  a step (`srun --jobid --overlap`); Slurm's cgroups enforce all three.
- **Placement.** Nodes of ours that fit now first (GPU work packed, CPU work kept off GPU nodes, then best fit);
  else a Slurm node, which queues it; else the request waits oldest first with the reason (who holds what, when the
  first holder's budget ends). The partition is the shortest time limit that covers the worker's minutes, and a
  request without GPUs goes to a partition without them.
- **Sharing.** A child spawned without `resources` shares its parent's allocation (it joins as a holder); the last
  holder to end releases it. An allocation whose holders' processes are gone is ended the next time anyone looks.
- **The person's OK.** A remote node without bwrap is offered only when its entry says `unsandboxed = "ok"`.
- **Record.** Agent events `allocation_request`, `allocation_wait`, `allocation_start`, `allocation_end` (hours,
  GPU-hours, core-hours), `node_run`, `node_run_end`. The allocations and runs themselves are in
  `allocations.db` beside box.toml.
- **Guardrail.** The PreToolUse hook refuses a spawned agent's `ssh`, `scp`, `sftp`, `mosh`, `sbatch`, `srun`,
  `salloc`, `scancel`, `sky`, `dstack`, `kubectl`, `nomad` and `rsync` to a remote path, pointing at `openlab node
  exec`.
- **Not yet.** GPU-hour budgets and `cap_gpu_hours`; the nodes panel in `openlab show`; study runs (`study_run`) on a
  remote allocation (they still run on the control machine; `openlab node exec` runs are events, not trials).
