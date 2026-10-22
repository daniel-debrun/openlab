# Nodes and allocations

A lead can give a worker its own share of a machine: cores, memory and whole GPUs on one node, held for the worker's
life. The lead says the shape; openlab picks the node.

```
openlab_agent_spawn(prompt="train the probe, seeds 0-4", role="experimenter", budget_usd=3, minutes=120,
                    resources={"gpus": 1, "cores": 4, "memory_gb": 24}, params={"seeds": [0, 1, 2, 3, 4]})
```

The answer says whether the worker is running (on which node, which GPU) or queued, and why. A child the worker
spawns without `resources` shares its allocation. `node="vulcan"` pins a node; `gpu_model="l40s"` asks for a model.
`openlab_tool("nodes")` shows the nodes, what each has free, and who holds what. Without `resources`, nothing changes:
the worker runs on this machine with no allocation.

## The inventory

This machine is always a node (`local`). Other machines are the person's, in `box.toml` in the box dir
(`~/.openlab/box`, or `$OPENLAB_BOX_DIR`), which agents cannot edit:

```toml
[[node]]                       # a machine reached over SSH
name = "vast-3060"
host = "root@66.96.91.70"
port = 41615
workdir = "/root/openlab-nodes"
unsandboxed = "ok"             # your OK for runs where there is no bwrap

[[node]]                       # a Slurm cluster, through its login node
name = "vulcan"
backend = "slurm"
host = "vulcan"                # an ~/.ssh/config alias works
account = "aip-mtaylor3"
workdir = "/scratch/USER/openlab-nodes"     # a filesystem the compute nodes share
partitions = ["gpubase_bygpu_b1", "gpubase_bygpu_b2", "cpubase_bycore_b1"]   # optional: default, all of them
```

Optional keys: `cores`, `memory_gb`, `gpus` (cut what the node reports), `keep_cores`, `keep_memory_gb`,
`gpu_model`, `exclude` (more patterns never copied there), `ssh_options`. The SSH login must work without a
prompt (a key, `BatchMode=yes`). `openlab node list` reads each node (cores and memory under its container's limits,
GPUs) and shows what is free; `openlab node probe` reads them again.

## Where things run

The worker's own process (Claude Code, its hooks, openlab's MCP server, the record and your key) always stays on
this machine. On a local allocation it runs inside its share: `CUDA_VISIBLE_DEVICES`, its cores (`taskset`), its
memory (a systemd user scope), and its study runs are admitted within it. On a remote node, its work runs through
`openlab node exec`:

```
openlab node exec -- python train.py --seed 3
```

copies the folder to the node (never `.openlab/`, `.claude/`, `.mcp.json`, `openlab.toml`, `.env`, keys or venvs),
runs the command inside the allocation (an SSH node: its cores and GPUs; Slurm: a job step), follows its output for
90 s, then copies back what is newer there. A longer command keeps running: `openlab node wait RUN`, `logs RUN`,
`stop RUN`, `runs` and `pull` reach it again. A spawned agent's own `ssh`, `scp`, `sbatch`, `srun` and the like are
refused by the hooks.

## Placement

A request goes to a node of ours that fits it now (GPU work packed onto the node with the fewest GPUs left free, CPU
work kept off GPU nodes); else to a Slurm node, which queues it (the worker's minutes plus 15 is the job's time
limit, in the shortest partition that covers it); else it waits, oldest first, saying who holds what. A request no
node could ever hold is refused at spawn. A worker that waits shows `waiting` and does not spend its minutes.

`openlab node allocations [--all]` lists them; `openlab node release ALLOC` ends one (yours to do: a worker's ends
with it). Every allocation and run is an event of the agent that held it (node, GPUs, GPU-hours).
