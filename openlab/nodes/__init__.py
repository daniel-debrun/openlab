"""Nodes and allocations: where a worker agent's work runs, and what it holds there.

A worker spawned with `resources` (openlab_agent_spawn) holds an **allocation** for its life: cores, memory and whole
GPUs on one node of the person's inventory (box.toml `[[node]]`, openlab.nodes.inventory). The lead chooses the shape;
openlab chooses the node (openlab.nodes.alloc.place). The agent's own process always runs on this machine, with the
record, the keys and the MCP server; on a remote node its work runs through `openlab node exec` (openlab.nodes.run),
which copies its folder there, runs the command inside the allocation, and copies the results back.

Backends: `local` (this machine: its admission queue and GPU leases, held for an agent's life), `ssh` (a machine
reached over SSH; openlab's own bookkeeping places work on it), `slurm` (a cluster's login node over SSH: one
allocation is one Slurm job, and each command a job step in it). docs/design/compute-allocations.md has the design.
"""
