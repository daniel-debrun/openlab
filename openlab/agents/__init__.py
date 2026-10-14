"""The agent control plane: openlab records, attributes, budgets, pauses, stops and monitors the research agents a
person runs (Claude Code, Codex), one by one and as trees. openlab holds no model key and runs no model itself: an
agent is the person's own `claude` or `codex` under their own login.

* `store`: the workspace control store, `<workspace>/.openlab/control.db` (agents, hash-chained events, inbox).
* `identity`: which agent is acting (`identity.current()`), for every record write.
* `spawn` / `supervisor`: start an agent CLI detached, parse its stream into events, enforce its budget, pause it.
* `hooks`: the Claude Code hooks (`openlab hook <Event>`): identity, the person gates, activity.
* `cli`: `openlab agents`, `openlab agent ...`, `openlab inbox`, `openlab watch`.

This package's top level imports nothing heavy: a hook runs on every tool call and must start fast.
"""
