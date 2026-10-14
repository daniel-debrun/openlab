# Supervising agents

Openlab is the control plane for the agents you already run. It records who each agent is and where it sits in the
tree, holds it to a budget, lets you pause or stop it, and collects everything that needs you in one inbox. It holds
no model key: the agents are `claude` and `codex` under your login.

## The workspace

A workspace is a folder with a `openlab.toml` (written by `openlab init`). Its studies, agents and inbox belong
together; the agent control store is `.openlab/control.db` at the workspace root. `openlab.toml` is edited by a person
only (the hooks refuse an agent's edit):

```toml
[agents]
stuck_minutes = 5          # openlab watch flags STUCK after this long with no event from a working agent
session_timeout_hours = 12 # a Claude Code session or sub-agent with no event this long is ended (its leases free)
max_depth = 3              # how deep a tree of delegated agents may go
max_children = 6           # live children per agent
loop_refusals = 3          # the same refusal this many times is flagged LOOP
budget_usd = 2.0           # default budget of a spawned agent ($, wall minutes, turns)
budget_minutes = 60
budget_turns = 200
cap_usd = 20.0             # the most one research proposal (its lead and every branch) may budget
cap_minutes = 480
permission_mode = "acceptEdits"
model = "haiku"            # the model spawned agents run on (claude --model / codex -m); unset: the CLI's default

[person]
fingerprint = "ed25519:..."   # written by `openlab person pin`
```

## Who is who

Every record write carries an agent id, so `openlab study why` names the agent behind each node.

- **A Claude Code session** you start in the workspace registers itself (SessionStart hook) as `cc-<session id>`.
  Its sub-agents join under it as `cc-<session>.<id>`, with their agent type as the role and the Task description as
  their question. A sub-agent's `openlab` Bash commands get its identity added automatically.
- **A spawned agent** (`openlab agent spawn`, `agent_spawn`, `propose execute`) is named `<role>-<id>`, with its
  parent, study, question and budget. It delegates only with `agent_spawn`, so every child gets a budget out of its
  parent's and a role prompt: a spawned Claude Code runs with `--disallowedTools Agent,Task`, and the PreToolUse hook
  refuses Agent/Task to a proposal's lead and to any spawned agent with a budget ("delegate with
  openlab_agent_spawn so the branch budget applies"). A plain interactive session keeps its native sub-agents, tracked
  as above. `[agents] allow_native_subagents = true` lifts both.
- **Your own terminal** is labelled `person:<handle>` (the person key's handle, else your login): any command run
  at a terminal with no agent variables, also with its input or output redirected. The label is attribution
  only; your acts are the signed ones. An unsigned amendment from your terminal reads `unsigned amendment`, a
  signed one `person (signed) amendment`.
- **`unattributed`** marks a write made with no agent id and no terminal, such as a cron job or a script run
  outside any agent. Run it with `OPENLAB_AGENT` set, or from an agent, if it should be attributed.
- Each agent gets a token from openlab. A write that claims another agent's id without its token is recorded as
  `unverified:<id>`, with a `spoof` event. This is attribution, recorded and cross-checked, not authentication: an
  agent on the same OS account can read a sibling's files.

## The tree and the live screen

```
$ openlab agents --tree
AGENT                        ROLE                 KIND            STATE     $ SPENT       TOKENS TRIALS CLAIMS REFUSED  LAST  STUDY / QUESTION
cc-7c0ffee1                  session              claude-session  working   ~$0.23          220k      0      0       0   11s
  └ cc-7c0ffee1.agent0       openlab-screener      claude-subagent working   $0.00              0      0      0       3    5s  find papers on lagged x features  [LOOP×3]
  └ cc-7c0ffee1.agent1       openlab-experimenter  claude-subagent waiting   $0.00              0      1      0       1    5s  does x alone beat persistence?  [WAITING BYPASS]
  └ cc-7c0ffee1.agent2       openlab-replicator    claude-subagent working   $0.00              0      0      0       0   11s  replicate table 2 of the ACI paper
reproducer-1e3a0f            reproducer           spawned         paused    ~$0.00/0.5       473      0      0       0    1s  demo / does run 1 reproduce?
```

`openlab watch` is the same tree, refreshed every few seconds, for a tmux pane. Under each row it prints the agent's
last refusal or bypass attempt. Keys: `j`/`k` select, `p` pause or resume, `s` stop, `a` approve the selected inbox
item, `q` quit. `openlab watch --once` prints one frame without colour.

- **States:** working, idle, waiting (on you: an inbox item or a permission prompt), stuck, paused, done, failed,
  stopped, lost, ended.
- **Quiet Claude Code sessions:** a session or sub-agent registered by the hooks gets no final event when Claude
  Code is killed. After `stuck_minutes` without an event it shows as idle, not STUCK. After
  `session_timeout_hours` it is ended (a sub-agent also ends with its session), and the frontier items it held are
  free again. `openlab agents` hides ended agents; `openlab agents --all` lists them. A session that comes back is
  running again at its next event.
- **Flags:** STUCK (no event for `stuck_minutes`), LOOP×N (the same refusal `loop_refusals` times or more),
  OVER (90% of a budget: $, minutes or turns), WAITING (a pending inbox item), BYPASS (a denied attempt at a person's
  act), DEAD-END (its question shares most of its words with a dead end another agent recorded in the same study).
- **Cost:** `~` marks an estimate from token counts. When the CLI reports its own bill (a spawned agent's result),
  the billed figure replaces the estimate. A Claude Code session's transcript has no bill, so its cost stays an
  estimate.
- **Trials, claims, refusals** per agent are read from the study records, counted the way `study status` counts
  them.

`openlab agent log ID` is one agent's timeline:

```
$ openlab agent log cc-7c0ffee1.agent1
20:00:35 cc-7c0ffee1.agent1       start          sub-agent openlab-experimenter of cc-7c0ffee1: does x alone beat persistence?
20:00:35 cc-7c0ffee1.agent1       tool           Bash openlab study run demo --note 'x only'
20:00:41 cc-7c0ffee1.agent1       bypass_attempt Bash: a holdout look (openlab study holdout) is a person's act; an agent cannot run it, in any wrapper
20:00:41 cc-7c0ffee1.agent1       request        #1 holdout: x-only model is final; one look
```

## The inbox

An agent that needs a person files `agent_request(kind, reason, study, args)` and carries on with other work. The
kinds are `holdout`, `amend`, `approval`, `budget`, `screen`, `fresh` (`{"why": ...}`: a new question on data
another study uses), `continues` (`{"continues": "../demo"}`: share that study's trial ledger), `compute`
(`{"gpus": 2, "hours": 6}`: another machine, see below), `question` (`{"choices": [...], "multi": false,
"other": true}`: answered in `openlab show`, unsigned; see [show.md](show.md#questions)) and `other`; a
disagreement between two screeners files a `screen_conflict` by itself. Approving `fresh` or `continues` writes it
into study.toml's `[study]` and signs the amendment. A request carries the exact change in `args` (a budget: `{"usd": 1.0}`), and a
budget request without one is refused.

```
$ openlab inbox
#1    20:00:41 pending   holdout   cc-7c0ffee1.agent1       demo           x-only model is final; one look

$ openlab inbox approve 1 --reason "x-only is final"
openlab person act: approve inbox #1 from cc-7c0ffee1.agent1: x-only model is final; one look -> holdout look 1 of 1 on demo
  spec sha effbd6859870
  code sha model.py f3b099f457b3
  data sha panel d74309e84d36
passphrase to sign (empty cancels):
#1 approved by person:sam; running the holdout in the background (pid 482833): `openlab inbox --all` shows its result, and the agent reads it with openlab_agent_inbox_item(1)
```

Approving shows exactly what will change, then asks for your passphrase. A long act (a holdout run) runs in the
background unless you pass `--wait`. `openlab inbox deny N --reason ...` refuses it. `--pick N` chooses which side of
a screening conflict stands. The agent reads the outcome with `agent_inbox_item(id)`. When the agent that filed an
item has ended, `openlab inbox` says so under it: "the requester has ended — run `openlab propose execute STUDY
--resume`" (see [Research from a prompt](research-from-prompt.md)).

### Questions

An agent that needs your view rather than an act files a `question`: `agent_request` kind `question` with args
`{"choices": [...], "multi": false, "other": true}` (2 to 8 choices; `multi` lets you pick several, `other` lets you
answer in your own words). `openlab inbox` lists it with its choices numbered:

```
$ openlab inbox
#4    14:02:10 pending   question  lead-1                   -              How many seeds?  (pick one or answer in your own words)
      1. 3 seeds
      2. 5 seeds
      answer with: openlab inbox answer 4 --pick N [--other TEXT] [-m MESSAGE]

$ openlab inbox answer 4 --pick 2 -m "then stop"
```

An answer is advice, not a person's act: it is not signed and asks for no passphrase, and the agent decides what to
do with it (it reads it with `agent_inbox_item(id)`, or wakes from `agent_wait(["inbox:4"])`). `--pick N` repeats on
a `multi` question; `--other TEXT` is your own answer. An unknown pick, several picks on a one-choice question, own
words on a question without `other`, or no pick and no words is refused. A question cannot be approved or denied
(`openlab inbox approve|deny` refuses it and points to `answer`), and a request cannot be answered. An agent cannot
answer a question, its own or another's: the hooks refuse the command. While a question waits, its agent shows as
`waiting` with the `WAITING` flag in `openlab watch`, and `openlab inbox wait` wakes on a new question as on any
request. You can also answer in `openlab show`.

### Stepping away

Set `notify` under `[agents]` in `openlab.toml` to a command, for example `notify = ["notify-send", "openlab"]`. It
runs, with the message as its last argument, when an agent asks you something (an inbox request) and when a spawned
agent ends; `OPENLAB_EVENT` says which. `openlab inbox wait` blocks until the next of those happens, which suits a
supervising terminal or an agent supervising others.

When you approve an item whose requester has ended, its work resumes within the budget it has left (the
replication's builder, the proposal's execution). `--no-resume` approves without resuming.

## Pause, resume, stop

```
$ openlab agent pause reproducer-1e3a0f --reason "checking the bundle first"
paused reproducer-1e3a0f: its next tool call is refused and its process is stopped (SIGSTOP)
```

- **Pause** refuses the agent's next tool call, Bash or MCP, with `paused by person: <reason> ... Stop here and
  wait; do not retry`. A paused parent pauses its children's calls too. For an agent openlab spawned, the process
  is also stopped.
- **Resume** is a person's act (`openlab agent resume ID --reason ...` asks for your passphrase).
- **Stop** (`openlab agent stop ID --reason ...`) tells the agent to end and kills a spawned agent's process.
- **Tell** (`openlab agent tell ID "TEXT"`) reaches a working agent at its next tool call: the hooks put the text in
  its context, and its log shows `tell` and, once it has seen it, `told`. You can change a lead's course without
  stopping it. A direction you sign (`--person`) is marked as yours; an unsigned one reaches the agent as advice.
  You tell leads only: telling an agent that works under a lead is refused (`only its lead directs it`), with the
  lead to tell instead. An agent supervising a child can still tell it.

## Leads take prompts

A lead is an agent at the top of its tree: a Claude Code session you started, an agent spawned with no parent, or an
agent with the role `lead` (a proposal's lead, a research's lead in `openlab show`). Only a lead takes your prompts.
The agents under a lead hear only from it, so the lead stays the one who directs them. Claude Code sub-agents are
never leads.

In `openlab show`, each lead's row has these buttons:

- **Prompt** opens a short chat with the lead, with two ways to send:
  - **Ask now (btw)**, like Claude Code's `/btw`: openlab forks the lead's Claude session (`claude -p --resume
    <session> --fork-session`, no tools, no MCP servers, no project hooks, nothing saved) and asks your question
    there. The answer comes in seconds from what the lead knows and did so far. The lead keeps working and never
    sees the question, so this is for asking, not directing. An ended lead can still be asked ("why did you
    stop?"). The fork's cost is kept apart from the lead's budget (`spent.btw_usd`). A Codex lead has no session
    to fork.
  - **Tell the lead**: your message reaches the lead itself at its next tool call (a long model turn or command
    finishes first; a lead in `agent_wait` wakes for it). It answers with `openlab_agent_reply(text, to=N)` and
    carries on; a message that asks for a change of course (finish sooner, drop a step) comes with the instruction
    to make it and report the new plan. Each told message shows as read or not read yet; a paused lead reads it
    once resumed, an ended one never.

  Either is advice, not a signed act.
- **⏸** pauses the lead and the agents under it. **▶** resumes them.
- The header's **Pause all leads** pauses every live lead at once; once all are paused it reads **Play all leads**.

Pausing is not signed. Resuming, one lead or all of them, is a person's act signed with your passphrase, as
`openlab agent resume` is. From a shell, `openlab agent tell LEAD "..."` reaches a lead the same way.

## Spawning agents and budgets

```
openlab agent spawn "reproduce run 1 from the bundle" --role reproducer --study demo --budget-usd 0.5 --minutes 30
openlab agent spawn "..." --cli codex --parent lead-1a5b28 --budget-usd 0.2
```

`openlab agent spawn` starts `claude -p` (or `codex exec` with `--cli codex`) under a detached supervisor, with an MCP
config that carries the agent's identity. The supervisor records its events, tokens and cost, and enforces the
budget in $, wall minutes (paused time not counted) and turns. A limit pauses the agent; it does not end it:
- At 80% of a limit, openlab tells the agent to ask for more (`agent_request(kind="budget")`) instead of wrapping up
  early, and never to end with its own run still going (the run dies with it).
- At the limit, openlab pauses it (the paused state reads "at its limit" in `openlab show`) and files a budget request
  for it in your inbox, proposing twice the limit. Approve it to raise the limit and resume the agent; `openlab inbox
  approve N --pick 600` (or the "New limit" field in `openlab show`) sets your own limit instead. Deny it and the
  agent resumes briefly (a short grace) told to wrap up and record what it has.
- The $ figure is the running estimate until the CLI bills it at the end. No `--max-budget-usd` or `--max-turns` is
  passed to the CLI: they would end the agent at the limit.

An agent
delegates with the `agent_spawn` tool: the child's budget comes out of the parent's remaining budget, and
`max_depth` and `max_children` bound the tree. `agent_wait(ids)` waits for children to end, or for the agent's
inbox items (`inbox:7`) to be answered.

`--model` (and the `agent_spawn` tool's `model`) picks the model: claude `--model`, codex `-m`. Without it a child
runs on its parent's model (same CLI), else `[agents] model`, else the CLI's default. `openlab agents --json` and
the spawn line show it.

### Leases

A frontier item taken with `study take` is leased to one agent for 30 minutes. Another agent's take is refused,
naming the holder. The lease ends when the holder releases it or ends, or when it expires. To free an agent's lease
yourself, for example one held by an agent you stopped:

```
openlab study release demo frontier:1a2b3c4d5e6f --person      # signed with your passphrase
```

## Claude Code hooks

`openlab init --claude` writes these into `.claude/settings.json` (merged with yours; a second run replaces only
openlab's entries):

| Event | What openlab does |
|---|---|
| SessionStart, SubagentStart | registers the session or sub-agent in the tree and exports its identity to its shell |
| PreToolUse | denies person-only acts and bypass attempts (below), refuses every call of a paused agent, attributes the call |
| PostToolUse, PostToolUseFailure | records openlab refusals (for LOOP) and tool errors |
| Stop, SubagentStop, SessionEnd | records the last message, tokens and estimated cost; marks the agent idle or done |
| Notification, UserPromptSubmit | marks the agent waiting or working |

PreToolUse denies, and logs as a `bypass_attempt`: person-only verbs (`study holdout`, `amend --person`,
`screen --person`, `study approve`, `inbox approve|deny`, `agent resume`, `person=True` from Python); wrappers that
fake a terminal (`script`, `expect`, `unbuffer`, `pty.spawn`, `tmux send-keys`, ...); decode-and-run pipelines
(`base64 -d | sh`, `eval "$(...)"`); unsetting or overriding `CLAUDECODE` or `OPENLAB_AGENT`; writes into `.openlab/`;
edits of `openlab.toml`, `.mcp.json`, `.claude/` (settings and role agents) and `~/.config/openlab/`; and reading the
person key. A hook never breaks Claude Code: an internal error is logged to `.openlab/hooks.log` and the call is
allowed. A hook costs about 60 ms.

These patterns are defence in depth, and a determined agent with a shell can find a command they do not match. The
gate is the signature (next section).

`openlab init --claude` also writes five role agents in `.claude/agents/`: `openlab-lead`, `openlab-screener`,
`openlab-replicator`, `openlab-experimenter` and `openlab-reviewer`. Each is a short prompt that already knows the
contract, so a lead can delegate to a role by name.

## Codex

`openlab init --codex` appends an openlab section to `AGENTS.md` and prints the snippet for `~/.codex/config.toml`
(`[mcp_servers.openlab]`). Codex has no hooks, so identity comes from the environment: set `OPENLAB_AGENT=<name>` where
you start `codex`, or spawn it with `openlab agent spawn --cli codex`, which sets it. Person acts are refused to Codex
the same way, by the signature.

## The person key

Your person acts are signed with an Ed25519 key whose private half is encrypted with your passphrase (argon2id,
then a SecretBox) at `~/.config/openlab/person.key`.

| Command | What it does |
|---|---|
| `openlab person init` | makes the key; asks for a handle and a passphrase (8+ characters) |
| `openlab person show` | the key's fingerprint, and what the workspace pins |
| `openlab person pin [STUDY]` | pins the fingerprint in `openlab.toml` and the control store, or in one study (a person act). Once a workspace pins a key, another key is refused unless the pinned key endorsed it |
| `openlab person rotate` | a new key, endorsed by the old one (asks for both passphrases); a workspace that pins the old key moves to the new one |

Every person node carries a signature over its content, its links, its session and the hash of the node before it.
The record refuses a person node without a valid signature by the pinned key, and `openlab study verify` checks every
signature again. The prompt shows what will be signed and asks for the passphrase on your terminal; typing it is the
confirmation.

| Person act | From your terminal | Through the inbox | An agent |
|---|---|---|---|
| holdout look | `openlab study holdout demo --reason ...` | `holdout` request | refused; files a request |
| amend a protected field | `openlab study amend demo --person --reason ...` | `amend` request | refused |
| screen over an agent, settle a conflict or a maybe | `openlab study screen demo SOURCE --person ...` | `screen_conflict` (`--pick N`), `screen_maybe` (`--pick include\|exclude`) | refused |
| approve, reject or waive | `openlab study approve demo ID --decision approve\|reject\|waive --reason ...` | `approval` request | refused |
| more budget | edit and `study amend --person` | `budget` request | refused |
| approve a proposal | `openlab propose approve FOLDER --reason ...` | | refused |
| resume a paused agent | `openlab agent resume ID --reason ...` | | refused |
| release another agent's lease | `openlab study release demo ITEM --person` | | refused |
| a fresh question on used data | `[study] fresh = "..."`, then `study amend --person` | `fresh` request | refused |

### On a box that keeps the records

When the records live on a server or a rented box, keep the key on your own machine and run each act with
`openlab person act HOST -- ARGS`: the box asks your machine for each signature over ssh, and your machine shows and
signs it. See [On a server, a container or a GPU box](box.md#person-acts-from-your-own-machine).

## What this holds against

An agent running as your OS user can read the encrypted key file, fake a terminal, and write the record's SQLite
directly. It cannot make a signature without your passphrase, so a forged or unsigned person node fails `verify`,
and a claim on an unsigned look stays `recorded`, never `measured`.

It cannot re-pin the workspace either:
- **A pinned workspace takes another key only through a rotation the pinned key signed.** `openlab person pin` with
  a key made under another `HOME` is refused (`this workspace pins the person key ...`), whatever `HOME` says, and
  so is every person act by a key the workspace does not pin. `openlab person rotate` (your current passphrase) is
  the one way to move the pin.
- **The pin is kept twice.** `openlab.toml [person] fingerprint` and a hash-chained `person_pin` event in the control
  store; each later pin event must carry the rotation signed by the key before it. If `openlab.toml` disagrees with
  that chain (it was edited), person acts are refused and `openlab study verify`, `openlab person show` and `openlab
  doctor` say `TAMPERED` until you restore it (by hand, or `openlab person pin` with the pinned key).
- **`verify` checks the study against the workspace.** It prints the study's key and the workspace pin, and fails
  when a study's person nodes are signed by a key the workspace never pinned.
- **The hooks deny the attempt.** `openlab person init|pin|rotate` and openlab commands run with `HOME=`,
  `XDG_CONFIG_HOME=` or `OPENLAB_*=` are refused to agents and logged as `bypass_attempt`.

What it can still do:
- **Start over under a key of its own.** Delete the control store, rewrite `openlab.toml`, and rebuild the workspace
  and every study's record from scratch (a study with no person act yet takes the workspace's pin, so it needs no
  rebuild). Everything then verifies against the new pin. The check that remains is yours: keep your fingerprint
  somewhere an agent cannot write (a commit, a note) and compare it with `openlab person show` or the `workspace pin:`
  line of `openlab study verify`. A lost pinned key means the same: a new workspace.
- **Read your terminal.** The passphrase is typed on `/dev/tty`; anything that can read your terminal can read it.
- **Read the files the sandbox hides from `model.py`.** The sandbox protects a run, not your account. The holdout
  rows live on the same machine.
- **Pass as a sibling.** Agent ids are attributed and cross-checked, not authenticated.

The record is tamper-evident, not tamper-proof. Custody, in which a third party holds the record's head, the holdout
and the key, is not built.
