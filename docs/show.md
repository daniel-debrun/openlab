# openlab show

`openlab show --root ~/research` serves a live view of the workspace at the printed URL (port 8321 by default). On
a box, forward the port: `ssh -L 8321:localhost:8321 HOST`, then open the link it printed. The page is the same
server as `openlab app`, opened on `/show`; the link carries the token, and the person's acts in it ask for your
passphrase.

| tab | what it shows | from |
|---|---|---|
| Activity | each agent and person: state, the tool call it is in (with a live timer), its last three actions, when it was last heard from (amber after 90 s of silence while working), spend, turns, tool calls, flags; on each live lead, **Prompt** and **⏸ / ▶** | the control store, as `openlab watch` reads it |
| Inbox | a person's acts to approve or deny (signed with your passphrase, with an optional message), and agents' questions to answer (their choices, or your own words) | the inbox |
| Roadmap | each agent's plan as a Gantt chart, by version; the plan changelog on the right picks the version; a step's panel opens the agent's conversation up to the call that marked it done | `agent_plan`, `agent_plan_node`, `show_conversation` |
| Board | every step of every plan in columns: backlog, ready, in progress, waiting, done, failed or dropped | the same plans |
| Experiments | one card per experiment of the studies: its state, purpose and Roadmap step, a bar of passes done against the target (failed in red) and of time against `until`, its start, last pass and duration, each job's schedule, next time and last runs (a click shows a run's output), failures, how it ended and the reaction agent with its conclusion; below, every registered instrument version with its capabilities, runs, last run, the studies pinning it and its usage. See [experiments.md](experiments.md) | `show_snapshot` (`experiments`, `instruments`), `instrument_output`, `instrument_show` |
| Burn | steps done against scope as the plan changes, and each agent's spend against its budget with when it runs out | the plans and the store |
| RAID | issues (a step the record disagrees with, a repeated refusal, a silent agent), dependencies (what waits for you or the box), risks (an agent near its budget), assumptions (the record's hypotheses, which hold once a test of them rejects the null, and its assumptions, until a break shows one does not) | derived from the above |
| Audit ledger | the studies' records, entry by entry, with **Verify chain** | each study's record |
| Workspace | the research's files as a tree (hide it with ‹); the file you pick on the right: LaTeX and Markdown in an editor with their rendering beside it (shown or hidden), a PDF or an image as itself, other text read-only; comments on any file for the lead; a study folder's card (**Start the paper**, its PDF, its README, **Sealed** or **Not sealed** with **Seal…**); **Compile now** for the study you are in | `show_tree`, `show_open`, `show_save`, `show_comment`, `show_snapshot` (`studies`), `show_study_seal` |
| Machine queue | each machine openlab runs on, local or remote, with its CPU, memory, GPU and disk use, then this machine's queue | `/proc`, `nvidia-smi` (over ssh for a remote one), `openlab box queue` |
| Settings | this research's settings with their defaults, changed in place; the remote machines | `openlab.toml`, `<home>/.openlab-show.json` |

## Researches: one tab each

Each research is its own workspace, with its own agents, inbox, records and paper. `openlab show --root ~/research`
draws a tab for the folder itself (when it is a workspace) and for every folder directly under it that has a
`openlab.toml`; a workspace nested deeper (inside a study folder, say) gets no tab. When a folder under the root has
the root's own name, the root's tab takes a `~` (`research~`) so the two never share one. `--root` must be an existing
folder: a file or a missing path is refused, never created. **New session** starts a new research: a folder under
the root, set up with `openlab init` and the Claude Code hooks, and its lead agent spawned there with your prompt and
budget. It opens in its own tab. Of two new sessions with one name, one makes the folder and the other is refused.
When the lead does not start (the agent CLI is not installed, say), the research is still made and the reply says
why. The research's Activity tab then shows **Start the lead**, with your prompt kept, to try again once the cause is
fixed; it also offers a new lead when the last one has ended. It calls `POST /api/workspaces/NAME/lead`, which is
refused while a lead runs. A new agent
*inside* a research is a different thing: the lead delegates with `agent_spawn`, or you add one from the palette
("New agent in this research").

A tab's **×** closes it: the research, its agents and its record are untouched (a live agent asks first, and keeps
running). **New session → Resume an existing one** lists every research under the root, closed or open, with its
live agents, what waits on you, its spend and its last lead: **Open its tab**, or **Continue with a new lead**, which
starts the lead form with a prompt that has the new lead read the record, the plans and your comments first. Closed
tabs are kept in `<home>/.openlab-show.json`, so every page of this server sees the same tabs.

## Prompt a lead, pause and play

A lead is an agent at the top of its tree: a Claude Code session, an agent spawned with no parent, or one with the
role `lead`. Only leads take your prompts; the agents under a lead hear only from it. Each lead's row in
Activity has:

- **Prompt**: a short chat with the lead (`agent_prompts` reads it).
  - **Ask now (btw)** (`agent_btw`, Ctrl+Enter): answered in seconds by a fork of the lead's Claude session with no
    tools; the lead keeps working and never sees it. Ended leads can be asked too.
  - **Tell the lead** (`agent_prompt`): reaches the lead at its next tool call; it answers with `openlab_agent_reply`
    and changes course if asked. Shown as read or not read yet; a paused lead reads it once resumed.

  Either is advice, not a signed act.
- **⏸ / ▶**: pause or resume the lead and the agents under it (`agent_control`).

The header's **Pause all leads** pauses every live lead at once, and reads **Play all leads** once every live lead is
paused (`agent_control_leads`). Pausing asks for nothing. Resuming, one lead or all, asks for your passphrase and is
signed, as `openlab agent resume` is. A message to an agent under a lead is refused, here and in
`openlab agent tell`, with the lead to send it to instead.

## Settings

Each setting shows its value, its default (marked **default** until you change it, then **changed** with **Reset**)
and where it lives. Those that may change at any time (the budgets and caps of spawned agents, the model, the STUCK
and LOOP flags, compiling on done) are edited in place (`show_settings_set`): written to `openlab.toml` with its
comments kept, and noted on the control store; they apply from the next agent or check on. Those marked **set at New
session** (the agents' permission mode, native sub-agents) are chosen in the New session form, under *Settings of this
research*, and hold for the research's life, so every agent of it ran under the same rules. Saving a setting needs
the link's reviewer secret, not your passphrase.

## Machines

Machine queue lists this machine (**local**) and every remote one (**remote**) declared in Settings by a name and an
ssh host, `user@host` or `~/.ssh/config` alias: its CPU load against its cores, memory, GPUs (use and memory, by
`nvidia-smi`) and disk, read by one small script, here with `sh` and there with `ssh -o BatchMode=yes` (it never asks
for a password or accepts an unknown host key: run `ssh HOST` once by hand). Readings are cached (5 s here, 30 s for a
remote) and refreshed in the background. openlab reads a machine's use; it never starts, stops or rents one. The queue
under it is this machine's (`openlab box queue`).

## The paper workspace

When a study's lead ends done (an agent with role `lead`, or with no parent, that has a study), the running server
compiles the study's release into `<study>/release/`. Nothing compiles while no `openlab show` or `openlab app` is
running; **Compile now** does it by hand.

```
release/
  README.md            the question, the data, the result as the record holds it, how to reproduce it
  paper.pdf            the white paper: every number bound to the record; built only when the paper's checks pass
  code/                model.py, metrics.py, recipes/, study.toml, README.md: the code the last result ran
  <study>.bundle.tar   the signed bundle a third party reproduces from
  MANIFEST.json        each file's sha256, the record head, and whether the paper was built
```

A paper the checks refuse (an unbound number, wording that outruns the claims, a stale binding) leaves no PDF; the
README lists why, and the code folder compiles anyway. A study with no `paper/paper.tex` (or `paper/paper.toml`) has no paper at all. Each
compile replaces the whole folder, so it never mixes two compiles. **Compile now** in the Workspace does the same at any time, for the study you are in.
`[show] compile_on_done = false` in `openlab.toml` turns the automatic compile off. The release is never part of the
study's code fingerprint or its bundle.

The page reads everything through one tool, `show_snapshot`, and reads it again whenever the server sees the
control store change: agents write it from their own processes, and the server publishes a `control` event on
`/ws/events`. Opened as a file, the page is a demo with made-up data.

## Plans: the fill is the state, the outline is the evidence

An agent reports its plan with `agent_plan(nodes, note)`: steps `{id, title, kind, needs, state, links, owner}`, a new
version whenever the plan changes. As a step moves it calls `agent_plan_node(id, state, links, note)`. Both are appended
to `.openlab/agents/<id>/plan.jsonl` and noted on the hash-chained control store.

A step's state (planned, active, waiting, done, failed, dropped) is the agent's report. Its outline is computed
from its links, never typed:

- **solid**: every link is on the record (a record node such as `run:41973e`, or a unique prefix of its id, on the
  agent's study or on another study of the same workspace, which the evidence names), or
  is an inbox item that exists (`inbox:7`), a study of the workspace that has a record (`study:red-good`), or a
  queue ticket (`ticket:43`, taken as the agent reports it);
- **dashed**: nothing is linked, so the state is the agent's word only;
- **red**: the record disagrees: a link is not on the record, or the agent says done and a linked run's
  truncation audit found look-ahead.

## People

Declare the people who work beside the agents in `openlab.toml`:

```toml
[team.sam]
role = "meteorologist"
hours = 10            # a week
does = "hand checks: outage labels and a forecaster's read of the maps"
# away_until = "Wed"
```

An optional `hours_used` is drawn against `hours`. They appear in Activity, the Roadmap and the Board. An agent assigns them a step with `owner: "sam"` in its plan.
`you` is the person who holds the key.

## The Workspace: the research's files, edited and commented on

The Workspace lists the research's folder as a tree (openlab's own `.openlab` and hidden files left out; a study is
marked, and a study with no `README.md` lists one in italics). Pick a file and it opens on the right; **‹** hides the
tree and **Files ›** brings it back. A `.tex` or `.md` file opens in an editor (a study's `release/` is read-only:
each compile rewrites it), with its rendering beside it; **Rendered** shows or hides it:

- a study's paper (`paper/paper.tex`, or an older layout's files) renders through the real LaTeX build of what is in
  the editor, saved or not (`show_draft_preview`): every binding resolved from the record, an unresolved one boxed as
  ??. Nothing is recorded or signed and `paper/build` is not touched; the signed paper is still only the one
  `study_paper` builds. Under the editor are the LaTeX errors and the paper's findings on their lines: click one to go
  to it. It renders after a pause as you type; Ctrl+Enter renders now. Another `.tex` file shows no rendering.
- Markdown renders as you type. A study's `README.md` shows the release README the next compile writes: the record's
  part between its two marks comes from the record, and edits inside the marks are replaced. With no `README.md` yet
  it opens as the README the last compile wrote; saved, it is the study's.

A PDF and an image show as themselves; other text (code, TOML, CSV) read-only; anything else by its size. A study
folder's card starts its paper from the template (**Start the paper**) and opens its PDF and README; **Compile now**
compiles the study you are in.

**Save** (Ctrl+S) writes the file (`show_save`), and only over the text you opened: when an agent changed the file
since, the save is refused and the editor offers theirs. A save is noted on the control store and told to the live
lead of its study (`openlab agent tell`, delivered at their next step), so it reads it again before editing.

**Comment** (Ctrl+Alt+M) works on every file (`show_comment`). With text selected, the comment is on that text: it
keeps the quote and where it stood, and is highlighted while open. With nothing selected, or on a PDF, an image or
data, it is on the whole file. The live lead of the file's study (else every live lead) is told at its next step:
the file's path, the quoted text and your comment. It may deal with it itself or hand it to an agent under it; it
lists the open comments with `agent_comments` (each with its path and the line it is on now) and answers with
`agent_comment_reply(id, text, resolve)`; you reply, resolve or reopen in the comment's card. With no lead live, the
comment waits open for the next one. Comments live in `.openlab/comments.jsonl`, appended and never rewritten, each
line's sha on the control store's chain. Saving and commenting are yours (the link's reviewer secret) but not
signed: the prose and the comments are advice to the agents, not a person's act.

## Sealing a study

A study is **Not sealed** until its first research run. Until then its plan, `study.toml`, can still change
without a trace. The page marks every unsealed study with an orange **Not sealed** badge: in the Workspace tree, on
the research's folder (a list of them) and on the study's card. The card says in plain words what sealing is: it
fixes the plan before any result is seen. It lists what the seal freezes, with this study's values: the split and
holdout, the budget, the sandbox, the primary metric, the instrument pins and the other protected fields. It also
says what the seal costs: trial 1 of the budget, with the holdout untouched. Afterwards an edit is an amendment, and
a change to a protected field needs your signature.

**Seal…** opens a confirm that names those consequences. Your passphrase signs it, and the server runs the study's
first research run as your act (`show_study_seal`). This is the same path, with the same protections, as an inbox
approval: the link's token and reviewer secret, then the passphrase. The dialog shows the result (the `plan` node,
`trial 1 of N`, the result id and the primary metric with its interval) or the refusal. A study already sealed or
whose data is not materialised is refused before anything runs; the card then says why and what to run first
(`openlab study materialise STUDY`). `openlab study seal STUDY` does the same in a terminal.

An experiment card in the Experiments tab, and an inbox item of kind `experiment`, whose study is not sealed carry a
line saying so, with a **Seal** button. `openlab experiment status`, `list`, the `experiment_request` reply and the
preview the person signs at the start say it too.

## Questions

An agent asks the person a question with `agent_request` kind `question` and args `{"choices": [...], "multi":
false, "other": true}`: 2 to 8 choices, one pick (`multi` false) or several (`multi` true), and `other` (true unless
set false) when the person may type an answer of their own. The person answers in the Inbox tab (the
`inbox_answer` tool): a choice, several, or their own words, with a message. The agent reads it with
`agent_inbox_item(id)`, or waits for it with `agent_wait(ids=["inbox:7"])`. An answer is advice, not a person's act: it is not signed, and a
question cannot be approved or denied.
