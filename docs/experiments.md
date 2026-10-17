# Experiments

An experiment runs a study's pinned instruments on a schedule, unattended, for a set number of passes. Then it stops,
and an agent is fired to react to it and write the conclusion. An agent sets an experiment up and asks for it; the
person starts it, because it runs outside the sandbox with nobody watching. Every run, pass and end is on the study's
record, and the **Experiments** tab of `openlab show` draws each one with its progress.

Planning still goes through the Roadmap. An experiment names the step of the owning agent's plan that it implements.
openlab moves that step to active when the experiment starts, and to done (or failed) when its conclusion is recorded.

## The spec

An experiment is a file of its study: `<study>/experiments/<name>.toml`.

```toml
[experiment]
name = "paper_trial"                     # the file's name
purpose = "Does the scan find tenders that the nightly export confirms? Judged on the exported rows of 20 nights."
plan_step = "paper-exp"                  # optional: the step of the owner's Roadmap plan it implements
# owner = "lead-1a2b3c"                  # optional: whose plan; default, the agent that asked for it

[[jobs]]                                 # each job runs one instrument alias the study pins, with fixed args
name = "scan"
instrument = "tender"                    # an alias under [instruments] in study.toml
args = ["scan"]
cron = "*/10 9-20 * * 1-5"
tz = "America/New_York"                  # optional: this machine's local time when empty
# timeout = 600                          # optional: seconds, else the instrument's own

[[jobs]]
name = "export"
instrument = "tender"
args = ["export"]
cron = "30 23 * * 1-5"
tz = "America/New_York"

[pass]
job = "export"                           # a pass = one successful run of this job; or "cycle"

[stop]
passes = 20                              # and/or until = 2026-12-31
max_failures = 3                         # failed passes in a row that end it, failed (default 3; 0: no limit)
# tz = "America/New_York"               # the zone of a date-only `until` (default: the first job's)

[react]
role = "reviewer"                        # an openlab role: lead, screener, replicator, experimenter, adversary, reviewer
prompt = "Read the runs and the exported rows; say whether the scan's tenders were confirmed."
budget = { usd = 5, minutes = 45, turns = 80 }
```

The spec is checked strictly. It is refused for any of these:

- a key that is not in the format, at any level;
- an alias the study does not pin, or a version that is not registered;
- a `live-order` instrument (each of its runs is a person's own act);
- a cron expression or time zone that does not parse (the same parser as `[schedule]`);
- no stop rule (`passes` or `until`);
- a pass job that is not one of the jobs;
- a role that is not an openlab role;
- a reaction budget without `usd` and `minutes`.

**A pass** is either one successful run (exit 0) of the named job, or, with `cycle`, every job having run once since
the last pass. A cycle passes when the latest run of each job succeeded. A run that failed, or was refused (drift, a
missing secret), makes the pass a failed one. Only passes that passed count toward `passes`.

**The stop rule** ends the experiment as soon as one of these holds:

- `passes` passes have passed;
- `max_failures` passes in a row have failed (the experiment ends **failed**);
- a job fires at or after `until`. A date-only `until` means the end of that day.

## Its life

```
draft -> requested -> running -> concluded | killed | failed -> reacting -> reacted
```

| State | What is on the study's record |
|---|---|
| draft | the file only |
| requested | an `experiment` node: the spec's sha256, the checked spec, who asked, the inbox item |
| running | an `experiment_start`: a **person** node, signed, with the crontab lines it installed |
| concluded, killed, failed | an `experiment_end`: why it ended, who ended it, its passes. Its lines are removed first |
| reacting | an `action` `experiment_react_spawn` names the reaction agent |
| reacted | an `experiment_reaction`: the verdict (`met`, `not_met`, `inconclusive`), the conclusion, what it rests on, what it wrote |

Every job run is also on the record:

- an `instrument_run`, as `openlab instrument run --study` records one;
- an `action` `experiment_job`, which names the run's slot;
- an `experiment_pass`, each time the pass rule is met: its number, ok or failed, and the runs it rests on.

An experiment runs the spec it was started with, because the start node holds it. A spec file edited after the start is
not run. The tab and `status` flag it, and `experiment_new` refuses to overwrite it. A changed experiment is a new
experiment, under a new name. A spec changed after its request but before its start is refused at the approval, and
the agent requests it again.

### Starting it: the person

An agent asks with `experiment_request(study, name, reason)`, which files `agent_request(kind="experiment")`. The
person then does one of two things:

- approves the inbox item (`openlab inbox approve ID`, or in `openlab show`);
- runs `openlab experiment start STUDY NAME --reason "..."` in their own terminal.

Either way, the person first reads what will run: each job's instrument version and capabilities, its args, schedule
and next run, the pass and stop rules, the reaction agent's role and budget, and the crontab lines. The passphrase
signs the start. The start is refused in these cases:

- without the person key;
- when an instrument version has drifted;
- when `until` has passed;
- when the experiment has already started.

A direct start cancels the experiment's other pending requests.

### Running: the crontab

The start installs one line per job in the user's crontab, tagged
`# openlab-experiment:<workspace>::<study>::<name>::<job>`. openlab only ever touches lines with its own tags, and it
edits the crontab under a lock, so two installs never drop each other's lines. Each line does the following:

- `cd`s to the workspace;
- sets the PATH of the shell that started the experiment;
- takes `flock -n` on the job's lock, so a run that is still going makes the next one skip;
- runs `openlab experiment job STUDY NAME JOB`;
- appends to `<study>/.openlab/experiments/<name>/cron.log`.

`job` runs only at a minute when its schedule fires, in its own zone, and only once for each such minute (a run that
cron started up to 2 minutes late still counts). cron reads lines in the machine's local time. A job in another zone
therefore gets a line that fires at its minutes every hour, or every minute when the zones differ by a fraction of an
hour, and `job` itself decides whether it is due. `job --now` runs at once. It is a person's act, and its pass is
marked off schedule.

**Secrets.** A cron job has cron's environment, not your shell's. An instrument that declares secrets needs them in
the crontab (`VAR=value` lines at its top), or its runs fail with `the secret X is not set`. The start's preview names
them.

### Ending: the stop rule, or a kill

When the stop rule is met, the job that met it removes the experiment's lines, writes the end and fires the reaction
agent.

`openlab experiment kill STUDY NAME --reason "..."` (or the `experiment_kill` tool) is open to agents and to the
person. It removes the lines at once. It waits for a job that is still running to finish and be counted, then writes
the end (`killed by <who>: <reason>`) and fires the reaction agent. Pause and resume are not supported: kill it, and
start a new experiment.

### Reacting: the agent

The reaction agent is spawned through openlab's own spawn: a top-level agent on the study, with the role, the budget
and the `[react]` prompt. Its prompt holds the following:

- the role's prompt;
- the experiment's purpose, its end and its passes;
- every run: job, slot, run id, and exit or error;
- how to read a run (`openlab_tool("instrument_output", {"run": ID})`, `experiment_status`);
- the instruction to plan on the Roadmap first, to record what the record allows on the way (a question, a
  hypothesis, a dead end, a claim grounded in results), and to close with `experiment_react`.

```
openlab_tool("experiment_react", {"study": "tender", "name": "paper_trial", "verdict": "not_met",
             "conclusion": "...", "rests_on": ["experiment_pass:...", "instrument_run:..."],
             "records": ["dead_end:..."]})
```

A spawn that failed (no agent CLI, say) is recorded, and the state stays at the end. To fire it again, run
`openlab experiment react STUDY NAME --spawn`. A reaction is recorded once.

## Commands and tools

| CLI | Tool | What it does |
|---|---|---|
| `openlab experiment new STUDY NAME [--file F]` | `experiment_new(study, name, toml \| spec)` | write a draft (the CLI: from a template with the study's pins), checked |
| `openlab experiment list` | `experiment_list` | every experiment of the workspace's studies |
| `openlab experiment status STUDY NAME` | `experiment_status(study, name)` | state, progress, jobs (next time, last runs), passes, failures, end, reaction |
| `openlab experiment request STUDY NAME --reason R` | `experiment_request(study, name, reason)` | ask the person to start it (an inbox item) |
| `openlab experiment start STUDY NAME --reason R [--sha S]` | none: the person's | start it, signed with the passphrase |
| `openlab experiment kill STUDY NAME --reason R` | `experiment_kill(study, name, reason)` | end it now; the reaction agent is fired |
| `openlab experiment job STUDY NAME JOB [--now]` | none | what a crontab line runs; `--now` is the person's |
| `openlab experiment react STUDY NAME --verdict V --conclusion C --rests-on ID... [--records ID...] \| --spawn` | `experiment_react(study, name, verdict, conclusion, rests_on, records)` | record the conclusion, or fire the reaction agent again |

## Worked example

This is a temp workspace with a fake instrument: a shell script with a `scan` and an `export` subcommand. The test
`tests/test_experiments.py::test_the_worked_example_of_the_docs` runs this spec verbatim through the CLI.

1. The person registers the instrument, and pins it in the study:

   ```
   openlab instrument register tender --source tender_src --entry ./run.sh --state --outputs out.json \
       --capabilities read,paper-order --reason "the tender scanner (fake)"
   # study.toml:  [instruments]  tender = "tender@<version>"
   ```

2. The agent writes the spec. `openlab experiment new tender_trial paper_trial` writes a template with the study's
   pins, and the agent edits it:

   ```toml
   [experiment]
   name = "paper_trial"
   purpose = "Does the scan find tenders that the export confirms? Judged on 2 exported nights."
   plan_step = "paper-exp"

   [[jobs]]
   name = "scan"
   instrument = "tender"
   args = ["scan"]
   cron = "*/10 9-20 * * 1-5"
   tz = "America/New_York"

   [[jobs]]
   name = "export"
   instrument = "tender"
   args = ["export"]
   cron = "30 23 * * 1-5"
   tz = "America/New_York"

   [pass]
   job = "export"

   [stop]
   passes = 2

   [react]
   role = "reviewer"
   prompt = "Say whether the scans' tenders were confirmed by the exports."
   budget = { usd = 2, minutes = 20 }
   ```

3. The agent links it on its Roadmap step and asks:
   `experiment_request(study="tender_trial", name="paper_trial", reason="the forward check")`. That gives inbox #1, and
   the agent's step links `inbox:1`.

4. The person approves inbox #1, or runs `openlab experiment start tender_trial paper_trial --reason "go"`. The
   crontab gains two lines:

   ```
   */10 9-20 * * 1-5 cd /ws && PATH=... flock -n /ws/tender_trial/.openlab/experiments/paper_trial/scan.lock \
     python -m openlab experiment job tender_trial paper_trial scan --root /ws \
     >> /ws/tender_trial/.openlab/experiments/paper_trial/cron.log 2>&1 # openlab-experiment:/ws::tender_trial::paper_trial::scan
   30 23 * * 1-5 cd /ws && ... experiment job tender_trial paper_trial export ... # openlab-experiment:/ws::tender_trial::paper_trial::export
   ```

   (Each is one line in the crontab.) This machine is in New York, so the lines carry the jobs' own schedules. On a
   UTC machine they would read `*/10 * * * *` and `30 * * * *`, and `job` would run only those that are due in New
   York.

5. The scans run every 10 minutes. Each export at 23:30 is a pass. After the second, that export's job removes the
   lines, ends the experiment (`concluded`, `stop rule met: 2 passes`) and spawns a reviewer agent with the runs in
   its prompt. `openlab experiment status tender_trial paper_trial` then reads:

   ```
   paper_trial (tender_trial): REACTING
     purpose: Does the scan find tenders that the export confirms? Judged on 2 exported nights.
     Roadmap: lead-1 / paper-exp
     asked by lead-1, inbox #1 (approved)
     passes: [####################] 2/2 ok, 0 failed
     started 2026-10-19T09:20:02-04:00, ran 1d 14h, last pass 2026-10-20T23:30:04-04:00
     job scan: tender = tender@eef053e7c380 scan  "*/10 9-20 * * 1-5" America/New_York
       2026-10-20T20:50 America/New_York: instrument_run:81c5e4fae66e exit 0
       ...
     job export: tender = tender@eef053e7c380 export  "30 23 * * 1-5" America/New_York
       2026-10-20T23:30 America/New_York: instrument_run:e36ab351fa20 exit 0
       2026-10-19T23:30 America/New_York: instrument_run:683cb7bfccb0 exit 0
     ended concluded at 2026-10-20T23:30:05-04:00: stop rule met: 2 passes
     reaction: reviewer-3d9a01 (running)
   ```

6. The reviewer reads the runs, plans on its Roadmap and records `experiment_react(verdict="met", ...)`. The state is
   then `reacted`, and lead-1's step `paper-exp` is done, with the start, end and reaction as its evidence.

## What openlab show draws

The **Experiments** tab holds one card per experiment of the workspace's studies:

- its state badge, its purpose, and its Roadmap step (a click opens it on the Roadmap);
- progress bars of passes done against the target (failed passes in red) and of time elapsed against `until`;
- when it started, its last pass, and how long it has run;
- who asked for it and the inbox item;
- per job: the instrument and its args, the schedule, the next time it fires, and the last runs. A click on a run
  shows its output through `instrument_output`;
- its recent failures, how it ended, and the reaction agent with its status and its conclusion.

Below the cards are the **Instruments**. Each registered version is shown with its capabilities, runs and last run,
the studies that pin it, and the head of its usage. **Whole usage** reads it with `instrument_show`. The data comes
from `show_snapshot` (`experiments`, `instruments`). The page computes nothing but the layout.

## Test seams

No test touches the real crontab or spawns a real agent:

- `OPENLAB_CRONTAB` names the crontab program, as for `[schedule]`;
- `openlab.study.experiments.SPAWNER` replaces the spawn;
- `OPENLAB_EXPERIMENT_SPAWN=register` registers the reaction agent without starting it.
