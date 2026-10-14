# What is in the box

The README shows the three workflows in brief. This page lists the rest, with links to the page for each.

## A study is a folder

```
demo/
  study.toml    the plan: question, data, target, split, metrics, baselines, budget
  model.py      your code: features(view, t), label(view, t), class Model with fit and predict
  recipes/      the scripts that make the data files
  data/         the files they make
  .openlab/      the record (record.db), its content store, the run snapshots
```

| You or your agent write | Openlab owns |
|---|---|
| `study.toml` | sealing it on the first run, and refusing an unrecorded edit after that |
| `model.py` | calling `features` with only what was known at `t`; training only on rows whose labels were known in time; running it in a sandbox; auditing it for look-ahead |
| `recipes/` | running them, registering each dataset by its sha256, checking the hash on every read |
| nothing | metrics, baselines, bootstrap intervals and paired tests |
| prose with bindings | every number, table, claim and citation the paper prints |

Every key of `study.toml` is in [The plan: study.toml](plan.md).

## Paper to code

`openlab replicate PAPER` takes an arXiv id, a DOI or a PDF. A planning agent reads the paper and drafts
`replication.toml`:

- the paper's method and the baselines it compares against
- the numbers to check, each quoted from its page
- the metric
- a tolerance with its justification from the paper (the spread over seeds, the rounding of the tables)
- the data and the protocol
- what the paper leaves unresolved

Openlab checks the draft before you see it (`openlab replicate lint`). Each quote must be on its page and write its
number, and the metric must be one openlab scores or the study's own: a function in `metrics.py` beside the plan,
shown in full on the page you approve, sealed by its sha and run only in the sandbox. You read one page
(`replication.md`) and approve it with your passphrase. A builder agent then rebuilds the baselines first, then the
paper's method, and binds the equations to the code.

The result is a folder: `methods/` (one file per method, citing its page and equations), `model.py`, the data
recipe, `paper/` (the PDF, its text and the checked equations) and `results/`. Its `README.md` has the verdict
table, and openlab computes every row from the record. `openlab replicate status FOLDER` shows progress, and
`openlab replicate report FOLDER` rewrites the README and `results/` from the record.

### By hand

```
$ openlab study replicate 1805.09091 rl18 --mode temporal --methods Raw,EMOS_gl,EMOS_loc,NN_aux_emb --metric crps --tolerance 0.02
replication study in ~/lab/rl18
arxiv:1805.09091 (replication): Neural networks for post-processing ensemble weather forecasts
rebuild: no fidelity yet
next: the steps in ~/lab/rl18/README.md
```

This pins the paper (its full text is on the record, page by page), fixes the tolerance before any run, and
scaffolds `model.py` with one class per method, a data recipe and a step-by-step README. The paper's baselines must
first be rebuilt within tolerance (`study fidelity`). A claim's verdict (`study replication`) stays inconclusive
while any baseline it depends on is not faithful, so a tuned headline number cannot stand in for a faithful
rebuild. The paper's equations are checked against your code on probe inputs. `study reproduce` reruns recorded
runs from the record alone, and `study bundle` writes a signed tar that someone else can reproduce from.
`--kind reproduction --code DIR` runs the authors' code instead of a rebuild.

In the Rasp and Lerch (2018) pilot, the raw ensemble (CRPS 1.152 against the paper's 1.16) and global EMOS (1.012
against 1.01) were rebuilt, and local EMOS (0.906) and the embedding network (0.834) replicated.

### Intervals, units and coverage

`predict` may return a `lower`/`upper` frame (with `evaluate.level = 0.9`) or quantiles. `width90` and
`interval_width` score the mean interval length (shorter is better), and `coverage90`, or `coverage` at
`evaluate.level`, the share covered. Coverage is target-valued: better is closer to its nominal level, in paired
tests and against the paper's baseline, and a coverage claim replicates when ours is within tolerance of the
paper's value.

Papers often print shares in percent. A number the quote writes with `%` is recorded as percent; otherwise declare
it (`study extract --unit coverage90=percent`, or `study replicate --paper-unit percent`). Fidelity and replication
convert to the metric's unit before comparing and print both, and they refuse a share above 1 with no unit rather
than compare 92.62 with 0.82. The tolerance is in the metric's unit (0.02 is two coverage points). A synthetic
rebuild of conformalized quantile regression (arXiv 1905.03222):

```
$ openlab study run cqr
  coverage90     0.9095 [0.8975, 0.922]
  width90        3.934 [3.892, 3.978]
  baseline model:QuantileReg: coverage90 0.8615 [0.846, 0.876], width90 3.16 [3.112, 3.207]
  model - model:QuantileReg (coverage90): 0.048 [0.036, 0.06], p 0.002, better: model (closer to 0.9)
$ openlab study fidelity cqr --paper evidence:9fe0808205c4 --ours result:6c5e084c32e8 --arm model:QuantileReg
fidelity:4df85fa72793  NOT rebuilt: model:QuantileReg coverage90 0.8615 vs the paper's 0.9262 (92.62% as printed), tolerance 0.02 as a fraction
$ openlab study replication cqr --paper evidence:8db9c6c42001 --ours result:6c5e084c32e8 --gated-by fidelity:4df85fa72793
replication:ccb187e6fe87  inconclusive (rebuild not faithful: fidelity:4df85fa72793 failed): model coverage90 0.9095 [0.8975, 0.922] vs the paper's 0.9033 (90.33% as printed), baseline 0.9262; ours closer to 0.9 than the baseline (0.0095 vs 0.0262 away)
```

## Writing the paper

The paper is yours (or your agent's): one LaTeX document, `paper/paper.tex`, started from a template by
`openlab study paper demo --init` (the abstract and the usual headings). Load any package and include any image from
the study's folder; the build copies each image in, records its sha256 and lists it as a figure from a file, which
openlab did not draw. `openlab study paper demo --draft` renders it as it stands, with its findings. Every number, table, figure, claim, citation and equation is a binding to the record, never typed:

| Binding | What it prints | What is checked |
|---|---|---|
| `\lab{result:…#metrics.mae}` | a number from a run | the run exists and still matches the study's code, spec and data |
| `\labtab{…}`, `\labfig{…}` | a table or a figure from the record | the same, with "selected from N trials" in the caption |
| `\labclaim{claim:…}` | a claim with its verdict and level | the claim's chain is complete; the wording does not outrun it |
| `\labcite{source:…}{quote}` | a citation | the quote is in the source's full text; a retracted source is printed as retracted |
| `\labeq{name}{model.fn}{latex}` | an equation | the LaTeX and your Python function agree on probe inputs |
| `\lab[ablation]{…}` | an earlier variant, on purpose | code may differ; spec and data may not |

Openlab generates the data, methods, literature, results, limitations and reproducibility sections from the record.
`openlab study paper-lint demo` lists every finding without building. `openlab study paper demo` refuses to build
while any remains, unless you waive it (a signed act, printed in the paper). The build checks for:

- an unbound number, or "significant" without a rejecting test
- "novel" anywhere
- a stale binding, or a quote not in its source
- an equation that disagrees with its function

The output is `paper/build/main.pdf`, a `paper` node and a signed bundle. `openlab study verify-paper` fails if the
record changed, or a cited source was retracted, after the build. More in [Claims, tests and the paper](claims.md).

## Your agent's side

Your agent works through the MCP server (`openlab-mcp`; `openlab init` writes `.mcp.json`) or the same verbs from a
shell (`openlab study ...`, `openlab tool NAME key=value`). The default MCP list holds the verbs an agent needs across
the three kinds of work (studies, literature, replication, claims, the frontier, delegation, requests and replies),
plus `openlab_tools`, which finds the other tools, and `openlab_tool`, which calls them under the same refusals. The
agent's contract is [For agents](AGENTS.md), served as `openlab://guide`.

Every refusal has three parts (error, meaning, fix), so an agent knows what to do instead of retrying. The same
refusal three times shows as LOOP in `openlab watch`.

## The web views

`openlab show --root DIR` is the live view for supervising agents: what each is doing now, its plan drawn against
the record, the inbox with agents' questions, the lead's Prompt conversation, burn, RAID, the audit ledger and the Workspace
(the research's files edited beside their rendering, comments on any file for the lead). See [openlab show](show.md).

`openlab app --root DIR` serves the workspace UI: the studies, each with tabs for runs, literature, claims, paper,
replication, frontier, forecasts, reproduction, the record and its agents; an Agents view with pause and stop; the
inbox, with approve and deny. Person acts ask for your passphrase in the browser. The terminal remains the main
interface.

## Also in the box

- **The question tree.** `study question`, `hypothesis` (with what would falsify it), `tree`. A question's status is
  computed from its children.
- **The research frontier.** `study frontier` ranks what to do next: untested hypotheses, claims at risk, open
  branches, empty cells in the evidence lattice. `study take` / `release` lease an item to one agent at a time.
- **The search tree.** `study run --op draft|improve|debug|fuse --parent RUN` records how each run was made from
  another, so the trials form a tree. `study search` scores it (UCT with a cost-shaped reward, exploring early and
  exploiting late) and the frontier offers expand, fuse, draft and select items. See [search.md](search.md).
- **The final candidate.** `[selection] rule` (best_val or seed_robust) is sealed with the plan; `study select`
  records the pick among N with its value adjusted for being the best of N, and the holdout look is on that run only.
- **Module checks.** `study check` runs lint, import, a smoke job on a little data and a determinism check in
  seconds; a run of code whose latest check is red is refused. `study ensemble` records a combination of runs as a
  counted run, and a reviewer's `critique` is advice with a computed track record. See [checks.md](checks.md).
- **Lessons.** `study lesson` records a rule whose direction must follow from a comparison of two runs; runs cite
  lessons, each lesson has a track record, and `openlab lessons find` searches every study's lessons.
- **The error budget.** `study test` spends alpha (alpha-investing). Once it is spent, the next test is refused,
  so an unattended loop cannot run unlimited tests.
- **Assumptions and dead ends.** `study assume` / `break` mark every claim that rests on a broken assumption as at
  risk. `study dead-end` keeps dropped ideas where sibling agents see them.
- **Variants.** `study runs` lists every trial with its code version and branch. `study compare` and `study test
  --baseline result:…` pair two of your own results on the same rows.
- **Forward runs and monitoring.** `study forecast` / `fill` write signed forecasts and record their outcomes.
  `study monitor` re-runs the sealed searches for new, contradicting and retracted sources, and can be put on a
  schedule.
- **Servers and containers.** Openlab sizes its work by the container's cgroup limits, not the host's counts, and every
  sandboxed child holds one of the machine's per-core slots, so concurrent studies never oversubscribe it. Each run
  records the machine it ran on. See [box.md](box.md).
- **Reproduction.** `study reproduce` reruns recorded runs from the record alone. `study bundle` writes a signed
  tar, and `openlab verify FILE.sig.json` checks a signed file offline.
- **A W&B mirror.** `openlab mirror` shows a study's runs in Weights & Biases, a view and never the record. See
  [mirror.md](mirror.md).

## The record is yours

Everything is plain files in your folder, readable without openlab:

- The record is one SQLite file per study (`.openlab/record.db`, two tables: nodes with a JSON payload, and links),
  plus a content-addressed store. `study new` runs `git init` in a folder that is not already in a repository (it
  makes no commit), and its `.gitignore` leaves the record in, so it can be committed with your code.
- The proposal is TOML with a rendered `proposal.md`; the evidence matrix is `--csv`; the paper is LaTeX and PDF.
- Agent activity is in `.openlab/control.db` (SQLite) in the workspace.
- `openlab study bundle demo out.tar` is a signed tar of the record, data and code that anyone can reproduce from.

## What openlab is not

- **Not a model provider.** It holds no API key and runs no model. Agents run under your own Claude Code or Codex
  login.
- **Not a search engine.** `discover` queries OpenAlex, arXiv and Crossref and records what it found; reading and
  judging are your agents' job and yours.
- **Not a hosted service.** Openlab itself sends nothing anywhere except the searches and paper downloads you ask
  for; your agents talk to their own providers as they always do.
