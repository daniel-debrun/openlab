# A literature review

Openlab records the search, every screening decision and every extracted number on the study record. A number is
accepted only if its quote is on the page it cites and the number is written in the quote, and each decision names
the agent that made it.

## With Claude Code

```
openlab init --claude                       # hooks, the openlab-lead and openlab-screener roles, .mcp.json
openlab study new lit --template review     # study.toml with the protocol, README.md with the steps
```

Edit `[review]` in `lit/study.toml`: the question, the indexes (openalex, arxiv, crossref), the queries, the
include and exclude criteria, the coverage floor, the citation hops and the fields to extract (`[review.extract]`).
Then ask your lead: "run the review in lit with two screeners". The lead runs `study_discover`, splits the
unscreened sources between `openlab-screener` sub-agents, and merges what they find.

## The commands

| Step | Command |
|---|---|
| search (seals the protocol) | `openlab study discover lit` |
| what is left | `openlab study sources lit --status unscreened` |
| screen one | `openlab study screen lit SOURCE --decision include --criterion I1 --rationale "..." --quote "<20+ characters of it>"` |
| screen many (agents) | `study_screen(folder="lit", decisions=[{source, decision, criterion, rationale, quote}, ...])` |
| full text | `openlab study fetch lit SOURCE --page 4` |
| find the exact quote | `openlab study find lit SOURCE "0.91"` (a phrase or a number: its page and sentence) |
| extract a number | `openlab study extract lit SOURCE --page 4 --section "Table 2" --quote "<the row>" --fields '{"method": "EMOS", "dataset": "...", "metric": "CRPS", "value": 0.91}'` |
| PRISMA | `openlab study prisma lit` (with `screened_by`: decisions per agent) |
| evidence matrix | `openlab study evidence lit --table` or `--csv` |
| who decided what | `openlab study sources lit` (each decision with its agent) |

`--quote` is required for an agent's screen: openlab finds it in the abstract or the fetched full text.

## The search

```
$ openlab study discover lit
openalex q1/4: 25 hits (0.9 s)
arxiv q3/4: 19 hits (3.1 s)
...
citations hop 1 79/79: 30 hits (1.5 s)
coverage:7d3071b11c55  sufficient: 2 sources, 4 queries, 1 hops, 2088 citation edges, 2188 records -> 1121 unique
162 queries recorded, 1121 new sources
took 67 s: the 83 index calls took 97.8 s between them, so running them side by side saved 30.8 s
```

`discover` runs every query on every index, follows the citation hops, records each work once (deduplicated by DOI,
arXiv and OpenAlex ids, preprints linked to their published versions, retractions flagged) and judges the coverage
against the floor. It prints a line per index call as it returns. The calls run side by side, six at a time
(`OPENLAB_DISCOVER_WORKERS`). Each index keeps its own rate limit and backs off on a 429, so the indexes' throttling
sets the pace: about a minute for the template protocol with one citation hop. The `study_discover` reply carries
each query's timing. The first `discover` seals the protocol; after that a change is an amendment, and the coverage
floor needs a person.

## Checked quotes

```
$ openlab study extract rl18 source:b4fdc0746771 --page 8 --section "5.1" --quote "For the local model, this results in a slightly improved mean CRPS of 0.88" --fields '{"method": "local model", "crps": 0.88}'
error: the quote is not on page 8
meaning: it is on page 9
fix: quote the page as study_fetch_text gives it (layout=true for a table row)

$ openlab study extract rl18 source:b4fdc0746771 --page 9 --section "5.1" --quote "..." --fields '{"method": "local model", "crps": 0.85}'
error: the value of `crps` is not in the quote
meaning: 0.85 is not written in it
fix: quote the text that states it
```

The check proves that the quote is on the page and the number is written in it. It does not prove that the number
means what the field says (a target coverage read as a realised one, say); that reading is still the agent's, and
yours to review. `openlab study fetch lit SOURCE --page N --layout` (the tool's `layout=true`) keeps a table row on
one line, which makes the quote easier to copy exactly.

`study find` does the copying for you. Give it a phrase, or a number (found by value: 0.9 finds 0.90). It prints
each page that has it, with the exact sentence from the running text or the row from the laid-out table. A hit
shorter than 20 characters takes the line or sentence before it. Each is a quote `extract` accepts for that page as
it stands:

```
$ openlab study find lit source:5e1f0c2a9b3d "0.9"
p.2 (text): "The realised coverage of ACI was 0.90 on the electricity data."
p.2 (table row): "Method Coverage Width ACI 0.90 1.23"
next: openlab study extract lit source:5e1f0c2a9b3d --page <page> --quote "<quote>" --section ...
```

Agents call it as `openlab_tool("study_find", {folder, source, text})`; `--around N` adds sentences of context.
`extract --kind` defaults to `table` or `figure` when `--section` says so ("Table 2", "Fig. 3") and to text
otherwise.

## Several screeners

Each screening decision carries the agent that made it. An agent may change its own decision. When two agents
disagree on a source (one includes, one excludes), no decision stands: the source shows as `CONFLICT` in `study
sources`, counts as waiting for a person in PRISMA, further agent screens of it are refused, and a `screen_conflict`
item lands in `openlab inbox`. Resolve it with `openlab inbox approve N --pick 1 --reason "..."` (the pick is recorded
as your screen), or screen it yourself with `openlab study screen lit SOURCE --person ...`. A `maybe` hands a source
to a person the same way: it files a `screen_maybe` item with the source, the agent's quote and its rationale, and
you decide it with `openlab inbox approve N --pick include --reason "..."` (or `--pick exclude`), signed and recorded
as your screen. Screening the source yourself settles the item too.

## Claims

A finding from the review is a claim that rests on evidence: `openlab study claim lit "..." --state observed
--rests-on evidence:...`. A gap is "not found in the searched sources", never "novel"; `--state unknown` rests on the
coverage of a search. `study recall` checks recall against a published review's references, and `study contra`
records a search for evidence that contradicts a claim.

## As a branch of a proposal

A proposal branch with role `screener` and a `[review]` protocol becomes this review inside the study (see
[research from a prompt](research-from-prompt.md)).
