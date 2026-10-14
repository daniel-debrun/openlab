# A literature review

`study.toml` holds the protocol: the question, the indexes, the queries, the inclusion and exclusion criteria, the
coverage floor and the fields to extract. Edit it first; the first `discover` seals it.

## Steps

1. Edit `[review]` in study.toml: the queries (at least `coverage_floor.queries_per_source` of them), the
   criteria, and `[review.extract]`.
2. `openlab study discover FOLDER`: runs every query on every index, follows the citation hops, records each work
   once and judges the coverage against the floor.
3. `openlab study sources FOLDER --status unscreened`: what is left to screen.
4. Screen: `openlab study screen FOLDER SOURCE --decision include|exclude|maybe --criterion I1 --rationale "..."
   --quote "<20+ characters of its abstract or full text>"` (the quote is required for an agent). Agents screen in
   batches with `study_screen(folder, decisions=[...])`, and each decision names the agent that made it. Two agents
   that disagree on a source make a `screen_conflict` item in `openlab inbox`; `maybe` hands a source to a person.
5. Full text: `openlab study fetch FOLDER SOURCE`, then extract each number with its page and quote:
   `openlab study extract FOLDER SOURCE --page 4 --section "Table 2" --quote "<the row>"
   --fields '{"method": "EMOS", "dataset": "...", "metric": "CRPS", "value": 0.91}'`.
   A quote that is not on that page, or a number that is not written in it, is refused.
6. Read it back: `openlab study prisma FOLDER` (the PRISMA flow) and `openlab study evidence FOLDER --table [--csv]`
   (the evidence matrix for a related-work section).
7. Claims: `openlab study claim FOLDER "..." --state observed --rests-on evidence:...`. A gap is "not found in the
   searched sources", never "novel".
