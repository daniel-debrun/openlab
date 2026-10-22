You compose openlab's harness for the next run of a task from evidence. This folder holds:
- ./reference: the run made with the base harness (brief, task, transcript, signals, deliverable);
- ./candidates/<id>/: for each edited harness, its diff against the base (diff.patch), its editor's notes
  (notes.md) and its probe run (transcript, signals, deliverable);
- ./contrast/<id>.txt: each probe's probe signals against the reference's, field by field;
- ./base: the base harness's files.

Weigh them as evidence, not as votes. A failure that repeats in the reference and a probe is real; a defect seen in
one run only may be noise. Candidates agreeing is not independent proof. A change that risks a regression counts as
much as one that promises a gain. Keeping the base harness is a sound decision when no change earns its place.

Decide one of: keep_base, adopt <candidate id>, or merge (some edits of several candidates). Write ./decision.json:
{{"decision": "keep_base" | "adopt" | "merge", "candidate": "<id, for adopt>",
  "edits": [{{"state": "kept" | "revised" | "dropped", "edit": "<file and change>", "evidence": "<the run, file and
  moment that justifies it>"}}]}}
For merge, also write the merged harness in ./merged (start from a copy of ./base). Every kept, revised or dropped
edit cites its evidence. Run nothing else and spawn no agent. End with one line: your decision.
