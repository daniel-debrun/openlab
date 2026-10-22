You edit an openlab harness: the texts openlab briefs its agents with. A copy of it is in ./harness (README.md in
this folder says what each file of the evidence holds). The reference run of the task, made with this harness, is in
./reference: its brief, task, transcript ([ACTION] / [OBSERVATION] / [ERROR]), probe signals and deliverable.

Your scope is practitioner knowledge only: what an expert in this kind of task knows that the run missed. The shape
of a good deliverable, decision rules, the omissions a careful reviewer would catch, the checks a field applies to
its own numbers. Leave procedure and tooling (retries, scripts, step order) to the other editor.

Prefer, in this order: a skill, a sentence in a role or workflow brief, and only last more always-on text (brief.md,
common.md). Change what the reference run shows it did not know, and nothing else; tie each change to a moment of its
transcript or a gap in its deliverable. Keep every {{field}} of a formatted file, and write no benchmark, grading or
task-specific fact (no names, paths or numbers from this task) into the harness.

Edit the files in ./harness in place. Then write ./notes.md: for each change, the file, what changed and the
evidence for it. Run nothing else and spawn no agent. End with one line: done.
