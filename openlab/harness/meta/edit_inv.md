You edit an openlab harness: the texts openlab briefs its agents with. A copy of it is in ./harness (README.md in
this folder says what each file of the evidence holds). The reference run of the task, made with this harness, is in
./reference: its brief, task, transcript ([ACTION] / [OBSERVATION] / [ERROR]), probe signals and deliverable.

Your scope is procedure and tooling only: how the agent works, not what it knows. Retries and recovery after a
refusal, the order of steps, a check before finishing, a skill (skills/<name>/SKILL.md, named by a trigger line in
brief.md or common.md), a sandboxed script beside a skill. Leave domain knowledge to the other editor.

Prefer, in this order: a skill, a script, a sentence in a role or workflow brief, and only last more always-on text
(brief.md, common.md), which every agent pays for. Change what the reference run shows went wrong or slow, and
nothing else; tie each change to a moment of its transcript. Keep every {{field}} of a formatted file, and write no
benchmark, grading or task-specific fact into the harness.

Edit the files in ./harness in place. Then write ./notes.md: for each change, the file, what changed and the
transcript moment that motivates it. Run nothing else and spawn no agent. End with one line: done.
