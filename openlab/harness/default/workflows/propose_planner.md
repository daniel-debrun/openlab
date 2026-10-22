Draft the research proposal for the study {rel} (an openlab study folder) from the person's prompt below.

1. Read {rel}/proposal.toml (a skeleton; each field says what goes in it) and {rel}/study.toml.
2. If it helps, scan the literature briefly (WebSearch, or openlab_tool("paper_search")) for the background, the
   baselines and the risks. Keep it short: a few searches, no full reading.
3. Fill proposal.toml: title, question, a short background, hypotheses (statement, prediction, falsifier: the result
   that would refute it), branches (question, the hypotheses it serves, role {roles}, method, data needs,
   budget {{usd, minutes, trials}}, a stop rule, a kill criterion), data, metrics (the primary first), baselines, a
   [review] protocol if a branch is a literature review (role screener), the total budget, risks. The total is at most
   ${cap_usd:g} and {cap_minutes:g} minutes; the branches' $ and trials add up to at most the total.
   Make study.toml's data, target and split fit the proposal (mode {mode}).
4. Run `openlab propose lint {rel}` (or openlab_tool("propose_lint")) and fix what it refuses, until it is clean.
5. Stop there. Approval is the person's act: do not approve, run, materialise or create any other study.
Finish with five lines: the question, the hypotheses, the branches with their budgets, and anything unresolved.

The person's prompt:
{prompt}
