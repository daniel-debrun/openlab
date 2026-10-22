Execute the approved research proposal of the study {rel}: {title}
It is signed off by a person (proposal {pid}, sha {sha}) and the plan is sealed: do not edit proposal.toml or the
sealed fields of study.toml (an edit is refused until a person records an amendment).

Question: {question} (question node {root})

Hypotheses (each decided by its falsifier):
{hyps}

Branches (each is one frontier item; give each to one child agent):
{branches}

How to run it:
1. Delegate each branch with openlab_agent_spawn(prompt, role=<the branch's role>, study="{rel}",
   question="<branch id>: <branch question>", budget_usd=<the branch's $>, minutes=<the branch's minutes>), never more
   than the branch's budget. Never delegate with Claude Code's own Agent or Task tool: it is refused to a lead,
   because such a sub-agent runs outside the branch budget and the role prompt. The child's prompt carries the branch's method, data needs, hypotheses (node ids and
   falsifiers), trial share, stop rule, kill criterion and frontier item, and tells it to lease the item first with
   openlab_study_take(folder="{rel}", item=<item>) and to stop at its trial share.
2. A child that meets its kill criterion records each hypothesis it killed as a dead end, with its evidence:
   openlab_study_dead_end(folder="{rel}", of=<hypothesis node>, statement, reason, tags, shown_by=[result, evidence or
   test ids]), then openlab_tool("propose_stop", {{folder: "{rel}", branch: <id>, reason, killed: true}}).
   Siblings read the dead ends before they start anything.
3. When a stop rule fires (or a branch's budget is spent), the branch stops: openlab_tool("propose_stop",
   {{folder: "{rel}", branch: <id>, reason}}) and the child ends.
4. Person acts (holdout looks, protected amendments, more budget, a fresh question) go to openlab_agent_request; do
   other work meanwhile. If nothing can go on without the person, say so and end: once they decide, they resume
   the run (`openlab propose execute --resume`), and a new lead gets the unfinished branches.
5. openlab_agent_wait waits for children; openlab_propose_status(folder="{rel}") shows each branch's agents, spend,
   trials and hypotheses.
6. Finish with a summary: the hypotheses tested, killed and untouched (with node ids), each branch's state and spend,
   and what is left open.
