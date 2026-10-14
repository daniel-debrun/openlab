/**
 * The words the interface uses before it has earned them: the Guide sheet lists them.
 */
export const GLOSSARY: { term: string; gloss: string }[] = [
  {
    term: "study",
    gloss:
      "A folder: study.toml (the question, the data, the split, the metrics and the budgets), model.py, and a hash-chained record of every plan, run, result, claim and holdout look. The agents and you act on it through the same tools.",
  },
  {
    term: "seal",
    gloss:
      "The first run seals study.toml. After that a change to it stops the next run until it is recorded as an amendment with a reason; the split, the budgets, the primary metric and the holdout can only be amended by a person.",
  },
  {
    term: "trial",
    gloss:
      "One distinct research run of model.py, counted against the study's trial budget. Running identical code on identical data again is answered from the record and is not a new trial.",
  },
  {
    term: "holdout look",
    gloss:
      "Scoring the held-back test set once. A person's act, counted against its own budget: the best of many looks at a test set is biased upward, so the count is what makes the score a measurement.",
  },
  {
    term: "claim",
    gloss:
      "A finding the paper may print, resting on results, evidence (a quote checked on its page) or other claims. Its verdict and level are computed from the record, never typed.",
  },
  {
    term: "agent",
    gloss:
      "A Claude Code or Codex session, a sub-agent it delegated to, or an agent openlab spawned. Each has an id, a parent, a budget and a timeline, and every node it writes to a study carries its id.",
  },
  {
    term: "inbox",
    gloss:
      "The requests agents filed for a person: a holdout look, an amendment, an approval, more budget. Approving performs the act as you and records it; denying records why.",
  },
  {
    term: "person's act",
    gloss:
      "An act an agent cannot make for you: a holdout look, a protected amendment, approving a claim, deciding an inbox item, resuming a paused agent. In this app it needs the reviewer link `openlab app` printed (its &r=), which the agent terminal never receives.",
  },
  {
    term: "flags",
    gloss:
      "What a supervising person should look at: STUCK (no event for a while), LOOP (the same refusal again and again), OVER (90% of a budget spent), WAITING (a request waits on you), BYPASS (it tried to get around a person's act), DEAD-END (its question matches a dead end another agent recorded).",
  },
];
