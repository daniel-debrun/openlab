"""A demo research for openlab's screenshots: a real study and the agent activity around it.

The study is real: three counted runs, a tested hypothesis, a claim, an assumption, a dead end, a paper bound to
the record and a compiled release. The agents are staged: a lead, an experimenter and an adversary with their
plans, notes and tool calls, a question in the inbox and a comment on the paper, on a clock spread over the
afternoon before now. Nothing is spent and no agent runs.

    python scripts/demo_workspace.py DIR       # then: openlab show --root DIR
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REAL_NOW = dt.datetime.now().astimezone()


class Clock:
    """Every timestamp openlab writes, from `at(minutes before now)` on, one second per write."""

    def __init__(self):
        self.t = REAL_NOW

    def at(self, minutes_ago: float):
        self.t = REAL_NOW - dt.timedelta(minutes=minutes_ago)

    def tick(self) -> dt.datetime:
        self.t += dt.timedelta(seconds=1)
        return self.t


CLOCK = Clock()


def _patch_clock():
    import openlab.agents.comments as comments
    import openlab.agents.plan as plan
    import openlab.agents.store as store
    import openlab.study.clock as sclock
    import openlab.study.record as record

    def utc():
        return CLOCK.tick().astimezone(dt.timezone.utc).isoformat(timespec="milliseconds")

    def local():
        return CLOCK.tick().isoformat(timespec="seconds")

    for m in (store, plan, comments):
        m.now = utc
    sclock.now = record.now = local
    import openlab.agents.identity as identity
    identity.person_label = lambda: "person:dana"           # the demo's person, not whoever runs the script


LEAD, EXP, ADV = "lead-4f2a", "experimenter-b71c", "adversary-9d3e"
STUDY = "next-day"
QUESTION = ("Does today's reading x predict tomorrow's temperature y at each station better than persistence and "
            "climatology?")

FEATURES_7D = '''

def features(view, t):
    """What is known at t, one row per site: today's x and y, and the mean of y over the last seven days."""
    return (view.frame("panel").sort("date").group_by("site")
            .agg(pl.col("x").last(), pl.col("y").last().alias("y_now"), pl.col("y").tail(7).mean().alias("y_7d"))
            .sort("site"))
'''

GBM = '''

class Model:
    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        self.m.fit(X.select(COLUMNS).to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select(COLUMNS).to_numpy())
'''

PAPER = r"""\documentclass[11pt]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage{graphicx,booktabs}

\title{Next-day station temperature from today's reading}
\author{openlab agents lead-4f2a, experimenter-b71c and adversary-9d3e}
\date{}

\begin{document}
\maketitle

\begin{abstract}
We ask whether today's reading at a station predicts tomorrow's temperature, against persistence and
climatology. A linear model on today's reading and temperature reaches a mean absolute error of
\lab{RUN1#metrics.mae.value}; persistence reaches \lab{RUN1#baselines.persistence.mae.value}, evaluated
walk-forward over the research year with every trial counted. Adding the seven-day mean and a gradient-boosted
model were tried as well and are recorded as dead ends.
\end{abstract}

\section{Introduction}
Persistence is a strong reference a day ahead. The question is whether a reading taken today carries
information about tomorrow that today's temperature does not.

\section{Data}
\labsection{data}

\section{Methods}
\labsection{methods}

\section{Results}
The selected model, linear on today's reading and temperature:

\labtab{RUN1}

\labclaim{CLAIM}

\section{Discussion}
The adversarial review looked for look-ahead in the features and for a single station carrying the result; what it
tried and what held is on the record as critiques and dead ends.

\section{Limitations}
\labsection{limitations}

\section{Reproducibility}
\labsection{reproducibility}

\end{document}
"""


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:] or ["openlab-demo"])[0]).resolve()
    res = root / "station-forecasts"
    if res.exists():
        sys.exit(f"{res} exists: pick an empty folder")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "openlab", "init", str(res)], check=True, capture_output=True)
    toml = res / "openlab.toml"
    toml.write_text(toml.read_text() + '\n[team.sam]\nrole = "meteorologist"\nhours = 10\nhours_used = 3.5\n')

    _patch_clock()
    from openlab import study
    from openlab.agents.comments import threads
    from openlab.agents.identity import acting_as
    from openlab.agents.store import Store
    from openlab.app import ToolContext, dispatch

    ctx, st = ToolContext(str(res)), Store(res)

    def tool(agent, name, args, human=False):
        os.environ["OPENLAB_AGENT"] = agent
        env = dispatch(ctx, name, args, human=human)
        if not env["ok"]:
            raise SystemExit(f"{name}: {env.get('error')}: {env.get('meaning')}")
        return env["result"]

    def ev(agent, kind, summary, tool_name=""):
        st.event(agent, kind, tool=tool_name, summary=summary)

    def run(agent, note):
        """(run id, result id) of a counted run made as `agent`."""
        with acting_as(agent):
            r = study.open(res / STUDY).run(note=note)
        rec = study.open(res / STUDY).record
        rid = r["id"]
        if rid.startswith("result:"):
            pair = (rec.links(rid, rel="of")[0][2], rid)
        else:
            pair = (rid, next(n.id for n in rec.find("result") if rec.links(n.id, rel="of")[0][2] == rid))
        ev(agent, "run_end", f"{STUDY}: {pair[0]} recorded, {pair[1]} ({note})")
        return pair

    # --- the lead starts: the study, its plan, the data ----------------------------------------------------------
    CLOCK.at(200)
    st.register(LEAD, "spawned", role="lead", status="running", study=STUDY, cli="claude", model="claude-opus-5-5",
                budget={"usd": 20, "minutes": 360, "turns": 400})
    ev(LEAD, "start", "claude lead: study next-day from the person's question")
    ev(LEAD, "tool", "openlab_study_new next-day", "mcp__openlab__openlab_study_new")
    tool(LEAD, "study_new", {"folder": STUDY, "template": "temporal"})
    spec = res / STUDY / "study.toml"
    spec.write_text(spec.read_text().replace(
        'question = "Does today\'s signal x predict tomorrow\'s y at each site better than persistence and climatology?"',
        f'question = "{QUESTION}"'))
    steps = [
        {"id": "data", "title": "Materialise the station panel", "kind": "data", "state": "active"},
        {"id": "base", "title": "Linear baseline on x and today's y", "kind": "run", "needs": ["data"]},
        {"id": "feat", "title": "Add the seven-day mean", "kind": "run", "needs": ["base"], "owner": EXP},
        {"id": "gbm", "title": "Gradient boosting on the same features", "kind": "run", "needs": ["feat"], "owner": EXP},
        {"id": "attack", "title": "Adversarial review of the best run", "kind": "critique", "needs": ["base"],
         "owner": ADV},
        {"id": "test", "title": "Test the hypothesis against persistence", "kind": "test", "needs": ["base"]},
        {"id": "paper", "title": "Write the paper", "kind": "paper", "needs": ["test", "attack"]},
        {"id": "sensors", "title": "Check the three relocated stations", "kind": "review", "owner": "sam"},
        {"id": "look", "title": "Holdout look", "kind": "person", "needs": ["paper"], "owner": "you"},
    ]
    tool(LEAD, "agent_plan", {"nodes": steps, "note": "first plan: baseline, two improvements, a review, the paper"})
    CLOCK.at(192)
    ev(LEAD, "tool", "openlab_study_materialise next-day", "mcp__openlab__openlab_study_materialise")
    data = tool(LEAD, "study_materialise", {"folder": STUDY})
    ds = next(iter(data.values()))["id"] if isinstance(data, dict) and data and isinstance(next(iter(data.values())), dict) else ""
    ds = ds or study.open(res / STUDY).record.find("dataset")[-1].id
    tool(LEAD, "agent_plan_node", {"id": "data", "state": "done", "links": [ds], "note": "2,192 station-days"})

    CLOCK.at(185)
    ev(LEAD, "tool", "openlab_study_run next-day linear on x and today's y", "mcp__openlab__openlab_study_run")
    (run1, res1) = run(LEAD, "linear on x and today's y")
    tool(LEAD, "agent_plan_node", {"id": "base", "state": "done", "links": [run1, res1]})
    ev(LEAD, "say", "The linear model beats both baselines on MAE; delegating the two improvements and a review.")

    # --- the experimenter and the adversary ------------------------------------------------------------------------
    CLOCK.at(178)
    for a, role, budget, minutes in ((EXP, "experimenter", 6, 240), (ADV, "adversary", 4, 120)):
        st.register(a, "spawned", role=role, status="running", study=STUDY, parent=LEAD, cli="claude",
                    model="claude-sonnet-5-5", budget={"usd": budget, "minutes": minutes, "turns": 200})
        ev(LEAD, "spawn", f"claude {role} for {LEAD}", "mcp__openlab__openlab_agent_spawn")
        ev(a, "start", f"claude {role}: {STUDY}")
    tool(EXP, "agent_plan", {"nodes": [
        {"id": "feat", "title": "Seven-day mean of y as a feature", "kind": "run", "state": "active"},
        {"id": "gbm", "title": "Gradient boosting on the same features", "kind": "run", "needs": ["feat"]},
        {"id": "ablation", "title": "Ablation table for the paper", "kind": "paper", "needs": ["gbm"]}],
        "note": "the lead's two runs, then the ablation table"})
    model = res / STUDY / "model.py"
    src = linear = model.read_text()
    head, _, rest = src.partition("\n\ndef features(view, t):")
    rest = rest.split("\n\ndef label(view, t):", 1)[1]
    model.write_text(head.replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x", "y_now", "y_7d"]') + FEATURES_7D
                     + "\n\ndef label(view, t):" + rest)
    CLOCK.at(170)
    ev(EXP, "tool", "Edit next-day/model.py", "Edit")
    ev(EXP, "tool", "openlab_study_run next-day seven-day mean of y", "mcp__openlab__openlab_study_run")
    (run2, res2) = run(EXP, "seven-day mean of y")
    tool(EXP, "agent_plan_node", {"id": "feat", "state": "done", "links": [run2, res2],
                                  "note": "no gain over the linear baseline"})

    text = model.read_text()
    model.write_text(text.split("\n\nclass Model:", 1)[0] + GBM)
    CLOCK.at(150)
    ev(EXP, "tool", "Edit next-day/model.py", "Edit")
    ev(EXP, "tool", "openlab_study_run next-day gradient boosting, same features", "mcp__openlab__openlab_study_run")
    (run3, res3) = run(EXP, "gradient boosting, same features")
    tool(EXP, "agent_plan_node", {"id": "gbm", "state": "done", "links": [run3, res3],
                                  "note": "worse than the linear baseline"})
    tool(EXP, "agent_plan_node", {"id": "ablation", "state": "active"})
    tool(EXP, "agent_note", {"text": "Neither the seven-day mean nor boosting improves on the linear baseline. "
                                     "Drafting the ablation table."})

    # --- the adversary attacks the first run --------------------------------------------------------------------------
    CLOCK.at(140)
    tool(ADV, "agent_plan", {"nodes": [
        {"id": "lookahead", "title": "Look-ahead in the features", "kind": "critique", "state": "active"},
        {"id": "one", "title": "One station carrying the result", "kind": "critique"}],
        "note": "two attacks"})
    ev(ADV, "tool", "Read next-day/model.py", "Read")
    ev(ADV, "tool", "Read next-day/recipes/make_data.py", "Read")
    q = tool(ADV, "study_question", {"folder": STUDY, "text": QUESTION})["id"]
    h = tool(LEAD, "study_hypothesis", {
        "folder": STUDY, "statement": "Today's reading and temperature predict tomorrow's better than persistence",
        "prediction": "the linear model's MAE is below persistence's with an interval excluding zero",
        "falsified_by": "a paired interval for model minus persistence that includes zero or is above it",
        "question": q, "tags": ["temperature", "next-day"]})["id"]
    dead = tool(ADV, "study_dead_end", {
        "folder": STUDY, "of": q, "statement": "Look-ahead through the feature view",
        "reason": "features() reads only rows dated on or before t; the truncation audit of every run is clean",
        "tags": ["look-ahead"], "shown_by": [res1]})["id"]
    tool(ADV, "agent_plan_node", {"id": "lookahead", "state": "done", "links": [dead]})
    CLOCK.at(125)
    tool(ADV, "agent_plan_node", {"id": "one", "state": "done", "links": [res1],
                                  "note": "leaving each station out keeps model minus persistence below zero"})
    ev(ADV, "say", "Both attacks held: no look-ahead, no single station carries it. Recorded the dead end.")
    st.set_spent(ADV, usd=1.18, turns=41)
    st.set_status(ADV, "done")
    ev(ADV, "end", "done (exit 0)")
    tool(LEAD, "agent_plan_node", {"id": "attack", "state": "done", "links": [dead]})

    for res_n, what, why in ((res2, "The seven-day mean of y as a feature", "its interval overlaps the baseline's"),
                             (res3, "Gradient boosting on the same features", "its MAE is above the linear model's")):
        tool(EXP, "study_dead_end", {"folder": STUDY, "of": q, "statement": what, "reason": why,
                                     "tags": ["features"], "shown_by": [res_n]})
    CLOCK.at(115)
    model.write_text(linear)
    ev(LEAD, "tool", "Edit next-day/model.py: back to the linear model", "Edit")
    ev(LEAD, "tool", "openlab_study_run next-day final: the linear model, selected", "mcp__openlab__openlab_study_run")
    (run4, res4) = run(LEAD, "final: the linear model, selected")
    steps_final = {"id": "final", "title": "Rerun the selected model on the study's code", "kind": "run",
                   "needs": ["gbm", "attack"], "state": "done", "links": [run4, res4]}
    tool(LEAD, "agent_plan", {"nodes": [*[{**x, "state": x.get("state", "planned")} for x in steps[:-3]],
                                        steps_final, *steps[-3:]],
                              "note": "the linear model goes forward; a final run on the study's code"})
    for nid, links in (("data", [ds]), ("base", [run1, res1]), ("feat", [run2, res2]), ("gbm", [run3, res3]),
                       ("attack", [dead])):
        tool(LEAD, "agent_plan_node", {"id": nid, "state": "done", "links": links})

    # --- the test, the claim, the assumption -------------------------------------------------------------------------
    CLOCK.at(110)
    ev(LEAD, "tool", "openlab_study_test of the hypothesis on the linear run", "mcp__openlab__openlab_study_test")
    test = tool(LEAD, "study_test", {"folder": STUDY, "of": h, "result": res4, "baseline": "persistence",
                                     "alpha_j": 0.04})
    test_id = test["id"]
    tool(LEAD, "agent_plan_node", {"id": "test", "state": "done", "links": [test_id]})
    claim = tool(LEAD, "study_claim", {
        "folder": STUDY, "statement": "The linear model on today's reading and temperature has a lower next-day MAE "
                                      "than persistence over the research year",
        "state": "observed", "rests_on": [res4]})["id"]
    tool(LEAD, "study_assume", {"folder": STUDY, "statement": "Station positions are fixed over the research year",
                                "rests_on": res4, "breaks_if": "a station moved within the year"})

    # --- the paper and the release ----------------------------------------------------------------------------------
    CLOCK.at(80)
    tool(LEAD, "agent_plan_node", {"id": "paper", "state": "active"})
    tool(LEAD, "study_paper_init", {"folder": STUDY})
    (res / STUDY / "paper" / "paper.tex").write_text(PAPER.replace("RUN1", res4).replace("CLAIM", claim))
    ev(LEAD, "tool", "Write next-day/paper/paper.tex", "Write")
    ev(LEAD, "tool", "openlab_study_paper next-day draft=true", "mcp__openlab__openlab_study_paper")
    (res / STUDY / "README.md").write_text(
        "# Next-day station temperature\n\nA walk-forward study of whether today's reading at a station predicts "
        "tomorrow's temperature better than persistence and climatology. The paper is `paper.pdf`; every number "
        "in it is bound to the record below.\n")
    CLOCK.at(60)
    tool(LEAD, "show_compile", {"study": STUDY})

    # --- the person: a comment on the paper, an answered and an open question ----------------------------------------
    CLOCK.at(45)
    quote = "Persistence is a strong reference a day ahead."
    tex = (res / STUDY / "paper" / "paper.tex").read_text()
    c = tool("person", "show_comment", {"path": f"{STUDY}/paper/paper.tex", "text": "Say by how much, with its "
             "interval, and cite the paired test.", "quote": quote, "start": tex.index(quote),
             "end": tex.index(quote) + len(quote)}, human=True)
    CLOCK.at(40)
    tool(LEAD, "agent_comment_reply", {"id": c["id"], "text": "Added the paired difference and its interval from "
                                       "the test; the sentence now cites it."})
    CLOCK.at(30)
    first = tool(EXP, "agent_request", {"kind": "question", "study": STUDY, "reason": "Which model goes into the "
                                        "paper's ablation table besides the selected linear one?",
                                        "args": {"choices": ["Both experiments", "Only gradient boosting"],
                                                 "other": True}})["item"]
    tool("person", "inbox_answer", {"id": first, "picks": ["Both experiments"],
                                    "message": "both, with their dead ends cited"}, human=True)
    CLOCK.at(8)
    tool(LEAD, "agent_request", {
        "kind": "question", "study": STUDY,
        "reason": "The paper builds and the adversary's attacks held. Look at the holdout now, or one more trial first?",
        "args": {"choices": ["Look at the holdout now", "One more trial first: five seeds of the linear model"],
                 "other": True}})

    # --- what each one is doing now -------------------------------------------------------------------------------------
    CLOCK.at(3)
    ev(LEAD, "note", "Paper builds with every binding resolved; asked the person about the holdout look.")
    st.set_spent(LEAD, usd=4.62, turns=118)
    st.set_spent(EXP, usd=2.07, turns=64)
    CLOCK.at(0.4)
    ev(EXP, "tool", "Edit next-day/paper/paper.tex: the ablation table", "Edit")
    _transcript(st, res, res1)
    os.environ.pop("OPENLAB_AGENT", None)
    print(json.dumps({"root": str(root), "research": str(res), "runs": [run1, run2, run3, run4],
                      "claim": claim, "comments": len(threads(st))}, indent=1))
    return 0


def _transcript(st, res, result):
    """The lead's conversation up to its baseline step, as the CLI's stream would have written it."""
    d = st.agent_dir(LEAD)
    (d / "prompt.txt").write_text(f"Study next-day: {QUESTION} Plan first, delegate the improvements and a review, "
                                  "test against persistence, write the paper.")
    t0 = REAL_NOW - dt.timedelta(minutes=186)

    def at(s):
        return (t0 + dt.timedelta(seconds=s)).astimezone(dt.timezone.utc).isoformat(timespec="milliseconds")

    lines = [
        {"type": "assistant", "timestamp": at(0), "message": {"content": [
            {"type": "text", "text": "The data is materialised. Running the template's linear model as the baseline."}]}},
        {"type": "assistant", "timestamp": at(2), "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "mcp__openlab__openlab_study_run",
             "input": {"folder": STUDY, "note": "linear on x and today's y"}}]}},
        {"type": "user", "timestamp": at(40), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": f"{result}: mae beats persistence and climatology"}]}},
        {"type": "assistant", "timestamp": at(42), "message": {"content": [
            {"type": "tool_use", "id": "t2", "name": "mcp__openlab__openlab_agent_plan_node",
             "input": {"id": "base", "state": "done", "links": [result]}}]}},
        {"type": "user", "timestamp": at(43), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": '{"ok": true, "step": "base", "state": "done"}'}]}},
    ]
    (d / "transcript.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


if __name__ == "__main__":
    sys.exit(main())
