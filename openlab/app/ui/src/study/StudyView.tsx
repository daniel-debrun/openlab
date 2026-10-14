/**
 * A study, open in the document pane: the question, the sealed spec and every amendment to it, then a tab per
 * area the record has something for: the approved proposal's branches (Plan), the runs (datasets, runs and their
 * results, the forking paths), the literature, the claims, the paper, the replication, the frontier, forecasts and
 * monitoring, reproductions, the agents working on the study (agent_tree) and the record log. Which tabs show is
 * the record's `kinds` (`study_status`); each tab reads its own tools.
 *
 * It reaches the pane by its path: the open path is `<folder>/study.toml`, dispatched in `Shell`.
 * Everything here is a study tool, the same ones an agent calls, and nothing is computed on this side.
 *
 * Run and Verify chain are anyone's. A holdout look, a person's amendment, screening a maybe, approving a claim
 * and waiving a finding are a person's acts: the server refuses them unless the request carries the reviewer secret
 * from the URL `openlab app` printed (the client sends it whenever it has it), which the agent terminal never
 * receives. They show only to the reviewer, behind a dialog that takes the decision and its reason.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { REVIEWER } from "../api/client";
import {
  STUDY_LOG,
  STUDY_RUNS,
  callTool,
  studyFolder,
  type StudyDataset,
  type StudyFrontier,
  type StudyLog,
  type StudyRuns,
  type StudyStatus,
  type StudyVerify,
  type WsList,
  type AgentTree,
} from "../api/tools";
import { isWorkspace, useEvents } from "../api/events";
import { Async, Button, Callout, Empty, Num, Table, Tabs, cx, fmtAgo, fmtDate, fmtDateFull, fmtHash, prompt, type Column } from "../ui";
import { ActorBadge, Section, useAct, useRead } from "./parts";
import Claims from "./Claims";
import Forecasts from "./Forecasts";
import Frontier, { FRONTIER_N } from "./Frontier";
import Literature from "./Literature";
import Paper from "./Paper";
import Plan from "./Plan";
import RecordLog from "./RecordLog";
import Replication from "./Replication";
import Reproduction from "./Reproduction";
import { RunsTable, SpecCurve } from "./StudyRuns";
import Agents from "../agents/Agents";
import { LIVE, useAgentTree } from "../agents/parts";

const DATASET_COLUMNS: Column<StudyDataset>[] = [
  { key: "name", header: "dataset", kind: "raw", render: (d) => <code>{d.name}</code> },
  { key: "rows", header: "rows", kind: "count" },
  { key: "sha", header: "sha256", kind: "raw", render: (d) => <code title={d.sha}>{fmtHash(d.sha, 12)}</code> },
  {
    key: "time", header: "time coverage", kind: "raw",
    render: (d) => (d.coverage.time ? <>{d.coverage.time.min} to {d.coverage.time.max} <span className="dim">({d.coverage.time.column})</span></> : <span className="dim">—</span>),
  },
  {
    key: "key", header: "entities", kind: "raw",
    render: (d) => (d.coverage.key ? <><Num value={d.coverage.key.distinct} kind="count" /> <span className="dim">{d.coverage.key.columns.join(", ")}</span></> : <span className="dim">—</span>),
  },
];

function VerifyNote({ v }: { v: StudyVerify }) {
  return v.ok ? (
    <Callout tone="info" title="the record's hash chain is intact">
      {v.nodes} nodes, each hashing the one before it; head <code title={v.head}>{fmtHash(v.head, 12)}</code>.
    </Callout>
  ) : (
    <Callout tone="danger" title={`the record was edited at node ${v.broken_at}`}>
      Node {v.broken_at} and every node after it no longer match their hashes: what the record says from there on
      cannot be trusted.
    </Callout>
  );
}

/** A tab of the view: whether the record has anything for it, its count, and whether it holds a state nobody may miss. */
interface Area {
  id: string;
  label: string;
  shown: boolean;
  count?: number;
  alarm?: string;
}

/** The view's tabs, in order, from what the record holds. Only which tabs show is decided here. */
function areas(s: StudyStatus, frontier: number, paperToml: boolean, agents?: AgentTree): Area[] {
  const k = s.kinds;
  const any = (...kinds: string[]) => kinds.some((x) => k[x]);
  return [
    { id: "plan", label: "Plan", shown: any("proposal") },
    { id: "runs", label: "Runs", shown: s.mode !== "review" || any("run", "dataset"), count: k.run },
    { id: "literature", label: "Literature", shown: any("source", "evidence"), count: k.source,
      alarm: k.retraction ? "a source this review rests on was retracted" : undefined },
    { id: "claims", label: "Claims", shown: any("claim", "question", "hypothesis", "test", "assumption", "dead_end"), count: k.claim,
      alarm: k.break ? "an assumption was broken: the claims that need it are at risk" : undefined },
    { id: "paper", label: "Paper", shown: any("paper") || paperToml, count: k.paper },
    { id: "replication", label: "Replication", shown: any("target") },
    { id: "frontier", label: "Frontier", shown: frontier > 0, count: frontier },
    { id: "forecasts", label: "Forecasts & monitor", shown: any("forecast", "watch", "retraction"), count: k.forecast,
      alarm: k.retraction ? "retractions found" : undefined },
    { id: "reproduction", label: "Reproduction", shown: any("reproduction"), count: k.reproduction },
    { id: "agents", label: "Agents", shown: true, count: agents?.agents.length,
      alarm: agents?.pending ? `${agents.pending} request${agents.pending === 1 ? "" : "s"} wait for you`
        : agents?.agents.some((a) => a.flags.some((f) => f !== "WAITING") && LIVE.includes(a.state))
          ? "an agent on this study is flagged" : undefined },
    { id: "record", label: "Record", shown: true, count: k.action },
  ].filter((a) => a.shown);
}

export default function StudyView({ doc }: { doc: string }) {
  const folder = studyFolder(doc);
  const status = useRead<StudyStatus>(folder, "study_status");
  const runs = useRead<StudyRuns>(folder, "study_runs", { n: STUDY_RUNS });
  const log = useRead<StudyLog>(folder, "study_log", { n: STUDY_LOG });
  const frontier = useRead<StudyFrontier>(folder, "study_frontier", { n: FRONTIER_N }, !!status.data?.sealed);
  const paperToml = useQuery({
    queryKey: ["study", folder, "paper.toml"],
    queryFn: async () => {
      const env = await callTool<WsList>("ws_list", { path: folder ? `${folder}/paper` : "paper" });
      return env.ok && env.result.entries.some((e) => e.name === "paper.toml");
    },
  });
  const agents = useAgentTree(folder || ".");
  const { busy, act, refresh } = useAct(folder);
  const [verified, setVerified] = useState<StudyVerify | null>(null);
  const [tab, setTab] = useState("");

  // a verb an agent ran (or this page did) is a new node on the record
  useEvents((ev) => { if (isWorkspace(ev) && String(ev.tool).startsWith("study_")) refresh(); }, undefined, [folder]);

  const verify = async () => setVerified((await act<StudyVerify>("study_verify")) ?? null);

  const amend = async (s: StudyStatus) => {
    const reason = await prompt({
      title: "Record a person's amendment",
      body: (
        <>
          <p>
            Records the edits made to <code>study.toml</code> since the seal, with your reason. Split, budget, sandbox,
            the primary metric and <code>known_at</code> can only be changed this way. Every run after it runs under
            the amended spec, and the amendment stays on the record for good.
          </p>
        </>
      ),
      label: "Reason",
      placeholder: "why the spec changes",
      confirmLabel: "Amend",
      typed: s.name,
    });
    if (reason?.trim()) await act("study_amend_person", { reason: reason.trim() });
  };

  const look = async (s: StudyStatus) => {
    const reason = await prompt({
      title: `Take holdout look ${s.looks.used + 1} of ${s.looks.budget}?`,
      body: (
        <>
          <p>
            Your look is recorded first, with the study&rsquo;s spec, code and data hashes, and counted against the
            budget. Then the holdout is scored once with exactly that code and data.
          </p>
          <p>
            It cannot be taken back: the look stays spent even if the run fails. Take it when the work is finished,
            not to decide what to try next.
          </p>
        </>
      ),
      label: "Why now (recorded with the look)",
      placeholder: "e.g. final model chosen after the research runs",
      confirmLabel: "Take the look",
      danger: true,
      typed: s.name,
    });
    if (reason?.trim()) await act("study_holdout", { reason: reason.trim() });
  };

  return (
    <div className="notebook study">
      <Async query={status} skeleton="lines" skeletonRows={6} context={`reading the study ${folder || "at the workspace root"}`}>
        {(s) => {
          const spent = s.looks.used >= s.looks.budget;
          const lookWhy = !s.sealed
            ? "the study is not sealed yet: its first run seals it"
            : spent
              ? `the holdout budget is spent: ${s.looks.used} of ${s.looks.budget} looks taken. A person's amendment can raise it`
              : `score the holdout once; look ${s.looks.used + 1} of ${s.looks.budget}`;
          const tabs = areas(s, frontier.data?.total ?? 0, !!paperToml.data, agents.data);
          const active = tabs.some((a) => a.id === tab) ? tab : tabs[0].id;
          return (
            <>
              <header className="notebook-head">
                <h2 className="notebook-title">
                  {s.name}
                  <span className="ds-kind">study</span>
                </h2>
                <div className="notebook-actions">
                  <Button size="sm" variant="primary" loading={busy === "study_run"} onClick={() => void act("study_run")}
                          title="run model.py in the sandbox over the research segment; never touches the holdout">
                    Run
                  </Button>
                  <Button size="sm" loading={busy === "study_verify"} onClick={() => void verify()}
                          title="recompute every node's hash and check the chain">
                    Verify chain
                  </Button>
                  {REVIEWER && (
                    <>
                      <Button size="sm" loading={busy === "study_amend_person"} disabled={!s.sealed}
                              title={s.sealed ? "record your edits to study.toml as a person's amendment" : "nothing to amend: the study is not sealed yet"}
                              onClick={() => void amend(s)}>
                        Amend…
                      </Button>
                      <Button size="sm" variant="danger" loading={busy === "study_holdout"} disabled={!s.sealed || spent}
                              title={lookWhy} onClick={() => void look(s)}>
                        Holdout look… {s.looks.used}/{s.looks.budget}
                      </Button>
                    </>
                  )}
                </div>
              </header>

              <p className="study-question">{s.question}</p>
              <div className="ds-facts">
                <span>{s.mode}</span>
                <span title={s.sealed_at ? fmtDateFull(s.sealed_at) : undefined}>
                  {s.sealed ? `sealed ${fmtAgo(s.sealed_at)}` : "not sealed: the first run seals study.toml"}
                </span>
                {s.amendments.length > 0 && <span>amended {s.amendments.length}×</span>}
                <span>trials <Num value={s.trials.used} kind="count" /> of <Num value={s.trials.budget} kind="count" /></span>
                <span className={cx(spent && "plan-warn")}>
                  holdout looks <Num value={s.looks.used} kind="count" /> of <Num value={s.looks.budget} kind="count" />
                </span>
                <code>{folder || "."}</code>
              </div>

              {verified && <VerifyNote v={verified} />}
              {!REVIEWER && (
                <Callout tone="info" title="a person's acts need the reviewer link">
                  Holdout looks, person amendments, screening, approvals and waivers show only at the full URL{" "}
                  <code>openlab app</code> printed, with its <code>&amp;r=</code>: the agent never gets that part, which is
                  what makes the act a person&rsquo;s. Or run <code>openlab study</code> in your own terminal.
                </Callout>
              )}

              {s.amendments.length > 0 && (
                <Section title="Amendments" hint="changes to the sealed spec, oldest first; each run after one runs under it">
                  <ol className="study-log">
                    {s.amendments.map((a) => (
                      <li key={a.id} className={cx("study-log-item", a.actor === "person" && "is-person")}>
                        <time className="study-log-at" dateTime={a.at} title={fmtDateFull(a.at)}>{fmtDate(a.at)}</time>
                        <ActorBadge actor={a.actor} />
                        <code className="study-log-verb">{a.changed.join(", ") || "no field"}</code>
                        <span className="study-log-outcome">{a.reason}</span>
                      </li>
                    ))}
                  </ol>
                </Section>
              )}

              <Tabs
                className="study-tabs"
                items={tabs.map((a) => ({
                  id: a.id,
                  label: a.alarm ? <>{a.label} <span className="study-tab-alarm" title={a.alarm} aria-label={a.alarm}>⚠</span></> : a.label,
                  badge: a.count || undefined,
                }))}
                value={active}
                onChange={setTab}
              />

              {active === "runs" && (
                <>
                  <Section title="Datasets">
                    <Table
                      columns={DATASET_COLUMNS}
                      rows={s.datasets}
                      dense
                      rowKey={(d) => d.name}
                      empty={
                        <Empty title="no datasets on record"
                               body="Materialise builds each [data] section of study.toml: it runs the recipe or fetches the source, stores the file and records its sha256 and card."
                               action={{ label: busy === "study_materialise" ? "Materialising…" : "Materialise", onClick: () => void act("study_materialise") }} />
                      }
                    />
                  </Section>

                  <Section title="Runs" hint="newest first; open a row for every metric, the baselines and the paired differences">
                    <Async query={runs} skeleton="table" context="reading the runs">
                      {(d) => <RunsTable folder={folder} d={d} />}
                    </Async>
                  </Section>

                  <Section title="Forking paths" hint="the specification curve: the primary metric of every research run">
                    <Async query={runs} skeleton="chart">{(d) => <SpecCurve d={d} />}</Async>
                  </Section>
                </>
              )}
              {active === "plan" && <Plan folder={folder} />}
              {active === "literature" && <Literature folder={folder} />}
              {active === "claims" && <Claims folder={folder} kinds={s.kinds} />}
              {active === "paper" && <Paper folder={folder} kinds={s.kinds} />}
              {active === "replication" && <Replication folder={folder} />}
              {active === "frontier" && <Frontier folder={folder} />}
              {active === "forecasts" && <Forecasts folder={folder} kinds={s.kinds} />}
              {active === "reproduction" && <Reproduction folder={folder} />}
              {active === "agents" && <Agents study={folder || "."} />}
              {active === "record" && (
                <Section title="Record" hint="every verb run on this study, newest first; Verify chain checks none was edited">
                  <Async query={log} skeleton="lines" context="reading the record">{(d) => <RecordLog d={d} />}</Async>
                </Section>
              )}
            </>
          );
        }}
      </Async>
    </div>
  );
}
