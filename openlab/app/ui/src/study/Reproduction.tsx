/**
 * Reproductions (`study_reproductions`): recorded runs executed again from what the record holds only, each
 * matched or naming what differs, the largest prediction difference, each metric again and the environment
 * differences. A claim whose results all have a matched reproduction is at level +reproduced.
 */
import type { StudyReproduction, StudyReproductions } from "../api/tools";
import { Async, Badge, Callout, Num, Table, fmtAgo, fmtDateFull, type Column } from "../ui";
import { Alarm, Ids, Section, useRead } from "./parts";

const COLUMNS: Column<StudyReproduction>[] = [
  { key: "at", header: "when", kind: "raw", render: (r) => <span title={fmtDateFull(r.at)}>{fmtAgo(r.at)}</span> },
  { key: "run", header: "run", kind: "raw", render: (r) => <Ids ids={[r.run]} /> },
  {
    key: "matched", header: "", kind: "raw",
    render: (r) => (r.matched ? <Badge tone="ok">✓ matched</Badge>
      : <Alarm title={`run again from the record, it does not match: ${r.differs}`}>differs</Alarm>),
  },
  { key: "max_abs_diff", header: "max |diff|", kind: "ratio", title: "the largest difference between the two runs' predictions" },
  { key: "differs", header: "what differs", kind: "raw", render: (r) => r.differs || <span className="dim">—</span> },
  {
    key: "metrics", header: "metrics: run → again", kind: "raw",
    render: (r) => (
      <span className="study-fields">
        {Object.entries(r.metrics).map(([k, m]) => (
          <span key={k}><span className="dim">{k}</span> <Num value={m.run} kind="ratio" /> → <Num value={m.reproduced} kind="ratio" /></span>
        ))}
      </span>
    ),
  },
  {
    key: "env_diff", header: "environment", kind: "raw",
    render: (r) => (Object.keys(r.env_diff).length ? (
      <span className="study-fields">
        {Object.keys(r.env_diff).map((k) => <span key={k} title={`run ${String(r.env_diff[k].run)}, now ${String(r.env_diff[k].now)}`}>{k} changed</span>)}
      </span>
    ) : <span className="dim">same</span>),
  },
  { key: "sandbox", header: "sandbox", kind: "raw" },
  { key: "seconds", header: "took", kind: "duration" },
];

export default function Reproduction({ folder }: { folder: string }) {
  const q = useRead<StudyReproductions>(folder, "study_reproductions");
  return (
    <Section title="Reproductions" hint="each run executed again in a fresh sandbox from the code, data, spec and seed the record holds">
      <Async query={q} skeleton="table" context="reading the reproductions">
        {(d) => {
          const differ = d.reproductions.filter((r) => !r.matched).length;
          return (
            <>
              <div className="ds-facts">
                <span><Num value={d.reproduced} kind="count" /> of <Num value={d.runs} kind="count" /> runs have a matched reproduction</span>
              </div>
              {differ > 0 && (
                <Callout tone="danger" title={`${differ} reproduction${differ === 1 ? "" : "s"} did not match`}>
                  Run again from what the record holds, the predictions came out different. Until a reproduction matches,
                  the claims resting on that run stay below +reproduced.
                </Callout>
              )}
              <Table columns={COLUMNS} rows={[...d.reproductions].reverse()} dense rowKey={(r) => r.id}
                     empty={<p className="dim small">No reproduction yet: study_reproduce runs a recorded run again.</p>} />
            </>
          );
        }}
      </Async>
    </Section>
  );
}
