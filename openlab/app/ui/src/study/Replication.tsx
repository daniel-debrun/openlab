/**
 * A replication of one paper (`study_replications`): the target and its tolerance policy, the fidelity checks
 * (did we rebuild the paper's baselines: the paper's value against ours, within tolerance) and each claim's verdict
 * (does it hold: ours with its interval against the paper's). All of it is computed by the server; a rebuild that
 * is not faithful makes every verdict inconclusive, so a bad rebuild cannot fail the paper.
 */
import type { Fidelity, Replication as ReplicationRow, StudyReplications } from "../api/tools";
import { Async, Badge, Callout, KeyValue, Num, Table, type Column } from "../ui";
import { Alarm, Ids, Interval, Section, Standing, useRead } from "./parts";

const where = (r: { page: number; section: string }) => <span className="small">{r.section}, p.{r.page}</span>;

const FIDELITY_COLUMNS: Column<Fidelity>[] = [
  { key: "method", header: "baseline", kind: "raw" },
  { key: "where", header: "in the paper", kind: "raw", render: where },
  { key: "arm", header: "our arm", kind: "raw", render: (f) => <code>{f.arm}</code> },
  { key: "paper", header: "paper", kind: "ratio" },
  { key: "ours", header: "ours", kind: "ratio" },
  { key: "tolerance", header: "tolerance", kind: "ratio" },
  {
    key: "passed", header: "", kind: "raw",
    render: (f) => (f.passed ? <Badge tone="ok">✓ rebuilt</Badge>
      : <Alarm title="our rebuild of this baseline is outside the tolerance: the replication verdicts it gates are inconclusive">not rebuilt</Alarm>),
  },
];

const REPLICATION_COLUMNS: Column<ReplicationRow>[] = [
  { key: "method", header: "claim", kind: "raw" },
  { key: "verdict", header: "verdict", kind: "raw", render: (r) => <Standing word={r.verdict} why={r.reason || undefined} /> },
  { key: "where", header: "in the paper", kind: "raw", render: where },
  { key: "arm", header: "our arm", kind: "raw", render: (r) => <code>{r.arm}</code> },
  { key: "paper", header: "paper", kind: "ratio" },
  { key: "ours", header: "ours [lo, hi]", kind: "ratio", render: (r) => <Interval s={{ value: r.ours, lo: r.lo, hi: r.hi }} /> },
  { key: "reference", header: "best gating baseline", kind: "ratio" },
  { key: "reason", header: "reason", kind: "raw", render: (r) => r.reason || <span className="dim">—</span> },
  { key: "gated_by", header: "gated by", kind: "raw", render: (r) => <Ids ids={r.gated_by} /> },
];

export default function Replication({ folder }: { folder: string }) {
  const rep = useRead<StudyReplications>(folder, "study_replications");
  return (
    <Async query={rep} skeleton="table" context="reading the replication">
      {(d) => (
        <>
          <Section title="Target" hint="the paper this study replicates, and the tolerance fixed before any run">
            <KeyValue rows={[
              { label: "paper", value: <>{d.target.title} <code>{d.target.key}</code></> },
              { label: "kind", value: d.target.kind === "reproduction" ? "reproduction (the authors' code)" : "replication (from the text)" },
              ...(d.target.code ? [{ label: "code", value: <code>{d.target.code}</code> }] : []),
              { label: "tolerance", value: <><Num value={d.target.tolerance} kind="ratio" /> <span className="dim">(at most <Num value={d.target.max_tolerance} kind="ratio" /> with a justification)</span></> },
              { label: "rebuild", value: d.rebuilt ? <Badge tone="ok">✓ faithful</Badge> : <Alarm title="a fidelity check failed: every verdict it gates is inconclusive">not faithful</Alarm> },
            ]} />
            {!d.rebuilt && (
              <Callout tone="danger" title="the rebuild is not faithful">
                At least one of the paper&rsquo;s baselines did not come out within tolerance. The verdicts it gates are
                inconclusive: fix the rebuild before reading them as the paper&rsquo;s.
              </Callout>
            )}
          </Section>
          <Section title="Fidelity" hint="did we rebuild it: each baseline the paper reports, against ours">
            <Table columns={FIDELITY_COLUMNS} rows={d.fidelity} dense rowKey={(f) => f.id}
                   empty={<p className="dim small">No fidelity check yet: study_fidelity compares a baseline of the paper with ours.</p>} />
          </Section>
          <Section title="Verdicts" hint="does it hold: each claim of the paper, against ours with its interval, gated by the fidelity checks">
            <Table columns={REPLICATION_COLUMNS} rows={d.replications} dense rowKey={(r) => r.id}
                   empty={<p className="dim small">No replication verdict yet.</p>} />
          </Section>
        </>
      )}
    </Async>
  );
}
