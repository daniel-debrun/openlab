/**
 * Forward use and monitoring: the forecasts a registered model made after the study's segments (`study_forecasts`:
 * the gate's checks, the signed file, the outcome once its label is known), the monitoring passes (`watch` nodes)
 * and the retractions they found, each with what rests on the retracted source (`study_impact`).
 */
import type { StudyForecast, StudyForecasts, StudyImpact, StudyNode, StudyNodes } from "../api/tools";
import { Async, Badge, Callout, Num, Table, fmtAgo, fmtDateFull, type Column } from "../ui";
import { Alarm, Section, useRead } from "./parts";

const FORECASTS_N = 100;
const NODES_N = 100;

/** The size of a list or dict the server cut to size (it ends in a "... n more" entry). */
function size(v: unknown): number {
  if (Array.isArray(v)) {
    const m = typeof v[v.length - 1] === "string" ? /^\.\.\. (\d+) more$/.exec(v[v.length - 1]) : null;
    return m ? v.length - 1 + Number(m[1]) : v.length;
  }
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    const m = /^(\d+) more keys$/.exec(String(o["..."] ?? ""));
    return m ? Object.keys(o).length - 1 + Number(m[1]) : Object.keys(o).length;
  }
  return 0;
}

const when = (at: string) => <span title={fmtDateFull(at)}>{fmtAgo(at)}</span>;

const FORECAST_COLUMNS: Column<StudyForecast>[] = [
  { key: "t", header: "decision", kind: "raw", render: (f) => <code>{f.t}</code> },
  { key: "model", header: "model", kind: "raw" },
  { key: "rows", header: "rows", kind: "count" },
  {
    key: "gate", header: "gate", kind: "raw",
    render: (f) => (f.gate ? <Badge tone="ok">✓ passed</Badge> : (
      <Alarm title={`failed: ${f.failed.map((c) => c.expect + (c.actual ? ` (was ${c.actual})` : c.error ? ` (${c.error})` : "")).join("; ")}`}>
        gate failed
      </Alarm>
    )),
  },
  {
    key: "prospective", header: "", kind: "raw",
    render: (f) => (f.prospective ? <Badge tone="info" title="made before its label could be known">prospective</Badge>
      : <Badge tone="neutral" title="written after its label time">after the fact</Badge>),
  },
  { key: "signed", header: "file", kind: "raw", render: (f) => (f.signed ? <code title={f.signed}>{f.file}</code> : <Badge tone="warn">unsigned</Badge>) },
  { key: "expires", header: "expires", kind: "raw", render: (f) => <code>{f.expires}</code> },
  {
    key: "outcome", header: "outcome", kind: "raw",
    render: (f) => (f.outcome ? (
      <span className="study-fields">
        <span><span className="dim">n</span> {f.outcome.n}</span>
        {Object.entries(f.outcome.metrics).map(([k, v]) => <span key={k}><span className="dim">{k}</span> <Num value={v as number} kind="ratio" /></span>)}
      </span>
    ) : <span className="dim">awaiting: label at {f.label_at}</span>),
  },
];

const WATCH_COLUMNS: Column<StudyNode>[] = [
  { key: "at", header: "pass", kind: "raw", render: (n) => when(n.at) },
  { key: "new", header: "new sources", kind: "count", value: (n) => size(n.payload.new_sources) },
  {
    key: "found", header: "retractions found", kind: "raw",
    render: (n) => {
      const ids = n.links.filter((l) => l.rel === "found").map((l) => l.to);
      return ids.length ? <Alarm title={ids.join(", ")}>{ids.length} retracted</Alarm> : <span className="dim">none</span>;
    },
  },
  { key: "contra", header: "contradicting evidence", kind: "count", value: (n) => size(n.payload.contradicting) },
  { key: "outcomes", header: "outcomes filled", kind: "count", value: (n) => size(n.payload.outcomes) },
  { key: "checked", header: "sources checked", kind: "count", value: (n) => size(n.payload.checked) },
];

function Impact({ folder, source }: { folder: string; source: string }) {
  const q = useRead<StudyImpact>(folder, "study_impact", { source });
  if (!q.data) return <span className="dim small">{q.isError ? "could not read" : "reading…"}</span>;
  const d = q.data;
  const part = (ids: string[], word: string) => <span title={ids.join(", ")}>{ids.length} {word}</span>;
  return (
    <span className="study-fields">
      {part(d.claims, "claims")}
      {part(d.conclusions, "conclusions")}
      {part(d.papers, "papers")}
      {part(d.evidence, "evidence")}
      {part(d.comparisons, "comparisons")}
    </span>
  );
}

function retractionColumns(folder: string): Column<StudyNode>[] {
  const source = (n: StudyNode) => n.links.find((l) => l.rel === "of")?.to ?? "";
  return [
    { key: "flag", header: "", kind: "raw", render: () => <Alarm title="no claim may rest on this source unless a person waives it">retracted</Alarm> },
    {
      key: "source", header: "source and notice", kind: "raw",
      render: (n) => (
        <span className="study-fields">
          <code>{source(n)}</code>
          <span className="dim small">{String(n.payload.via)} {String(n.payload.type)} <code>{String(n.payload.notice)}</code></span>
        </span>
      ),
    },
    { key: "impact", header: "what rests on it", kind: "raw", render: (n) => <Impact folder={folder} source={source(n)} /> },
    { key: "valid_from", header: "dated", kind: "raw", render: (n) => String(n.payload.valid_from || "—") },
    { key: "at", header: "detected", kind: "raw", render: (n) => when(n.at) },
  ];
}

export default function Forecasts({ folder, kinds }: { folder: string; kinds: Record<string, number> }) {
  const forecasts = useRead<StudyForecasts>(folder, "study_forecasts", { n: FORECASTS_N }, !!kinds.forecast);
  const watches = useRead<StudyNodes>(folder, "study_nodes", { kind: "watch", n: NODES_N }, !!kinds.watch);
  const retractions = useRead<StudyNodes>(folder, "study_nodes", { kind: "retraction", n: NODES_N }, !!kinds.retraction);
  return (
    <>
      {kinds.forecast ? (
        <Section title="Forecasts" hint="newest first: each made after the study's segments by a pinned model, checked by the gate, signed for a program downstream">
          <Async query={forecasts} skeleton="table" context="reading the forecasts">
            {(d) => {
              const failed = d.forecasts.filter((f) => !f.gate).length;
              return (
                <>
                  {failed > 0 && (
                    <Callout tone="danger" title={`${failed} forecast${failed === 1 ? "" : "s"} failed the gate`}>
                      The file is signed as failed: nothing downstream should act on it. The failed checks are in the row&rsquo;s title.
                    </Callout>
                  )}
                  <Table columns={FORECAST_COLUMNS} rows={d.forecasts} total={d.total} dense rowKey={(f) => f.id} />
                </>
              );
            }}
          </Async>
        </Section>
      ) : null}

      {kinds.retraction ? (
        <Section title="Retractions" hint="found by monitoring after the source was recorded: the notice's date, when openlab learned of it, and what rests on the source">
          <Async query={retractions} skeleton="table" context="reading the retractions">
            {(d) => (
              <>
                <Callout tone="danger" title={`${d.total} retraction${d.total === 1 ? "" : "s"} on record`}>
                  Evidence from a retracted source cannot ground a new claim, and a paper citing it is stale until built again.
                </Callout>
                <Table columns={retractionColumns(folder)} rows={d.nodes} total={d.total} dense rowKey={(n) => n.id} />
              </>
            )}
          </Async>
        </Section>
      ) : null}

      {kinds.watch ? (
        <Section title="Monitoring" hint="each watch pass, newest first: the review's search run again, retraction checks, outcomes filled">
          <Async query={watches} skeleton="table" context="reading the watch passes">
            {(d) => <Table columns={WATCH_COLUMNS} rows={d.nodes} total={d.total} dense rowKey={(n) => n.id} />}
          </Async>
        </Section>
      ) : null}
    </>
  );
}
