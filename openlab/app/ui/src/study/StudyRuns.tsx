/**
 * A study's runs: one row per run of model.py, newest first, with the truncation audit's verdict, the primary
 * metric with its interval, each baseline's, and the paired difference against it. A row opens the run and its
 * result in full. Cached runs (an identical run answered from the record, not counted as a trial) are folded
 * away until asked for.
 *
 * Then the forking paths: every research run's primary metric with its interval, ordered by value, the newest
 * highlighted. It is the specification curve, and it shows how far the answer moved with the choices tried.
 */
import { useMemo, useState } from "react";
import type { Stats, StudyRun, StudyRuns } from "../api/tools";
import { Async, Badge, Button, Callout, Empty, KeyValue, Sheet, Table, fmtAgo, fmtDateFull, fmtHash, type Column } from "../ui";
import { Generic } from "./Generic";
import { AuditBadge, Interval, LOOK_AHEAD, useNode } from "./parts";

/** A paired difference's `better`, as the runner records it ("model" or "baseline"). */
const BETTER_TONE: Record<string, "pos" | "neg"> = { model: "pos", baseline: "neg" };
const BETTER_WORD: Record<string, string> = { model: "model", baseline: "baseline" };

function columns(d: StudyRuns): Column<StudyRun>[] {
  return [
    { key: "at", header: "when", kind: "raw", render: (r) => <span title={fmtDateFull(r.at)}>{fmtAgo(r.at)}</span> },
    {
      key: "segment", header: "segment", kind: "raw",
      render: (r) => (r.segment === "holdout" ? <Badge tone="accent">holdout</Badge> : <span className="dim">research</span>),
    },
    { key: "audit", header: "audit", kind: "raw", render: (r) => <AuditBadge audit={r.audit} /> },
    {
      key: "metric", header: d.primary, kind: "ratio", title: `the primary metric, ${d.primary}, with its interval`,
      render: (r) => <Interval s={r.result?.metric} struck={r.audit === LOOK_AHEAD} />,
    },
    ...d.baselines.flatMap((b): Column<StudyRun>[] => [
      { key: `b:${b}`, header: b, kind: "ratio", title: `${b}'s ${d.primary} (its interval is in the run's detail)`, value: (r) => r.result?.baselines[b]?.value },
      {
        key: `p:${b}`, header: `vs ${b}`, kind: "ratio",
        title: `the paired difference, model minus ${b}, on the same rows, with its interval`,
        render: (r) => <Interval s={r.result?.paired[b]} paired />,
        tone: (r) => BETTER_TONE[r.result?.paired[b]?.better ?? ""],
      },
    ]),
    { key: "seconds", header: "took", kind: "duration" },
    { key: "note", header: "note", kind: "raw", render: (r) => r.note || <span className="dim">—</span> },
  ];
}

/** Every `{value…}` or `{diff…}` leaf of a result section, with the keys that lead to it. */
function statRows(v: unknown, path: string[] = []): { of: string; s: Stats }[] {
  if (!v || typeof v !== "object" || Array.isArray(v)) return [];
  const o = v as Record<string, unknown>;
  if (typeof o.value === "number" || typeof o.diff === "number") return [{ of: path.join(" · "), s: o as Stats }];
  return Object.entries(o).flatMap(([k, x]) => statRows(x, [...path, k]));
}

function StatTable({ title, section, paired }: { title: string; section: unknown; paired?: boolean }) {
  const rows = statRows(section);
  if (!rows.length) return null;
  const cols: Column<{ of: string; s: Stats }>[] = [
    { key: "of", header: paired ? "against · metric" : "", kind: "raw" },
    { key: "v", header: paired ? "difference" : "value", kind: "ratio", value: (r) => (paired ? r.s.diff : r.s.value) },
    { key: "lo", header: "lo", kind: "ratio", value: (r) => r.s.lo },
    { key: "hi", header: "hi", kind: "ratio", value: (r) => r.s.hi },
    ...(paired
      ? ([
          { key: "p", header: "p", kind: "pvalue", value: (r) => r.s.p },
          { key: "better", header: "better", kind: "raw", render: (r) => BETTER_WORD[r.s.better ?? ""] ?? "tie" },
        ] as Column<{ of: string; s: Stats }>[])
      : []),
  ];
  return (
    <section className="study-section">
      <h4 className="panel-h">{title}</h4>
      <Table columns={cols} rows={rows} dense stickyFirst rowKey={(r) => r.of} />
    </section>
  );
}

function RunDetail({ folder, run }: { folder: string; run: StudyRun }) {
  const node = useNode(folder, run.id);
  const result = useNode(folder, run.result?.id);
  return (
    <div className="study-detail">
      {run.audit === LOOK_AHEAD && (
        <Callout tone="danger" title="look-ahead: this result cannot back a claim">
          Replayed with only the data known at each sampled step, model.py predicted differently. Its features read
          data from after the decision time.
        </Callout>
      )}
      <Async query={node} skeleton="lines" context={`reading ${run.id}`}>
        {(n) => {
          const p = n.payload;
          const audit = p.audit as Record<string, unknown>;
          const choices = p.choices as Record<string, unknown> | undefined;
          return (
            <KeyValue rows={[
              { label: "run", value: <code>{n.id}</code> },
              { label: "segment", value: String(p.segment) },
              { label: "when", value: `${fmtDateFull(n.at)}, by ${n.actor}` },
              { label: "audit", value: <><AuditBadge audit={run.audit} />{Object.keys(audit).length > 0 && <Generic value={audit} />}</> },
              { label: "sandbox", value: String(p.sandbox) },
              { label: "cached", value: p.cached ? "yes: an identical run, answered from the record" : "no" },
              { label: "key", value: <code title={String(p.key)}>{fmtHash(p.key, 12)}</code> },
              { label: "spec", value: <code title={String(p.spec_sha)}>{fmtHash(p.spec_sha, 12)}</code> },
              { label: "branches from", value: run.branches_from ? <code>{run.branches_from}</code> : <span className="dim">—</span> },
              { label: "note", value: run.note || <span className="dim">—</span> },
              ...(choices && Object.keys(choices).length ? [{ label: "choices", value: <Generic value={choices} /> }] : []),
            ]} />
          );
        }}
      </Async>
      {run.result ? (
        <Async query={result} skeleton="table" context={`reading ${run.result.id}`}>
          {(n) => {
            const policy = (n.payload.policy ?? {}) as Record<string, unknown>;
            return (
              <>
                <StatTable title="Metrics" section={n.payload.metrics} />
                <StatTable title="Baselines" section={n.payload.baselines} />
                <StatTable title="Paired differences" section={n.payload.paired} paired />
                {Object.keys(policy).length > 0 && (
                  <section className="study-section">
                    <h4 className="panel-h">Policy</h4>
                    <Generic value={policy} />
                  </section>
                )}
              </>
            );
          }}
        </Async>
      ) : (
        <p className="dim small">This run has no result on record.</p>
      )}
    </div>
  );
}

export function RunsTable({ folder, d }: { folder: string; d: StudyRuns }) {
  const [showCached, setShowCached] = useState(false);
  const [open, setOpen] = useState<StudyRun | null>(null);
  if (!d.runs.length) {
    return (
      <Empty variant="terminal" title="no runs yet"
             body="Run executes model.py in the sandbox over the research segment. The first run seals study.toml; each run after it counts as a trial." />
    );
  }
  const cached = d.runs.filter((r) => r.cached).length;
  const bad = d.runs.filter((r) => r.audit === LOOK_AHEAD).length;
  return (
    <>
      {bad > 0 && (
        <Callout tone="danger" title={`${bad} run${bad === 1 ? "" : "s"} failed the truncation audit: look-ahead`}>
          Replayed with only the data known at each sampled step, model.py predicted differently, so its features read
          the future. Those results cannot back a claim: fix the features and run again.
        </Callout>
      )}
      <Table
        columns={columns(d)}
        rows={showCached ? d.runs : d.runs.filter((r) => !r.cached)}
        dense
        stickyFirst
        rowKey={(r) => r.id}
        onRowClick={setOpen}
        empty={<p className="dim small">Every run on record is cached.</p>}
        note={
          <>
            {d.total > d.runs.length && <span>newest {d.runs.length} of {d.total} runs · </span>}
            {cached > 0 && (
              <Button size="sm" variant="ghost" onClick={() => setShowCached((v) => !v)}
                      title="a cached run is an identical run answered from the record; it is not a new trial">
                {showCached ? "hide" : "show"} {cached} cached
              </Button>
            )}
          </>
        }
      />
      <Sheet open={!!open} onClose={() => setOpen(null)} width={600}
             title={open ? `${open.segment} run · ${fmtDateFull(open.at)}` : ""}>
        {open && <RunDetail folder={folder} run={open} />}
      </Sheet>
    </>
  );
}

/** The curve's drawing box, in SVG units: it scales to the column's width. */
const W = 640;
const H = 180;
const PAD = { l: 56, r: 12, t: 10, b: 24 };

/** Round ticks over [lo, hi], four or five of them. */
function ticks(lo: number, hi: number): number[] {
  const span = hi - lo || Math.abs(hi) || 1;
  const step = 10 ** Math.floor(Math.log10(span / 3));
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => span / s <= 5) ?? step * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / nice) * nice; v <= hi + nice * 1e-9; v += nice) out.push(+v.toPrecision(12));
  return out;
}

const short = (v: number | null | undefined) => (v == null ? "—" : String(+v.toPrecision(4)));

/** The specification curve over the research runs on record (cached and look-ahead runs left out): points with
 *  their intervals, drawn as plain SVG in the design tokens (app.css .spec-*), since it is the app's only chart. */
export function SpecCurve({ d }: { d: StudyRuns }) {
  const research = d.runs.filter((r) => r.segment === "research" && !r.cached && r.result?.metric?.value != null);
  const lookAhead = research.filter((r) => r.audit === LOOK_AHEAD).length;
  const values = useMemo(() => {
    const newest = research[0]?.id;
    return research
      .filter((r) => r.audit !== LOOK_AHEAD)
      .map((r) => ({ ...r.result!.metric!, value: r.result!.metric!.value!, newest: r.id === newest, id: r.id, note: r.note }))
      .sort((a, b) => a.value - b.value);
  }, [d]); // eslint-disable-line react-hooks/exhaustive-deps
  if (values.length < 2) {
    return <p className="dim small">The curve appears once two research runs have a result{lookAhead ? " that passed the audit" : ""}.</p>;
  }
  const better = d.higher_is_better === true ? "higher is better" : d.higher_is_better === false ? "lower is better" : "";
  const lo0 = Math.min(...values.map((v) => v.lo ?? v.value));
  const hi0 = Math.max(...values.map((v) => v.hi ?? v.value));
  const pad = (hi0 - lo0 || Math.abs(hi0) || 1) * 0.08;
  const [lo, hi] = [lo0 - pad, hi0 + pad];
  const y = (v: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - (v - lo) / (hi - lo));
  const x = (i: number) => PAD.l + (W - PAD.l - PAD.r) * ((i + 0.5) / values.length);
  return (
    <>
      <svg className="spec-curve" viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`specification curve of ${d.primary} over ${values.length} research runs`}>
        {ticks(lo, hi).map((t) => (
          <g key={t}>
            <line className="spec-grid" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />
            <text className="spec-tick" x={PAD.l - 6} y={y(t)} dy="0.32em" textAnchor="end">{short(t)}</text>
          </g>
        ))}
        <text className="spec-axis" x={PAD.l} y={H - 6}>research runs, ordered by {d.primary}</text>
        {values.map((v, i) => (
          <g key={v.id} className={v.newest ? "is-newest" : undefined}>
            <title>{`${v.id}${v.note ? ` · ${v.note}` : ""}: ${d.primary} ${short(v.value)} [${short(v.lo)}, ${short(v.hi)}]${v.newest ? " (newest)" : ""}`}</title>
            {v.lo != null && v.hi != null && <line className="spec-ci" x1={x(i)} x2={x(i)} y1={y(v.lo)} y2={y(v.hi)} />}
            <circle className="spec-pt" cx={x(i)} cy={y(v.value)} r={4} />
          </g>
        ))}
      </svg>
      <p className="dim small">
        Each point is one research run&rsquo;s {d.primary}{better && ` (${better})`} with its interval, ordered by value; the
        newest run is highlighted. A wide spread means the answer depends on the choices tried.
        {lookAhead > 0 && ` ${lookAhead} look-ahead run${lookAhead === 1 ? " is" : "s are"} left out.`}
      </p>
    </>
  );
}
