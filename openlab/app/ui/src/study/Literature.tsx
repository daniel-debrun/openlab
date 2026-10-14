/**
 * The review: the PRISMA flow and whether the search met the protocol's coverage floor (`study_prisma`), the
 * sources with their standing screening decision (`study_sources`), the evidence quoted from them
 * (`study_evidence`), and the comparison lattice whose empty cells are the gaps (`study_lattice`).
 *
 * An agent screens the clear cases and marks the rest maybe; a maybe waits for a person, whose decision stands
 * over any agent's. That decision (`study_screen_person`) is a person's act: it shows only at the reviewer link.
 */
import { useState } from "react";
import { REVIEWER } from "../api/client";
import type { Criterion, StudyEvidence, StudyEvidenceRow, StudyLattice, StudyPrisma, StudySource, StudySources } from "../api/tools";
import { Async, Badge, Button, Callout, Field, Num, Table, explainError, form, type Column } from "../ui";
import { ActorBadge, Alarm, Section, useAct, useNode, useRead } from "./parts";

const SOURCES_N = 200;
const EVIDENCE_N = 200;
const STATUSES = ["all", "unscreened", "include", "exclude", "maybe", "retracted"];
const DECISION_TONE = { include: "ok", exclude: "neutral", maybe: "warn" } as const;

function Flow({ p }: { p: StudyPrisma }) {
  const why = (id: string) => [...p.criteria.include, ...p.criteria.exclude].find((c) => c.id === id)?.text;
  const stages = [
    { label: "identified", n: p.identified, detail: Object.entries(p.by_source).map(([k, v]) => `${k} ${v}`).join(" · ") },
    { label: "duplicates removed", n: p.duplicates_removed },
    { label: "records", n: p.records, detail: p.earlier_only ? `${p.earlier_only} more only an earlier search found` : "" },
    { label: "screened", n: p.screened, detail: p.not_screened ? `${p.not_screened} not screened yet` : "" },
    {
      label: "excluded", n: p.excluded,
      detail: Object.entries(p.excluded_by_criterion).map(([k, v]) => <span key={k} title={why(k)}><code>{k}</code> {v} </span>),
    },
    { label: "waiting for a person", n: p.awaiting_person },
    { label: "included", n: p.included },
  ];
  return (
    <ol className="study-flow">
      {stages.map((s) => (
        <li key={s.label} className="study-flow-stage">
          <span className="study-flow-label">{s.label}</span>
          <span className="study-flow-n"><Num value={s.n} kind="count" /></span>
          <span className="study-flow-detail">{s.detail}</span>
        </li>
      ))}
    </ol>
  );
}

function Coverage({ folder, id }: { folder: string; id: string | null }) {
  const node = useNode(folder, id);
  if (!id) return <Badge tone="neutral">no discovery on record</Badge>;
  const p = node.data?.payload;
  if (!p) return null;
  const floor = Object.entries((p.floor ?? {}) as Record<string, unknown>).map(([k, v]) => `${k} ${v}`).join(", ");
  return p.verdict === "sufficient" ? (
    <Badge tone="ok" title={`met the review's floor (${floor})`}>coverage sufficient</Badge>
  ) : (
    <Badge tone="warn" title={`below the review's floor (${floor}): absence of evidence here is not evidence`}>coverage insufficient</Badge>
  );
}

function sourceColumns(onDecide: ((s: StudySource) => void) | undefined, busy: string): Column<StudySource>[] {
  return [
    {
      key: "title", header: "source", kind: "raw",
      render: (s) => (
        <span className="study-cell-text">
          {s.title}
          <span className="study-fields small">
            <code className="dim">{s.key}</code>
            <span>{s.year ?? "—"}</span>
            <span className="dim" title="trust tier">{s.tier.replace(/_/g, " ")}</span>
            <span className="dim">{s.version}</span>
          </span>
        </span>
      ),
    },
    {
      key: "retracted", header: "", kind: "raw",
      render: (s) => (s.retracted ? <Alarm title="retracted: no claim may rest on it unless a person waives it">retracted</Alarm> : null),
    },
    {
      key: "decision", header: "decision", kind: "raw",
      render: (s) => (s.decision ? (
        <span className="study-decision">
          <Badge tone={DECISION_TONE[s.decision]}>{s.decision}</Badge> <code>{s.criterion}</code> {s.by && <ActorBadge actor={s.by} />}
        </span>
      ) : <span className="dim">unscreened</span>),
    },
    {
      key: "rationale", header: "rationale and quote", kind: "raw",
      render: (s) => (s.decision ? (
        <span className="study-cell-text">{s.rationale}{s.quote && <q className="study-quote">{s.quote}</q>}</span>
      ) : null),
    },
    ...(onDecide
      ? [{
          key: "act", header: "", kind: "raw" as const,
          render: (s: StudySource) => (s.decision === "maybe" ? (
            <Button size="sm" loading={busy === `screen:${s.id}`} onClick={() => onDecide(s)}
                    title="a person's decision: include or exclude, under a criterion of the sealed review">Decide…</Button>
          ) : null),
        }]
      : []),
  ];
}

const EVIDENCE_COLUMNS: Column<StudyEvidenceRow>[] = [
  {
    key: "key", header: "source", kind: "raw",
    render: (e) => (
      <span className="study-fields">
        <code title={e.source}>{e.key}</code>
        <span className="small">p.{e.page} · {e.section} · {e.kind}</span>
      </span>
    ),
  },
  { key: "quote", header: "quote", kind: "raw", render: (e) => <q className="study-cell-text study-quote">{e.quote}</q> },
  {
    key: "fields", header: "fields", kind: "raw",
    render: (e) => (Object.keys(e.fields).length ? (
      <span className="study-cell-text study-fields">{Object.entries(e.fields).map(([k, v]) => <span key={k}><span className="dim">{k}</span> {String(v)}</span>)}</span>
    ) : <span className="dim">—</span>),
  },
  { key: "valid_from", header: "published", kind: "raw", render: (e) => e.valid_from || <span className="dim">—</span> },
];

type LatticeRow = StudyLattice["rows"][number];

function Lattice({ d }: { d: StudyLattice }) {
  const cols: Column<LatticeRow>[] = [
    {
      key: "protocol", header: "protocol", kind: "raw",
      render: (r) => (
        <span className="study-cell-text study-fields">
          {d.coordinates.map((c) => <span key={c}><span className="dim">{c}</span> {r.coordinates[c]}</span>)}
        </span>
      ),
    },
    ...d.columns.map((c): Column<LatticeRow> => ({
      key: c, header: c, kind: "raw",
      render: (r) => (r.cells[c]?.length ? (
        <span className="study-fields">
          {r.cells[c].map((e) => (
            <span key={e.evidence} title={`${e.evidence}, p.${e.page} of ${e.source}`}>
              {d.values.map((v) => <span key={v}>{e[v] == null ? "—" : String(e[v])} </span>)}
            </span>
          ))}
        </span>
      ) : (
        <span className="study-gap" title="no evidence fills this cell: not found in the extracted sources, which is not the same as novel">gap</span>
      )),
    })),
  ];
  return (
    <>
      <p className="dim small">
        {d.filled} of {d.cells} cells filled, {d.gaps} gaps; each cell is the {d.values.join(", ")} read for that {d.by}{" "}
        under that protocol.
      </p>
      <Table columns={cols} rows={d.rows} dense stickyFirst rowKey={(r) => JSON.stringify(r.coordinates)}
             note={d.rows_withheld ? `${d.rows_withheld} more protocols withheld` : undefined} />
    </>
  );
}

export default function Literature({ folder }: { folder: string }) {
  const [status, setStatus] = useState("all");
  const prisma = useRead<StudyPrisma>(folder, "study_prisma");
  const sources = useRead<StudySources>(folder, "study_sources", { status, n: SOURCES_N });
  const evidence = useRead<StudyEvidence>(folder, "study_evidence", { n: EVIDENCE_N });
  const lattice = useRead<StudyLattice>(folder, "study_lattice");
  const { busy, act } = useAct(folder);

  const decide = async (s: StudySource, criteria: { include: Criterion[]; exclude: Criterion[] }) => {
    const opt = (kind: string) => (c: Criterion) => ({ value: c.id, label: `${c.id} (${kind}): ${c.text}` });
    const v = await form({
      title: "Decide a source the agent left to you",
      body: (
        <>
          <p><strong>{s.title}</strong> <code>{s.key}</code></p>
          <p>The agent marked it maybe under <code>{s.criterion}</code>: {s.rationale}</p>
          <p>Your decision stands over any agent&rsquo;s and stays on the record.</p>
        </>
      ),
      fields: [
        { key: "decision", label: "Decision", options: [{ value: "include", label: "include" }, { value: "exclude", label: "exclude" }] },
        { key: "criterion", label: "Criterion (of the decision's kind)", options: [...criteria.include.map(opt("include")), ...criteria.exclude.map(opt("exclude"))] },
        { key: "rationale", label: "Reason", placeholder: "why, in a sentence" },
        { key: "quote", label: "Quote from the source (optional)", placeholder: "at least 20 characters, as the source has it", optional: true },
      ],
      confirmLabel: "Record decision",
    });
    if (v) await act("study_screen_person", { source: s.id, ...v }, `screen:${s.id}`);
  };

  return (
    <>
      <Section title="PRISMA flow" hint="the latest discovery, derived from the record: what was found, removed, screened and kept">
        <Async query={prisma} skeleton="lines" context="reading the PRISMA flow">
          {(p) => (
            <>
              <div className="ds-facts"><Coverage folder={folder} id={p.coverage} /></div>
              {p.retracted > 0 && (
                <Callout tone="danger" title={`${p.retracted} source${p.retracted === 1 ? " is" : "s are"} retracted${p.retracted_included.length ? `, ${p.retracted_included.length} of them included` : ""}`}>
                  A retracted source cannot ground a claim unless a person waives it. Filter the sources by retracted to see which.
                </Callout>
              )}
              {p.awaiting_person > 0 && !REVIEWER && (
                <Callout tone="info" title={`${p.awaiting_person} source${p.awaiting_person === 1 ? " waits" : "s wait"} for a person`}>
                  The agent marked {p.awaiting_person === 1 ? "it" : "them"} maybe. Open the reviewer link <code>openlab app</code> printed
                  to decide, or run <code>openlab study screen --person</code> in your own terminal.
                </Callout>
              )}
              <Flow p={p} />
            </>
          )}
        </Async>
      </Section>

      <Section title="Sources" hint="each source's standing decision, oldest first; a person's decision is marked and stands over an agent's">
        <Field label="show">
          <select value={status} onChange={(e) => setStatus(e.target.value)} className="study-filter">
            {STATUSES.map((s) => <option key={s} value={s}>{s === "maybe" ? "maybe (waiting for a person)" : s}</option>)}
          </select>
        </Field>
        <Async query={sources} skeleton="table" context="reading the sources">
          {(d) => (
            <Table
              columns={sourceColumns(REVIEWER && prisma.data ? (s) => void decide(s, prisma.data!.criteria) : undefined, busy)}
              rows={d.sources}
              total={d.total}
              dense
              collapseTo={50}
              rowKey={(s) => s.id}
              empty={<p className="dim small">No source is {status}.</p>}
            />
          )}
        </Async>
      </Section>

      <Section title="Evidence" hint="what was read from the sources, newest first: each quote is checked on its page, each field against the quote">
        <Async query={evidence} skeleton="table" context="reading the evidence">
          {(d) => (
            <Table columns={EVIDENCE_COLUMNS} rows={d.evidence} total={d.total} dense collapseTo={30} rowKey={(e) => e.id}
                   empty={<p className="dim small">No evidence extracted yet.</p>} />
          )}
        </Async>
      </Section>

      <Section title="Comparison lattice" hint="results keyed by protocol: only cells on the same protocol compare; an empty cell is a gap">
        {lattice.isError ? (
          // a review that declares no coordinates has no lattice: the server's reason, not an error
          <p className="dim small">{explainError(lattice.error).error}: {explainError(lattice.error).try}</p>
        ) : (
          <Async query={lattice} skeleton="table" context="reading the comparison lattice">{(d) => <Lattice d={d} />}</Async>
        )}
      </Section>
    </>
  );
}
