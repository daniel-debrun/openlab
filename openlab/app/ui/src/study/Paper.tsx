/**
 * The paper: its last build (the Paper node: when, signed, the record head it was built at, its bindings and the
 * findings a person waived), and the check of the prose as it is now (`study_paper_lint`): each binding and each
 * finding of the number and wording lints and the citation and equation gates.
 *
 * Only a person can waive a finding: an approval with decision waive whose reason names the finding's id, on the
 * paper or on a claim bound in the finding's sentence (`study_approve`). It shows only at the reviewer link.
 */
import { REVIEWER } from "../api/client";
import type { LintFinding, StudyNodes, StudyPaperLint } from "../api/tools";
import { Async, Badge, Button, Callout, Empty, KeyValue, Table, fmtAgo, fmtDateFull, fmtHash, form, type Column } from "../ui";
import { ActorBadge, Ids, Section, useAct, useRead } from "./parts";

/** Where `study_paper` writes the PDF, under the study folder. */
const PDF = "paper/build/main.pdf";

type Binding = StudyPaperLint["bindings"][number];

const BINDING_COLUMNS: Column<Binding>[] = [
  { key: "macro", header: "binding", kind: "raw", render: (b) => <code className="study-cell-text">{b.macro}</code> },
  { key: "at", header: "at", kind: "raw", render: (b) => <code>{b.at}</code> },
  { key: "node", header: "node", kind: "raw", render: (b) => (b.node ? <code>{b.node}</code> : <span className="dim">—</span>) },
  {
    key: "state", header: "", kind: "raw",
    render: (b) => (b.error ? <Badge tone="warn" title={b.error}>unresolved</Badge>
      : b.stale ? <Badge tone="warn" title={b.stale}>stale</Badge> : <Badge tone="ok">resolved</Badge>),
  },
];

function findingColumns(waive?: (f: LintFinding) => void, why?: (f: LintFinding) => string, busy?: string): Column<LintFinding>[] {
  return [
    { key: "rule", header: "finding", kind: "raw", render: (f) => <span className="study-fields"><code>{f.rule}</code><code className="dim">{f.location}</code></span> },
    { key: "message", header: "message", kind: "raw", render: (f) => <span className="study-cell-text">{f.message}</span> },
    {
      key: "waive", header: "", kind: "raw",
      render: (f) => (f.waived_by ? <span className="small">waived <code>{f.waived_by}</code></span>
        : !f.waivable ? <Badge tone="warn" title="an unresolved or stale binding: fix the prose or run again">not waivable</Badge>
          : waive ? (
            <Button size="sm" disabled={!!why?.(f)} title={why?.(f) || "a person's waiver, naming this finding's id"}
                    loading={busy === `waive:${f.id}`} onClick={() => waive(f)}>
              Waive…
            </Button>
          ) : <span className="dim small">waivable</span>),
    },
    { key: "claims", header: "claims in the sentence", kind: "raw", render: (f) => <Ids ids={f.claims} /> },
  ];
}

export default function Paper({ folder, kinds }: { folder: string; kinds: Record<string, number> }) {
  const papers = useRead<StudyNodes>(folder, "study_nodes", { kind: "paper", n: 1 }, !!kinds.paper);
  const lint = useRead<StudyPaperLint>(folder, "study_paper_lint");
  const { busy, act } = useAct(folder);
  const last = papers.data?.nodes[0];

  /** A waiver goes on the paper once one is built, else on a claim bound in the finding's sentence. */
  const target = (f: LintFinding) => last?.id ?? f.claims[0];
  const why = (f: LintFinding) => (target(f) ? "" : "no paper is built yet and no claim is bound in this sentence: a waiver needs one of them to stand on");
  const waive = async (f: LintFinding) => {
    const v = await form({
      title: "Waive this finding",
      body: (
        <>
          <p><code>{f.rule}</code> at <code>{f.location}</code>: {f.message}</p>
          <p>
            The waiver is recorded on <code>{target(f)}</code> as <code>{f.id}: </code>your reason. The paper then prints
            as written.
          </p>
        </>
      ),
      fields: [{ key: "reason", label: "Reason", placeholder: "why the prose is right as written" }],
      confirmLabel: "Waive",
    });
    if (v) await act("study_approve", { on: target(f), decision: "waive", reason: `${f.id}: ${v.reason.trim()}` }, `waive:${f.id}`);
  };

  return (
    <>
      <Section title="Last build" hint="the Paper node study_paper wrote: what it bound, at which record head, and the findings a person waived">
        {!kinds.paper ? (
          <Empty variant="terminal" title="not built yet"
                 body="study_paper builds paper/build/main.pdf once every finding below is fixed or waived, records the Paper node and signs the bundle." />
        ) : (
          <Async query={papers} skeleton="lines" context="reading the last build">
            {(d) => {
              const n = d.nodes[0];
              const p = n.payload;
              const waived = Array.isArray(p.lint) ? p.lint.length : 0;
              return (
                <KeyValue rows={[
                  { label: "paper", value: <code>{n.id}</code> },
                  { label: "built", value: <><span title={fmtDateFull(n.at)}>{fmtAgo(n.at)}</span> <ActorBadge actor={n.actor} /></> },
                  {
                    label: "signed",
                    value: p.signed ? <><Badge tone="ok">signed</Badge> <code>{folder ? `${folder}/` : ""}{String(p.signed)}</code></>
                      : <Badge tone="warn">not signed</Badge>,
                  },
                  { label: "record head", value: <code title={String(p.head)}>{fmtHash(p.head, 12)}</code> },
                  { label: "bindings", value: n.links.length + (n.links_withheld ?? 0) },
                  { label: "waived findings", value: waived },
                  { label: "PDF", value: <><code>{folder ? `${folder}/` : ""}{PDF}</code> <span className="dim small">open it from your file manager or PDF viewer</span></> },
                ]} />
              );
            }}
          </Async>
        )}
      </Section>

      <Section title="Check" hint="the prose as it is now, checked without building: what study_paper would refuse">
        <Async query={lint} skeleton="table" context="checking the paper">
          {(d) => (
            <>
              {d.ok ? (
                <Callout tone="info" title="the paper would build">Every binding resolves and every finding is fixed or waived.</Callout>
              ) : (
                <Callout tone="warn" title={`${d.findings.length} finding${d.findings.length === 1 ? "" : "s"} stop the build`}>
                  Fix the prose; a finding that is right as written can be waived by a person only.
                </Callout>
              )}
              <Table columns={findingColumns(REVIEWER ? (f) => void waive(f) : undefined, why, busy)} rows={d.findings} dense
                     rowKey={(f) => f.id} empty={<p className="dim small">No open findings.</p>} />
              {d.waived.length > 0 && (
                <>
                  <h4 className="panel-h">Waived</h4>
                  <Table columns={findingColumns()} rows={d.waived} dense rowKey={(f) => f.id} />
                </>
              )}
              <h4 className="panel-h">Bindings</h4>
              <Table columns={BINDING_COLUMNS} rows={d.bindings} dense collapseTo={10} rowKey={(b, i) => `${b.at}:${i}`}
                     empty={<p className="dim small">The prose binds nothing to the record yet.</p>} />
            </>
          )}
        </Async>
      </Section>
    </>
  );
}
