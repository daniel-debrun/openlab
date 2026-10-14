/**
 * How the research is organised and what it claims: the question tree (`study_tree`, each status computed by the
 * server), the claims with their verdict, level, risk and contradiction search (`study_claims`), the error budget
 * spent by each test, the assumption register with its breaks, and the dead ends (`study_nodes`).
 *
 * A claim opens its provenance chain (`study_trace`): what it rests on down to the source or the run, and anything
 * missing. A person approves, rejects or waives a claim there (`study_approve`), only at the reviewer link.
 */
import { useState } from "react";
import { REVIEWER } from "../api/client";
import type { StudyClaim, StudyClaims, StudyFrontier, StudyNode, StudyNodes, StudyTrace, StudyTraceability, StudyTree, TraceNode, TreeNode } from "../api/tools";
import { Async, Badge, Button, Callout, Num, Sheet, Table, fmtAgo, fmtDateFull, form, type Column } from "../ui";
import { FRONTIER_N } from "./Frontier";
import { Alarm, AuditBadge, Ids, Section, Standing, useAct, useRead } from "./parts";

const CLAIMS_N = 200;
const NODES_N = 200;
const CONTRA = {
  found: { tone: "warn", word: "contradicting evidence found", why: "a search found evidence against it; read it before relying on the claim" },
  none_found: { tone: "ok", word: "none found", why: "a search the review's floor judges sufficient found nothing against it" },
  insufficient: { tone: "neutral", word: "search insufficient", why: "the search fell below the review's coverage floor" },
} as const;

function Tree({ nodes }: { nodes: TreeNode[] }) {
  return (
    <ul className="study-tree">
      {nodes.map((n) => (
        <li key={n.id}>
          <div className="study-tree-row">
            <Standing word={n.status} />
            <span className="dim small">{n.kind}</span>
            <span>{n.text}</span>
            {n.dead_ends?.length ? <Badge tone="neutral" title={n.dead_ends.join(", ")}>{n.dead_ends.length} dead end{n.dead_ends.length === 1 ? "" : "s"}</Badge> : null}
            {n.claims.length > 0 && <span className="dim small">{n.claims.length} claim{n.claims.length === 1 ? "" : "s"}</span>}
          </div>
          {n.children?.length ? <Tree nodes={n.children} /> : null}
        </li>
      ))}
    </ul>
  );
}

function claimColumns(): Column<StudyClaim>[] {
  return [
    { key: "statement", header: "claim", kind: "raw", render: (c) => <span className="study-cell-text">{c.statement}</span> },
    { key: "state", header: "state", kind: "raw" },
    {
      key: "verdict", header: "verdict", kind: "raw",
      render: (c) => <Standing word={c.verdict} why={c.verdict_written ? `written ${c.verdict_written}; tests since make it ${c.verdict}` : undefined} />,
    },
    {
      key: "level", header: "level", kind: "raw",
      render: (c) => (c.level ? <Badge tone={c.level.endsWith("+reproduced") ? "ok" : "neutral"} title="recorded from research, measured from the holdout; +reproduced once every result it rests on reproduced">{c.level}</Badge> : <span className="dim">—</span>),
    },
    {
      key: "at_risk", header: "", kind: "raw",
      render: (c) => (c.at_risk.length ? <Alarm title={`it needs a broken assumption: ${c.at_risk.join(", ")}`}>at risk</Alarm> : null),
    },
    {
      key: "contra", header: "contradictions", kind: "raw",
      render: (c) => {
        const k = c.contra ? CONTRA[c.contra as keyof typeof CONTRA] : undefined;
        return k ? <Badge tone={k.tone} title={k.why}>{k.word}</Badge> : <span className="dim">not searched</span>;
      },
    },
    { key: "approval", header: "person", kind: "raw", render: (c) => (c.approval ? <Badge tone="accent">{c.approval}</Badge> : <span className="dim">—</span>) },
  ];
}

const FACTS = ["key", "code", "env", "sha256", "recipe_sha"];

function Chain({ n, rel }: { n: TraceNode; rel?: string }) {
  if (n.shown) return <li className="study-chain-item"><span className="dim small">{rel}</span> <code>{n.id}</code> <span className="dim small">(shown above)</span></li>;
  const facts = FACTS.filter((f) => n[f] != null && n[f] !== "");
  return (
    <li className="study-chain-item">
      <div className="study-tree-row">
        {rel && <span className="dim small">{rel}</span>}
        <Badge tone="neutral">{n.kind}</Badge>
        <code>{n.id}</code>
        {typeof n.audit === "string" && <AuditBadge audit={n.audit} />}
        {n.retracted ? <Alarm title={String(n.retracted)}>retracted</Alarm> : null}
      </div>
      <div className="small">{n.summary}</div>
      {facts.length > 0 && (
        <div className="study-fields small">
          {facts.map((f) => <span key={f}><span className="dim">{f}</span> {typeof n[f] === "object" ? Object.entries(n[f] as object).map(([k, v]) => `${k} ${v}`).join(", ") : String(n[f])}</span>)}
        </div>
      )}
      {n.links?.length ? <ul className="study-chain">{n.links.map((l, i) => <Chain key={i} n={l.node} rel={l.rel} />)}</ul> : null}
    </li>
  );
}

function Trace({ folder, claim }: { folder: string; claim: StudyClaim }) {
  const trace = useRead<StudyTrace>(folder, "study_trace", { claim: claim.id });
  const { busy, act } = useAct(folder);
  const decide = async () => {
    const v = await form({
      title: "Decide on this claim",
      body: <p>&ldquo;{claim.statement}&rdquo; Your decision stays on the record, with your reason.</p>,
      fields: [
        { key: "decision", label: "Decision", options: ["approve", "reject", "waive"].map((d) => ({ value: d, label: d })) },
        { key: "reason", label: "Reason", placeholder: "why, in a sentence" },
      ],
      confirmLabel: "Record decision",
    });
    if (v) await act("study_approve", { on: claim.id, ...v });
  };
  return (
    <div className="study-detail">
      {REVIEWER && (
        <div><Button size="sm" variant="primary" loading={busy === "study_approve"} onClick={() => void decide()}>Approve, reject or waive…</Button></div>
      )}
      {claim.at_risk.length > 0 && (
        <Callout tone="danger" title="at risk">
          It needs an assumption shown not to hold: <Ids ids={claim.at_risk} />
        </Callout>
      )}
      <Async query={trace} skeleton="lines" context={`tracing ${claim.id}`}>
        {(t) => (
          <>
            {t.complete ? (
              <Callout tone="info" title="the chain is complete">Every link down to the sources and runs is on record and still holds.</Callout>
            ) : (
              <Callout tone="danger" title="the chain is broken">
                <ul className="small">{t.missing.map((m) => <li key={m}>{m}</li>)}</ul>
              </Callout>
            )}
            <ul className="study-chain"><Chain n={t.chain} /></ul>
          </>
        )}
      </Async>
    </div>
  );
}

const TEST_COLUMNS: Column<StudyNode>[] = [
  { key: "at", header: "when", kind: "raw", render: (n) => <span title={fmtDateFull(n.at)}>{fmtAgo(n.at)}</span> },
  { key: "of", header: "of", kind: "raw", render: (n) => <Ids ids={n.links.filter((l) => l.rel === "of").map((l) => l.to)} /> },
  { key: "family", header: "family", kind: "raw", value: (n) => n.payload.family },
  { key: "baseline", header: "against", kind: "raw", render: (n) => <code>{String(n.payload.baseline)}</code> },
  { key: "alpha", header: "alpha", kind: "ratio", value: (n) => n.payload.alpha_j },
  { key: "p", header: "p", kind: "pvalue", value: (n) => n.payload.p },
  { key: "rejected", header: "", kind: "raw", render: (n) => (n.payload.rejected ? <Badge tone="info">rejected</Badge> : <span className="dim">not rejected</span>) },
  {
    key: "wealth", header: "wealth before → after", kind: "raw",
    render: (n) => <span className="study-interval"><Num value={n.payload.wealth_before as number} kind="ratio" /> → <Num value={n.payload.wealth_after as number} kind="ratio" /></span>,
  },
];

function Budget({ folder }: { folder: string }) {
  const tests = useRead<StudyNodes>(folder, "study_nodes", { kind: "test", n: NODES_N });
  const frontier = useRead<StudyFrontier>(folder, "study_frontier", { n: FRONTIER_N });
  const b = frontier.data?.budget;
  return (
    <Section title="Error budget" hint="alpha-investing, oldest test first: each test spends wealth, a rejection earns some back; a spent budget refuses the next test">
      {b && (
        <div className="ds-facts">
          <span>wealth now <Num value={b.wealth} kind="ratio" /> of <Num value={b.initial} kind="ratio" /> at the start</span>
          <span>trials <Num value={b.trials.used} kind="count" /> of <Num value={b.trials.budget} kind="count" /></span>
        </div>
      )}
      <Async query={tests} skeleton="table" context="reading the tests">
        {(d) => <Table columns={TEST_COLUMNS} rows={[...d.nodes].reverse()} total={d.total} dense rowKey={(n) => n.id}
                       empty={<p className="dim small">No test has spent the budget yet.</p>} />}
      </Async>
    </Section>
  );
}

function Assumptions({ folder }: { folder: string }) {
  const assumptions = useRead<StudyNodes>(folder, "study_nodes", { kind: "assumption", n: NODES_N });
  const breaks = useRead<StudyNodes>(folder, "study_nodes", { kind: "break", n: NODES_N });
  const brokenBy = (id: string) => (breaks.data?.nodes ?? []).filter((b) => b.links.some((l) => l.rel === "breaks" && l.to === id));
  const cols: Column<StudyNode>[] = [
    {
      key: "broken", header: "", kind: "raw",
      render: (n) => (brokenBy(n.id).length
        ? <Alarm title="every claim that needs it, and every claim resting on those, is at risk">broken</Alarm>
        : <span className="dim">holds</span>),
    },
    {
      key: "statement", header: "assumption", kind: "raw",
      render: (n) => {
        const b = brokenBy(n.id);
        return (
          <span className="study-cell-text">
            {String(n.payload.statement)}
            {b.length > 0 && <span className="study-quote">broken: {b.map((x) => String(x.payload.reason)).join("; ")}</span>}
          </span>
        );
      },
    },
    { key: "rests_on", header: "rests on", kind: "raw", render: (n) => <span className="study-cell-text">{String(n.payload.rests_on)}</span> },
    { key: "breaks_if", header: "breaks if", kind: "raw", render: (n) => <span className="study-cell-text">{String(n.payload.breaks_if)}</span> },
  ];
  return (
    <Section title="Assumptions" hint="what the claims take for granted, and the test that could break it">
      <Async query={assumptions} skeleton="table" context="reading the assumptions">
        {(d) => <Table columns={cols} rows={d.nodes} total={d.total} dense rowKey={(n) => n.id} />}
      </Async>
    </Section>
  );
}

const DEAD_END_COLUMNS: Column<StudyNode>[] = [
  { key: "statement", header: "dead end", kind: "raw", render: (n) => <span className="study-cell-text">{String(n.payload.statement)}</span> },
  { key: "reason", header: "reason", kind: "raw", render: (n) => <span className="study-cell-text">{String(n.payload.reason)}</span> },
  { key: "tags", header: "tags", kind: "raw", render: (n) => <span className="small">{(n.payload.tags as string[]).join(", ")}</span> },
  { key: "of", header: "of", kind: "raw", render: (n) => <Ids ids={n.links.filter((l) => l.rel === "of").map((l) => l.to)} /> },
  { key: "shown_by", header: "shown by", kind: "raw", render: (n) => <Ids ids={n.links.filter((l) => l.rel === "shown_by").map((l) => l.to)} /> },
];

export default function Claims({ folder, kinds }: { folder: string; kinds: Record<string, number> }) {
  const tree = useRead<StudyTree>(folder, "study_tree");
  const claims = useRead<StudyClaims>(folder, "study_claims", { n: CLAIMS_N });
  const trace = useRead<StudyTraceability>(folder, "study_traceability");
  const deadEnds = useRead<StudyNodes>(folder, "study_nodes", { kind: "dead_end", n: NODES_N }, !!kinds.dead_end);
  const [open, setOpen] = useState<StudyClaim | null>(null);
  return (
    <>
      {(kinds.question || kinds.hypothesis) ? (
        <Section title="Questions" hint="the question tree; a parent is no better than its weakest in-scope child, a hypothesis stands by its tests">
          <Async query={tree} skeleton="lines" context="reading the question tree">{(t) => <Tree nodes={t.questions} />}</Async>
        </Section>
      ) : null}

      <Section title="Claims" hint="what the paper may print, oldest first; the verdict and level are computed, never typed; open one for its chain">
        <Async query={trace} skeleton="lines">
          {(t) => (
            <>
              <div className="ds-facts">
                <span>chain complete for <Num value={t.traceable} kind="count" /> of <Num value={t.claims} kind="count" /> claims</span>
                <span>approved by a person: <Num value={t.approved.traceable} kind="count" /> of <Num value={t.approved.claims} kind="count" /> traceable</span>
              </div>
              {t.broken.length > 0 && (
                <Callout tone="danger" title={`${t.broken.length} claim${t.broken.length === 1 ? "'s" : "s'"} chain is broken`}>
                  <ul className="small">{t.broken.map((b) => <li key={b.claim}><code>{b.claim}</code>: {b.missing.join("; ")}</li>)}</ul>
                </Callout>
              )}
            </>
          )}
        </Async>
        <Async query={claims} skeleton="table" context="reading the claims">
          {(d) => {
            const risky = d.claims.filter((c) => c.at_risk.length).length;
            return (
              <>
                {risky > 0 && (
                  <Callout tone="danger" title={`${risky} claim${risky === 1 ? " is" : "s are"} at risk`}>
                    {risky === 1 ? "It needs" : "They need"} an assumption that was shown not to hold, directly or through a
                    claim {risky === 1 ? "it rests" : "they rest"} on. Do not report {risky === 1 ? "it" : "them"} until the assumption is tested again.
                  </Callout>
                )}
                <Table columns={claimColumns()} rows={d.claims} total={d.total} dense rowKey={(c) => c.id} onRowClick={setOpen}
                       empty={<p className="dim small">No claims yet.</p>} />
              </>
            );
          }}
        </Async>
      </Section>

      {kinds.test ? <Budget folder={folder} /> : null}
      {kinds.assumption ? <Assumptions folder={folder} /> : null}
      {kinds.dead_end ? (
        <Section title="Dead ends" hint="falsified hypotheses and abandoned paths, kept with their reason; a new hypothesis sharing a tag must say what differs">
          <Async query={deadEnds} skeleton="table" context="reading the dead ends">
            {(d) => <Table columns={DEAD_END_COLUMNS} rows={d.nodes} total={d.total} dense rowKey={(n) => n.id} />}
          </Async>
        </Section>
      ) : null}

      <Sheet open={!!open} onClose={() => setOpen(null)} width={640} title={open ? `trace · ${open.id}` : ""}>
        {open && <Trace folder={folder} claim={open} />}
      </Sheet>
    </>
  );
}
