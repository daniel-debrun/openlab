/**
 * The approved proposal's progress (`propose_status`): the plan's title, question and budget against what was
 * spent, then one row per branch with its state, its agents, its spend against its own budget, its hypotheses
 * tested, killed and untouched, its stop rule and who holds it. An agent's name opens it in the Agents view.
 * Approving a proposal is a person's act and is done in the terminal (`openlab propose approve`), not here.
 */
import { AGENTS, useView } from "../state/view";
import type { BranchBudget, ProposalBranch, ProposeStatus } from "../api/tools";
import { Async, Badge, Num, Table, fmtAgo, fmtDateFull, type Column } from "../ui";
import { useAgentPick } from "../agents/parts";
import { Section, useRead } from "./parts";

const STATE_TONE: Record<string, "ok" | "neutral" | "warn" | "info" | "accent"> = {
  working: "ok", held: "info", idle: "neutral", unassigned: "warn", done: "neutral", stopped: "neutral", killed: "neutral",
};

/** Spend against a budget, one line per unit when `stacked` (a table cell), else on one line. */
function Spend({ spent, budget, stacked }: { spent: { usd: number; minutes?: number; trials: number }; budget: BranchBudget;
                                             stacked?: boolean }) {
  const part = (v: number, of: number | null | undefined, unit: (x: number) => string) =>
    of ? `${unit(v)} of ${unit(of)}` : unit(v);
  const parts = [part(spent.usd, budget.usd, (x) => `$${x.toFixed(2)}`),
    spent.minutes !== undefined ? part(spent.minutes, budget.minutes, (x) => `${Math.round(x)} min`) : null,
    part(spent.trials, budget.trials, (x) => `${x} trials`)].filter(Boolean) as string[];
  return stacked ? (
    <span className="plan-spend">{parts.map((p) => <span key={p}>{p}</span>)}</span>
  ) : (
    <span className="agt-budget-num">{parts.join(" · ")}</span>
  );
}

export default function Plan({ folder }: { folder: string }) {
  const q = useRead<ProposeStatus>(folder, "propose_status");
  const setView = useView((v) => v.setView);
  const pick = useAgentPick((v) => v.pick);
  const agent = (id: string) => (
    <button key={id} type="button" className="agt-link" onClick={() => { pick(id); setView(AGENTS); }}><code>{id}</code></button>
  );
  const cols: Column<ProposalBranch>[] = [
    { key: "id", header: "branch", kind: "raw", render: (b) => <span className="agt-who"><code>{b.id}</code><span className="agt-role">{b.role}</span></span> },
    { key: "state", header: "state", kind: "raw", render: (b) => <Badge tone={STATE_TONE[b.state] ?? "neutral"}>{b.state}</Badge> },
    { key: "question", header: "question", kind: "raw", render: (b) => <span className="study-cell-text">{b.question}</span> },
    { key: "spent", header: "spent", kind: "raw", render: (b) => <Spend spent={b.spent} budget={b.budget} stacked /> },
    {
      key: "hyp", header: "hypotheses", kind: "raw", title: "tested · killed · untouched",
      render: (b) => {
        const n = (s: string) => b.hypotheses.filter((h) => h.state === s).length;
        return <span className="num">{n("tested")} · {n("killed")} · {n("untouched")}</span>;
      },
    },
    {
      key: "agents", header: "agents", kind: "raw",
      render: (b) => (b.agents.length ? <span className="agt-children">{b.agents.map((a) => agent(a.id))}</span>
        : b.held_by ? <span className="small">held by {agent(b.held_by.agent)}</span> : <span className="dim">none</span>),
    },
    {
      key: "stop", header: "stop rule", kind: "raw",
      render: (b) => (
        <span className="study-cell-text" title={`stop: ${b.stop_rule || "none"}${b.kill ? `; kill: ${b.kill}` : ""}`}>
          {b.stop_state.startsWith("not") ? <span className="dim">{b.stop_state}</span> : <Badge tone="warn">{b.stop_state}</Badge>}
        </span>
      ),
    },
  ];
  return (
    <Section title="Plan" hint="the approved proposal: each branch with its agents, its budget and its stop rule">
      <Async query={q} skeleton="table" context="reading the proposal's progress">
        {(p) => (
          <>
            <p className="study-question">{p.title}</p>
            <div className="ds-facts">
              <span title={fmtDateFull(p.approved_at)}>approved {fmtAgo(p.approved_at)} by {p.approved_by}</span>
              {p.amendments > 0 && <span>amended {p.amendments}×</span>}
              <span>spent <Spend spent={p.spent} budget={p.budget} /></span>
              <span>hypotheses <Num value={p.hypotheses.tested} kind="count" /> tested, <Num value={p.hypotheses.killed} kind="count" /> killed, <Num value={p.hypotheses.untouched} kind="count" /> untouched</span>
              {p.lead.map((l) => <span key={l.id}>lead {agent(l.id)} {l.status}</span>)}
            </div>
            <Table columns={cols} rows={p.branches} dense rowKey={(b) => b.id}
                   empty={<p className="dim small">The proposal has no branches.</p>} />
          </>
        )}
      </Async>
    </Section>
  );
}
