/**
 * What to do next (`study_frontier`): the open items the record shows (untested hypotheses and questions, claims at
 * risk, contradictions not decided, insufficient searches, inconclusive tests, lattice gaps, a stale paper,
 * forecasts awaiting outcomes), ranked by expected information per hour. Take records the choice as an action
 * (`study_take`, the same verb an agent calls) and leases the item to the taker; an item another agent holds says
 * who and until when, and a click on the holder opens it in the Agents view. The item names the verb that acts on it.
 */
import type { FrontierItem, StudyFrontier } from "../api/tools";
import { AGENTS, useView } from "../state/view";
import { Async, Badge, Button, Num, Table, fmtDateFull, type Column } from "../ui";
import { useAgentPick } from "../agents/parts";
import { Ids, Section, useAct, useRead } from "./parts";

/** Items the view asks `study_frontier` for (also read by the Claims tab for the error budget). */
export const FRONTIER_N = 50;

export default function Frontier({ folder }: { folder: string }) {
  const frontier = useRead<StudyFrontier>(folder, "study_frontier", { n: FRONTIER_N });
  const { busy, act } = useAct(folder);
  const setView = useView((v) => v.setView);
  const pick = useAgentPick((v) => v.pick);
  const cols: Column<FrontierItem>[] = [
    { key: "kind", header: "kind", kind: "raw", render: (i) => <Badge tone="neutral">{i.kind.replace(/_/g, " ")}</Badge> },
    { key: "target", header: "target", kind: "raw", render: (i) => <Ids ids={[i.target]} /> },
    { key: "why", header: "why", kind: "raw", render: (i) => <span className="study-cell-text">{i.why}</span> },
    { key: "verb", header: "verb", kind: "raw", render: (i) => <code>{i.verb}</code> },
    { key: "info", header: "info", kind: "ratio", title: "expected information, 0 to 1" },
    { key: "cost", header: "cost", kind: "duration", title: "expected seconds" },
    { key: "score", header: "score", kind: "ratio", title: "information per hour: the rank" },
    {
      key: "take", header: "", kind: "raw",
      render: (i) => (i.held_by ? (
        <span className="small" title={`leased until ${fmtDateFull(i.held_by.expires)}; another agent's take is refused until then`}>
          held by{" "}
          <button type="button" className="agt-link" onClick={() => { pick(i.held_by!.agent); setView(AGENTS); }}>
            <code>{i.held_by.agent}</code>
          </button>
        </span>
      ) : (
        <Button size="sm" loading={busy === `take:${i.id}`} onClick={() => void act("study_take", { item: i.id }, `take:${i.id}`)}
                title={`record that this item is being worked on (a lease to you); then act on it with ${i.verb}${i.taken ? `. Taken before (${i.taken}); that lease has ended` : ""}`}>
          Take
        </Button>
      )),
    },
  ];
  return (
    <Section title="Frontier" hint="ranked by expected information per hour of work; derived from the record, never stored">
      <Async query={frontier} skeleton="table" context="reading the frontier">
        {(d) => (
          <>
            <div className="ds-facts">
              <span>error budget <Num value={d.budget.wealth} kind="ratio" /> of <Num value={d.budget.initial} kind="ratio" /></span>
              <span>trials <Num value={d.budget.trials.used} kind="count" /> of <Num value={d.budget.trials.budget} kind="count" /></span>
            </div>
            <Table columns={cols} rows={d.items} total={d.total} dense rowKey={(i) => i.id}
                   empty={<p className="dim small">Nothing open: the record shows no untested leaf, risk, gap or pending outcome.</p>} />
          </>
        )}
      </Async>
    </Section>
  );
}
