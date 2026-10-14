/**
 * The start screen's line on the agents: how many, in which states, what they spent, how many requests wait on
 * you, and the agents with a flag (stuck, looping, over budget, waiting, a bypass attempt, a dead end), each one
 * click from its log in the Agents view.
 */
import { AGENTS, useView } from "../state/view";
import { Badge, Button, ErrorPanel } from "../ui";
import { Flags, LIVE, StateChip, money, useAgentPick, useAgentTree } from "./parts";

const SHOWN = 5;

export default function AgentsSummary() {
  const q = useAgentTree();
  const setView = useView((s) => s.setView);
  const pick = useAgentPick((s) => s.pick);
  if (q.isError) return <ErrorPanel error={q.error} context="reading the agent tree" onRetry={() => void q.refetch()} />;
  if (!q.data) return null;
  const t = q.data;
  const live = t.agents.filter((a) => LIVE.includes(a.state));
  const flagged = t.agents.filter((a) => a.flags.length && LIVE.includes(a.state));
  const open = (id?: string) => {
    if (id) pick(id);
    setView(AGENTS);
  };
  return (
    <section className="start-agents" aria-label="agents">
      <h2 className="start-sub">Agents</h2>
      {t.agents.length === 0 ? (
        <p className="start-hint">
          None yet. Every agent that works here — spawned with <code>openlab agent spawn</code>, delegated by a lead
          agent, or a Claude Code session set up with <code>openlab init --claude</code> — appears on one tree with
          its budget, its spend and what it waits on you for.
        </p>
      ) : (
        <>
          <p className="start-agents-line">
            <span>{live.length} running of {t.agents.length}</span>
            {(["working", "waiting", "stuck", "paused"] as const).filter((k) => t.counts[k]).map((k) => (
              <span key={k}>{t.counts[k]} {k}</span>
            ))}
            <span>{money(t.usd)} spent</span>
            {t.pending > 0 && <Badge tone="accent">{t.pending} wait{t.pending === 1 ? "s" : ""} for you</Badge>}
          </p>
          {flagged.length > 0 && (
            <ul className="start-agents-flagged">
              {flagged.slice(0, SHOWN).map((a) => (
                <li key={a.id}>
                  <button type="button" className="agt-link" onClick={() => open(a.id)} title="open its log">
                    <code>{a.id}</code>
                  </button>
                  <StateChip state={a.state} />
                  <Flags flags={a.flags} row={a} />
                </li>
              ))}
              {flagged.length > SHOWN && <li className="dim small">and {flagged.length - SHOWN} more</li>}
            </ul>
          )}
        </>
      )}
      <Button size="sm" onClick={() => open()}>Open the agents{t.pending ? " and the inbox" : ""}</Button>
    </section>
  );
}
