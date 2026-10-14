/**
 * The agents: the tree (indented by parent) with each agent's state, role, question, spend against its budget,
 * tokens, trials, claims, refusals, last action and the flags a supervising person cares about; the inbox of
 * what waits on a person; and, for the agent picked, its live log, brief, budget and children, with Pause,
 * Resume and Stop.
 *
 * It is `openlab watch` in the browser and reads the same snapshot (`agent_tree`). Top level it shows the whole
 * workspace; inside a study's view (`study`) only the agents working on that study.
 *
 * Pause, stop, resume and an inbox decision are a person's acts (personAct.ts): they show only at the reviewer
 * link, and the server refuses them without it.
 */
import { useQueryClient } from "@tanstack/react-query";
import { REVIEWER } from "../api/client";
import type { AgentRow, AgentTree, InboxItem } from "../api/tools";
import { Async, Badge, Button, Callout, Empty, KeyValue, Table, cx, fmtAgo, fmtDateFull, type Column } from "../ui";
import { useWorkspace } from "../state/workspace";
import { personAct } from "./personAct";
import {
  BudgetBar,
  Flags,
  LIVE,
  StateChip,
  ago,
  money,
  tokens,
  useAgentLog,
  useAgentPick,
  useAgentTree,
  useInbox,
} from "./parts";

function columns(pick: (id: string) => void, picked: string, wide: boolean): Column<AgentRow>[] {
  return [
    {
      key: "id", header: "agent", kind: "raw",
      render: (r) => (
        <button type="button" className={cx("agt-pick", r.id === picked && "is-picked")}
                style={{ paddingLeft: `calc(${r.depth} * var(--sp-6))` }}
                onClick={(e) => { e.stopPropagation(); pick(r.id); }}
                aria-pressed={r.id === picked} title={`${r.kind} ${r.id}${r.parent ? `, child of ${r.parent}` : ""}`}>
          {r.depth > 0 && <span className="agt-branch" aria-hidden="true">└</span>}
          <span className="agt-who">
            <code>{r.id}</code>
            <span className="agt-role">{r.role || r.kind}</span>
          </span>
        </button>
      ),
    },
    {
      key: "state", header: "state", kind: "raw",
      render: (r) => <span className="agt-states"><StateChip state={r.state} /><Flags flags={r.flags} row={r} /></span>,
    },
    {
      key: "question", header: wide ? "study · question" : "question", kind: "raw",
      render: (r) => (
        <span className="agt-q" title={r.question ?? undefined}>
          {wide && r.study && <code className="agt-study">{r.study}</code>}
          {r.question ?? <span className="dim">—</span>}
        </span>
      ),
    },
    { key: "usd", header: "spent", kind: "raw", render: (r) => <BudgetBar row={r} /> },
    { key: "tokens", header: "tokens", kind: "raw", render: (r) => <span className="num">{tokens(r.tokens)}</span> },
    { key: "trials", header: "trials", kind: "count" },
    { key: "claims", header: "claims", kind: "count" },
    { key: "refusals", header: "refused", kind: "count", title: "refusals: the record or the hooks said no",
      tone: (r) => (r.refusals ? "neg" : undefined) },
    {
      key: "last", header: "last action", kind: "raw",
      render: (r) => (
        <span className="agt-last" title={r.last ? `${r.last.kind}: ${r.last.summary}` : undefined}>
          <span className="agt-age">{ago(r.idle_s)}</span> {r.last ? r.last.summary : <span className="dim">nothing yet</span>}
        </span>
      ),
    },
  ];
}

function Summary({ t }: { t: AgentTree }) {
  const c = t.counts;
  const parts = (["working", "waiting", "stuck", "paused", "idle"] as const).filter((k) => c[k]);
  return (
    <div className="agt-summary">
      <span><strong>{t.agents.length}</strong> agent{t.agents.length === 1 ? "" : "s"}</span>
      {parts.map((k) => <span key={k}>{c[k]} {k}</span>)}
      <span title="the $ the agents report (or openlab estimates from tokens)">{money(t.usd)} spent</span>
      {t.pending > 0 && <Badge tone="accent">{t.pending} wait{t.pending === 1 ? "s" : ""} for you</Badge>}
    </div>
  );
}

const CONTROL_SAID = {
  pause: { title: "Pause", body: "Its next tool call is refused (a spawned agent's process is also stopped) until a person resumes it. Its children are held with it.", label: "Pause" },
  resume: { title: "Resume", body: "It may continue. Resuming is a person's act.", label: "Resume" },
  stop: { title: "Stop", body: "It is told to end now (a spawned agent's process is killed). This cannot be undone: a stopped agent does not resume.", label: "Stop" },
} as const;

function Controls({ row, onDone }: { row: AgentRow; onDone: () => void }) {
  const live = LIVE.includes(row.state);
  const paused = row.state === "paused" || row.control === "pause";
  const control = async (action: "pause" | "resume" | "stop") => {
    const s = CONTROL_SAID[action];
    const env = await personAct("agent_control", { id: row.id, action }, {
      title: `${s.title} ${row.id}?`, body: s.body, confirmLabel: s.label, danger: action === "stop",
      placeholder: action === "resume" ? "why it may go on" : "why",
    });
    if (env) onDone();
  };
  if (!REVIEWER) {
    return (
      <p className="dim small">
        Pause and stop are a person&rsquo;s acts: open the full URL <code>openlab app</code> printed (with its{" "}
        <code>&amp;r=</code>), or run <code>openlab agent pause {row.id} --reason …</code> in your own terminal.
      </p>
    );
  }
  const why = live ? undefined : `${row.id} has ended (${row.state})`;
  return (
    <div className="agt-controls">
      {paused ? (
        <Button size="sm" onClick={() => void control("resume")} disabled={!live} title={why ?? "let it continue"}>Resume…</Button>
      ) : (
        <Button size="sm" onClick={() => void control("pause")} disabled={!live} title={why ?? "hold it (and its children) before its next tool call"}>Pause…</Button>
      )}
      <Button size="sm" variant="danger" onClick={() => void control("stop")} disabled={!live} title={why ?? "end it now"}>Stop…</Button>
    </div>
  );
}

const LOG_TONE: Record<string, string> = {
  refusal: "is-warn", blocked: "is-warn", bypass_attempt: "is-danger", lost: "is-warn", request: "is-accent",
  decision: "is-accent", control_pause: "is-accent", control_resume: "is-accent", control_stop: "is-accent",
};

function Detail({ id, onPick }: { id: string; onPick: (id: string) => void }) {
  const qc = useQueryClient();
  // the whole tree, not the study's: a parent or child picked from here may work on another study
  const tree = useAgentTree(undefined, id);
  const log = useAgentLog(id);
  const row = tree.data?.agents.find((a) => a.id === id);
  const d = tree.data?.detail;
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["agent_tree"] });
    void qc.invalidateQueries({ queryKey: ["agent_log", id] });
  };
  return (
    <section className="agt-detail" aria-label={`agent ${id}`}>
      <header className="agt-detail-head">
        <h3><code>{id}</code></h3>
        {row && <StateChip state={row.state} />}
        {row && <Flags flags={row.flags} row={row} />}
      </header>
      {row && <Controls row={row} onDone={refresh} />}
      {d?.control && (
        <Callout tone="info" title={`${d.control === "pause" ? "paused" : "stopped"} by a person`}>{d.control_reason}</Callout>
      )}
      {row && d && (
        <KeyValue rows={[
          { label: "role", value: row.role || "—" },
          { label: "study", value: row.study ? <code>{row.study}</code> : "—" },
          { label: "question", value: row.question ?? "—" },
          { label: "spent", value: <BudgetBar row={row} /> },
          {
            label: "budget",
            value: [d.budget.usd != null && `$${d.budget.usd}`, d.budget.minutes != null && `${d.budget.minutes} min`,
                    d.budget.turns != null && `${d.budget.turns} turns`].filter(Boolean).join(" · ") || "none set",
          },
          { label: "tokens", value: `${tokens(row.tokens)}${typeof d.spent.turns === "number" ? ` · ${d.spent.turns} turns` : ""}` },
          { label: "record", value: `${row.trials} trials · ${row.claims} claims · ${row.refusals} refusals` },
          { label: "runs as", value: [row.kind, d.cli, d.model].filter(Boolean).join(" · ") },
          { label: "started", value: <span title={fmtDateFull(d.created)}>{fmtAgo(d.created)}</span> },
          ...(d.exit_code != null ? [{ label: "exit code", value: String(d.exit_code) }] : []),
          ...(row.parent ? [{ label: "parent", value: <button type="button" className="agt-link" onClick={() => onPick(row.parent!)}><code>{row.parent}</code></button> }] : []),
          ...(d.children.length ? [{
            label: "children",
            value: <span className="agt-children">{d.children.map((c) => (
              <button key={c} type="button" className="agt-link" onClick={() => onPick(c)}><code>{c}</code></button>
            ))}</span>,
          }] : []),
        ]} />
      )}
      {d?.prompt && (
        <details className="agt-brief">
          <summary>Brief</summary>
          <pre>{d.prompt}</pre>
        </details>
      )}
      <h4 className="agt-sub">Log <span className="dim small">live, newest last</span></h4>
      <Async query={log} skeleton="lines" context={`reading ${id}'s log`}>
        {(l) => l.events.length ? (
          <ol className="agt-log">
            {l.events.map((e) => (
              <li key={e.seq} className={LOG_TONE[e.kind]}>
                <time dateTime={e.at} title={fmtDateFull(e.at)}>{new Date(e.at).toLocaleTimeString()}</time>
                <span className="agt-log-kind">{e.kind}</span>
                <span className="agt-log-what">{e.summary || e.tool}</span>
              </li>
            ))}
          </ol>
        ) : <p className="dim small">No events yet.</p>}
      </Async>
    </section>
  );
}

const KIND_SAID: Record<string, string> = {
  holdout: "a holdout look", amend: "an amendment of the sealed plan", approval: "an approval",
  budget: "more budget", screen: "a screening decision", screen_conflict: "which screening decision stands",
  other: "a decision",
};

/** A screen_conflict's options: the agents' disagreeing decisions on one source. */
type ScreenOption = { agent: string; decision: string; criterion: string };

function Inbox({ study, onPick }: { study?: string; onPick: (id: string) => void }) {
  const qc = useQueryClient();
  const q = useInbox(study);
  const decide = async (it: InboxItem, approve: boolean) => {
    const env = await personAct("inbox_decide", { id: it.id, approve }, {
      title: `${approve ? "Approve" : "Deny"} #${it.id}: ${KIND_SAID[it.kind] ?? it.kind}?`,
      body: (
        <>
          <p><code>{it.agent}</code> asks: {it.reason}</p>
          {it.cost_note && <p>Cost: {it.cost_note}</p>}
          <p>{approve ? "Approving performs the act now, as you, and records both." : "Denying records why; the agent reads it."}</p>
        </>
      ),
      confirmLabel: approve ? "Approve" : "Deny",
      danger: approve && it.kind === "holdout",
      fields: approve && it.kind === "screen_conflict" ? [{
        key: "pick", label: "The decision that stands",
        options: ((it.args.options as ScreenOption[] | undefined) ?? []).map((o, i) => ({
          value: String(i + 1), label: `${o.agent}: ${o.decision} (${o.criterion})`,
        })),
      }] : undefined,
    });
    if (env?.ok) {
      void qc.invalidateQueries({ queryKey: ["inbox_list"] });
      void qc.invalidateQueries({ queryKey: ["agent_tree"] });
    }
  };
  return (
    <section className="agt-inbox" aria-label="inbox">
      <h3 className="agt-sub">Inbox <span className="dim small">what waits on you</span></h3>
      <Async query={q} skeleton="lines" skeletonRows={2} context="reading the inbox">
        {(d) => d.items.length === 0 ? (
          <p className="dim small">Nothing waits on you. An agent asks here (agent_request) when it needs a person&rsquo;s act.</p>
        ) : (
          <ul className="agt-items">
            {d.items.map((it) => (
              <li key={it.id} className="agt-item">
                <div className="agt-item-head">
                  <Badge tone="accent">#{it.id} {it.kind}</Badge>
                  <button type="button" className="agt-link" onClick={() => onPick(it.agent)} title="show this agent">
                    <code>{it.agent}</code>
                  </button>
                  {it.study && <code className="agt-study">{it.study}</code>}
                  <span className="dim small" title={fmtDateFull(it.created)}>{fmtAgo(it.created)}</span>
                </div>
                <p className="agt-item-why">{it.reason}</p>
                {it.cost_note && <p className="dim small">cost: {it.cost_note}</p>}
                {REVIEWER ? (
                  <div className="agt-controls">
                    <Button size="sm" variant="primary" onClick={() => void decide(it, true)}>Approve…</Button>
                    <Button size="sm" onClick={() => void decide(it, false)}>Deny…</Button>
                  </div>
                ) : (
                  <p className="dim small">Decide at the reviewer link, or <code>openlab inbox approve {it.id} --reason …</code></p>
                )}
              </li>
            ))}
          </ul>
        )}
      </Async>
    </section>
  );
}

export default function Agents({ study }: { study?: string }) {
  const tree = useAgentTree(study);
  const { id: picked, pick } = useAgentPick();
  const { agentOpen, toggleAgent } = useWorkspace();
  const rows = tree.data?.agents ?? [];
  const current = rows.some((r) => r.id === picked) ? picked : "";
  return (
    <div className={cx("agents", study !== undefined && "is-study")}>
      {study === undefined && (
        <header className="agents-head">
          <h2 className="agents-title">Agents</h2>
          {tree.data && <Summary t={tree.data} />}
        </header>
      )}
      <div className="agents-grid">
        <div className="agents-main">
          <Async query={tree} skeleton="table" context="reading the agent tree">
            {(t) => (
              <Table
                columns={columns(pick, current, study === undefined)}
                rows={t.agents}
                dense
                stickyFirst
                rowKey={(r) => r.id}
                onRowClick={(r) => pick(r.id)}
                empty={
                  <Empty
                    title={study !== undefined ? "no agent works on this study yet" : "no agents yet"}
                    action={{ label: agentOpen ? "Hide the agent pane" : "Open the agent pane", onClick: toggleAgent }}
                    body="An agent appears here when it is spawned (openlab agent spawn, or agent_spawn from a lead agent), or when a Claude Code session starts in a workspace set up with openlab init --claude. openlab watch shows the same tree in a terminal."
                  />
                }
              />
            )}
          </Async>
          <Inbox study={study} onPick={pick} />
        </div>
        <aside className="agents-side">
          {current ? (
            <Detail id={current} onPick={pick} />
          ) : (
            <p className="dim small agt-hint">{rows.length ? "Pick an agent for its live log, brief, budget and children." : ""}</p>
          )}
        </aside>
      </div>
    </div>
  );
}
