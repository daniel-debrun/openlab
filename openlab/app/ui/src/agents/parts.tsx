/**
 * What every agent surface draws the same way: an agent's state, its flags, its budget bar, money and tokens;
 * and the reads (agent_tree, agent_log, inbox_list), polled, because agents write to the control store from
 * their own processes and the app's event socket does not see those writes.
 */
import { useQuery } from "@tanstack/react-query";
import { create } from "zustand";
import {
  AGENT_POLL_MS,
  runTool,
  type AgentLog,
  type AgentRow,
  type AgentState,
  type AgentTree,
  type InboxList,
} from "../api/tools";
import { Badge, cx } from "../ui";

/** The agent picked in the Agents view, shared so the start screen can open the view on one agent. */
export const useAgentPick = create<{ id: string; pick: (id: string) => void }>((set) => ({
  id: "",
  pick: (id) => set({ id }),
}));

export function useAgentTree(study?: string, id?: string) {
  return useQuery({
    queryKey: ["agent_tree", study ?? "", id ?? ""],
    queryFn: () => runTool<AgentTree>("agent_tree", { ...(study ? { study } : {}), ...(id ? { id } : {}) }),
    refetchInterval: AGENT_POLL_MS,
  });
}

export function useAgentLog(id: string, n = 60) {
  return useQuery({
    queryKey: ["agent_log", id, n],
    queryFn: () => runTool<AgentLog>("agent_log", { id, n }),
    enabled: !!id,
    refetchInterval: AGENT_POLL_MS,
  });
}

export function useInbox(study?: string, all = false) {
  return useQuery({
    queryKey: ["inbox_list", study ?? "", all],
    queryFn: () => runTool<InboxList>("inbox_list", { ...(study ? { study } : {}), ...(all ? { all } : {}) }),
    refetchInterval: AGENT_POLL_MS * 2,
  });
}

/** The system channel only: red is for what broke (failed), never for a state that merely needs a look. */
const STATE_TONE: Record<AgentState, "neutral" | "ok" | "danger" | "warn" | "info" | "accent"> = {
  working: "ok",
  idle: "neutral",
  waiting: "accent",
  stuck: "warn",
  paused: "info",
  done: "neutral",
  failed: "danger",
  stopped: "neutral",
  lost: "warn",
};

const STATE_WHY: Record<AgentState, string> = {
  working: "running, with a recent event",
  idle: "registered, not running a turn",
  waiting: "waiting on a person: an inbox item or a permission prompt",
  stuck: "running, but no event for longer than [agents] stuck_minutes",
  paused: "paused by a person: its next tool call is refused",
  done: "ended normally",
  failed: "ended with an error",
  stopped: "stopped by a person",
  lost: "its process is gone and it never reported an end",
};

export const LIVE: AgentState[] = ["working", "idle", "waiting", "stuck", "paused"];

export function StateChip({ state }: { state: AgentState }) {
  return (
    <Badge tone={STATE_TONE[state] ?? "neutral"} title={STATE_WHY[state]} className="agt-state">
      {state}
    </Badge>
  );
}

const FLAG_TONE: Record<string, "warn" | "danger" | "accent" | "info"> = {
  STUCK: "warn",
  LOOP: "warn",
  OVER: "warn",
  WAITING: "accent",
  BYPASS: "danger",
  "DEAD-END": "info",
};

const FLAG_WHY: Record<string, string> = {
  STUCK: "no event for longer than [agents] stuck_minutes while running",
  LOOP: "the same refusal again and again among its recent events",
  OVER: "90% or more of a budget ($, minutes or turns) is spent",
  WAITING: "it filed a request that waits on you: see the inbox",
  BYPASS: "it tried to get around a person's act; the attempt was denied and recorded",
  "DEAD-END": "its question or hypothesis matches a dead end another agent recorded in the same study",
};

export function Flags({ flags, row }: { flags: string[]; row?: AgentRow }) {
  if (!flags.length) return null;
  return (
    <span className="agt-flags">
      {flags.map((f) => {
        const base = f.startsWith("LOOP") ? "LOOP" : f;
        const extra =
          base === "DEAD-END" && row?.dead_end ? `: ${row.dead_end.agent} recorded "${row.dead_end.text}"`
          : base === "LOOP" && row?.last_refusal ? `: ${row.last_refusal.summary}`
          : "";
        return (
          <Badge key={f} tone={FLAG_TONE[base] ?? "warn"} title={`${FLAG_WHY[base] ?? f}${extra}`} className="agt-flag">
            {f}
          </Badge>
        );
      })}
    </span>
  );
}

export function money(usd: number | null | undefined, estimated = false): string {
  if (usd == null) return "—";
  return `${estimated ? "~" : ""}$${usd < 10 ? usd.toFixed(2) : usd.toFixed(0)}`;
}

export function tokens(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}

export function ago(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

/** $ spent against the budget: a bar of the largest share spent ($, minutes or turns), warn from 90%. */
export function BudgetBar({ row }: { row: AgentRow }) {
  const frac = Math.max(0, row.frac || 0);
  const label = `${money(row.usd, row.usd_estimated)}${row.budget_usd != null ? ` of ${money(row.budget_usd)}` : ""}`;
  const title = `${label}${row.usd_estimated ? " (estimated from tokens)" : ""}; ${Math.round(frac * 100)}% of its tightest budget spent`;
  return (
    <span className="agt-budget" title={title}>
      <span className="agt-budget-num">{label}</span>
      {(row.budget_usd != null || frac > 0) && (
        <span className="agt-bar" aria-hidden="true">
          <span className={cx("agt-bar-fill", frac >= 0.9 && "is-over")} style={{ width: `${Math.min(1, frac) * 100}%` }} />
        </span>
      )}
    </span>
  );
}
