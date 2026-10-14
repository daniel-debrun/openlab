/**
 * What the agents just did, above the terminal: each tool call and refusal from the control store, with the
 * agent that made it, and how many requests wait in the inbox (decided in the Agents view or `openlab inbox`).
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { isWorkspace, useEvents } from "../api/events";
import { runTool } from "../api/tools";
import { cx, toDate } from "../ui";

interface Event { at: string; agent: string; kind: string; tool: string; ok: boolean; error?: string }

export default function AgentActivity() {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["agent_activity"],
    queryFn: () => runTool<{ events: Event[]; waiting: number; refused: number }>("agent_activity", { last: 8 }),
    refetchInterval: 5000,
  });
  useEvents((ev) => { if (isWorkspace(ev)) qc.invalidateQueries({ queryKey: ["agent_activity"] }); }, undefined, []);
  const events = q.data?.events ?? [];
  const waiting = q.data?.waiting ?? 0;
  if (!events.length) return <p className="agent-line">Your agents&rsquo; calls to openlab appear here as they work.</p>;
  return (
    <ul className="agent-activity" aria-label="the agents' recent calls">
      {waiting > 0 && (
        <li className="aa-row is-waiting">
          <span className="aa-what">{waiting} request{waiting === 1 ? "" : "s"} wait for you in the inbox</span>
        </li>
      )}
      {events.map((e, i) => (
        <li key={`${e.at}-${i}`} className={cx("aa-row", !e.ok && "is-refused")}>
          <span className="aa-time">{toDate(e.at)?.toLocaleTimeString()}</span>
          <span className="aa-agent">{e.agent}</span>
          <code className="aa-tool">{e.tool}</code>
          <span className="aa-what">{e.ok ? "ok" : `${e.kind}: ${e.error ?? ""}`}</span>
        </li>
      ))}
    </ul>
  );
}
