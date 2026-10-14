/**
 * The view tabs after the workspace name: Workspace (the files and the open study), Agents (the tree of agents
 * and the inbox of what needs you), and Tools under "More" (every tool, callable directly). The Agents tab
 * carries the inbox's pending count, so a request that waits on you is visible from every screen.
 */
import { useQuery } from "@tanstack/react-query";
import { runTool, type InboxList } from "../api/tools";
import { AGENTS, TOOLS, useView } from "../state/view";
import { Menu, MenuItem, cx } from "../ui";

export default function ViewTabs() {
  const { view, setView } = useView();
  const inbox = useQuery({
    queryKey: ["inbox_list", "", false],
    queryFn: () => runTool<InboxList>("inbox_list", {}),
    refetchInterval: 5000,
  });
  const pending = inbox.data?.pending ?? 0;

  return (
    <nav className="topbar-views" aria-label="views">
      <button className={cx("topbar-view", !view && "is-active")} onClick={() => setView("")}
        title="the files, the studies and the open document">
        Workspace
      </button>
      <button className={cx("topbar-view", view === AGENTS && "is-active")} onClick={() => setView(AGENTS)}
        title={pending ? `the agent tree; ${pending} request${pending === 1 ? "" : "s"} wait for you` : "the agent tree, and the inbox of what needs you"}>
        Agents
        {pending > 0 && <span className="topbar-view-count" aria-label={`${pending} waiting`}>{pending}</span>}
      </button>
      {view === TOOLS && (
        <button className="topbar-view is-active" onClick={() => setView(TOOLS)}>Tools</button>
      )}
      <Menu label="More" align="left" className="topbar-more">
        <MenuItem onClick={() => setView(TOOLS)} hint="every tool with its arguments, callable directly">
          Tools
        </MenuItem>
      </Menu>
    </nav>
  );
}
