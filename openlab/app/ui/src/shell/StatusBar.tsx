/**
 * The status bar: the quiet line that says what is open and what just happened.
 *
 * The last event is shown as the act rather than summarised, because the useful
 * fact is *who* moved — "study run" when you did not press Run means
 * the agent is working in this workspace right now.
 */
import { useState } from "react";
import { isWorkspace, useEvents } from "../api/events";
import { useWorkspace } from "../state/workspace";
import { Button } from "../ui";

/** The common tools in a person's words; the rest lose their prefix and underscores. */
const SAID: Record<string, string> = {
  study_run: "ran a study",
  study_holdout: "took a holdout look",
  ws_write: "saved a file",
  ws_delete: "deleted a file",
};
const said = (tool: string) => SAID[tool] ?? tool.replace(/^(ws|study)_/, "").replace(/_/g, " ");

export default function StatusBar() {
  const { doc, agentOpen, agentDisabled, toggleAgent } = useWorkspace();
  const [last, setLast] = useState<string>("");

  useEvents((ev) => {
    if (isWorkspace(ev)) setLast(said(String(ev.tool)));
  });

  return (
    <footer className="statusbar">
      <span className="statusbar-doc">{doc || "nothing open"}</span>
      <span className="statusbar-spacer" />
      {last && (
        <span className="statusbar-last" title="the most recent thing to change this workspace">
          {last}
        </span>
      )}
      <Button
        variant="ghost"
        size="sm"
        onClick={toggleAgent}
        aria-expanded={agentOpen}
        title={
          agentDisabled
            ? "this server was started without a terminal — the pane explains what to do"
            : agentOpen
              ? "hide the agent terminal"
              : "show the agent terminal"
        }
      >
        {agentOpen ? "▾" : "▸"} agent{agentDisabled ? " · off" : ""}
      </Button>
    </footer>
  );
}
