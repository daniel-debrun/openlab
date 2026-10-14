/**
 * The whole app, in one grid.
 *
 *   ┌────────────────────────────────────────────────┐
 *   │ top bar: root · views · agent · guide · theme   │
 *   ├──────────┬─────────────────────────────────────┤
 *   │ tree     │ the open document (a study, a file) │   or a whole view: Agents, Tools
 *   ├──────────┴═════════════════════════════════════┤  ═ = drag to resize
 *   │ agent terminal (a drawer)                       │
 *   ├─────────────────────────────────────────────────┤
 *   │ status bar                                      │
 *   └─────────────────────────────────────────────────┘
 *
 * The agent drawer's height is clamped so the document keeps at least `DOC_MIN`, whatever was dragged or
 * restored.
 *
 * There is no router. The one navigational fact — which document is open — is a workspace path in the URL
 * hash. Pane SIZES are deliberately not in the URL: they belong to this person at this screen, not to the
 * document, so a copied link must not carry them.
 *
 * THE PANE IS CHOSEN BY THE PATH: a study (its folder's `study.toml`), the start screen for no path, and any
 * other file as plain text.
 */
import { useEffect, useState } from "react";
import { AGENT_MIN, useWorkspace, watchHash } from "../state/workspace";
import { isStudyDoc } from "../api/tools";
import { ErrorBoundary, Resizer } from "../ui";
import TextFile from "../editor/TextFile";
import StudyView from "../study/StudyView";
import Agents from "../agents/Agents";
import ToolsPanel from "../panels/ToolsPanel";
import AgentPane from "./AgentPane";
import FileTree from "./FileTree";
import Start from "./Start";
import StatusBar from "./StatusBar";
import TabStrip from "./TabStrip";
import TokenLost from "./TokenLost";
import TopBar from "./TopBar";
import { AGENTS, TOOLS, useView } from "../state/view";

/** The document never goes below this, no matter what the drawer was dragged to. */
const DOC_MIN = 240;

export default function Shell() {
  const { doc, agentOpen, agentHeight, agentDisabled, setAgentHeight, toggleAgent } = useWorkspace();
  const view = useView((s) => s.view);

  useEffect(() => watchHash(), []);

  // the viewport height, watched: a window resized while the drawer is open must not leave the document at nothing
  const [vh, setVh] = useState(window.innerHeight);
  useEffect(() => {
    const onResize = () => setVh(window.innerHeight);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  /* A disabled terminal gets no dragged height at all: it has nothing to show but one paragraph. */
  const agentPx = agentDisabled ? 0 : Math.min(agentHeight, Math.max(AGENT_MIN, vh - DOC_MIN - 120));

  return (
    <div className="shell">
      <TopBar />

      {view === AGENTS || view === TOOLS ? (
        <main className="view-main">
          <ErrorBoundary where={view.slice(1)} resetKey={view}>
            {view === AGENTS ? <Agents /> : <div className="tools-view"><ToolsPanel /></div>}
          </ErrorBoundary>
        </main>
      ) : (
        <div className="shell-main">
          <div className="rail">
            <ErrorBoundary where="workspace tree">
              <FileTree />
            </ErrorBoundary>
          </div>

          <main className="shell-doc">
            <ErrorBoundary where="tab strip">
              <TabStrip />
            </ErrorBoundary>
            <div className="doc-pane">
              {!doc ? (
                <ErrorBoundary where="start screen">
                  <Start />
                </ErrorBoundary>
              ) : isStudyDoc(doc) ? (
                <ErrorBoundary where="study" resetKey={doc}>
                  <StudyView doc={doc} />
                </ErrorBoundary>
              ) : (
                <ErrorBoundary where="file" resetKey={doc}>
                  <TextFile doc={doc} />
                </ErrorBoundary>
              )}
            </div>
          </main>
        </div>
      )}

      {agentOpen && (
        <>
          {!agentDisabled && (
            <Resizer
              orientation="horizontal"
              value={agentPx}
              min={AGENT_MIN}
              max={Math.max(AGENT_MIN, vh - DOC_MIN - 120)}
              onChange={setAgentHeight}
              onToggle={toggleAgent}
              invert
              label="agent terminal height"
            />
          )}
          <div
            className={`agent-drawer${agentDisabled ? " is-notice" : ""}`}
            style={agentDisabled ? undefined : { height: agentPx }}
          >
            <ErrorBoundary where="agent pane">
              <AgentPane />
            </ErrorBoundary>
          </div>
        </>
      )}

      <StatusBar />
      <TokenLost />
    </div>
  );
}
