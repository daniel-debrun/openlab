/**
 * The first screen: what openlab is, in one line; the studies in this workspace (`study_list`); the agents
 * working on them (`agent_tree`), with what waits on you; and the way to hand work to your own agent.
 *
 * Two columns: the statement, the studies and the agent on the left; the agents and what a study gives you on
 * the right. Nothing on this screen is invented: every row is a tool's answer about this workspace.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { Button } from "../ui";
import AgentsSummary from "../agents/AgentsSummary";
import Studies from "../study/Studies";
import { sendToAgent } from "../state/agent";
import { useWorkspace } from "../state/workspace";

interface DoctorRow {
  check: string;
  ok: boolean | null;
  detail: string;
  fix: string;
}

/** Three first messages, typed into the agent and left for the person to send or edit. */
const PROMPTS = [
  "Read openlab://guide, then report the state of every study here (study_list, study_status), using the tools, not memory.",
  "Start a temporal study called demo for the question at hand, materialise its data and run the first model once; then stop and report the result.",
  "Take the top item of the frontier of the study here and delegate it to a child agent with a $1 budget (agent_spawn); wait for it and report.",
];

/** Where a newcomer installs and signs in to the agent openlab works with; openlab ships none of its own. */
const CLAUDE_CODE = "https://code.claude.com/docs/en/quickstart";

const useDoctor = () =>
  useQuery({
    queryKey: ["doctor"],
    queryFn: () => api.get<{ ok: boolean; rows: DoctorRow[] }>("/api/doctor"),
    staleTime: 60_000,
  });

/** `openlab doctor` for this workspace, as one line: ready, or what is missing and the command that fixes it. */
function Readiness() {
  const q = useDoctor();
  if (!q.data) return <p className="start-ready is-checking">checking this install…</p>;
  // an extra is a feature somebody may never want: listed, but not "missing from this install"
  const optional = q.data.rows.filter((r) => r.ok === false && r.check.startsWith("extra: "));
  const broken = q.data.rows.filter((r) => r.ok === false && !r.check.startsWith("extra: "));
  const extras = optional.length ? (
    <span className="start-optional">
      {" "}Not installed (optional): {optional.map((r) => r.check.slice(7)).join(", ")} —{" "}
      <code>pip install "openlab-research[{optional.map((r) => r.check.slice(7)).join(",")}]"</code>
    </span>
  ) : null;
  if (!broken.length) return <p className="start-ready is-ok">✓ Ready: engine, web UI, agent CLI and MCP all found.{extras}</p>;
  return (
    <div className="start-ready is-warn">
      <p>
        {broken.length} thing{broken.length > 1 ? "s" : ""} missing from this install — the rest works without{" "}
        {broken.length > 1 ? "them" : "it"}:
      </p>
      <ul>
        {broken.map((r) => (
          <li key={r.check}>
            <strong>{r.check}</strong>: {r.detail}
            {r.fix && (
              <>
                {" "}
                <code>{r.fix}</code>
              </>
            )}
          </li>
        ))}
      </ul>
      {extras && <p>{extras}</p>}
    </div>
  );
}

export default function Start() {
  const { agentOpen, agentDisabled, toggleAgent } = useWorkspace();
  // no agent CLI: the prompts would have nothing to go to; the pane says how to install one
  const noAgent = useDoctor().data?.rows.some((r) => r.check === "agent CLI" && r.ok === false) ?? false;

  return (
    <div className="start">
      <div className="start-main">
        <h1 className="start-title">Research you can audit.</h1>
        <Readiness />
        <p className="start-lede">
          A study is a folder: the question and its plan, sealed by the first run, then the model, every run and a
          held-back test set, in a record nobody can quietly rewrite — not you, not your agents. Every agent working
          here is on one tree, with its budget, its spend, and what it waits on you for.
        </p>

        <Studies />

        <div className="start-agent">
          <h2 className="start-sub">Or hand it to your Claude Code</h2>
          <p className="start-hint">
            openlab has no agent of its own. It works with your{" "}
            <a href={CLAUDE_CODE} target="_blank" rel="noreferrer">Claude Code</a> (or another agent CLI), installed
            and signed in by you; openlab never holds a key.
          </p>
          {agentDisabled ? (
            <p className="start-hint">
              Your own agent CLI can run in a pane here, acting on this workspace through openlab&rsquo;s tools.
              This server was started with <code>OPENLAB_APP_TERMINAL=0</code>: restart without it, or run the
              agent from any terminal with the openlab MCP.
            </p>
          ) : (
            <>
              <Button onClick={toggleAgent} aria-expanded={agentOpen}>
                {agentOpen ? "Hide the agent pane" : "Open the agent pane"}
              </Button>
              <span className="start-hint">
                It runs in a pane here, acting on this workspace through openlab&rsquo;s tools — so what it reports
                was checked the same way.{" "}
                {noAgent
                  ? "None is installed here yet: install Claude Code (link above) and sign in once. These first prompts work once it is:"
                  : "Or start it with one of these; it is typed in for you to edit or send:"}
              </span>
              <ul className="start-prompts">
                {PROMPTS.map((p) => (
                  <li key={p}>
                    <button
                      type="button"
                      className="start-prompt"
                      disabled={noAgent}
                      title={noAgent ? "no agent CLI is installed: open the agent pane for how to add one" : undefined}
                      onClick={() => {
                        sendToAgent(p, false);
                        if (!agentOpen) toggleAgent();
                      }}
                    >
                      {p}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>

      <aside className="start-aside" aria-label="the agents, and what a study gives you">
        <AgentsSummary />
        <dl className="start-gets">
          <div>
            <dt>A sealed plan and counted trials</dt>
            <dd>the first run seals study.toml; a change after it is an amendment on the record, and every distinct run counts against the budget.</dd>
          </div>
          <div>
            <dt>A held-back test set a person opens</dt>
            <dd>a holdout look is a person&rsquo;s act, counted against its budget; an agent asks for it in your inbox.</dd>
          </div>
          <div>
            <dt>Claims bound to the record</dt>
            <dd>every number in the paper is bound to a run or a quote checked on its page, and the record says which agent made it.</dd>
          </div>
        </dl>
      </aside>
    </div>
  );
}
