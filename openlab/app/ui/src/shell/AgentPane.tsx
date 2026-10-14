/**
 * The agent pane: a terminal on `/ws/terminal`, running the user's own agent CLI.
 *
 * Nothing about the agent is hosted here. The CLI is theirs, signed in with
 * their own credentials, and it reaches this workspace through the openlab MCP —
 * the same dispatch the buttons above call. That is why its work shows up in the
 * workspace as it happens rather than afterwards: it is not a copy of the
 * session, it is the session.
 *
 * The socket carries raw terminal bytes as UTF-8 text frames in both directions;
 * a resize is the one structured frame, `{"resize":[cols,rows]}`.
 *
 * If the upgrade is refused the terminal is simply off, which is the default.
 * Close code 4403 says so explicitly; anything else that closes before opening
 * gets the same note, because from the user's side "there is no terminal here"
 * and "the terminal is disabled" call for exactly the same next step.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { socketUrl } from "../api/client";
import { useTheme } from "../state/theme";
import { useWorkspace } from "../state/workspace";
import { registerAgentSocket } from "../state/agent";
import { Callout } from "../ui";
import AgentActivity from "./AgentActivity";

type Status = "connecting" | "open" | "disabled" | "missing" | "closed";

/** xterm owns its own palette, so it is handed the resolved tokens. */
function themeFor(): Record<string, string> {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    background: v("--surface-0"),
    foreground: v("--text-1"),
    cursor: v("--accent"),
    cursorAccent: v("--surface-0"),
    selectionBackground: v("--surface-3"),
    black: v("--surface-2"),
    red: v("--sys-danger"),
    green: v("--sys-ok"),
    yellow: v("--sys-warn"),
    blue: v("--ch-1"),
    magenta: v("--ch-5"),
    cyan: v("--ch-3"),
    white: v("--text-2"),
    brightBlack: v("--text-3"),
    brightRed: v("--sys-danger"),
    brightGreen: v("--sys-ok"),
    brightYellow: v("--sys-warn"),
    brightBlue: v("--ch-1"),
    brightMagenta: v("--ch-7"),
    brightCyan: v("--ch-3"),
    brightWhite: v("--text-1"),
  };
}

export default function AgentPane() {
  const host = useRef<HTMLDivElement>(null);
  const term = useRef<Terminal | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const theme = useTheme();
  const setAgentDisabled = useWorkspace((s) => s.setAgentDisabled);
  /** The effect's teardown, so the disabled branch can run it early. */
  const teardown = useRef<(() => void) | null>(null);

  /** Idempotent: unmounting after this is a no-op. */
  const disposeTerminal = useCallback(() => {
    teardown.current?.();
    teardown.current = null;
  }, []);

  /* The shell needs to know, because a terminal that is off must not be given
     28vh of the screen to say so. Reported up rather than probed on load: we
     find out by opening the socket the user asked us to open, and not a moment
     earlier. */
  useEffect(() => {
    const off = status === "disabled";
    setAgentDisabled(off);
    /* And tear the terminal down when it turns out there is none. The notice
       branch below removes the host div, which used to leave a live `Terminal`
       — its renderer, its buffers and an observed node — attached to nothing
       until the drawer was closed. It cannot recover from here (the socket is
       gone), so there is nothing to keep it for.
       In the effect, not in render: render must stay free of side effects, and
       under StrictMode it runs twice. */
    if (off || status === "missing") disposeTerminal();
  }, [status, setAgentDisabled, disposeTerminal]);

  useEffect(() => {
    const node = host.current;
    if (!node) return;

    const t = new Terminal({
      fontFamily: getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim(),
      fontSize: 12,
      convertEol: true,
      cursorBlink: true,
      theme: themeFor(),
    });
    const fit = new FitAddon();
    t.loadAddon(fit);
    t.open(node);
    fit.fit();
    term.current = t;

    const ws = new WebSocket(socketUrl("/ws/terminal"));
    let opened = false;

    const sendResize = () => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ resize: [t.cols, t.rows] }));
    };

    ws.onopen = () => {
      opened = true;
      setStatus("open");
      sendResize();
      registerAgentSocket(ws);
    };
    ws.onmessage = (m) => t.write(typeof m.data === "string" ? m.data : "");
    ws.onclose = (e) => {
      registerAgentSocket(null);
      // 4403 is the server saying the terminal is off; a close before the socket
      // ever opened means the route is not there, which needs the same note; 4404: no agent CLI to run
      setStatus(e.code === 4404 ? "missing" : e.code === 4403 || !opened ? "disabled" : "closed");
    };
    ws.onerror = () => {
      if (!opened) setStatus("disabled");
    };

    const dataSub = t.onData((d) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(d);
    });

    const ro = new ResizeObserver(() => {
      try {
        fit.fit();
        sendResize();
      } catch {
        /* the terminal may already be disposed */
      }
    });
    ro.observe(node);

    const stop = () => {
      ro.disconnect();
      dataSub.dispose();
      ws.close();
      t.dispose();
      term.current = null;
    };
    teardown.current = stop;
    return () => {
      teardown.current = null;
      stop();
    };
  }, []);

  // the palette lives in CSS; push it across when the theme flips
  useEffect(() => {
    if (term.current) term.current.options.theme = themeFor();
  }, [theme]);

  /* Off: one paragraph and nothing else. No standing intro line, no terminal
     host taking up the rest of the drawer — the explanation IS the content, and
     the drawer sizes itself to it. It is only ever seen because the user opened
     the pane, which is the only moment the sentence is an answer to anything. */
  if (status === "disabled") {
    return (
      <section className="agent-pane is-notice" aria-label="agent terminal">
        <Callout tone="info" title="the terminal is off">
          This server was started with <code>OPENLAB_APP_TERMINAL=0</code>; start it without that to run your agent CLI in this pane.
          Everything else on this screen works without it — and so does the agent, from any terminal of
          your own, through the openlab MCP.
        </Callout>
      </section>
    );
  }

  if (status === "missing") {
    return (
      <section className="agent-pane is-notice" aria-label="agent terminal">
        <Callout tone="info" title="no agent installed">
          <p>
            This pane runs your own AI agent CLI in this workspace, where it works through openlab&rsquo;s tools;
            openlab has no agent of its own and holds no key. Everything else here works without one.
          </p>
          <p>
            For Claude Code: install it with <code>npm install -g @anthropic-ai/claude-code</code>, run{" "}
            <code>claude</code> once in a terminal to sign in, then reopen this pane.
          </p>
          <p>
            For another CLI: start openlab with <code>OPENLAB_AGENT_CMD="your-cli --its-flags" openlab app</code>.
          </p>
        </Callout>
      </section>
    );
  }

  return (
    <section className="agent-pane" aria-label="agent terminal">
      <AgentActivity />
      {status === "closed" && (
        <Callout tone="warn" title="the terminal closed">
          The shell exited. Reload the page to start another.
        </Callout>
      )}
      <div className={`agent-term${status === "open" ? "" : " is-idle"}`} ref={host} />
    </section>
  );
}
