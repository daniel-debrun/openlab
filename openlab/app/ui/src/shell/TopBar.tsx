/**
 * The top bar: where you are, how the page is dressed, and whether it is still
 * listening.
 *
 * The connection dot is not decoration. This app is live — a cell can finish
 * because an agent ran it — so "the socket is down" means "what you are looking
 * at may be stale", and that has to be visible without being alarming.
 */
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { Health } from "../api/tools";
import { LinkState, watchLink } from "../api/events";
import { THEME_GLYPH, THEME_LABEL, useThemeStore } from "../state/theme";
import { Button, cx } from "../ui";
import { useWorkspace } from "../state/workspace";
import About from "./About";
import ViewTabs from "./ViewTabs";

const LINK_TITLE: Record<LinkState, string> = {
  open: "live — cell outputs arrive as they happen",
  connecting: "reconnecting…",
  closed: "not connected — this page may be showing stale output",
};

const LINK_WORD: Record<Exclude<LinkState, "open">, string> = {
  connecting: "reconnecting…",
  closed: "not connected: this page may be stale",
};

const basename = (p: string) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

export default function TopBar() {
  const { choice, cycle } = useThemeStore();
  const { agentOpen, agentDisabled, toggleAgent } = useWorkspace();
  const [link, setLink] = useState<LinkState>("closed");
  const [about, setAbout] = useState(false);
  // until it has been live once there is nothing to have lost, and the words would only flash on load
  const [wasLive, setWasLive] = useState(false);
  useEffect(() => watchLink((s) => { setLink(s); if (s === "open") setWasLive(true); }), []);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<Health>("/api/health"),
    staleTime: 60_000,
  });

  return (
    <header className="topbar">
      <span className="topbar-brand">openlab</span>
      {/* the folder's name, not its path: the path is the one string on the
          bar nobody reads twice, and it is still here, in the title */}
      <span className="topbar-root" title={`workspace: ${health.data?.root ?? "…"} — every path is relative to it`}>
        {health.data ? basename(health.data.root) : "…"}
      </span>
      <ViewTabs />
      <span className="topbar-spacer" />
      {/* Reachable from every screen, not only the empty one: half this app's
          vocabulary is invented here (carve, budget, vault) and the question
          "what does that word mean" arrives the first time you meet each word,
          not once at the start. */}
      {/* the agent is the point of the product, so it is a word on the bar,
          not a glyph in the corner of the status line */}
      <Button
        variant={agentOpen ? "default" : "ghost"}
        size="sm"
        className="topbar-agent"
        onClick={toggleAgent}
        aria-expanded={agentOpen}
        title={
          agentDisabled
            ? "this server was started without a terminal — the pane explains what to do"
            : agentOpen
              ? "hide the agent pane"
              : "show the agent pane: your own agent CLI, in this workspace"
        }
      >
        Agent{agentDisabled ? " · off" : ""}
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="topbar-about"
        onClick={() => setAbout(true)}
        title="what openlab is, and what the words on this screen mean"
      >
        {/* a bare "?" is a guess; the words are the affordance. Both ship and
            CSS picks — no JS, so it is right on the first paint at any width. */}
        <span className="topbar-about-long">Guide</span>
        <span className="topbar-about-short" aria-hidden="true">
          ?
        </span>
      </Button>
      <About open={about} onClose={() => setAbout(false)} version={health.data?.version} />
      <Button variant="ghost" size="sm" onClick={cycle} title={THEME_LABEL[choice]} aria-label={THEME_LABEL[choice]}>
        {THEME_GLYPH[choice]}
      </Button>
      {/* the words whenever it is not live: a red dot alone does not say that the page may be stale */}
      <span className={cx("link", `link-${link}`)} title={LINK_TITLE[link]} role="status">
        {link !== "open" && wasLive && <span className="link-word">{LINK_WORD[link]}</span>}
        {link === "open" && <span className="sr-only">{LINK_TITLE.open}</span>}
        <span className="link-dot" aria-hidden="true" />
      </span>
    </header>
  );
}
