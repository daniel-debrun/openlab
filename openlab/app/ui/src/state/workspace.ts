/**
 * What the shell needs to remember, and nothing more.
 *
 * There is no router: the only navigational fact in this app is which document
 * is open, and that is a workspace path. It lives in the URL hash so a reload
 * and a copied link both land back on the same document — which is all a router
 * would have bought, without the router.
 *
 * The strip of OPEN documents is the second half of that sentence and it is
 * deliberately not in the hash: which document you are reading is a fact about
 * the link, which documents you happen to have open is a fact about you at this
 * screen — so a copied link carries the one and never the other. It lives
 * beside the pane sizes in `localStorage`, for the same reason they do.
 *
 * Everything else on screen is server state and belongs to react-query, not
 * here. The rule that keeps this file small: if a tool can answer it, it is not
 * stored.
 */
import { create } from "zustand";

function docFromHash(): string {
  const h = decodeURIComponent(window.location.hash.replace(/^#/, ""));
  return h || "";
}

/* ---------------------------------------------------------------- layout */
/**
 * How big the panes are, remembered across reloads.
 *
 * This is the one kind of state that breaks the "if a tool can answer it, it is
 * not stored here" rule, and it breaks it on purpose: no tool can answer "how
 * tall did this person want their agent drawer". A dragged pane that springs back
 * on reload is the single most irritating thing a resizable layout can do, and
 * RStudio users drag panes constantly.
 *
 * localStorage rather than the URL: it is a property of this person at this
 * screen, not of the document, so it must NOT travel in a copied link.
 */
const LS = "openlab.layout.v1";

interface Layout {
  agentOpen: boolean;
  agentHeight: number;
  /** The open documents, oldest first. The active one is in the hash, not here. */
  tabs: string[];
}

export const AGENT_MIN = 120;

/**
 * How many documents the strip will hold at once.
 *
 * A strip is furniture, but it is furniture with a stored size, and nothing here
 * is unbounded on principle — an app you leave open for a week of research would
 * otherwise accumulate every file you ever clicked, in a scroller nobody can
 * read and a `localStorage` entry that only grows. Opening past the cap drops
 * the OLDEST tab, never the one you just asked for.
 */
export const MAX_TABS = 20;

/**
 * The restored strip, made safe to render.
 *
 * `agentHeight` is clamped on read because a screen can shrink between visits;
 * a tab list is clamped because a *workspace* can change between visits, and a
 * tab is not a number — it is a claim that a file is there. Two kinds of
 * claim are dropped here, each one a tab that would otherwise render and then
 * error:
 *
 *   - anything that is not a non-empty string (a half-written or hand-edited
 *     value: this is a key in someone's browser, not a private format);
 *   - duplicates, because the strip is keyed by path and two tabs for one file
 *     are the same tab.
 *
 * The third — does the file still exist — cannot be answered without asking the
 * server, so it is not guessed at here: `TabStrip` sweeps the restored tabs
 * against `ws_list` once, on mount, and drops the ones the workspace does not
 * have.
 */
function clampTabs(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  for (const p of v) {
    if (typeof p === "string" && p && !out.includes(p)) out.push(p);
  }
  return out.slice(-MAX_TABS);
}

/** The strip with `doc` in it, added at the end if it was not already there. */
function withTab(tabs: string[], doc: string): string[] {
  if (!doc || tabs.includes(doc)) return tabs;
  return [...tabs, doc].slice(-MAX_TABS);
}

/** ~28vh, capped — big enough to read a few lines of output, small enough to leave the document. */
function defaultAgentHeight(): number {
  return Math.max(AGENT_MIN, Math.min(320, Math.round(window.innerHeight * 0.28)));
}

function loadLayout(): Layout {
  const fallback: Layout = {
    agentOpen: false,
    agentHeight: defaultAgentHeight(),
    tabs: [],
  };
  try {
    const raw = window.localStorage.getItem(LS);
    if (!raw) return fallback;
    const v = JSON.parse(raw) as Partial<Layout>;
    return {
      agentOpen: typeof v.agentOpen === "boolean" ? v.agentOpen : fallback.agentOpen,
      // clamped on read: a window that shrank since last time must not restore a
      // pane taller than the screen
      agentHeight: clamp(v.agentHeight ?? fallback.agentHeight, AGENT_MIN, window.innerHeight - 200),
      tabs: clampTabs(v.tabs),
    };
  } catch {
    return fallback;
  }
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(n)));
}

function saveLayout(s: Layout) {
  try {
    window.localStorage.setItem(LS, JSON.stringify(s));
  } catch {
    /* private mode, a full quota — a forgotten pane size is not worth an error */
  }
}

interface WorkspaceState {
  /** The open document's workspace-relative path, or "" for none. */
  doc: string;
  /**
   * Every document the strip is holding, oldest first, `doc` among them.
   *
   * A research session is more than one document at a time — a study, the
   * dataset it reads, the script that made the dataset — and before this
   * existed, following any one of those links threw away what you were reading.
   */
  tabs: string[];
  agentOpen: boolean;
  /** The drawer's height in px, as dragged. Persisted. */
  agentHeight: number;
  /**
   * The terminal is not available on this server.
   *
   * Learned from the socket (close 4403, or a close before it ever opened) the
   * first time the pane is opened — NOT probed on load, because opening a
   * socket to find out would be work done on behalf of a question nobody asked.
   * Session-only: the server could be restarted with the terminal on, and a
   * remembered "off" would then be a lie that survives the fix.
   */
  agentDisabled: boolean;
  /** Read a document: it becomes the active tab, and joins the strip if it is new. */
  open: (doc: string) => void;
  /** The same, without touching the hash — for the hash itself having changed. */
  adopt: (doc: string) => void;
  /** Close one tab. If it was the one you were reading, its neighbour takes over. */
  closeTab: (path: string) => void;
  /**
   * Forget tabs whose files are gone. Never the active one: the hash is the
   * user's own claim about what to show, and a missing document is worth one
   * honest error rather than silently vanishing from under them.
   */
  dropTabs: (paths: string[]) => void;
  toggleAgent: () => void;
  setAgentHeight: (px: number) => void;
  setAgentDisabled: (v: boolean) => void;
}

const layout = loadLayout();

export const useWorkspace = create<WorkspaceState>((set, get) => {
  /** Persist the layout facts after any change to one of them. */
  const remember = () => {
    const s = get();
    saveLayout({
      agentOpen: s.agentOpen,
      agentHeight: s.agentHeight,
      tabs: s.tabs,
    });
  };

  return {
    doc: docFromHash(),
    /* the link wins over the stored strip: a pasted URL opens its document, and
       the strip it lands in is the one this browser already had */
    tabs: withTab(layout.tabs, docFromHash()),
    agentOpen: layout.agentOpen,
    agentHeight: layout.agentHeight,
    agentDisabled: false,
    open: (doc) => {
      window.location.hash = doc ? encodeURIComponent(doc) : "";
      get().adopt(doc);
    },
    adopt: (doc) => {
      set((s) => ({ doc, tabs: withTab(s.tabs, doc) }));
      remember();
    },
    closeTab: (path) => {
      const s = get();
      const i = s.tabs.indexOf(path);
      if (i < 0) return;
      const tabs = s.tabs.filter((t) => t !== path);
      if (s.doc === path) {
        /* the tab to the RIGHT, then the one to the left, then nothing. Closing a
           run of tabs left to right should keep moving forward; landing on the
           left neighbour each time walks you back through what you just shut. */
        const next = tabs[i] ?? tabs[tabs.length - 1] ?? "";
        window.location.hash = next ? encodeURIComponent(next) : "";
        set({ tabs, doc: next });
      } else {
        set({ tabs });
      }
      remember();
    },
    dropTabs: (paths) => {
      const gone = new Set(paths);
      const s = get();
      const tabs = s.tabs.filter((t) => t === s.doc || !gone.has(t));
      if (tabs.length === s.tabs.length) return;
      set({ tabs });
      remember();
    },
    toggleAgent: () => {
      set((s) => ({ agentOpen: !s.agentOpen }));
      remember();
    },
    setAgentHeight: (px) => {
      set({ agentHeight: clamp(px, AGENT_MIN, Math.max(AGENT_MIN, window.innerHeight - 200)) });
      remember();
    },
    setAgentDisabled: (agentDisabled) => set({ agentDisabled }),
  };
});

/**
 * Keep the store in step with the back button. Call once, in the shell.
 *
 * It `adopt`s rather than just assigning, so a document arrived at by the back
 * button or by a pasted link is in the strip like any other — the alternative is
 * an active document with no tab of its own, which reads as no tab being active.
 */
export function watchHash(): () => void {
  const onHash = () => useWorkspace.getState().adopt(docFromHash());
  window.addEventListener("hashchange", onHash);
  return () => window.removeEventListener("hashchange", onHash);
}

