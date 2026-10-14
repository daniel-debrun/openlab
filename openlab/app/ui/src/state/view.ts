/**
 * Which top-level view fills the main area: the workspace (tree, document), the agents, or the tool list.
 * A person's preference, like the theme: kept in localStorage, never in the URL, because the URL's one
 * navigational fact is the open document.
 */
import { create } from "zustand";

const KEY = "openlab_view";

/** "@" is never in a workspace path, so a view id cannot collide with a document. */
export const AGENTS = "@agents";
export const TOOLS = "@tools";
const VIEWS = new Set(["", AGENTS, TOOLS]);

function stored(): string {
  const v = localStorage.getItem(KEY) ?? "";
  return VIEWS.has(v) ? v : "";             // a view this build no longer has (a dashboard, Runs) is the workspace
}

/** "" is the workspace. */
export const useView = create<{ view: string; setView: (v: string) => void }>((set) => ({
  view: stored(),
  setView: (view) => {
    localStorage.setItem(KEY, view);
    set({ view });
  },
}));
