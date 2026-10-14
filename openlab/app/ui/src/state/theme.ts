/**
 * Theme: follow the OS, or override it.
 *
 * "system" is the default and writes NO `data-theme` attribute, so the CSS
 * `@media (prefers-color-scheme)` block does the work. An explicit choice stamps
 * `data-theme` on `<html>`, which the token layer is built to let win in both
 * directions (see styles/tokens.css).
 *
 * Also exports a resolved "dark" | "light" for the consumers that own their own
 * theme and cannot read CSS: Vega, CodeMirror and xterm.
 */
import { useEffect } from "react";
import { create } from "zustand";

export type ThemeChoice = "system" | "dark" | "light";
export type Resolved = "dark" | "light";

const KEY = "openlab_theme";

function osPrefers(): Resolved {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function apply(choice: ThemeChoice): Resolved {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", choice);
  return choice === "system" ? osPrefers() : choice;
}

function stored(): ThemeChoice {
  const v = localStorage.getItem(KEY);
  return v === "dark" || v === "light" || v === "system" ? v : "system";
}

interface ThemeState {
  choice: ThemeChoice;
  resolved: Resolved;
  set: (c: ThemeChoice) => void;
  /** Cycle system -> dark -> light -> system, for a one-key toggle. */
  cycle: () => void;
}

const initial = stored();

export const useThemeStore = create<ThemeState>((set) => ({
  choice: initial,
  resolved: apply(initial),
  set: (choice) => {
    localStorage.setItem(KEY, choice);
    set({ choice, resolved: apply(choice) });
  },
  cycle: () =>
    set((s) => {
      const next: ThemeChoice = s.choice === "system" ? "dark" : s.choice === "dark" ? "light" : "system";
      localStorage.setItem(KEY, next);
      return { choice: next, resolved: apply(next) };
    }),
}));

/** The resolved theme, kept in sync with the OS while the choice is "system". */
export function useTheme(): Resolved {
  const { choice, resolved } = useThemeStore();
  useEffect(() => {
    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => useThemeStore.setState({ resolved: osPrefers() });
    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, [choice]);
  return resolved;
}

export const THEME_LABEL: Record<ThemeChoice, string> = {
  system: "theme: system",
  dark: "theme: dark",
  light: "theme: light",
};

export const THEME_GLYPH: Record<ThemeChoice, string> = {
  system: "◐",
  dark: "●",
  light: "○",
};
