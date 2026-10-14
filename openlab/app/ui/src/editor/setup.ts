/**
 * CodeMirror 6, deliberately plain.
 *
 * A cell body is LaTeX, markdown, SQL, a path or Python depending on its kind,
 * and guessing wrong is worse than not guessing: a highlighter that mis-parses
 * a statement teaches the reader that the statement is malformed when it is
 * not. So this is monospace text with line numbers, history, search and the two
 * keys that matter — Ctrl/Cmd-S saves, Shift-Enter runs.
 *
 * The theme is `var(--token)` throughout, so the editor flips with the app and
 * this file stays free of colour.
 */
import { EditorState, Extension } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { searchKeymap, highlightSelectionMatches } from "@codemirror/search";

const theme = EditorView.theme({
  "&": {
    backgroundColor: "var(--surface-0)",
    color: "var(--text-1)",
    fontSize: "var(--fs-3)",
    borderRadius: "var(--r-2)",
  },
  ".cm-content": { fontFamily: "var(--font-mono)", padding: "var(--sp-4) 0" },
  ".cm-gutters": {
    backgroundColor: "var(--surface-1)",
    color: "var(--text-3)",
    border: "none",
    fontFamily: "var(--font-mono)",
  },
  ".cm-activeLine": { backgroundColor: "var(--accent-wash)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--text-2)" },
  ".cm-cursor": { borderLeftColor: "var(--text-1)" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": {
    backgroundColor: "color-mix(in oklab, var(--accent) 28%, transparent)",
  },
  ".cm-selectionMatch": { backgroundColor: "var(--accent-wash)" },
  ".cm-panels": { backgroundColor: "var(--surface-2)", color: "var(--text-1)" },
  "&.cm-focused": { outline: "none" },
});

export interface Keys {
  onSave: () => void;
  onRun: () => void;
}

export function cellExtensions(keys: () => Keys): Extension[] {
  return [
    lineNumbers(),
    history(),
    highlightActiveLine(),
    highlightSelectionMatches(),
    theme,
    keymap.of([
      {
        key: "Mod-s",
        preventDefault: true,
        run: () => {
          keys().onSave();
          return true;
        },
      },
      {
        key: "Shift-Enter",
        preventDefault: true,
        run: () => {
          keys().onRun();
          return true;
        },
      },
      ...defaultKeymap,
      ...historyKeymap,
      ...searchKeymap,
      indentWithTab,
    ]),
    EditorView.lineWrapping,
    EditorState.tabSize.of(2),
  ];
}
