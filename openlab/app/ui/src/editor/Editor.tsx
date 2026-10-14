/**
 * The cell editor.
 *
 * The remount key is the cell's INDEX, not its uid — and that is deliberate, for
 * a reason worth writing down. A cell's `uid` is a hash of its kind, flags and
 * body, so it changes every time the text changes: keying on it would tear down
 * and rebuild the editor on every save, dropping the cursor mid-word on
 * Ctrl/Cmd-S. Keying on the index keeps one editor per slot for as long as that
 * slot exists.
 *
 * The cost of keying on a slot is that a structural edit — delete the cell above
 * — moves different text into the same slot without a remount. That is what the
 * sync effect below is for: when the text arriving from the server differs from
 * what is in the editor, the document is replaced. It does NOT replace while the
 * editor has focus, because the one thing worse than a stale buffer is a refetch
 * deleting a sentence out from under someone's hands.
 *
 * `onBlur` is where the save happens (plus Ctrl/Cmd-S, plus running). The file on
 * disk is the document; nothing in this component may be the only copy of your
 * text.
 */
import { useEffect, useRef } from "react";
import { EditorView } from "@codemirror/view";
import { Annotation, EditorState } from "@codemirror/state";
import { cellExtensions } from "./setup";

/**
 * "This change came from the server, not from a person."
 *
 * The sync effect below replaces the document when the file changes under us,
 * and CodeMirror's update listener cannot tell that apart from typing — so
 * every server-driven update was reported as a user edit. The cell was then
 * marked `unsaved` immediately after being saved (the badge stuck, claiming a
 * cell on disk was not), and a phantom draft was held that a later save could
 * write back over newer text. Annotating the transaction is the standard way to
 * say which is which.
 */
const FromServer = Annotation.define<boolean>();

interface Props {
  /** Remount key: the cell's slot in the notebook. */
  slot: number;
  /** The cell's text as the server last reported it. */
  text: string;
  onChange: (text: string) => void;
  onSave: () => void;
  onRun: () => void;
  onBlur: () => void;
  /** Take focus on mount: a prose cell opened from its rendering. */
  autoFocus?: boolean;
}

export default function Editor({ slot, text, onChange, onSave, onRun, onBlur, autoFocus }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const keys = useRef({ onSave, onRun });
  keys.current = { onSave, onRun };
  const changeRef = useRef(onChange);
  changeRef.current = onChange;
  const blurRef = useRef(onBlur);
  blurRef.current = onBlur;
  // the text as of the last time this editor and the server agreed
  const settled = useRef(text);

  useEffect(() => {
    if (!host.current) return;
    const state = EditorState.create({
      doc: settled.current,
      extensions: [
        ...cellExtensions(() => keys.current),
        EditorView.contentAttributes.of({ "aria-label": `source of cell ${slot}` }),
        EditorView.updateListener.of((u) => {
          if (!u.docChanged) return;
          // a replacement we dispatched ourselves is not something the user typed
          if (u.transactions.some((t) => t.annotation(FromServer))) return;
          changeRef.current(u.state.doc.toString());
        }),
        EditorView.domEventHandlers({
          blur: () => {
            blurRef.current();
            return false;
          },
        }),
      ],
    });
    const v = new EditorView({ state, parent: host.current });
    view.current = v;
    if (autoFocus) v.focus();
    return () => {
      v.destroy();
      view.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slot]);

  // the server's text changed under us (an agent edited this cell, or a
  // structural edit moved a different cell into this slot)
  useEffect(() => {
    const v = view.current;
    if (!v) {
      settled.current = text;
      return;
    }
    if (text === settled.current) return;
    settled.current = text;
    const current = v.state.doc.toString();
    if (current === text) return;
    if (v.hasFocus) return; // never rewrite what someone is typing into
    v.dispatch({
      changes: { from: 0, to: v.state.doc.length, insert: text },
      annotations: FromServer.of(true),
    });
  }, [text]);

  return <div className="cell-editor" ref={host} />;
}
