/**
 * Notes: where an idea goes.
 *
 * A strip at the foot of the workspace tree — one input, always there, and a
 * dated list above it. `Ctrl/Cmd-I` from anywhere focuses the input, and what
 * you type is filed with the document you were looking at.
 *
 * THE PRODUCT IS THREE KEYSTROKES AND NO CONTEXT SWITCH. Ideas arrive while you
 * are reading a chart, and anything that makes you leave the chart to record
 * them loses them — so there is no dialog, no dropdown for the document, no
 * second field for a tag, and no navigation. The strip is where you already are.
 *
 * Each line has one affordance: DONE. The idea was answered, or was never going
 * to be. `- [x]` in the file, where it is legible without this app.
 *
 * The store is `NOTES.md` in the workspace root — the user's own markdown, not a
 * database and not something under `.data`. `notes_add` appends; MARKING DONE IS
 * AN EDIT TO A MARKDOWN FILE, so it is `ws_write`, deliberately, and the read
 * before it passes `chars: -1`: a windowed read followed by a whole-file write
 * is exactly how a scratchpad loses its tail.
 */
import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { NOTES_LAST, Note, NotesList, WsRead, runTool } from "../api/tools";
import { isWorkspace, useEvents } from "../api/events";
import { Button, Num, cx, dayKey, fmtDateFull, fmtDay, fmtTime, toast } from "../ui";
import { useWorkspace } from "../state/workspace";

/** The scratchpad. One path, in the workspace root, so `ws_read` and `ws_write` address the same file. */
const NOTES_PATH = "NOTES.md";

/** Notes on screen while the strip is collapsed: enough to see the last one land. */
const COLLAPSED = 3;

/**
 * A trailing `#word` is a tag; a `#` anywhere else is prose.
 *
 * The tool takes a tag and the strip has one input, so the convention has to
 * come out of what you type. Only at the END, because "run #3 failed" is a
 * sentence about run 3 and filing it under the tag `3` would be worse than
 * filing it under nothing.
 */
function splitTag(raw: string): { text: string; tag: string } {
  const m = /^(.*\S)\s+#([A-Za-z][\w-]*)$/.exec(raw.trim());
  return m ? { text: m[1], tag: m[2] } : { text: raw.trim(), tag: "" };
}

/** The document, shortened to what tells it apart — the whole path is the `title`. */
function shortDoc(doc: string): string {
  return doc.replace(/\/study\.toml$/, "");
}

export default function NotesStrip() {
  const qc = useQueryClient();
  const doc = useWorkspace((s) => s.doc);
  const [text, setText] = useState("");
  const [expanded, setExpanded] = useState(false);
  /** Done notes are history, not a to-do list: one click away, never the default. */
  const [withDone, setWithDone] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const q = useQuery({
    queryKey: ["notes", withDone],
    queryFn: () => runTool<NotesList>("notes_list", { last: NOTES_LAST, open_only: !withDone }),
  });

  /* The agent in the drawer can add a note, and a person can edit `NOTES.md` in
     another editor and save it — both arrive as a workspace event. */
  useEvents((ev) => {
    if (isWorkspace(ev)) qc.invalidateQueries({ queryKey: ["notes"] });
  });

  /**
   * `Ctrl/Cmd-I` focuses the input, from anywhere in the app.
   *
   * On `window` rather than on the strip, because the whole point is that it
   * works while you are reading a chart three panes away. It expands the list
   * too: typing into a strip that shows nothing back reads as typing into
   * nothing.
   */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey) return;
      if (e.key.toLowerCase() !== "i") return;
      e.preventDefault();
      setExpanded(true);
      input.current?.focus();
      input.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const add = async () => {
    const { text: said, tag } = splitTag(text);
    if (!said) return;
    setBusy(true);
    try {
      /* The document as it is NOW: the note is about what you were looking at,
         which is the one fact the strip can supply and you would have to type. */
      await runTool("notes_add", { text: said, doc, tag });
      setText("");
      qc.invalidateQueries({ queryKey: ["notes"] });
    } catch (e) {
      toast.err(e, { context: "could not write the note" });
    } finally {
      setBusy(false);
      input.current?.focus();
    }
  };

  /**
   * Flip a note's box in `NOTES.md`.
   *
   * Addressed by line, then checked before it is written: the file is the user's
   * own, so it may have been reordered or edited between the list and the click,
   * and a rewrite of a line we can no longer recognise would put `[x]` against
   * somebody else's idea. Unrecognised is a refetch, never a guess.
   */
  const mark = async (n: Note, done: boolean) => {
    const file = await runTool<WsRead>("ws_read", { path: NOTES_PATH, chars: -1 });
    const lines = file.text.split("\n");
    const at = n.line - 1;
    const line = lines[at] ?? "";
    if (!/^\s*[-*]\s*\[[ xX]\]/.test(line) || !line.includes(n.text.slice(0, 24))) {
      toast.warn("NOTES.md has moved on", { hint: "the list is reloaded — try that note again" });
      qc.invalidateQueries({ queryKey: ["notes"] });
      return;
    }
    lines[at] = line.replace(/\[[ xX]\]/, done ? "[x]" : "[ ]");
    await runTool("ws_write", { path: file.path, text: lines.join("\n") });
    qc.invalidateQueries({ queryKey: ["notes"] });
  };

  const done = async (n: Note) => {
    setBusy(true);
    try {
      await mark(n, !n.done);
    } catch (e) {
      toast.err(e, { context: "could not mark the note" });
    } finally {
      setBusy(false);
    }
  };

  const notes = q.data?.notes ?? [];
  const shown = expanded ? notes : notes.slice(0, COLLAPSED);
  const today = dayKey(new Date().toISOString());

  return (
    <section className="notes" aria-label="notes">
      <div className="notes-head">
        <button
          className="notes-toggle"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          title={expanded ? "show fewer" : "show the whole list"}
        >
          <span className="tree-glyph" aria-hidden="true">
            {expanded ? "▾" : "▸"}
          </span>
          <span className="tree-head-label">notes</span>
          {q.data ? <Num value={q.data.open} kind="count" /> : null}
        </button>
        {q.data?.done ? (
          <button
            className={cx("chip", withDone && "is-active")}
            onClick={() => setWithDone((v) => !v)}
            title={`${q.data.done} done — kept in NOTES.md, out of the way`}
          >
            done
          </button>
        ) : null}
      </div>

      {/* A scratchpad that cannot be read is worth one quiet line: the input still
          works, because an append does not depend on the list. */}
      {q.isError ? <div className="tree-empty">NOTES.md could not be read</div> : null}

      {shown.length > 0 && (
        <ul className="notes-list">
          {shown.map((n) => (
            <li className={cx("note", n.done && "is-done")} key={`${n.line}:${n.at}`}>
              <div className="note-line">
                <span className="note-when" title={fmtDateFull(n.at) || "no date on this line"}>
                  {!n.at ? "—" : dayKey(n.at) === today ? fmtTime(n.at) : fmtDay(n.at)}
                </span>
                {n.doc ? (
                  <span className="note-doc" title={n.doc}>
                    {shortDoc(n.doc)}
                  </span>
                ) : null}
                {n.tag ? <span className="note-tag">#{n.tag}</span> : null}
                <span className="note-text">{n.text}</span>
              </div>
              <div className="note-acts">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy}
                  title={n.done ? "open it again" : "done — `- [x]` in NOTES.md"}
                  onClick={() => void done(n)}
                >
                  {n.done ? "reopen" : "done"}
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {q.data && notes.length > shown.length ? (
        <button className="notes-more" onClick={() => setExpanded(true)}>
          <Num value={notes.length - shown.length} kind="count" /> more
        </button>
      ) : null}

      <form
        className="notes-form"
        onSubmit={(e) => {
          e.preventDefault();
          void add();
        }}
      >
        <input
          ref={input}
          className="notes-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") input.current?.blur();
          }}
          placeholder={doc ? `note on ${shortDoc(doc)}…` : "note…"}
          aria-label={doc ? `a note about ${doc}` : "a note"}
          title="Ctrl/Cmd-I from anywhere. A trailing #tag is a tag; Enter files it."
          disabled={busy}
        />
      </form>
    </section>
  );
}
