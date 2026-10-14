/**
 * The open documents, as a strip above the document pane.
 *
 * A tab is a workspace path and nothing else, so studies and files sit
 * in the same strip — the shell already dispatches the pane on the path, which
 * means "kinds of document" never became "kinds of tab". Following a link — a
 * row in the tree, a journal line — ADDS a tab instead of replacing what you
 * were reading, and that is
 * the whole feature: a research session is several documents at once, and before
 * this the app could only ever be looking at one.
 *
 * There is no tool behind any of it, deliberately. A tab list is a property of
 * this person at this screen, exactly like a pane width, so it lives in
 * `localStorage` beside the pane widths; the ACTIVE tab stays in the URL hash,
 * where the one navigational fact already lives, so a copied link carries the
 * document and not somebody else's open files. An agent has no use for a tab:
 * `ws_list` already tells it what documents exist, and nothing it does depends
 * on which ones a human has arranged on screen.
 *
 * Two things this component owns beyond drawing:
 *
 *   - **the existence sweep.** The store's clamp can drop a tab that is not a
 *     string or not a document, but not one whose file was deleted between
 *     visits, and a restored tab that errors the moment you click it is exactly
 *     what the clamp exists to prevent. So the restored tabs are checked ONCE,
 *     on mount, against the same `ws_list` query the tree uses — so a strip of
 *     root-level tabs costs no request at all, it reads the tree's cache — and
 *     the ones the workspace does not have are forgotten.
 *   - **the live delete.** A file can also go while you are looking at it, and
 *     usually not by your hand: the agent in the drawer below deletes through the
 *     same dispatch, and says so on the event socket. A tab for a file that no
 *     longer exists is closed when that arrives.
 *
 * On ARIA: this is a real `tablist` with `aria-selected` and a roving tabindex,
 * but the tabs carry no `aria-controls`, because the pane they control is the
 * `<main>` landmark. Relabelling `main` as a `tabpanel` would trade a landmark a
 * screen-reader user navigates by for a relationship they can already hear in
 * the tab's own state.
 */
import { useEffect, useMemo, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { WsList, isStudyDoc, runTool, studyFolder } from "../api/tools";
import { isWorkspace, useEvents } from "../api/events";
import { cx } from "../ui";
import { useWorkspace } from "../state/workspace";

/** The folder part of a workspace path, "" for the root. */
function folderOf(path: string): string {
  const i = path.lastIndexOf("/");
  return i < 0 ? "" : path.slice(0, i);
}

/** The file part of a workspace path. */
function nameOf(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

/**
 * What a tab is called.
 *
 * The same rule as the tree: a file wears its file name, a study its folder's name.
 */
function labelOf(path: string): string {
  if (isStudyDoc(path)) return nameOf(studyFolder(path)) || "study";
  return nameOf(path);
}

export default function TabStrip() {
  const { doc, tabs, open, closeTab, dropTabs } = useWorkspace();
  const qc = useQueryClient();
  const activeRef = useRef<HTMLButtonElement | null>(null);

  /**
   * Labels, disambiguated only where they collide.
   *
   * `trial2/model.openlab` and `model.openlab` are one word apart and the same word
   * on a tab, which is the one case where the file name is not enough — so those
   * two, and only those two, gain their folder.
   */
  const labels = useMemo(() => {
    const seen = new Map<string, number>();
    for (const p of tabs) seen.set(labelOf(p), (seen.get(labelOf(p)) ?? 0) + 1);
    return new Map(
      tabs.map((p) => {
        const short = labelOf(p);
        const folder = folderOf(p);
        return [p, (seen.get(short) ?? 0) > 1 && folder ? `${nameOf(folder)}/${short}` : short];
      }),
    );
  }, [tabs]);

  /* The restored strip, checked against the workspace once. Deliberately not on
     every change: a tab opened just now is a file the app has just seen, and
     re-sweeping would race a new file against its own tree invalidation and
     close the document it had only this moment created. */
  const swept = useRef(false);
  useEffect(() => {
    if (swept.current) return;
    swept.current = true;
    const restored = useWorkspace.getState().tabs;
    /* A path inside a dot-folder (`.qa/…`) is only listed with `hidden`, and the
       tree's cached listing is the plain one — checking it here would read
       "absent" for a file that is simply not being shown. */
    const checkable = restored.filter((p) => !p.split("/").some((s) => s.startsWith(".")));
    const folders = Array.from(new Set(checkable.map(folderOf)));
    void (async () => {
      const gone: string[] = [];
      for (const folder of folders) {
        let entries: WsList["entries"];
        try {
          const list = await qc.fetchQuery({
            // the tree's own key, so an expanded folder is free and the root
            // listing is shared with the query already in flight beside us
            queryKey: ["ws", folder, false],
            queryFn: () => runTool<WsList>("ws_list", { path: folder, hidden: false }),
            staleTime: 10_000,
          });
          entries = list.entries ?? [];
        } catch {
          /* the folder itself may be gone, or the server may be down; a tab is
             not worth closing on a failed request, and the next reload asks
             again */
          continue;
        }
        const have = new Set(entries.map((e) => e.path));
        for (const p of checkable) if (folderOf(p) === folder && !have.has(p)) gone.push(p);
      }
      if (gone.length) dropTabs(gone);
    })();
  }, [qc, dropTabs]);

  /* Deleted while we watch — by the tree's ✕, or by the agent in the drawer,
     which goes through the same dispatch and reports on the same socket. */
  useEvents((ev) => {
    if (!isWorkspace(ev) || ev.tool !== "ws_delete") return;
    const path = ev.args?.path;
    if (typeof path === "string") closeTab(path);
  });

  /**
   * `Ctrl/Cmd-1..9`: the nth tab.
   *
   * Read from the store rather than from this render's `tabs`, so the listener
   * can be installed once and still be right after a tab is opened or closed.
   */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey) return;
      if (e.key.length !== 1 || e.key < "1" || e.key > "9") return;
      const target = useWorkspace.getState().tabs[Number(e.key) - 1];
      if (!target) return;
      e.preventDefault();
      useWorkspace.getState().open(target);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /* A tab switched from the keyboard, or opened from a link in a long tree, may
     be outside the scroller — the strip is one line and it scrolls. */
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [doc]);

  /* Nothing open, nothing to draw: an empty bar is a row of pixels that says
     only that a feature exists. */
  if (!tabs.length) return null;

  const activeIndex = Math.max(0, tabs.indexOf(doc));

  return (
    <div className="tabstrip" role="tablist" aria-label="open documents">
      {tabs.map((path, i) => {
        const active = path === doc;
        const label = labels.get(path) ?? labelOf(path);
        return (
          <div
            key={path}
            role="presentation"
            className={cx("tabstrip-tab", active && "is-active")}
            /* Middle-click closes. `mousedown` is where the default has to be
               refused: on X11 the middle button pastes the selection, and on
               every platform it starts autoscroll. */
            onMouseDown={(e) => {
              if (e.button === 1) e.preventDefault();
            }}
            onAuxClick={(e) => {
              if (e.button !== 1) return;
              e.preventDefault();
              closeTab(path);
            }}
          >
            <button
              role="tab"
              aria-selected={active}
              tabIndex={i === activeIndex ? 0 : -1}
              ref={active ? activeRef : undefined}
              className="tabstrip-open"
              title={`${path} (Delete closes it)`}
              aria-keyshortcuts="Delete"
              onClick={() => open(path)}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                  e.preventDefault();
                  const d = e.key === "ArrowRight" ? 1 : -1;
                  open(tabs[(activeIndex + d + tabs.length) % tabs.length]);
                } else if (e.key === "Delete" || e.key === "Backspace") {
                  e.preventDefault();
                  closeTab(path);
                }
              }}
            >
              <span className="tabstrip-glyph" aria-hidden="true">
                {isStudyDoc(path) ? "◈" : "◆"}
              </span>
              <span className="tabstrip-name">{label}</span>
            </button>
            {/* a pointer affordance only: a tablist may hold nothing but tabs, and from the keyboard the tab
                itself closes with Delete */}
            <button
              className="tabstrip-close"
              aria-hidden="true"
              tabIndex={-1}
              title="close this tab (or middle-click it)"
              onClick={() => closeTab(path)}
            >
              ✕
            </button>
          </div>
        );
      })}
    </div>
  );
}
