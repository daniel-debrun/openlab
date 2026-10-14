/**
 * The workspace, as a tree.
 *
 * `ws_list` already sorts folders first and hides dotfiles, so this component
 * does not re-sort: what you see is the order the server gave, which is the same
 * order `openlab tool ws_list` prints. A folder fetches its own children the first
 * time it is opened, so a deep workspace costs one request per folder you
 * actually look at.
 *
 * A study's folder carries a `study` badge and opens its Study view; a file opens as text.
 */
import { ReactNode, createContext, useContext, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Entry,
  StudyList,
  WsList,
  callConfirmed,
  isStudyDoc,
  runTool,
  studyDoc,
  uploadFile,
} from "../api/tools";
import { isWorkspace, useEvents } from "../api/events";
import { Async, Badge, Button, Num, confirm, cx, toast } from "../ui";
import NotesStrip from "./NotesStrip";
import { useWorkspace } from "../state/workspace";

/** The study folders (`study_list`, one request for the workspace): such a folder's row opens its Study view. */
const Studies = createContext<Set<string>>(new Set());

function Row({ entry, depth }: { entry: Entry; depth: number }) {
  const { doc, open, closeTab } = useWorkspace();
  const qc = useQueryClient();
  const studies = useContext(Studies);
  const [expanded, setExpanded] = useState(false);

  if (entry.kind === "folder") {
    const study = studies.has(entry.path);
    return (
      <>
        <button
          className={cx("tree-row", "tree-folder", study && doc === studyDoc(entry.path) && "is-open")}
          style={{ paddingLeft: `calc(var(--sp-4) + ${depth} * var(--sp-5))` }}
          onClick={() => {
            setExpanded((e) => !e);
            if (study) open(studyDoc(entry.path));
          }}
          aria-expanded={expanded}
          title={study ? `open the study ${entry.path}` : undefined}
        >
          <span className="tree-glyph" aria-hidden="true">
            {expanded ? "▾" : "▸"}
          </span>
          <span className="tree-name">{entry.name}</span>
          {study && <Badge tone="accent">study</Badge>}
        </button>
        {expanded && <Folder path={entry.path} depth={depth + 1} />}
      </>
    );
  }

  const isStudy = isStudyDoc(entry.path);

  return (
    <div
      className={cx(
        "tree-row",
        "tree-file",
        doc === entry.path && "is-open",
      )}
      style={{ paddingLeft: `calc(var(--sp-4) + ${depth} * var(--sp-5))` }}
    >
      <button
        className="tree-open"
        onClick={() => open(entry.path)}
        title={`open ${entry.path}`}
      >
        <span className="tree-glyph" aria-hidden="true">
          {isStudy ? "◈" : "·"}
        </span>
        <span className="tree-name">{entry.name}</span>
      </button>
      <span className="tree-bytes">
        <Num value={entry.bytes} kind="bytes" nullAs="" />
      </span>
      <button
        className="tree-del"
        title={`delete ${entry.path}`}
        aria-label={`delete ${entry.name}`}
        onClick={async () => {
          const env = await callConfirmed("ws_delete", { path: entry.path }, {
            title: `Delete ${entry.name}?`,
            body: `${entry.path} is removed from the workspace. This cannot be undone.`,
          });
          if (!env.ok) {
            if (env.error !== "confirm") toast.err(env, { context: "could not delete" });
            return;
          }
          /* The tab goes with the file, and if it was the one on screen its
             neighbour takes over: a document pane still pointing at a path that
             has just been deleted is the one state the strip must never leave
             behind. `open("")` is the fallback for a file that had no tab. */
          closeTab(entry.path);
          if (useWorkspace.getState().doc === entry.path) open("");
          qc.invalidateQueries({ queryKey: ["ws"] });
        }}
      >
        ✕
      </button>
    </div>
  );
}

/** What the upload button offers: the data formats a study's recipes read (the server's UPLOAD_SUFFIXES). */
const UPLOAD_ACCEPT =
  ".csv,.tsv,.parquet,.json,.ndjson,.jsonl,.xlsx,.xlsm,.xls,.sav,.zsav,.por,.dta,.sas7bdat,.xpt,.nc,.nc4,.h5,.hdf5";

/** Whether dot-names (`.data`, `.qa`) are listed. Off by default: they are the workspace's plumbing. */
const ShowHidden = createContext(false);

function Folder({
  path,
  depth,
  emptyState,
}: {
  path: string;
  depth: number;
  /** Only the root passes one: a nested empty folder is not a dead end. */
  emptyState?: ReactNode;
}) {
  const hidden = useContext(ShowHidden);
  const list = useQuery({
    queryKey: ["ws", path, hidden],
    queryFn: () => runTool<WsList>("ws_list", { path, hidden }),
  });
  return (
    <Async
      query={list}
      skeleton="lines"
      skeletonRows={3}
      context={`listing ${path || "the workspace root"}`}
      isEmpty={(d) => !d.entries?.length}
      emptyState={emptyState ?? <div className="tree-empty">empty</div>}
    >
      {(data) => (
        <>
          {data.entries.map((e) => (
            <Row entry={e} depth={depth} key={e.path} />
          ))}
        </>
      )}
    </Async>
  );
}

export default function FileTree() {
  const qc = useQueryClient();

  // anything that mutates the workspace invalidates the whole tree; ws_list is
  // a stat() per entry, so refetching it is cheaper than reasoning about which
  // folder a tool touched
  useEvents((ev) => {
    if (!isWorkspace(ev)) return;
    qc.invalidateQueries({ queryKey: ["ws"] });
    qc.invalidateQueries({ queryKey: ["study_list"] });
  });

  const [hidden, setHidden] = useState(false);
  const [dropping, setDropping] = useState(false);
  const picker = useRef<HTMLInputElement>(null);

  /** Bring your own file: each is written into the workspace (asking before one replaces another). */
  const upload = async (files: File[]) => {
    for (const f of files) {
      try {
        let env = await uploadFile(f);
        if (!env.ok && env.error === "exists") {
          const agreed = await confirm({
            title: `Replace ${f.name}?`,
            body: `${env.meaning}. The file in the workspace is overwritten; this cannot be undone.`,
            confirmLabel: "Replace",
            danger: true,
          });
          if (!agreed) continue;
          env = await uploadFile(f, true);
        }
        if (!env.ok) {
          toast.err(env, { context: `could not upload ${f.name}` });
          continue;
        }
        toast.ok(`uploaded ${env.result.path}`);
        qc.invalidateQueries({ queryKey: ["ws"] });
      } catch (e) {
        toast.err(e, { context: `could not upload ${f.name}` });
      }
    }
  };

  const studyList = useQuery({ queryKey: ["study_list"], queryFn: () => runTool<StudyList>("study_list", {}) });
  const studies = useMemo(
    () => new Set((studyList.data?.studies ?? []).map((s) => (s.folder === "." ? "" : s.folder))),
    [studyList.data],
  );

  return (
    <nav
      className={cx("tree", dropping && "is-drop")}
      aria-label="workspace"
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        setDropping(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDropping(false);
      }}
      onDrop={(e) => {
        setDropping(false);
        if (!e.dataTransfer.files.length) return;
        e.preventDefault();
        void upload(Array.from(e.dataTransfer.files));
      }}
    >
      <div className="tree-head">
        <span className="tree-head-label">workspace</span>
        <Button
          size="sm"
          variant="ghost"
          title={hidden ? "hide dot-folders (.data, .qa)" : "show dot-folders (.data, .qa)"}
          aria-pressed={hidden}
          onClick={() => setHidden((h) => !h)}
        >
          {hidden ? "· hidden shown" : "· hidden"}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          title="upload a data file (csv, tsv, xlsx, parquet, json…) into the workspace, or drop one on this panel"
          onClick={() => picker.current?.click()}
        >
          + file
        </Button>
        <input
          ref={picker}
          type="file"
          multiple
          hidden
          accept={UPLOAD_ACCEPT}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = "";
            void upload(files);
          }}
        />
      </div>
      <div className="tree-body">
        <Studies.Provider value={studies}>
        <ShowHidden.Provider value={hidden}>
          <Folder
            path=""
            depth={0}
            /* One line. The starts live on the start screen, which is the
               largest thing on screen and where a newcomer's eye lands; the
               tree only has to say that it is empty rather than loading. */
            emptyState={<div className="tree-empty">no files yet — start a study, or drop a data file here</div>}
          />
        </ShowHidden.Provider>
        </Studies.Provider>
      </div>
      {/* The foot of the tree, and it stays in view while the tree scrolls: an
          idea arrives while you are reading something, so the place to put it
          must not be somewhere you scroll to find. */}
      <NotesStrip />
    </nav>
  );
}
