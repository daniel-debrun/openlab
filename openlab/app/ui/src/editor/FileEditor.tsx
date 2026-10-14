/**
 * A workspace text file in an editor, saved on blur and on Ctrl/Cmd-S.
 *
 * `ws_read` and `ws_write` already do this job, so nothing new was added for it
 * — but `chars: ALL_TEXT` is not optional: `ws_read` is a window by default, and
 * an editor that loaded a truncated file and then saved it would delete the
 * tail of someone's pull script. An editor reads whole or not at all.
 */
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ALL_TEXT, WsRead, runTool } from "../api/tools";
import { Async, Badge, Button, toast } from "../ui";
import Editor from "./Editor";

export default function FileEditor({
  path,
  onRun,
  runHint,
}: {
  path: string;
  /** Shift-Enter in the editor: for a generator, "regenerate". */
  onRun?: () => void;
  runHint?: string;
}) {
  const qc = useQueryClient();
  const file = useQuery({
    queryKey: ["file", path],
    queryFn: () => runTool<WsRead>("ws_read", { path, chars: ALL_TEXT }),
  });
  const [draft, setDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // a different file starts clean: a draft carried across would write one
  // file's text into another's path
  useEffect(() => setDraft(null), [path]);

  const save = async () => {
    if (draft === null || draft === file.data?.text) return;
    setSaving(true);
    try {
      await runTool("ws_write", { path, text: draft });
      setDraft(null);
      qc.invalidateQueries({ queryKey: ["file", path] });
      qc.invalidateQueries({ queryKey: ["dsrc"] });
      toast.ok(`saved ${path}`);
    } catch (e) {
      toast.err(e, { context: `could not save ${path}` });
    } finally {
      setSaving(false);
    }
  };

  const dirty = draft !== null && draft !== file.data?.text;

  return (
    <Async query={file} skeleton="lines" skeletonRows={8} context={`reading ${path}`}>
      {(data) => (
        <div className="ds-editor">
          <div className="ds-editor-head">
            <code className="ds-editor-path">{path}</code>
            {dirty && <Badge tone="warn">unsaved</Badge>}
            <span className="ds-spacer" />
            <Button size="sm" onClick={() => void save()} loading={saving} disabled={!dirty}>
              Save
            </Button>
          </div>
          <Editor
            slot={0}
            text={data.text}
            onChange={setDraft}
            onSave={() => void save()}
            onRun={() => {
              void save().then(() => onRun?.());
            }}
            onBlur={() => void save()}
          />
          {runHint && <p className="ds-hint">{runHint}</p>}
        </div>
      )}
    </Async>
  );
}
