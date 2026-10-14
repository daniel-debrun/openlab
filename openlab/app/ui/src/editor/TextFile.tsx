/**
 * Any other workspace file, open in the document pane as plain text: a statement
 * file (`openlab FILE.openlab` runs it), `OPENLAB.md`, a script.
 * Edits are saved through `ws_write`; a change made elsewhere (the agent, another
 * editor) arrives as a workspace event and is read again.
 */
import { useQueryClient } from "@tanstack/react-query";
import { isWorkspace, useEvents } from "../api/events";
import FileEditor from "./FileEditor";

export default function TextFile({ doc }: { doc: string }) {
  const qc = useQueryClient();
  useEvents((ev) => {
    if (isWorkspace(ev)) qc.invalidateQueries({ queryKey: ["file", doc] });
  });
  return (
    <div className="notebook">
      <FileEditor path={doc} />
    </div>
  );
}
