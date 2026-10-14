/**
 * The start screen's way into a study: every study folder in the workspace (`study_list`), with its question,
 * whether its spec is sealed, its runs and the last result's primary metric. One opens its Study view.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { runTool, studyDoc, type StudyList } from "../api/tools";
import { Badge, Button, ErrorPanel, prompt, toast } from "../ui";
import { useWorkspace } from "../state/workspace";
import { Interval } from "./parts";

export default function Studies() {
  const qc = useQueryClient();
  const open = useWorkspace((s) => s.open);
  const q = useQuery({ queryKey: ["study_list"], queryFn: () => runTool<StudyList>("study_list", {}) });

  const create = async () => {
    const folder = (await prompt({
      title: "New study",
      body: "A study is a folder: study.toml (the question, data, split, metrics and budgets, sealed by the first run), model.py, and the record of every run. It starts from the temporal template.",
      label: "Folder",
      placeholder: "my_study",
      confirmLabel: "Create",
    }))?.trim();
    if (!folder) return;
    try {
      const made = await runTool<{ folder: string }>("study_new", { folder });
      void qc.invalidateQueries({ queryKey: ["study_list"] });
      void qc.invalidateQueries({ queryKey: ["ws"] });
      open(studyDoc(made.folder));
    } catch (e) {
      toast.err(e, { context: "the study was not created" });
    }
  };

  if (q.isError) return <ErrorPanel error={q.error} context="listing the studies" onRetry={() => void q.refetch()} />;
  const studies = q.data?.studies ?? [];
  return (
    <section className="start-studies" aria-label="studies">
      <h2 className="start-sub">Studies</h2>
      {q.data && studies.length === 0 && (
        <p className="start-hint">
          None in this workspace yet. A study is a folder with study.toml, model.py and a hash-chained record of every
          plan, run, result and holdout look.
        </p>
      )}
      {studies.length > 0 && (
        <ul className="start-study-list">
          {studies.map((s) => (
            <li key={s.folder}>
              <button type="button" className="start-study" onClick={() => open(studyDoc(s.folder))}>
                <span className="start-study-head">
                  <span className="start-study-name">{s.name ?? s.folder}</span>
                  {s.error ? (
                    <Badge tone="warn">does not load</Badge>
                  ) : (
                    <Badge tone={s.sealed ? "ok" : "neutral"}>{s.sealed ? "sealed" : "not sealed"}</Badge>
                  )}
                  {!s.error && <span className="dim small">{s.runs} run{s.runs === 1 ? "" : "s"}</span>}
                  {s.last && (
                    <span className="small">
                      {s.last.metric} <Interval s={s.last} /> <span className="dim">{s.last.segment}</span>
                    </span>
                  )}
                </span>
                <span className="start-study-q">{s.error ? `${s.error}${s.meaning ? `: ${s.meaning}` : ""}` : s.question}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {q.data?.studies_withheld ? <p className="dim small">and {q.data.studies_withheld} more</p> : null}
      <Button size="sm" onClick={() => void create()}>New study…</Button>
    </section>
  );
}
