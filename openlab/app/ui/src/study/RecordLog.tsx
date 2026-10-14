/**
 * The study's flight recorder: one line per verb run on it, newest first (`study_log`): when, who, the verb, and
 * what came of it. A person's acts (holdout looks, person amendments) are set apart, because they are the ones
 * an agent cannot make; a refusal is kept, because it happened.
 */
import { useState } from "react";
import type { StudyLog } from "../api/tools";
import { Button, cx, fmtDate, fmtDateFull } from "../ui";
import { ActorBadge } from "./parts";

/** Lines shown before "show all": the log of a long study is a reference, not the page. */
const SHOWN = 15;

export default function RecordLog({ d }: { d: StudyLog }) {
  const [all, setAll] = useState(false);
  const shown = all ? d.actions : d.actions.slice(0, SHOWN);
  return (
    <>
      <ol className="study-log">
        {shown.map((a) => (
          <li key={a.id} className={cx("study-log-item", a.actor === "person" && "is-person", !a.ok && "is-refused")}>
            <time className="study-log-at" dateTime={a.at} title={fmtDateFull(a.at)}>{fmtDate(a.at)}</time>
            <ActorBadge actor={a.actor} />
            <code className="study-log-verb">{a.verb}</code>
            <span className="study-log-outcome" title={a.id}>{a.ok ? "ok" : a.summary.slice(a.verb.length + 2)}</span>
          </li>
        ))}
      </ol>
      <div className="study-foot">
        {d.actions.length > SHOWN && (
          <Button size="sm" variant="ghost" onClick={() => setAll((v) => !v)}>
            {all ? "show fewer" : `show all ${d.actions.length}`}
          </Button>
        )}
        {d.total > d.actions.length && <span className="dim small">newest {d.actions.length} of {d.total} on record</span>}
      </div>
    </>
  );
}
