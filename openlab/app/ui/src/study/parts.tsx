/**
 * What every part of the Study view draws or does the same way: a number with its interval, who acted, the
 * states nobody may miss, a standing as a verdict chip, a section, and the hooks that read the record and act on
 * it. Every read is a study tool, the same one an agent calls, and nothing here computes a verdict.
 */
import { useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { runTool, type Actor, type Stats, type StudyNode } from "../api/tools";
import { Badge, Num, Verdict, cx, toast, verdictFrom, type VerdictProps } from "../ui";

/** The audit verdict whose results cannot back a claim. */
export const LOOK_AHEAD = "look-ahead";

/** A metric as `value [lo, hi]`, or a paired difference as `+diff [lo, hi]`. `struck`: a result that does not count. */
export function Interval({ s, paired, struck }: { s: Stats | null | undefined; paired?: boolean; struck?: boolean }) {
  const v = paired ? s?.diff : s?.value;
  return (
    <span className={cx("study-interval", struck && "is-struck")}
          title={struck ? "look-ahead: this result cannot back a claim" : undefined}>
      <Num value={v} kind="ratio" sign={paired ? "always" : "auto"} />
      {v != null && s?.lo != null && s?.hi != null && (
        <span className="study-band">
          {" "}[<Num value={s.lo} kind="ratio" />, <Num value={s.hi} kind="ratio" />]
        </span>
      )}
    </span>
  );
}

const ACTOR_TONE = { person: "accent", agent: "neutral", openlab: "info" } as const;
const ACTOR_WHY = {
  person: "a person's act, made at the reviewer link or in their own terminal",
  agent: "done by an agent (or the app on its behalf)",
  openlab: "written by openlab itself",
};

export function ActorBadge({ actor }: { actor: Actor }) {
  return (
    <Badge tone={ACTOR_TONE[actor] ?? "neutral"} title={ACTOR_WHY[actor]} className="study-actor">
      {actor}
    </Badge>
  );
}

/**
 * A state nobody may miss, drawn one way everywhere: look-ahead, retracted, at risk, a failed gate, a rebuild
 * that is not faithful, a reproduction that differs. `title` says what it means for the reader.
 */
export function Alarm({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Badge tone="danger" className="study-alarm" title={title}>
      ⚠ {children}
    </Badge>
  );
}

/** The truncation audit's verdict. Look-ahead is the one nobody may miss: its result cannot back a claim. */
export function AuditBadge({ audit }: { audit: string }) {
  if (audit === LOOK_AHEAD) {
    return (
      <Alarm title="replayed with only the data known at each sampled step, model.py predicted differently: its features read the future">
        look-ahead
      </Alarm>
    );
  }
  if (audit === "clean") {
    return <Badge tone="ok" title="every sampled step replayed with only the data known then gave the same predictions">✓ clean</Badge>;
  }
  return <Badge tone={audit === "not audited" ? "neutral" : "warn"}>{audit}</Badge>;
}

/**
 * The words the record uses for where a claim, question or replication stands, and the verdict chip each wears.
 * The server decides the word; this only picks the chip. Out of scope is not a verdict and has none.
 */
const STANDING: Record<string, VerdictProps> = {
  answered: { kind: "supported", label: "answered" },
  replicated: { kind: "supported", label: "replicated" },
  not_replicated: { kind: "disproved", label: "not replicated" },
  weaker: { kind: "inconclusive", label: "weaker", why: "ours holds, but by less than the paper reports" },
  untested: { kind: "unproved", label: "untested", why: "nothing on record decides it yet" },
};

export function Standing({ word, why }: { word: string; why?: string }) {
  if (word === "out_of_scope") return <Badge tone="neutral" title={why}>out of scope</Badge>;
  const v = STANDING[word] ?? verdictFrom(word, why);
  return <Verdict {...({ ...v, why: why || v.why } as VerdictProps)} />;
}

export function Section({ title, hint, children }: { title: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <section className="study-section">
      <h3 className="study-h">{title}</h3>
      {hint && <p className="ds-hint">{hint}</p>}
      {children}
    </section>
  );
}

/** A read tool of the study, keyed under the study so any verb on it refreshes every read. */
export function useRead<T>(folder: string, tool: string, args: Record<string, unknown> = {}, enabled = true) {
  return useQuery({
    queryKey: ["study", folder, tool, args],
    queryFn: () => runTool<T>(tool, { folder, ...args }),
    enabled,
  });
}

export function useNode(folder: string, id: string | null | undefined) {
  return useQuery({
    queryKey: ["study", folder, "node", id],
    queryFn: () => runTool<StudyNode>("study_node", { folder, id }),
    enabled: !!id,
    staleTime: Infinity,                                // a record node never changes
  });
}

/**
 * A verb on the study, then the record re-read: a refusal is an action node too. Success is silent, the page
 * shows it. `busy` is the key of the verb in flight (the tool, or the caller's own for a row's button).
 */
export function useAct(folder: string) {
  const qc = useQueryClient();
  const [busy, setBusy] = useState("");
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: ["study", folder] });
    void qc.invalidateQueries({ queryKey: ["study_list"] });
  };
  const act = async <T,>(tool: string, args: Record<string, unknown> = {}, key = tool): Promise<T | undefined> => {
    setBusy(key);
    try {
      return await runTool<T>(tool, { folder, ...args });
    } catch (e) {
      toast.err(e, { context: `${tool} did not finish` });
      return undefined;
    } finally {
      setBusy("");
      refresh();
    }
  };
  return { busy, act, refresh };
}

/** Node ids, as the record writes them. */
export function Ids({ ids }: { ids: string[] }) {
  if (!ids.length) return <span className="dim">—</span>;
  return <span className="study-ids">{ids.map((i) => <code key={i}>{i}</code>)}</span>;
}
