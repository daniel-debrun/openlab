/**
 * The dispatch, typed.
 *
 * Every action in the workspace is one POST to /api/tool/{name} whose reply is
 * the envelope, verbatim:
 *
 *     {ok: true,  tool, result, took_ms}
 *     {ok: false, tool, error, meaning, try}
 *
 * A failed envelope is NOT an exception in the HTTP sense — the request
 * succeeded and the server is telling you something useful. `callTool` therefore
 * hands the whole envelope back, and `runTool` is the variant that throws an
 * `EnvelopeError` so react-query surfaces it through `<ErrorPanel>` with the
 * `meaning` and `try` lines intact.
 *
 * `error: "confirm"` is the destructive-tool ceremony: the first call explains
 * what would be lost, the second carries `confirm: true`. `callConfirmed` runs
 * exactly that, with the server's own `meaning` as the dialog body — the UI
 * never writes its own warning, because the tool knows what it is about to do
 * and the UI does not.
 */
import { TOKEN, api } from "./client";
import { confirm } from "../ui/Dialog";

export interface Ok<T> {
  ok: true;
  tool: string;
  result: T;
  took_ms: number;
}

export interface Fail {
  ok: false;
  tool: string;
  error: string;
  meaning?: string;
  try?: string;
}

export type Envelope<T> = Ok<T> | Fail;

/** A failed envelope, thrown. `explainError` unwraps `.body` into the triple. */
export class EnvelopeError extends Error {
  body: Fail;
  constructor(body: Fail) {
    super(body.error);
    this.body = body;
  }
}

export type Args = Record<string, unknown>;

/** Call a tool. Never throws for a refusal — the envelope is the answer. */
export function callTool<T>(name: string, args: Args = {}): Promise<Envelope<T>> {
  return api.post<Envelope<T>>(`/api/tool/${encodeURIComponent(name)}`, args);
}

/** Call a tool, throwing a failed envelope so a panel can render the triple. */
export async function runTool<T>(name: string, args: Args = {}): Promise<T> {
  const env = await callTool<T>(name, args);
  if (!env.ok) throw new EnvelopeError(env);
  return env.result;
}

/**
 * Call a tool, performing the confirm ceremony if the tool is destructive.
 *
 * Returns the envelope. A cancelled confirm returns the original refusal, so the
 * caller can distinguish "the user said no" (`error === "confirm"`) from "it
 * failed". `ask` puts the question in a person's words instead of the tool's.
 */
export async function callConfirmed<T>(name: string, args: Args = {},
                                       ask?: { title: string; body: string }): Promise<Envelope<T>> {
  const first = await callTool<T>(name, args);
  if (first.ok || first.error !== "confirm") return first;
  const agreed = await confirm({
    title: ask?.title ?? `${name} cannot be undone`,
    body: ask?.body ?? first.meaning ?? "this tool is destructive",
    confirmLabel: "Yes, do it",
    danger: true,
  });
  if (!agreed) return first;
  return callTool<T>(name, { ...args, confirm: true });
}

/**
 * Write a file the person chose into the workspace root (`POST /api/upload`, the raw bytes). The reply is an
 * envelope like a tool's; `error: "exists"` means the name is taken and `overwrite` was not given.
 */
export async function uploadFile(file: File, overwrite = false): Promise<Envelope<{ path: string; bytes: number }>> {
  const q = `path=${encodeURIComponent(file.name)}${overwrite ? "&overwrite=1" : ""}`;
  const res = await fetch(`/api/upload?${q}`, {
    method: "POST",
    headers: TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {},
    body: file,
  });
  return res.json();
}

/* ------------------------------------------------------------------ shapes */
/* The result shapes the tools return. These mirror openlab/app/tools.py; nothing
   here is derived, every field arrives on the wire. */

export interface Health {
  ok: boolean;
  version: string;
  root: string;
  tools: number;
}

export type Section = "workspace" | "meta" | "research" | "study";

export interface JsonSchema {
  type?: string;
  properties?: Record<string, { type?: string; default?: unknown }>;
  required?: string[];
}

export interface ToolSpec {
  name: string;
  section: Section | string;
  description: string;
  mutates: boolean;
  destructive: boolean;
  input_schema: JsonSchema;
}

export type EntryKind = "folder" | "file";

export interface Entry {
  path: string;
  name: string;
  kind: EntryKind;
  bytes: number | null;
}

export interface WsList {
  path: string;
  entries: Entry[];
}

export interface WsRead {
  path: string;
  text: string;
  /** the file's whole length, when the reply is a window onto it */
  total_chars?: number;
  offset?: number;
  truncated?: boolean;
  chars_withheld?: number;
  read_with?: string;
}

/**
 * A whole file, asked for explicitly (`ws_read(chars: -1)`).
 *
 * `ws_read` hands back a 200k-character window by default, because a workspace
 * holds the data as well as the text and `ws_read` of a csv is `ws_read` of
 * however many megabytes that csv is. An editor or an export must have every
 * byte — half an HTML file is not a smaller HTML file — so this screen says so
 * at the call.
 */
export const ALL_TEXT = -1;

export interface ToolsList {
  tools: ToolSpec[];
}

/**
 * `notes_list`: one line of `NOTES.md`.
 *
 * The scratchpad is a markdown file in the workspace root, so every field here
 * is something a person may have typed or deleted by hand. `at` and `doc` can
 * both be empty — a hand-written line is still a note — and `line` is where it
 * sits in the file, which is how the two edits this pair of tools does NOT do
 * (done, reworded) address one note without guessing at its text.
 */
export interface Note {
  /** `2026-09-25 16:44`, as written in the file; "" for a line nobody dated */
  at: string;
  /** the document that was open when it was written, or "" */
  doc: string;
  /** one word, without its `#`, or "" */
  tag: string;
  text: string;
  /** `- [x]`, in plain text, in the file */
  done: boolean;
  /** 1-based, in `NOTES.md` */
  line: number;
  /** the document is still there to open */
  openable?: boolean;
  /** a hand-written essay on one line, cut for the reply */
  truncated?: boolean;
}

export interface NotesList {
  path: string;
  notes: Note[];
  /** every note in the file, before `doc`, `tag` and `open_only` narrowed anything */
  total: number;
  open: number;
  done: number;
  notes_withheld?: number;
}

export interface NotesAdd {
  path: string;
  at: string;
  doc: string;
  tag: string;
  text: string;
  /** the line exactly as it was appended */
  markdown: string;
}

/** Notes the strip asks for: a scratchpad you scroll for a minute, not a database. */
export const NOTES_LAST = 60;

/* ---------------------------------------------------------------- studies */
/* A study is a folder (study.toml, model.py, .openlab/record.db). It opens in the document pane as
   `<folder>/study.toml`, so the open path stays the only navigational state, and back and reload work for free. */

/** The file that makes a folder a study. */
export const STUDY_FILE = "study.toml";

/** Runs the Study view asks `study_runs` for. */
export const STUDY_RUNS = 100;

/** Actions the Study view's record log asks `study_log` for. */
export const STUDY_LOG = 100;

export function isStudyDoc(path: string): boolean {
  return path === STUDY_FILE || path.endsWith(`/${STUDY_FILE}`);
}

/** The folder a study doc names, "" for a study at the workspace root. */
export function studyFolder(doc: string): string {
  return doc.slice(0, -STUDY_FILE.length).replace(/\/$/, "");
}

/** The doc that opens a study folder (`study_list` names the root "."). */
export function studyDoc(folder: string): string {
  return folder && folder !== "." ? `${folder}/${STUDY_FILE}` : STUDY_FILE;
}

export type Actor = "agent" | "person" | "openlab";

/** A metric's numbers, `{value, lo, hi}`, or a paired difference `{diff, lo, hi, p, better}`. */
export interface Stats {
  value?: number;
  lo?: number;
  hi?: number;
  diff?: number;
  p?: number;
  /** which side of a paired difference is better: "a" the model, "b" the baseline */
  better?: string | null;
}

export interface StudyItem {
  folder: string;
  name?: string;
  question?: string;
  sealed?: boolean;
  runs?: number;
  last?: (Stats & { metric: string; segment: string; at: string }) | null;
  /** a study.toml that does not load */
  error?: string;
  meaning?: string;
}

export interface StudyList {
  studies: StudyItem[];
  studies_withheld?: number;
}

export interface StudyDataset {
  name: string;
  rows: number;
  sha: string;
  id: string;
  coverage: {
    time: { column: string; min: string; max: string } | null;
    key: { columns: string[]; distinct: number } | null;
  };
}

export interface StudyStatus {
  name: string;
  question: string;
  mode: string;
  sealed: boolean;
  plan: string | null;
  sealed_at: string | null;
  amendments: { id: string; at: string; actor: Actor; reason: string; changed: string[] }[];
  datasets: StudyDataset[];
  trials: { used: number; budget: number };
  looks: { used: number; budget: number };
  last_result: { id: string; at: string; metrics: Record<string, Stats> } | null;
  /** how many nodes of each kind the record holds: which of the view's tabs have anything to show */
  kinds: Record<string, number>;
}

export interface StudyRun {
  id: string;
  at: string;
  actor: Actor;
  segment: "research" | "holdout";
  cached: boolean;
  /** the truncation audit's verdict: clean, look-ahead, or not audited */
  audit: string;
  seconds: number;
  note: string;
  branches_from: string | null;
  /** the primary metric only; `study_node` on the id has every metric */
  result: {
    id: string;
    metric: Stats | null;
    baselines: Record<string, Stats | null>;
    paired: Record<string, Stats | null>;
  } | null;
}

export interface StudyRuns {
  primary: string;
  higher_is_better: boolean | null;
  baselines: string[];
  runs: StudyRun[];
  total: number;
}

export interface StudyNode {
  id: string;
  seq: number;
  kind: string;
  actor: Actor;
  at: string;
  /** the body, cut to size: a long list or dict ends in a "... n more" entry */
  payload: Record<string, unknown>;
  links: { rel: string; to: string }[];
  /** links past the first 40 (`study_nodes`) */
  links_withheld?: number;
}

/** `study_nodes`: the nodes of one kind, newest first. */
export interface StudyNodes {
  nodes: StudyNode[];
  total: number;
}

export interface StudyAction {
  id: string;
  actor: Actor;
  at: string;
  /** "verb: ok" or "verb: refused: why" */
  summary: string;
  verb: string;
  ok: boolean;
}

export interface StudyLog {
  actions: StudyAction[];
  total: number;
}

export interface StudyVerify {
  ok: boolean;
  nodes: number;
  broken_at: number | null;
  head: string;
}

/* The replies of the read tools behind the Study view's tabs. Each is what the record says, derived on the
   server; the view draws it and computes no verdict of its own. */

export interface Criterion {
  id: string;
  text: string;
}

export interface StudyPrisma {
  /** the coverage node the flow is of; its body holds the verdict against the review's floor */
  coverage: string | null;
  identified: number;
  by_source: Record<string, number>;
  criteria: { include: Criterion[]; exclude: Criterion[] };
  duplicates_removed: number;
  records: number;
  screened: number;
  awaiting_person: number;
  not_screened: number;
  excluded: number;
  excluded_by_criterion: Record<string, number>;
  included: number;
  retracted: number;
  retracted_included: string[];
  earlier_only: number;
}

export interface StudySource {
  id: string;
  key: string;
  title: string;
  year: number | null;
  version: string;
  tier: string;
  retracted: boolean;
  /** the standing decision, "" when unscreened */
  decision: "" | "include" | "exclude" | "maybe";
  by: "" | Actor;
  criterion: string;
  rationale: string;
  quote: string;
  full_text: boolean;
}

export interface StudySources {
  sources: StudySource[];
  total: number;
}

export interface StudyEvidenceRow {
  id: string;
  source: string;
  key: string;
  page: number;
  section: string;
  kind: string;
  quote: string;
  fields: Record<string, unknown>;
  valid_from: string;
  at: string;
}

export interface StudyEvidence {
  evidence: StudyEvidenceRow[];
  total: number;
}

/** One number read for a lattice cell: the evidence, its source and page, and the value fields. */
export type LatticeEntry = { evidence: string; source: string; page: number } & Record<string, unknown>;

export interface StudyLattice {
  coordinates: string[];
  by: string;
  values: string[];
  columns: string[];
  rows: { coordinates: Record<string, string>; cells: Record<string, LatticeEntry[]>; gaps: string[] }[];
  cells: number;
  filled: number;
  gaps: number;
  rows_withheld?: number;
}

export interface TreeNode {
  id: string;
  kind: "question" | "hypothesis";
  text: string;
  /** answered, refuted, inconclusive, untested or out_of_scope: computed by the server */
  status: string;
  children?: TreeNode[];
  claims: string[];
  tests?: string[];
  dead_ends?: string[];
}

export interface StudyTree {
  questions: TreeNode[];
}

export interface StudyClaim {
  id: string;
  statement: string;
  state: string;
  verdict: string;
  verdict_written?: string;
  /** recorded, measured, or either +reproduced; "" when not applicable */
  level: string;
  level_written?: string;
  value: number | null;
  /** the broken assumptions that put it at risk */
  at_risk: string[];
  /** a person's last decision on it */
  approval: string | null;
  /** its last contradiction search's state: found, none_found or insufficient */
  contra: string | null;
}

export interface StudyClaims {
  claims: StudyClaim[];
  total: number;
}

/** A node of a claim's provenance chain: its line, a few facts (code, env, key, retracted), and what it rests on. */
export interface TraceNode {
  id: string;
  kind?: string;
  actor?: Actor;
  at?: string;
  summary?: string;
  /** a node met earlier in the chain is shown once, then by id */
  shown?: "above";
  links?: { rel: string; node: TraceNode }[];
  [fact: string]: unknown;
}

export interface StudyTrace {
  claim: string;
  statement: string;
  complete: boolean;
  missing: string[];
  chain: TraceNode;
}

export interface StudyTraceability {
  claims: number;
  traceable: number;
  share: number | null;
  approved: { claims: number; traceable: number; share: number | null };
  broken: { claim: string; statement: string; missing: string[] }[];
}

export interface LintFinding {
  id: string;
  rule: string;
  location: string;
  message: string;
  waivable: boolean;
  /** the claims bound in the finding's sentence: a waiver may be on one of them */
  claims: string[];
  waived_by: string | null;
}

export interface StudyPaperLint {
  ok: boolean;
  bindings: { macro: string; at: string; node: string; error: string; stale: string }[];
  findings: LintFinding[];
  waived: LintFinding[];
}

export interface FrontierItem {
  id: string;
  kind: string;
  target: string;
  targets: string[];
  why: string;
  verb: string;
  info: number;
  /** seconds */
  cost: number;
  score: number;
  /** the action that took it */
  taken: string | null;
  /** the agent it is leased to now, and until when: another agent's take is refused meanwhile */
  held_by?: { agent: string; expires: string } | null;
}

export interface StudyFrontier {
  items: FrontierItem[];
  total: number;
  budget: { wealth: number; initial: number; trials: { used: number; budget: number } };
  costs: Record<string, number>;
}

export interface Fidelity {
  id: string;
  method: string;
  page: number;
  section: string;
  evidence: string;
  result: string;
  arm: string;
  metric: string;
  paper: number;
  ours: number;
  tolerance: number;
  justification: string;
  passed: boolean;
}

export interface Replication extends Omit<Fidelity, "passed"> {
  gated_by: string[];
  lo: number;
  hi: number;
  reference: number;
  /** replicated, weaker, not_replicated or inconclusive */
  verdict: string;
  reason: string;
}

export interface StudyReplications {
  target: { id: string; source: string; key: string; title: string; code: string; kind: string; tolerance: number; max_tolerance: number };
  /** every fidelity passed: the rebuild is faithful */
  rebuilt: boolean;
  fidelity: Fidelity[];
  replications: Replication[];
}

export interface StudyForecast {
  id: string;
  at: string;
  actor: Actor;
  t: string;
  model: string;
  rows: number;
  gate: boolean;
  failed: { expect: string; ok: boolean; actual?: string; error?: string }[];
  prospective: boolean;
  label_at: string;
  file: string;
  signed: string;
  expires: string;
  outcome: { id: string; at: string; n: number; metrics: Record<string, unknown> } | null;
}

export interface StudyForecasts {
  forecasts: StudyForecast[];
  total: number;
}

/** What rests on a source: the impact of its retraction. */
export interface StudyImpact {
  source: string;
  evidence: string[];
  claims: string[];
  comparisons: string[];
  papers: string[];
  conclusions: string[];
}

export interface StudyReproduction {
  id: string;
  run: string;
  matched: boolean;
  max_abs_diff: number | null;
  /** what does not match, "" when it all does */
  differs: string;
  metrics: Record<string, { run: number; reproduced: number; diff: number }>;
  env_diff: Record<string, { now: unknown; run: unknown }>;
  sandbox: string;
  seconds: number;
  at: string;
}

export interface StudyReproductions {
  reproductions: StudyReproduction[];
  runs: number;
  reproduced: number;
}

/* --------------------------------------------------------------- proposal */
/* `propose_status`: progress per branch of the approved proposal (openlab/study/proposal.py status). */

export interface BranchBudget {
  usd?: number | null;
  minutes?: number | null;
  trials?: number | null;
}

export interface ProposalBranch {
  id: string;
  role: string;
  question: string;
  question_node: string;
  /** the branch's frontier item (leased as a whole) */
  item: string;
  /** working, held, idle, unassigned, done, stopped, killed */
  state: string;
  budget: BranchBudget;
  spent: { usd: number; minutes: number; trials: number; tokens: number };
  agents: { id: string; role: string; kind: string; status: string; parent: string; usd: number; tokens: number; minutes: number }[];
  hypotheses: { id: string; node: string; state: "tested" | "killed" | "untouched"; verdict?: string | null; by?: string | null }[];
  dead_ends: unknown[];
  stop_rule: string;
  kill: string;
  /** "not fired", "near its budget (...)", "budget spent (...)", "fired: ..." */
  stop_state: string;
  live: string[];
  held_by: { agent: string; expires: string } | null;
}

export interface ProposeStatus {
  proposal: string;
  title: string;
  question: string;
  approved_at: string;
  approved_by: string;
  amendments: number;
  budget: BranchBudget;
  spent: { usd: number; trials: number; tokens: number };
  hypotheses: { tested: number; killed: number; untouched: number };
  lead: { id: string; status: string; usd: number; minutes: number }[];
  branches: ProposalBranch[];
  trials_used: number;
}

/* ----------------------------------------------------------------- agents */
/* The agent control plane (openlab/app/agent_tools.py): `agent_tree` is the snapshot `openlab watch` draws. */

/** What an agent is doing, derived by openlab.agents.view. */
export type AgentState =
  | "working" | "idle" | "waiting" | "stuck" | "paused" | "done" | "failed" | "stopped" | "lost";

export interface AgentEvent {
  at: string;
  kind: string;
  tool: string | null;
  summary: string;
}

export interface AgentRow {
  id: string;
  parent: string | null;
  /** its place in the tree: 0 for a root (or the topmost agent of a study) */
  depth: number;
  role: string;
  kind: string;
  status: string;
  state: AgentState;
  /** STUCK, LOOP×N, OVER, WAITING, BYPASS, DEAD-END */
  flags: string[];
  /** what a person asked for: pause or stop */
  control: string | null;
  study: string | null;
  question: string | null;
  usd: number;
  usd_estimated: boolean;
  budget_usd: number | null;
  /** the largest share of a budget spent ($, minutes or turns), 0..1+ */
  frac: number;
  tokens: number;
  trials: number;
  claims: number;
  refusals: number;
  /** inbox items it filed that wait on a person */
  pending: number;
  last: AgentEvent | null;
  last_refusal: AgentEvent | null;
  /** seconds since its last event; null when it has none */
  idle_s: number | null;
  dead_end: { agent: string; text: string } | null;
}

export interface AgentDetail {
  id: string;
  /** a spawned agent's brief; null for a session a hook registered */
  prompt: string | null;
  budget: { usd?: number | null; minutes?: number | null; turns?: number | null };
  spent: Record<string, number | boolean>;
  control: string | null;
  control_reason: string | null;
  cli: string | null;
  model: string | null;
  cwd: string | null;
  created: string;
  exit_code: number | null;
  children: string[];
}

export interface AgentTree {
  agents: AgentRow[];
  pending: number;
  usd: number;
  counts: Partial<Record<AgentState, number>>;
  detail?: AgentDetail;
}

export interface AgentLog {
  events: (AgentEvent & { seq: number })[];
}

export interface InboxItem {
  id: number;
  agent: string;
  study: string;
  /** holdout, amend, approval, budget, screen, other */
  kind: string;
  args: Record<string, unknown>;
  reason: string;
  cost_note: string;
  status: "pending" | "approved" | "denied" | "cancelled";
  created: string;
  decided_by: string;
  decided_at: string;
  decision_reason: string;
  result: Record<string, unknown>;
}

export interface InboxList {
  items: InboxItem[];
  pending: number;
}

/** How often the Agents view asks again: agents write to the control store from their own processes, which the
 *  app's event socket does not see. */
export const AGENT_POLL_MS = 2500;
