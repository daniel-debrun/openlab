/**
 * The only module in the UI permitted to call `toFixed` or `new Date`.
 *
 * Pure functions, no React, no DOM — so they are trivially checkable and there
 * is exactly one place where "how does a number look" is answered.
 */
import { DP, Kind, NOT_APPLICABLE, NOT_REPORTED } from "./kinds";

export interface FormatOptions {
  dp?: number;
  /** "always" prefixes a + on positives — for deltas. */
  sign?: "auto" | "always";
  /** suffix override. */
  unit?: string;
  /** what to show when the value is null/undefined. */
  nullAs?: string;
}

/** A value that is null, undefined, or an empty string — "not reported". */
export function isMissing(v: unknown): boolean {
  return v === null || v === undefined || v === "";
}

/**
 * Coerce to a finite number, or null.
 *
 * Returns `null` for both "missing" and "not a number", so callers must check
 * `isMissing` first when they need to tell those apart — which `<Num>` does,
 * because the distinction is load-bearing.
 */
export function toNumber(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/** True for a value that IS a number but not a finite one (NaN, ±Inf). */
export function isNonFinite(v: unknown): boolean {
  if (typeof v === "number") return !Number.isFinite(v);
  if (typeof v === "string") {
    const t = v.trim();
    if (t === "") return false;
    // JSON can carry these as bare tokens
    if (t === "NaN" || t === "Infinity" || t === "-Infinity") return true;
    const n = Number(t);
    return Number.isNaN(n) ? false : !Number.isFinite(n);
  }
  return false;
}

function group(intPart: string): string {
  return intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function fixed(n: number, dp: number, grouped: boolean): string {
  const s = Math.abs(n).toFixed(dp);
  if (!grouped) return s;
  const [i, f] = s.split(".");
  return f ? `${group(i)}.${f}` : group(i);
}

function withSign(n: number, body: string, sign: FormatOptions["sign"]): string {
  if (n < 0) return `−${body}`; // U+2212 minus, not a hyphen: it aligns
  if (sign === "always") return `+${body}`;
  return body;
}

/** Format a number according to its kind. Never returns "NaN" or "Infinity". */
export function fmtNumber(value: unknown, kind: Kind, opts: FormatOptions = {}): string {
  const nullAs = opts.nullAs ?? NOT_REPORTED;
  if (isMissing(value)) return nullAs;
  if (isNonFinite(value)) {
    if (typeof value === "number") {
      if (value === Infinity) return "∞";
      if (value === -Infinity) return "−∞";
    }
    if (value === "Infinity") return "∞";
    if (value === "-Infinity") return "−∞";
    return NOT_APPLICABLE;
  }

  const n = toNumber(value);
  if (n === null) return String(value); // an unparseable string renders verbatim
  const dp = opts.dp ?? DP[kind];

  switch (kind) {
    case "percent":
      return withSign(n, `${fixed(n * 100, dp, false)}%`, opts.sign);
    case "count":
      // grouped only above 10k, so "3 statements" is not ceremonial
      return withSign(n, fixed(n, 0, Math.abs(n) >= 10000), opts.sign);
    case "duration":
      return fmtDuration(n);
    case "date":
      return fmtDate(n);
    case "bytes":
      return fmtBytes(n);
    case "pvalue":
      return fmtPValue(n, dp);
    case "ratio":
    case "raw":
    default:
      return withSign(n, fixed(n, dp, Math.abs(n) >= 10000), opts.sign) + (opts.unit ?? "");
  }
}

/**
 * A p-value, at the resolution the report actually has.
 *
 * `0.000` is the lie this function exists to prevent: it reads as "the
 * probability is zero", which no test ever says. Below the last reportable
 * decimal the honest rendering is the inequality — `<0.001` — and above 1 or
 * below 0 the value is not a p-value at all and is shown verbatim so the
 * malformed number is visible rather than laundered into a plausible one.
 *
 * Note what is NOT here: no star, no bold, no threshold. openlab does not mark a
 * p-value as "significant"; the interval beside it says more, and a cutoff on
 * screen is an invitation to fish for it.
 */
export function fmtPValue(p: number, dp = DP.pvalue): string {
  if (p < 0 || p > 1) return String(p);
  const floor = Math.pow(10, -dp);
  if (p < floor) return `<${floor.toFixed(dp)}`;
  return p.toFixed(dp);
}

/** Bytes -> "1.4 kB". SI units, because a file listing is not a memory dump. */
export function fmtBytes(v: unknown): string {
  const n = toNumber(v);
  if (n === null) return isMissing(v) ? NOT_REPORTED : String(v);
  if (n < 1000) return `${Math.round(n)} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let x = n / 1000;
  let i = 0;
  while (x >= 1000 && i < units.length - 1) {
    x /= 1000;
    i++;
  }
  return `${x < 10 ? x.toFixed(1) : Math.round(x)} ${units[i]}`;
}

/** Epoch seconds, epoch millis, or an ISO string -> a Date, or null. */
export function toDate(v: unknown): Date | null {
  if (isMissing(v)) return null;
  if (typeof v === "number" && Number.isFinite(v)) {
    // the server sends epoch SECONDS (time.time()); anything past ~1e11 is millis
    return new Date(v > 1e11 ? v : v * 1000);
  }
  const s = String(v);
  if (/^\d+(\.\d+)?$/.test(s)) return toDate(Number(s));
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function fmtDate(v: unknown): string {
  const d = toDate(v);
  if (!d) return isMissing(v) ? NOT_REPORTED : String(v);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * "Thu 25 Sep" — a day header, for a list of events grouped by the day they
 * happened on.
 *
 * No year, on purpose: the journal reads downwards from today, so the year is
 * the same as the reader's until it is not, and a header carrying it on every
 * day of a week spends the widest column in the list on the least news. The
 * full timestamp is one hover away on every row under it.
 */
export function fmtDay(v: unknown): string {
  const d = toDate(v);
  if (!d) return isMissing(v) ? NOT_REPORTED : String(v);
  return d.toLocaleDateString(undefined, { weekday: "short", day: "2-digit", month: "short" });
}

/** "16:41" — the time within a day whose header already said which day it is. */
export function fmtTime(v: unknown): string {
  const d = toDate(v);
  if (!d) return isMissing(v) ? NOT_REPORTED : String(v);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/**
 * The day an event belongs to, as a key a list can group on.
 *
 * The reader's LOCAL day, which is why it is not `String(at).slice(0, 10)`: the
 * server writes its own clock's time (with its offset), so slicing the string
 * gives the server's day, not the reader's — and two events one minute apart
 * could land under two different headers.
 */
export function dayKey(v: unknown): string {
  const d = toDate(v);
  if (!d) return "";
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/** ISO-ish absolute form, for a `title=` beside a shortened one. */
export function fmtDateFull(v: unknown): string {
  const d = toDate(v);
  return d ? d.toLocaleString() : "";
}

/**
 * "2h ago". Every caller shows something that already happened, so a time
 * ahead of this clock (another machine's clock, or an old log's time with no
 * zone) reads "just now", never "in 4h".
 */
export function fmtAgo(v: unknown, now = Date.now()): string {
  const d = toDate(v);
  if (!d) return isMissing(v) ? NOT_REPORTED : String(v);
  const a = Math.max(0, Math.round((now - d.getTime()) / 1000));
  let body: string;
  if (a < 45) body = "just now";
  else if (a < 90) body = "1m";
  else if (a < 3600) body = `${Math.round(a / 60)}m`;
  else if (a < 86400) body = `${Math.round(a / 3600)}h`;
  else if (a < 86400 * 30) body = `${Math.round(a / 86400)}d`;
  else if (a < 86400 * 365) body = `${Math.round(a / (86400 * 30))}mo`;
  else body = `${Math.round(a / (86400 * 365))}y`;
  if (body === "just now") return body;
  return `${body} ago`;
}

/** Seconds -> "0.42s" / "00:47" / "1h 12m". */
export function fmtDuration(secs: unknown): string {
  const n = toNumber(secs);
  if (n === null) return isMissing(secs) ? NOT_REPORTED : String(secs);
  if (n > 0 && n < 10) return `${n.toFixed(2)}s`;
  const s = Math.max(0, Math.round(n));
  if (s < 3600) {
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Milliseconds -> the same, for a running elapsed clock. */
export function fmtElapsed(ms: number): string {
  return fmtDuration(ms / 1000);
}

/** First 8 characters of a digest — enough to recognise, short enough to read. */
export function fmtHash(v: unknown, chars = 8): string {
  if (isMissing(v)) return NOT_REPORTED;
  const s = String(v);
  return s.length <= chars ? s : s.slice(0, chars);
}

/**
 * Unwrap an error into the dispatcher's teaching triple.
 *
 * `openlab.app.dispatch` never raises: it returns `{ok:false, error, meaning,
 * try}`. That envelope is a product asset — it is the difference between
 * "something went wrong" and "here is what to do" — so flattening it into
 * `String(e)` throws away the most useful thing the workspace produced.
 */
export interface Explained {
  error: string;
  meaning?: string;
  try?: string;
  status?: number;
}

export function explainError(e: unknown): Explained {
  // fetch() rejects with a TypeError when the server is not there; "Failed to fetch" tells the reader nothing
  if (e instanceof TypeError && /fetch|network|load failed/i.test(e.message)) {
    return {
      error: "the workspace server did not answer",
      meaning: "this page could not reach `openlab app`: it has stopped, or the connection dropped. The request may not have reached it",
      try: "start `openlab app` again and reload this page; your work is the files on disk",
    };
  }
  const anyE = e as Record<string, unknown> | null | undefined;
  const body = (anyE?.body ?? anyE) as Record<string, unknown> | null | undefined;
  const status: number | undefined = typeof anyE?.status === "number" ? anyE.status : undefined;

  // FastAPI wraps HTTPException detail; dispatch returns the triple at top level
  const detail = (body?.detail ?? body) as Record<string, unknown> | string | null | undefined;
  if (detail && typeof detail === "object") {
    const err = detail.error ?? detail.message ?? detail.msg;
    if (typeof err === "string") {
      return {
        error: err,
        meaning: typeof detail.meaning === "string" ? detail.meaning : undefined,
        try: typeof detail.try === "string" ? detail.try : undefined,
        status,
      };
    }
  }
  if (typeof detail === "string") return { error: detail, status };
  if (e instanceof Error) return { error: e.message, status };
  if (typeof e === "string") return { error: e, status };
  return { error: "unexpected error", status };
}
