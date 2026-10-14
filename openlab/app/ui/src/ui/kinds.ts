/**
 * What a number IS, and therefore how precisely it deserves to be shown.
 *
 * The rule this file exists to enforce: never render a value more precisely
 * than it deserves. `0.4230000000000001` is not detail, it is noise, and a
 * workspace that shows sixteen significant figures of a sample mean reads as
 * amateur. Precision lives here so that no screen ever decides again.
 *
 * `KIND_BY_COLUMN` maps common column names onto kinds, so a table over an
 * arbitrary frame or tool result formats itself sensibly; anything unmapped
 * falls to "raw", which is honest rather than falsely precise.
 *
 * This module FORMATS. It never derives, aggregates or compares — the web layer
 * computes nothing; every number on screen came from a tool result.
 */

export type Kind =
  /** a plain count: rows, statements, cells. Integer. */
  | "count"
  /** a fraction on the wire (0.0423) shown as a percentage (4.2%) */
  | "percent"
  /** a dimensionless quantity: a mean, a bound, a score. 3dp. */
  | "ratio"
  /** epoch seconds, epoch millis, or an ISO string, shown absolute */
  | "date"
  /** a span in seconds */
  | "duration"
  /** a size in bytes */
  | "bytes"
  /**
   * a p-value. Its own kind because its formatting rule is not a number of
   * decimals, it is an honesty rule: below the resolution of the report it is
   * shown as `<0.001` and NEVER as `0.000`. A rounded zero claims a certainty
   * the number does not carry, and a coefficient table is exactly where that
   * claim does damage.
   */
  | "pvalue"
  /** last resort: shown as given, at 4dp if it is a number */
  | "raw";

/** Default decimal places per kind. `<Num dp>` overrides for a specific pairing. */
export const DP: Record<Kind, number> = {
  count: 0,
  percent: 1,
  ratio: 3,
  date: 0,
  duration: 0,
  bytes: 0,
  pvalue: 3,
  raw: 4,
};

/**
 * Column name -> kind.
 *
 * Deliberately small and general. This is a workspace for arbitrary data: the
 * map covers the names the tools themselves emit (`rows`, `bytes`, `seconds`,
 * `took_ms`, …) plus the handful of universally-understood ones. A column
 * nobody mapped renders at "raw" rather than pretending to a precision it has
 * not earned.
 */
export const KIND_BY_COLUMN: Record<string, Kind> = {
  // --- counts
  n: "count",
  count: "count",
  rows: "count",
  n_rows: "count",
  total: "count",
  index: "count",
  line: "count",
  cells: "count",
  columns: "count",
  statements: "count",
  verdicts: "count",
  clean: "count",
  hits: "count",
  missing: "count",
  distinct: "count",
  dropped: "count",
  attempt: "count",
  budget: "count",
  remaining: "count",
  iterations: "count",
  df_model: "count",
  df_resid: "count",

  // --- fractions on the wire, shown as percentages
  rate: "percent",
  share: "percent",
  fraction: "percent",
  ratio: "ratio",
  pct: "percent",
  percent: "percent",

  // --- dimensionless quantities
  mean: "ratio",
  std: "ratio",
  sd: "ratio",
  value: "ratio",
  numeric: "ratio",
  score: "ratio",
  min: "ratio",
  max: "ratio",
  sum: "ratio",
  median: "ratio",
  q1: "ratio",
  q3: "ratio",
  estimate: "ratio",
  se: "ratio",
  sigma: "ratio",
  // --- p-values, wherever they appear. Bare `p` is deliberately NOT here: this
  // map also types the columns of an arbitrary user frame, where `p` is as
  // likely to be a price or a probability as a test statistic. The fit table
  // passes kind="pvalue" explicitly instead.
  p_value: "pvalue",
  pvalue: "pvalue",
  f_p: "pvalue",

  // --- sizes
  bytes: "bytes",
  size: "bytes",
  nbytes: "bytes",

  // --- time
  date: "date",
  time: "date",
  timestamp: "date",
  ts: "date",
  created: "date",
  modified: "date",
  mtime: "date",
  seconds: "duration",
  duration: "duration",
  elapsed: "duration",
  took_ms: "duration",
};

/** The kind for a column name, or `fallback` when nothing is mapped. */
export function kindForColumn(name: string, fallback: Kind = "raw"): Kind {
  return KIND_BY_COLUMN[String(name).toLowerCase()] ?? fallback;
}

/**
 * Why a null is a null. `—` (not reported) and `n/a` (a real non-result) are
 * deliberately different glyphs: collapsing NaN into `—` would hide that a
 * computation failed, and a failure that leaves no trace is the one thing a
 * verified workspace may not do. See `<Num>`.
 */
export const NOT_REPORTED = "—"; // em dash
export const NOT_APPLICABLE = "n/a";
