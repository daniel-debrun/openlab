/**
 * A number, rendered at the precision it deserves.
 *
 * `kind` is REQUIRED. A number without a kind is a bug: it is the difference
 * between `0.0423` and `4.2%`, and between a mean shown to 3dp and one shown to
 * 16. The whole point of this component is that no screen decides again.
 *
 * This component FORMATS. It never computes — no aggregation, no ratios, no
 * comparisons that produce a new number. `compare` only *tints*; the threshold
 * it tints against must already have come from a tool.
 */
import { Kind } from "./kinds";
import { cx } from "./cx";
import { fmtDateFull, fmtNumber, isMissing, isNonFinite, toNumber } from "./format";

export interface NumProps {
  value: number | string | null | undefined;
  kind: Kind;
  /** Override the kind's default decimals. Use when pairing two numbers. */
  dp?: number;
  sign?: "auto" | "always";
  unit?: string;
  /** Text for a missing value. Defaults to `—`. */
  nullAs?: string;
  /**
   * Why this value is absent, shown on hover.
   *
   * Load-bearing for honesty: a null `numeric` on an openlab record does not mean
   * "missing", it means "this statement's verdict is not a number". Passing
   * that sentence here is the difference between an informative blank and a
   * suspicious one.
   */
  nullMeans?: string;
  /** Show the unrounded value on hover. Never in the glyph. */
  full?: boolean;
  /** Tint against a threshold a TOOL produced. Non-finite values never tint. */
  compare?: number | null;
  /** Higher is better (default). */
  higherIsBetter?: boolean;
  className?: string;
  title?: string;
}

export default function Num({
  value,
  kind,
  dp,
  sign,
  unit,
  nullAs,
  nullMeans,
  full,
  compare,
  higherIsBetter = true,
  className,
  title,
}: NumProps) {
  const missing = isMissing(value);
  const nonFinite = !missing && isNonFinite(value);
  const text = fmtNumber(value, kind, { dp, sign, unit, nullAs });

  let tone: "pos" | "neg" | undefined;
  if (!missing && !nonFinite && compare !== null && compare !== undefined) {
    const n = toNumber(value);
    if (n !== null) tone = (higherIsBetter ? n > compare : n < compare) ? "pos" : "neg";
  }

  const hoverParts: string[] = [];
  if (title) hoverParts.push(title);
  if (missing && nullMeans) hoverParts.push(nullMeans);
  if (nonFinite) {
    hoverParts.push(
      nullMeans ?? "not a number — the computation did not produce a finite value, which is kept on purpose",
    );
  }
  if (full && !missing && !nonFinite) {
    hoverParts.push(kind === "date" ? fmtDateFull(value) : `exact: ${String(value)}`);
  }

  return (
    <span
      className={cx(
        "num",
        (missing || nonFinite) && "num-absent",
        tone === "pos" && "num-pos",
        tone === "neg" && "num-neg",
        className,
      )}
      title={hoverParts.join(" · ") || undefined}
    >
      {text}
      {/* the glyph carries meaning for sighted users; spell it out for the rest (an aria-label on a bare span
          is not announced) */}
      {text && (missing || nonFinite) && <span className="sr-only"> {missing ? "not reported" : "not a number"}</span>}
    </span>
  );
}
