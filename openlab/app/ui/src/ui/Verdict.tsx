/**
 * The one place a verdict is drawn.
 *
 * openlab's claim is that a verdict is not an opinion, so the states are equal in
 * visual weight by construction: identical type size, weight, geometry and
 * border. They differ on three redundant channels — hue, glyph and the word —
 * never on hue alone. That redundancy is what keeps teal-vs-violet legible
 * under deuteranopia (where the two hues converge) and in greyscale print.
 *
 * `unproved` is drawn hatched and chromaless rather than in a further colour,
 * because "not proved" is the ABSENCE of a verdict, not a third opinion. It is
 * not a disproof, and the interface must never let it read as one.
 *
 * ---------------------------------------------------------------------------
 * The type below is the interesting part.
 *
 * `why` is OPTIONAL for the states that assert something and REQUIRED for the
 * three that do not — `unproved`, `inconclusive`, `error`. So `tsc` refuses to
 * compile an unexplained "not proved". An unexplained non-verdict reads as a
 * fact about the mathematics when it is usually a fact about how the statement
 * was written; the engine already returns the diagnosis (`note`, `hint`,
 * `message`), and this makes it impossible to drop on the floor.
 * ---------------------------------------------------------------------------
 */
import { ReactNode } from "react";
import { cx } from "./cx";

/** The display states. Several engine verdicts share one state — see `VERDICT_STATE`. */
export type VerdictKind =
  | "proved"
  | "supported"
  | "disproved"
  | "unproved"
  | "inconclusive"
  | "value"
  | "error";

/** Glyphs. `error` is the machine channel's mark, not a verdict's: it did not run. */
const MARK: Record<VerdictKind, string> = {
  proved: "✓",
  supported: "≈",
  disproved: "✗",
  unproved: "?",
  inconclusive: "◐",
  value: "=",
  error: "!",
};

const SPOKEN: Record<VerdictKind, string> = {
  proved: "verdict: proved",
  supported: "verdict: supported",
  disproved: "verdict: disproved",
  unproved: "no verdict — the statement was not proved",
  inconclusive: "verdict: inconclusive",
  value: "a value, not a claim",
  error: "this statement did not run",
};

/**
 * Every verdict the engine emits, and the display state it wears.
 *
 * This object is the vocabulary. A screen may not invent a fourteenth word, and
 * an unknown string degrades to `unproved` with a why that says so — saying "we
 * do not know" is honest, whereas guessing `disproved` would invent a finding.
 */
export const VERDICT_STATE: Record<string, VerdictKind> = {
  proved: "proved",
  true: "proved",
  disproved: "disproved",
  refuted: "disproved",
  false: "disproved",
  supported: "supported",
  "not proved": "unproved",
  undecided: "unproved",
  note: "unproved",
  inconclusive: "inconclusive",
  value: "value",
  solved: "value",
  bound: "value",
  error: "error",
};

interface Common {
  /** inline = in a sentence · chip = a badge · hero = a headline */
  size?: "inline" | "chip" | "hero";
  /** The word. Defaults to the engine's own verdict string, upper-cased. */
  label?: string;
  className?: string;
  children?: ReactNode;
}

export type VerdictProps =
  | (Common & { kind: "proved" | "supported" | "disproved" | "value"; why?: string })
  | (Common & { kind: "unproved" | "inconclusive" | "error"; why: string });

export default function Verdict(props: VerdictProps) {
  const { kind, size = "chip", label, className } = props;
  const why = props.why;
  return (
    <span
      className={cx("vd", `vd-${kind}`, `vd-${size}`, className)}
      title={why || undefined}
      role="status"
      aria-label={why ? `${SPOKEN[kind]} — ${why}` : SPOKEN[kind]}
    >
      <span className="vd-mark" aria-hidden="true">
        {MARK[kind]}
      </span>
      <span className="vd-word">{(label ?? kind).toUpperCase()}</span>
    </span>
  );
}

/**
 * Map an engine verdict string onto a display state, keeping the engine's own
 * word as the label.
 *
 * `note` maps to `unproved` deliberately: a note counts for nothing, so it must
 * carry exactly the weight of "not proved" — which is none.
 */
export function verdictFrom(raw: unknown, why?: string): VerdictProps {
  const word = String(raw ?? "").trim().toLowerCase();
  const kind = VERDICT_STATE[word];
  if (kind === "unproved" || kind === "inconclusive" || kind === "error") {
    return {
      kind,
      label: word,
      why:
        why ||
        (kind === "unproved"
          ? "no proof was found — this is not a disproof; name a technique (by induction, by exhaustion, witness n = …) and run it again"
          : kind === "inconclusive"
            ? "the evidence did not settle it — read the interpretation to see how the statement was read"
            : "the statement did not run"),
    };
  }
  if (kind) return { kind, label: word, why };
  return {
    kind: "unproved",
    label: word || "no verdict",
    why: word
      ? `unrecognised verdict ${JSON.stringify(word)} — shown as unproved rather than guessed`
      : "no verdict on the record yet",
  };
}

/**
 * A hypothesis test's verdict as a person reads it. The engine's `refuted` and
 * `inconclusive` are about the null, and "refuted" beside a hypothesis reads as
 * the hypothesis failing; records and agents keep the engine's words.
 */
export function testVerdict(raw: unknown, why?: string, test?: unknown): VerdictProps {
  const word = String(raw ?? "").trim().toLowerCase();
  const of = ["pearson r", "spearman rho", "chi-squared"].includes(String(test ?? "")) ? "association" : "difference";
  if (word === "refuted") {
    return {
      kind: "supported",
      label: `${of} found`,
      why: why || "the data are too unlikely under the null for it to stand at the corrected threshold",
    };
  }
  if (word === "inconclusive" || !word) {
    return {
      kind: "inconclusive",
      label: `no ${of} shown`,
      why: why || "the null was not refuted — which is not evidence that it is true; the test simply did not settle it",
    };
  }
  return verdictFrom(raw, why);
}

/**
 * The ledger's statuses. Same triad, different vocabulary: a submission
 * is `verified` only when the statements behind it carry verdicts that hold.
 */
export type SubmissionStatus = "verified" | "supported" | "recorded" | "unverified" | "measured" | "tested";

export function verdictFromStatus(status: string, reason?: string): VerdictProps {
  const s = status.trim().toLowerCase();
  if (s === "verified") return { kind: "proved", label: "verified", why: reason };
  if (s === "supported") return { kind: "supported", label: "supported", why: reason };
  // a holdout score: not proved and not merely claimed — measured on labels the session never held
  if (s === "measured") return { kind: "value", label: "measured", why: reason };
  /*
   * A hypothesis test was run. This wears the `inconclusive` state, and the
   * choice is deliberate on three counts:
   *
   *   - it must not be `proved`/`supported`. A test can refute a null; it can
   *     never establish one, and a green chip on a row whose test failed to
   *     refute would teach exactly the error the engine's own notes exist to
   *     prevent ("a large p is not evidence of equality").
   *   - it must not be `disproved` either. `tested` records that a reading was
   *     TAKEN; whether that reading refuted anything is in `reason`, and
   *     stamping the row with a refutation the reader has not read yet would
   *     put the interface's word ahead of the engine's.
   *   - it must not be `value`, which is spent on `measured` — a number the
   *     session did not choose. A test is a decision procedure, not a number.
   *
   * `inconclusive` is the one state that means "a reading was taken and it does
   * not by itself settle the claim", which is precisely what a ledger row for a
   * test is. Its state REQUIRES a `why`, so the engine's reason — verdict, p,
   * threshold, trials — cannot be dropped from the screen by accident.
   */
  if (s === "tested") {
    return {
      kind: "inconclusive",
      label: "tested",
      why:
        reason ||
        "a test was run in this session; its verdict, p and the corrected threshold are on the record",
    };
  }
  return {
    kind: "unproved",
    label: s || "recorded",
    why: reason || (s === "unverified" ? "a statement behind this answer failed or went against it"
                                       : "handed in with nothing openlab checked behind it"),
  };
}
