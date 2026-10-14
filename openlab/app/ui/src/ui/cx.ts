/**
 * Conditional class names. Six lines instead of a dependency.
 *
 * Accepts the falsy values `a && "cls"` actually produces — including `0` and
 * `""`, which show up when the guard is a ReactNode or a count rather than a
 * boolean.
 */
export type ClassArg = string | number | false | null | undefined;

export function cx(...parts: ClassArg[]): string {
  return parts.filter((p): p is string => typeof p === "string" && p !== "").join(" ");
}
