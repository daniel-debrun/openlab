/**
 * The design system's public surface. Screens import from here, never from the
 * individual modules, so the file layout can change without touching them.
 */
export { cx } from "./cx";

export type { Kind } from "./kinds";
export { DP, KIND_BY_COLUMN, kindForColumn, NOT_APPLICABLE, NOT_REPORTED } from "./kinds";

export {
  dayKey,
  explainError,
  fmtAgo,
  fmtBytes,
  fmtDate,
  fmtDateFull,
  fmtDay,
  fmtDuration,
  fmtElapsed,
  fmtHash,
  fmtNumber,
  fmtPValue,
  fmtTime,
  isMissing,
  isNonFinite,
  toDate,
  toNumber,
} from "./format";
export type { Explained, FormatOptions } from "./format";

export { default as Num } from "./Num";
export type { NumProps } from "./Num";

export { default as Verdict, testVerdict, verdictFrom, verdictFromStatus, VERDICT_STATE } from "./Verdict";
export type { SubmissionStatus, VerdictKind, VerdictProps } from "./Verdict";


export { default as Resizer } from "./Resizer";
export type { ResizerProps } from "./Resizer";

export { Badge, Button, Callout, Card, Field, KeyValue, Stat, Tabs } from "./basics";
export type {
  BadgeProps,
  ButtonProps,
  CalloutProps,
  CardProps,
  FieldProps,
  KeyValueProps,
  StatProps,
  TabsProps,
} from "./basics";

export { Menu, MenuItem } from "./Menu";
export { GLOSSARY } from "./glossary";
export type { MenuItemProps, MenuProps } from "./Menu";

export { default as Table } from "./Table";
export type { Column, TableProps } from "./Table";

export { default as Async, Empty, ErrorPanel, Skeleton } from "./Async";
export type { AsyncProps, EmptyProps, ErrorPanelProps, QueryLike, SkeletonKind } from "./Async";

export { toast, ToastHost, useToasts } from "./Toast";
export type { Toast, ToastTone } from "./Toast";

export { confirm, DialogHost, form, prompt, Sheet } from "./Dialog";
export type { FormField, SheetProps } from "./Dialog";

export { default as ErrorBoundary } from "./ErrorBoundary";
