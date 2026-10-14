/**
 * The loading / empty / error trifecta, once.
 *
 * `<Async>` is how every panel gets all three without a bespoke design each —
 * and how the dispatcher's teaching triple (`{error, meaning, try}`) reaches the
 * screen intact instead of being flattened into `String(e)`.
 *
 * A failed envelope is a first-class panel here, never a toast: a toast that
 * vanishes takes the "try" line with it, and the "try" line is the whole point.
 */
import { ReactNode } from "react";
import { cx } from "./cx";
import { explainError } from "./format";
import { Button } from "./basics";

/* ------------------------------------------------------------------- Empty */

interface EmptyBase {
  title: ReactNode;
  body?: ReactNode;
  className?: string;
  children?: ReactNode;
}

/**
 * An empty state must say what to do — so `action` is required unless the state
 * is genuinely terminal (a table with no rows has no call to action; an empty
 * workspace does).
 */
export type EmptyProps =
  | (EmptyBase & { variant?: "default"; action: { label: string; onClick?: () => void; href?: string } })
  | (EmptyBase & { variant: "terminal"; action?: never });

export function Empty(props: EmptyProps) {
  const { title, body, className, children } = props;
  const action = "action" in props ? props.action : undefined;
  return (
    <div className={cx("ui-empty", props.variant === "terminal" && "is-terminal", className)}>
      <div className="ui-empty-title">{title}</div>
      {body && <div className="ui-empty-body">{body}</div>}
      {children}
      {action &&
        (action.href ? (
          <a className="btn btn-primary btn-sm" href={action.href}>
            {action.label}
          </a>
        ) : (
          <Button variant="primary" size="sm" onClick={action.onClick}>
            {action.label}
          </Button>
        ))}
    </div>
  );
}

/* -------------------------------------------------------------- ErrorPanel */

export interface ErrorPanelProps {
  error: unknown;
  /** What the user was trying to do, for context above the engine's words. */
  context?: ReactNode;
  onRetry?: () => void;
  className?: string;
}

/**
 * The teaching triple, rendered as three labelled lines.
 *
 * `error` is what went wrong, `meaning` is why, `try` is what to do about it.
 * The dispatcher guarantees the shape; showing only the first line throws away
 * two thirds of the value.
 */
export function ErrorPanel({ error, context, onRetry, className }: ErrorPanelProps) {
  const e = explainError(error);
  return (
    <div className={cx("ui-error", className)} role="alert">
      {context && <div className="ui-error-context">{context}</div>}
      <div className="ui-error-msg">{e.error}</div>
      {e.meaning && (
        <div className="ui-error-line">
          <span className="ui-error-tag">meaning</span>
          {e.meaning}
        </div>
      )}
      {e.try && (
        <div className="ui-error-line">
          <span className="ui-error-tag">try</span>
          <code>{e.try}</code>
        </div>
      )}
      {onRetry && (
        <div className="ui-error-actions">
          <Button size="sm" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- Skeleton */

export type SkeletonKind = "table" | "cards" | "chart" | "lines" | "stat";

export function Skeleton({ kind = "lines", rows = 5 }: { kind?: SkeletonKind; rows?: number }) {
  if (kind === "chart") return <div className="skel skel-chart" aria-hidden="true" />;
  if (kind === "stat") {
    return (
      <div className="skel-row" aria-hidden="true">
        {Array.from({ length: 3 }).map((_, i) => (
          <div className="skel skel-stat" key={i} />
        ))}
      </div>
    );
  }
  if (kind === "cards") {
    return (
      <div className="skel-row" aria-hidden="true">
        {Array.from({ length: 3 }).map((_, i) => (
          <div className="skel skel-card" key={i} />
        ))}
      </div>
    );
  }
  return (
    <div aria-hidden="true">
      {Array.from({ length: rows }).map((_, i) => (
        <div className={cx("skel", kind === "table" ? "skel-tr" : "skel-line")} key={i} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------- Async */

/** The subset of a react-query result this needs. Keeps the primitive testable. */
export interface QueryLike<T> {
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  data: T | undefined;
  refetch?: () => void;
}

export interface AsyncProps<T> {
  query: QueryLike<T>;
  /** Treat a successful-but-empty response as empty rather than as data. */
  isEmpty?: (data: T) => boolean;
  emptyState?: ReactNode;
  skeleton?: SkeletonKind;
  skeletonRows?: number;
  context?: ReactNode;
  children: (data: T) => ReactNode;
}

export default function Async<T>({
  query,
  isEmpty,
  emptyState,
  skeleton = "lines",
  skeletonRows,
  context,
  children,
}: AsyncProps<T>) {
  if (query.isLoading) return <Skeleton kind={skeleton} rows={skeletonRows} />;
  if (query.isError) return <ErrorPanel error={query.error} context={context} onRetry={query.refetch} />;
  if (query.data === undefined) return <Skeleton kind={skeleton} rows={skeletonRows} />;
  if (isEmpty?.(query.data)) {
    return <>{emptyState ?? <Empty variant="terminal" title="nothing here yet" />}</>;
  }
  return <>{children(query.data)}</>;
}
