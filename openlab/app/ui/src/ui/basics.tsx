/**
 * The small presentational primitives. Grouped in one module because each is a
 * dozen lines and the barrel (`./index`) makes the file layout invisible to
 * consumers anyway.
 *
 * Note what is NOT here: no verdict tones on `<Badge>`. A badge speaks the
 * SYSTEM channel (a tool finished, a tool crashed); a verdict is `<Verdict>`.
 * Keeping them in different components is what stops a disproved statement from
 * ever being rendered as a red pill.
 */
import { ButtonHTMLAttributes, ReactNode, cloneElement, isValidElement, useId } from "react";
import { cx } from "./cx";
import { Kind } from "./kinds";
import Num from "./Num";

/* ------------------------------------------------------------------ Button */

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className"> {
  variant?: "default" | "primary" | "danger" | "ghost" | "link";
  size?: "sm" | "md";
  /** Keeps the label (no width jump) and blocks the click. */
  loading?: boolean;
  /** A literal glyph — there is no icon set and we may not add one. */
  icon?: string;
  className?: string;
  children?: ReactNode;
}

export function Button({
  variant = "default",
  size = "md",
  loading,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cx("btn", `btn-${variant}`, `btn-${size}`, loading && "btn-loading", className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {icon && (
        <span className="btn-icon" aria-hidden="true">
          {icon}
        </span>
      )}
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------- Card */

export interface CardProps {
  title?: ReactNode;
  /** A quiet line under the title — where a "what this is" belongs. */
  hint?: ReactNode;
  actions?: ReactNode;
  pad?: "none" | "sm" | "md";
  className?: string;
  children?: ReactNode;
}

export function Card({ title, hint, actions, pad = "md", className, children }: CardProps) {
  return (
    <section className={cx("ui-card", `ui-card-pad-${pad}`, className)}>
      {(title || actions || hint) && (
        <header className="ui-card-head">
          <div className="ui-card-titles">
            {title && <h3 className="ui-card-title">{title}</h3>}
            {hint && <div className="ui-card-hint">{hint}</div>}
          </div>
          {actions && <div className="ui-card-actions">{actions}</div>}
        </header>
      )}
      <div className="ui-card-body">{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------- Badge */

export interface BadgeProps {
  /** The SYSTEM channel only. There is deliberately no verdict tone. */
  tone?: "neutral" | "ok" | "danger" | "warn" | "info" | "accent";
  title?: string;
  className?: string;
  children?: ReactNode;
}

export function Badge({ tone = "neutral", title, className, children }: BadgeProps) {
  return (
    <span className={cx("ui-badge", `ui-badge-${tone}`, className)} title={title}>
      {children}
    </span>
  );
}

/* -------------------------------------------------------------------- Stat */

export interface StatProps {
  label: string;
  value: number | string | null | undefined;
  kind: Kind;
  /** A qualifying line: what the number is of, over what. */
  sub?: ReactNode;
  size?: "sm" | "md" | "lg";
  dp?: number;
  className?: string;
}

export function Stat({ label, value, kind, sub, size = "md", dp, className }: StatProps) {
  return (
    <div className={cx("stat", `stat-${size}`, className)}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        <Num value={value} kind={kind} dp={dp} full />
      </div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

/* ----------------------------------------------------------------- Callout */

export interface CalloutProps {
  tone?: "info" | "warn" | "danger" | "neutral";
  title?: ReactNode;
  action?: ReactNode;
  className?: string;
  children?: ReactNode;
}

export function Callout({ tone = "info", title, action, className, children }: CalloutProps) {
  return (
    <div className={cx("callout", `callout-${tone}`, className)}>
      <div className="callout-body">
        {title && <div className="callout-title">{title}</div>}
        {children}
      </div>
      {action && <div className="callout-action">{action}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- KeyValue */

export interface KeyValueProps {
  rows: { label: ReactNode; value: ReactNode }[];
  className?: string;
}

export function KeyValue({ rows, className }: KeyValueProps) {
  return (
    <dl className={cx("ui-kv", className)}>
      {rows.map((r, i) => (
        <div className="ui-kv-row" key={i}>
          <dt>{r.label}</dt>
          <dd>{r.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------- Field */

export interface FieldProps {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  htmlFor?: string;
  className?: string;
  children?: ReactNode;
}

export function Field({ label, hint, error, htmlFor, className, children }: FieldProps) {
  // a label that names no control is read as loose text: with no `htmlFor`, the one control inside gets an id
  const auto = useId();
  const own = !htmlFor && isValidElement<{ id?: string }>(children) && !children.props.id;
  return (
    <div className={cx("ui-field", error && "ui-field-error", className)}>
      <label className="ui-field-label" htmlFor={htmlFor ?? (own ? auto : undefined)}>
        {label}
      </label>
      {own ? cloneElement(children, { id: auto }) : children}
      {hint && !error && <div className="ui-field-hint">{hint}</div>}
      {error && <div className="ui-field-err">{error}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------- Tabs */

export interface TabsProps {
  items: { id: string; label: ReactNode; badge?: ReactNode }[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
}

/** Roving tabindex + arrow keys — the accessible minimum. */
export function Tabs({ items, value, onChange, className }: TabsProps) {
  const idx = Math.max(0, items.findIndex((i) => i.id === value));
  return (
    <div className={cx("ui-tabs", className)} role="tablist">
      {items.map((it, i) => (
        <button
          key={it.id}
          role="tab"
          aria-selected={it.id === value}
          tabIndex={i === idx ? 0 : -1}
          className={cx("ui-tab", it.id === value && "is-active")}
          onClick={() => onChange(it.id)}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
              e.preventDefault();
              const d = e.key === "ArrowRight" ? 1 : -1;
              onChange(items[(idx + d + items.length) % items.length].id);
            }
          }}
        >
          {it.label}
          {it.badge != null && <span className="ui-tab-badge">{it.badge}</span>}
        </button>
      ))}
    </div>
  );
}
