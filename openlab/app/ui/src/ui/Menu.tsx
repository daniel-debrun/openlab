/**
 * A disclosure menu: one button, a list of actions under it.
 *
 * Holds everything that is not a primary task of its screen, so a header can
 * show one filled button and a handful of words instead of nine equal pills.
 * Native <details> does the open/close, so it works with no JS state; this
 * component adds what <details> lacks — close on Escape, on an outside click,
 * and after an item is chosen.
 */
import { ReactNode, useEffect, useRef } from "react";
import { cx } from "./cx";

export interface MenuProps {
  label: ReactNode;
  /** Where the list opens from. */
  align?: "left" | "right";
  className?: string;
  children: ReactNode;
}

export function Menu({ label, align = "right", className, children }: MenuProps) {
  const ref = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    const close = () => d.removeAttribute("open");
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && d.open) {
        e.preventDefault();
        close();
        d.querySelector("summary")?.focus();
      }
    };
    const onDown = (e: MouseEvent) => {
      if (d.open && !d.contains(e.target as Node)) close();
    };
    // a chosen item closes the menu; the item's own handler has already run
    const onClick = (e: MouseEvent) => {
      if ((e.target as HTMLElement).closest(".ui-menu-item")) close();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    d.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
      d.removeEventListener("click", onClick);
    };
  }, []);

  return (
    <details ref={ref} className={cx("ui-menu", `ui-menu-${align}`, className)}>
      <summary className="btn btn-default btn-sm ui-menu-button">
        {label}
        <span className="ui-menu-caret" aria-hidden="true">
          ▾
        </span>
      </summary>
      <div className="ui-menu-list" role="menu">
        {children}
      </div>
    </details>
  );
}

export interface MenuItemProps {
  onClick: () => void;
  /** A quiet second line: what the action does. */
  hint?: ReactNode;
  disabled?: boolean;
  /** Keeps the label and blocks the click, as a button does. */
  loading?: boolean;
  title?: string;
  children: ReactNode;
}

export function MenuItem({ onClick, hint, disabled, loading, title, children }: MenuItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      className={cx("ui-menu-item", loading && "is-loading")}
      onClick={onClick}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      title={title}
    >
      <span className="ui-menu-item-label">{children}</span>
      {hint && <span className="ui-menu-item-hint">{hint}</span>}
    </button>
  );
}
