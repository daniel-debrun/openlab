/**
 * Modal dialogs — the replacement for `window.confirm` / `window.prompt`.
 *
 * Promise-returning, so a call site reads exactly like the blocking builtin it
 * replaces:
 *
 *     if (await confirm({ title: "Delete this file?", danger: true })) …
 *
 * What the builtins do not give you: a focus trap, Escape-to-cancel, a styled
 * surface that matches the app in either theme, and a `body` long enough to say
 * what the consequence actually is — which is exactly where the dispatcher's
 * `meaning` line goes when a destructive tool asks to be confirmed.
 * `window.confirm` also blocks the whole tab, which with a live event socket
 * means dropped frames.
 */
import { ReactNode, RefObject, useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { cx } from "./cx";
import { Button } from "./basics";

interface ConfirmSpec {
  title: ReactNode;
  body?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

interface PromptSpec {
  title: ReactNode;
  label?: ReactNode;
  body?: ReactNode;
  initial?: string;
  placeholder?: string;
  confirmLabel?: string;
  danger?: boolean;
  /**
   * For a person's irreversible act: OK stays off until the answer is given and this text (e.g. the study's
   * name) is typed in a second field, so the act cannot be one stray click or Enter.
   */
  typed?: string;
}

/** One answer a form asks for: a line of text, or a choice when it has `options`. */
export interface FormField {
  key: string;
  label?: ReactNode;
  /** A choice, made on purpose: it starts unchosen. */
  options?: { value: string; label: string }[];
  placeholder?: string;
  initial?: string;
  /** May be left empty; every other field must be answered before OK. */
  optional?: boolean;
  /** A passphrase: typed hidden, never echoed. */
  secret?: boolean;
}

interface FormSpec {
  title: ReactNode;
  body?: ReactNode;
  fields: FormField[];
  confirmLabel?: string;
  danger?: boolean;
  /** As for `prompt`: OK stays off until this text is typed in a last field. */
  typed?: string;
}

type Pending =
  | { kind: "confirm"; spec: ConfirmSpec; resolve: (v: boolean) => void }
  | { kind: "form"; spec: FormSpec; resolve: (v: Record<string, string> | null) => void };

interface DialogState {
  pending: Pending | null;
  open: (p: Pending) => void;
  close: () => void;
}

const useDialog = create<DialogState>((set) => ({
  pending: null,
  open: (p) => set({ pending: p }),
  close: () => set({ pending: null }),
}));

export function confirm(spec: ConfirmSpec): Promise<boolean> {
  return new Promise((resolve) => useDialog.getState().open({ kind: "confirm", spec, resolve }));
}

export function prompt(spec: PromptSpec): Promise<string | null> {
  const { label, placeholder, initial, ...rest } = spec;
  const field = { key: "text", label, placeholder, initial, optional: spec.typed === undefined };
  return form({ ...rest, fields: [field] }).then((v) => (v ? v.text : null));
}

/** Several answers at once (e.g. a decision and its reason), keyed by field; null when cancelled. */
export function form(spec: FormSpec): Promise<Record<string, string> | null> {
  return new Promise((resolve) => useDialog.getState().open({ kind: "form", spec, resolve }));
}

/** Keep focus inside the dialog while it is open. */
function useFocusTrap(ref: RefObject<HTMLDivElement | null>, onEscape: () => void) {
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const previously = document.activeElement as HTMLElement | null;
    const focusables = () =>
      Array.from(
        node.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => !el.hasAttribute("disabled"));

    focusables()[0]?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onEscape();
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusables();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      previously?.focus?.();
    };
  }, [ref, onEscape]);
}

export function DialogHost() {
  const { pending, close } = useDialog();
  const ref = useRef<HTMLDivElement>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [typed, setTyped] = useState("");

  useEffect(() => {
    if (pending?.kind === "form") setValues(Object.fromEntries(pending.spec.fields.map((f) => [f.key, f.initial ?? ""])));
    setTyped("");
  }, [pending]);

  const cancel = () => {
    if (!pending) return;
    if (pending.kind === "confirm") pending.resolve(false);
    else pending.resolve(null);
    close();
  };

  useFocusTrap(ref, cancel);

  if (!pending) return null;
  const { spec } = pending;
  const need = pending.kind === "form" ? pending.spec.typed : undefined;
  const answered = pending.kind !== "form" || pending.spec.fields.every((f) => f.optional || !!values[f.key]?.trim());
  const ready = answered && (need === undefined || typed === need);
  const set = (key: string, v: string) => setValues((o) => ({ ...o, [key]: v }));

  const accept = () => {
    if (!ready) return;
    if (pending.kind === "confirm") pending.resolve(true);
    else pending.resolve(values);
    close();
  };

  return (
    <div className="ui-scrim" onMouseDown={(e) => e.target === e.currentTarget && cancel()}>
      <div
        className="ui-dialog"
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={typeof spec.title === "string" ? spec.title : undefined}
      >
        <h3 className="ui-dialog-title">{spec.title}</h3>
        {spec.body && <div className="ui-dialog-body">{spec.body}</div>}
        {pending.kind === "form" &&
          pending.spec.fields.map((f, i) => (
            <div className="ui-dialog-field" key={f.key}>
              {f.label && <label className="ui-field-label" htmlFor={`ui-dialog-${f.key}`}>{f.label}</label>}
              {f.options ? (
                <select id={`ui-dialog-${f.key}`} value={values[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)}
                        autoFocus={i === 0}>
                  <option value="" disabled>choose…</option>
                  {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              ) : (
                <input
                  id={`ui-dialog-${f.key}`}
                  type={f.secret ? "password" : "text"}
                  autoComplete={f.secret ? "current-password" : undefined}
                  value={values[f.key] ?? ""}
                  placeholder={f.placeholder}
                  onChange={(e) => set(f.key, e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && accept()}
                  autoFocus={i === 0}
                />
              )}
            </div>
          ))}
        {need !== undefined && (
          <div className="ui-dialog-field">
            <label className="ui-field-label" htmlFor="ui-dialog-typed">
              Type <code>{need}</code> to confirm
            </label>
            <input
              id="ui-dialog-typed"
              value={typed}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && accept()}
            />
          </div>
        )}
        <div className="ui-dialog-actions">
          <Button size="sm" onClick={cancel}>
            {(pending.kind === "confirm" && pending.spec.cancelLabel) || "Cancel"}
          </Button>
          <Button
            size="sm"
            variant={spec.danger ? "danger" : "primary"}
            disabled={!ready}
            title={ready ? undefined : need === undefined ? "answer every field first" : `give the answer and type ${need} first`}
            onClick={accept}
          >
            {spec.confirmLabel ?? (pending.kind === "confirm" ? "Confirm" : "OK")}
          </Button>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- Sheet ---- */

export interface SheetProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  width?: number;
  className?: string;
  children?: ReactNode;
}

/** A right-hand panel, for a detail that a modal would over-weight. */
export function Sheet({ open, onClose, title, width = 420, className, children }: SheetProps) {
  const ref = useRef<HTMLDivElement>(null);
  const nullRef = useRef<HTMLDivElement>(null);
  useFocusTrap(open ? ref : nullRef, onClose);
  if (!open) return null;
  return (
    <div className="ui-scrim ui-scrim-right" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={cx("ui-sheet", className)} style={{ width }} ref={ref} role="dialog" aria-modal="true">
        <header className="ui-sheet-head">
          <div className="ui-sheet-title">{title}</div>
          <button className="ui-sheet-close" onClick={onClose} aria-label="close">
            ×
          </button>
        </header>
        <div className="ui-sheet-body">{children}</div>
      </div>
    </div>
  );
}
