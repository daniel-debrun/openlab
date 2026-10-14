/**
 * Toasts — for the things that are genuinely transient: "saved", "3 cells ran".
 *
 * A zustand store outside React so any module (the API layer, a socket handler)
 * can raise one without a hook, plus a `<ToastHost/>` mounted once in the shell.
 *
 * A FAILED TOOL ENVELOPE DOES NOT BELONG HERE. `{error, meaning, try}` is a
 * panel (`<ErrorPanel>`), attached to the thing that failed, because "what to do
 * about it" needs reading and a toast is by construction something that goes
 * away. `toast.err` exists for the few failures with nowhere to land — a save on
 * blur, a socket drop — and even then it is sticky.
 */
import { useEffect } from "react";
import { create } from "zustand";
import { cx } from "./cx";
import { explainError } from "./format";

export type ToastTone = "ok" | "danger" | "warn" | "info";

export interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
  meaning?: string;
  hint?: string;
  /** ms; 0 = sticky (errors are sticky — they need reading). */
  ttl: number;
  action?: { label: string; onClick: () => void };
}

interface ToastState {
  items: Toast[];
  push: (t: Omit<Toast, "id">) => number;
  dismiss: (id: number) => void;
}

let seq = 1;

export const useToasts = create<ToastState>((set) => ({
  items: [],
  push: (t) => {
    const id = seq++;
    set((s) => ({ items: [...s.items, { ...t, id }] }));
    return id;
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((i) => i.id !== id) })),
}));

/** Imperative API, usable from non-React code. */
export const toast = {
  ok(message: string, opts: { hint?: string; ttl?: number; action?: Toast["action"] } = {}) {
    return useToasts.getState().push({ tone: "ok", message, ttl: opts.ttl ?? 4000, ...opts });
  },
  info(message: string, opts: { hint?: string; ttl?: number } = {}) {
    return useToasts.getState().push({ tone: "info", message, ttl: opts.ttl ?? 4000, ...opts });
  },
  warn(message: string, opts: { hint?: string; ttl?: number } = {}) {
    return useToasts.getState().push({ tone: "warn", message, ttl: opts.ttl ?? 7000, ...opts });
  },
  /** Errors are sticky: a message that vanishes before it is read is not a message. */
  err(error: unknown, opts: { context?: string; ttl?: number } = {}) {
    const e = explainError(error);
    return useToasts.getState().push({
      tone: "danger",
      message: opts.context ? `${opts.context}: ${e.error}` : e.error,
      meaning: e.meaning,
      hint: e.try,
      ttl: opts.ttl ?? 0,
    });
  },
  dismiss(id: number) {
    useToasts.getState().dismiss(id);
  },
};

function ToastItem({ t }: { t: Toast }) {
  const dismiss = useToasts((s) => s.dismiss);
  useEffect(() => {
    if (!t.ttl) return;
    const h = window.setTimeout(() => dismiss(t.id), t.ttl);
    return () => window.clearTimeout(h);
  }, [t.id, t.ttl, dismiss]);

  return (
    <div className={cx("toast", `toast-${t.tone}`)} role={t.tone === "danger" ? "alert" : "status"}>
      <div className="toast-body">
        <div className="toast-msg">{t.message}</div>
        {t.meaning && <div className="toast-line">{t.meaning}</div>}
        {t.hint && (
          <div className="toast-line">
            <code>{t.hint}</code>
          </div>
        )}
      </div>
      {t.action && (
        <button
          className="toast-action"
          onClick={() => {
            t.action!.onClick();
            dismiss(t.id);
          }}
        >
          {t.action.label}
        </button>
      )}
      <button className="toast-close" onClick={() => dismiss(t.id)} aria-label="dismiss">
        ×
      </button>
    </div>
  );
}

export function ToastHost() {
  const items = useToasts((s) => s.items);
  if (!items.length) return null;
  return (
    <div className="toast-host" aria-live="polite">
      {items.map((t) => (
        <ToastItem t={t} key={t.id} />
      ))}
    </div>
  );
}
