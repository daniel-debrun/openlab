/**
 * A draggable pane separator — with a keyboard.
 *
 * Almost every resizable layout on the web is mouse-only: a 4px strip with a
 * `mousedown` handler and no tabindex, which means the person who cannot use a
 * mouse cannot change the shape of the application. That is not an edge case
 * here — a notebook is a reading surface, and "make the reading surface bigger"
 * is one of the most ordinary things anyone will want to do.
 *
 * So this is a real `role="separator"`: focusable, with `aria-valuenow` so a
 * screen reader can say how big the pane is, arrow keys to move it, Home/End to
 * jump to the limits, and Enter or a double-click to collapse and restore. The
 * pointer path uses pointer capture, so a fast drag that leaves the strip keeps
 * resizing instead of stopping dead.
 *
 * It owns no size. The parent holds the number and passes it back in, because
 * the size is persisted and two components disagreeing about the current width
 * is how a pane ends up jumping on reload.
 */
import { useCallback, useRef } from "react";
import { cx } from "./cx";

export interface ResizerProps {
  /** `vertical` = a vertical bar you drag left/right to change a WIDTH. */
  orientation: "vertical" | "horizontal";
  value: number;
  min: number;
  max: number;
  onChange: (px: number) => void;
  /** Enter, or a double-click. Omit to make the separator resize-only. */
  onToggle?: () => void;
  /**
   * Which way the pane grows relative to the pointer.
   *
   * The inspector is on the RIGHT, so dragging its separator left makes it
   * wider — the delta is inverted. The agent drawer is at the BOTTOM, same
   * story vertically. Getting this backwards is the classic resizer bug, so it
   * is a named prop rather than a sign buried in the maths.
   */
  invert?: boolean;
  label: string;
  className?: string;
}

/** Arrow keys move by this; with shift, by five times as much. */
const STEP = 16;

export default function Resizer({
  orientation,
  value,
  min,
  max,
  onChange,
  onToggle,
  invert,
  label,
  className,
}: ResizerProps) {
  const start = useRef<{ pos: number; value: number } | null>(null);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      // a secondary button must not start a drag the user cannot see
      if (e.button !== 0) return;
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      start.current = {
        pos: orientation === "vertical" ? e.clientX : e.clientY,
        value,
      };
    },
    [orientation, value],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const s = start.current;
      if (!s) return;
      const now = orientation === "vertical" ? e.clientX : e.clientY;
      const delta = (now - s.pos) * (invert ? -1 : 1);
      onChange(s.value + delta);
    },
    [orientation, invert, onChange],
  );

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    start.current = null;
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* capture may already be gone if the pointer left the window */
    }
  }, []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      // the arrow that means "bigger" is the one pointing at the pane
      const grow = orientation === "vertical" ? "ArrowLeft" : "ArrowUp";
      const shrink = orientation === "vertical" ? "ArrowRight" : "ArrowDown";
      const step = e.shiftKey ? STEP * 5 : STEP;
      if (e.key === grow) {
        e.preventDefault();
        onChange(value + (invert ? step : -step));
      } else if (e.key === shrink) {
        e.preventDefault();
        onChange(value - (invert ? step : -step));
      } else if (e.key === "Home") {
        e.preventDefault();
        onChange(min);
      } else if (e.key === "End") {
        e.preventDefault();
        onChange(max);
      } else if ((e.key === "Enter" || e.key === " ") && onToggle) {
        e.preventDefault();
        onToggle();
      }
    },
    [orientation, invert, value, min, max, onChange, onToggle],
  );

  return (
    <div
      className={cx("rsz", `rsz-${orientation}`, className)}
      role="separator"
      tabIndex={0}
      aria-orientation={orientation}
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={max}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={onToggle}
      onKeyDown={onKeyDown}
      title={`${label} — drag, or use the arrow keys${onToggle ? "; Enter to collapse" : ""}`}
    >
      <span className="rsz-grip" aria-hidden="true" />
    </div>
  );
}
