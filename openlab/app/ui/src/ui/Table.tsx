/**
 * A column-spec table — the one place alignment, precision and null-rendering
 * are decided for tabular data.
 *
 * A column declares its `kind`; that drives precision (via `<Num>`), right
 * alignment and tabular figures. `tone` returns a token NAME, never a colour,
 * so a table cannot introduce a palette.
 */
import { ReactNode, useMemo, useState } from "react";
import { cx } from "./cx";
import { Kind, kindForColumn } from "./kinds";
import Num from "./Num";
import { toNumber } from "./format";
import { Empty } from "./Async";

/** Kinds that are numbers, and therefore right-aligned in a column. */
function isNumeric(kind: Kind): boolean {
  return (
    kind === "count" ||
    kind === "percent" ||
    kind === "ratio" ||
    kind === "bytes" ||
    kind === "duration" ||
    kind === "pvalue"
  );
}

export interface Column<Row> {
  key: string;
  header: ReactNode;
  kind: Kind;
  dp?: number;
  width?: number;
  /** Pull the value out of the row; defaults to `row[key]`. */
  value?: (row: Row) => unknown;
  /** Render something other than a number (a link, a badge, a verdict). */
  render?: (row: Row) => ReactNode;
  /** A token name — "pos" | "neg" | "muted" — never a colour. */
  tone?: (row: Row) => "pos" | "neg" | "muted" | undefined;
  sortable?: boolean;
  title?: string;
}

export interface TableProps<Row> {
  columns: Column<Row>[];
  rows: Row[];
  /** The full count when `rows` is a page of it. */
  total?: number;
  truncated?: boolean;
  note?: ReactNode;
  empty?: ReactNode;
  dense?: boolean;
  sticky?: boolean;
  maxHeight?: number;
  rowKey?: (row: Row, i: number) => string;
  onRowClick?: (row: Row) => void;
  /** Cap the visible rows with a "show all" toggle. */
  collapseTo?: number;
  /**
   * Pin the first column while the rest scrolls sideways.
   *
   * For tables whose first column is the row's NAME — a model term, a test
   * group. Scrolling right to read an estimate is fine; losing which term it
   * belongs to while you do it is not. Off by default because an arbitrary
   * frame's first column is not necessarily an identifier.
   */
  stickyFirst?: boolean;
  className?: string;
}

function cellValue<Row>(col: Column<Row>, row: Row): unknown {
  if (col.value) return col.value(row);
  return (row as Record<string, unknown>)?.[col.key];
}

export default function Table<Row>({
  columns,
  rows,
  total,
  truncated,
  note,
  empty,
  dense,
  sticky = true,
  maxHeight,
  rowKey,
  onRowClick,
  collapseTo,
  stickyFirst,
  className,
}: TableProps<Row>) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const [expanded, setExpanded] = useState(false);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    return [...rows].sort((a, b) => {
      const av = cellValue(col, a);
      const bv = cellValue(col, b);
      const an = toNumber(av);
      const bn = toNumber(bv);
      if (an !== null && bn !== null) return (an - bn) * sort.dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * sort.dir;
    });
  }, [rows, sort, columns]);

  const visible = collapseTo && !expanded ? sorted.slice(0, collapseTo) : sorted;
  const hidden = sorted.length - visible.length;

  if (!rows.length) {
    return <>{empty ?? <Empty variant="terminal" title={note ?? "no rows"} />}</>;
  }

  return (
    <div
      className={cx("ui-table-wrap", "ui-table-scroll-shadows", className)}
      style={maxHeight ? { maxHeight } : undefined}
      /* a region that scrolls must be reachable without a mouse, or its right-hand columns are not */
      tabIndex={0}
      role="group"
      aria-label="table (scrolls)"
    >
      <table
        className={cx("ui-table", dense && "is-dense", sticky && "is-sticky", stickyFirst && "is-sticky-first")}
      >
        <thead>
          <tr>
            {columns.map((c) => {
              const num = isNumeric(c.kind);
              const active = sort?.key === c.key;
              return (
                <th
                  key={c.key}
                  className={cx(num && "is-num", c.sortable && "is-sortable", active && "is-sorted")}
                  style={c.width ? { width: c.width } : undefined}
                  title={c.title}
                  aria-sort={active ? (sort!.dir === 1 ? "ascending" : "descending") : undefined}
                  onClick={
                    c.sortable
                      ? () =>
                          setSort((s) =>
                            s?.key === c.key ? { key: c.key, dir: s.dir === 1 ? -1 : 1 } : { key: c.key, dir: -1 },
                          )
                      : undefined
                  }
                >
                  {c.header}
                  {active ? (
                    <span className="ui-sort" aria-hidden="true">
                      {sort!.dir === 1 ? " ↑" : " ↓"}
                    </span>
                  ) : null}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {visible.map((row, i) => (
            <tr
              key={rowKey ? rowKey(row, i) : i}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={cx(onRowClick && "is-clickable")}
            >
              {columns.map((c) => {
                const tone = c.tone?.(row);
                return (
                  <td key={c.key} className={cx(isNumeric(c.kind) && "is-num", tone && `tone-${tone}`)}>
                    {c.render ? (
                      c.render(row)
                    ) : (
                      <Num value={cellValue(c, row) as never} kind={c.kind} dp={c.dp} full />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {(hidden > 0 || note || truncated || total !== undefined) && (
        <div className="ui-table-foot">
          {hidden > 0 && (
            <button className="ui-table-more" onClick={() => setExpanded(true)}>
              show all {sorted.length} rows
            </button>
          )}
          {total !== undefined && total !== rows.length && (
            <span>
              {rows.length} of {total} rows
            </span>
          )}
          {truncated && <span>truncated by the server</span>}
          {note && <span>{note}</span>}
        </div>
      )}
    </div>
  );
}

/** Re-exported for screens that build their own columns. */
export { kindForColumn };
