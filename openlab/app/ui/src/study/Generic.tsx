/**
 * A plain rendering of a tool result whose shape the view does not pin down: lists of records as tables,
 * everything else as label/value rows.
 */
import type { ReactNode } from "react";
import { KeyValue, Table, kindForColumn, type Column } from "../ui";

function text(v: unknown): ReactNode {
  if (v == null) return <span className="dim">—</span>;
  if (Array.isArray(v) && v.length === 0) return <span className="dim">none</span>;
  if (typeof v === "object") return <code className="small">{JSON.stringify(v)}</code>;
  return String(v);
}

/** A table over a list of records, one column per key seen. */
export function Records({ rows, empty = "none", maxHeight = 360 }: { rows: Record<string, unknown>[]; empty?: string; maxHeight?: number }) {
  if (!rows.length) return <p className="dim small">{empty}</p>;
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const columns: Column<Record<string, unknown>>[] = keys.map((k) => ({
    key: k,
    header: k.replace(/_/g, " "),
    kind: rows.some((r) => typeof r[k] === "number") ? kindForColumn(k, "ratio") : "raw",
    render: rows.some((r) => r[k] != null && typeof r[k] === "object") ? (r) => text(r[k]) : undefined,
  }));
  return <Table columns={columns} rows={rows} dense stickyFirst maxHeight={maxHeight} rowKey={(_, i) => String(i)} />;
}

function isRecordList(v: unknown): v is Record<string, unknown>[] {
  return Array.isArray(v) && v.length > 0 && v.every((x) => x && typeof x === "object" && !Array.isArray(x));
}

/** Any tool result: lists of records as tables, everything else as label/value rows. */
export function Generic({ value }: { value: unknown }) {
  if (isRecordList(value)) return <Records rows={value} />;
  if (Array.isArray(value)) return <ul className="small">{value.map((x, i) => <li key={i}>{text(x)}</li>)}</ul>;
  if (!value || typeof value !== "object") return <>{text(value)}</>;
  const entries = Object.entries(value as Record<string, unknown>);
  // a nested record is read as its own block of labelled values, not as a line of JSON
  const isNested = (v: unknown) => !!v && typeof v === "object" && !Array.isArray(v);
  const flat = entries.filter(([, v]) => !isRecordList(v) && !isNested(v));
  const lists = entries.filter(([, v]) => isRecordList(v));
  return (
    <div className="view-stack">
      {flat.length > 0 && <KeyValue rows={flat.map(([k, v]) => ({ label: k.replace(/_/g, " "), value: text(v) }))} />}
      {entries.filter(([, v]) => isNested(v)).map(([k, v]) => (
        <div key={k}>
          <div className="dim small">{k.replace(/_/g, " ")}</div>
          <Generic value={v} />
        </div>
      ))}
      {lists.map(([k, v]) => (
        <div key={k}>
          <div className="dim small">{k.replace(/_/g, " ")}</div>
          <Records rows={v as Record<string, unknown>[]} />
        </div>
      ))}
    </div>
  );
}
