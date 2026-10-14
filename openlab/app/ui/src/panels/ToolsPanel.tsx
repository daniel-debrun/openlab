/**
 * The tool console: every tool the server has, with a form built from its own
 * input schema, and the raw envelope shown as it came back.
 *
 * This panel exists to prove a claim the rest of the UI only asserts: the
 * interface has no private powers. Every button elsewhere in this app is one of
 * these calls, and so is every action the agent takes over the MCP. If a tool is
 * here, you can run it; if it is not here, the UI cannot do it either.
 *
 * So the form is generated, never hand-written — a hand-written form would drift
 * from the schema and the proof would quietly stop being true.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { Args, Envelope, ToolSpec, ToolsList, callConfirmed, callTool } from "../api/tools";
import { Async, Badge, Button, ErrorPanel, Field, Num, cx } from "../ui";

/** Field bodies wide enough to deserve a textarea rather than an input. */
const LONG = new Set(["src", "text", "body", "query", "statements", "answer", "note", "definition"]);

function defaultsFor(t: ToolSpec): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, p] of Object.entries(t.input_schema.properties ?? {})) {
    if (name === "confirm") continue; // the ceremony owns this one
    out[name] = p.default === undefined || p.default === null ? "" : String(p.default);
  }
  return out;
}

/** Text in the form -> the JSON value the schema asks for. */
function coerce(raw: string, type: string | undefined): unknown {
  if (type === "integer") return raw === "" ? undefined : Number.parseInt(raw, 10);
  if (type === "number") return raw === "" ? undefined : Number(raw);
  if (type === "boolean") return raw === "true";
  if (type === "object" || type === "array") {
    if (raw.trim() === "") return undefined;
    try {
      return JSON.parse(raw);
    } catch {
      return raw; // let the server say what is wrong with it
    }
  }
  return raw;
}

function ToolForm({ tool }: { tool: ToolSpec }) {
  const [values, setValues] = useState<Record<string, string>>(() => defaultsFor(tool));
  const [envelope, setEnvelope] = useState<Envelope<unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const required = new Set(tool.input_schema.required ?? []);

  const run = async () => {
    setBusy(true);
    setEnvelope(null);
    const args: Args = {};
    for (const [name, p] of Object.entries(tool.input_schema.properties ?? {})) {
      if (name === "confirm") continue;
      const v = coerce(values[name] ?? "", p.type);
      // an empty optional is left out so the server's own default applies
      if (v === undefined || (v === "" && !required.has(name))) continue;
      args[name] = v;
    }
    try {
      setEnvelope(tool.destructive ? await callConfirmed(tool.name, args) : await callTool(tool.name, args));
    } catch (e) {
      setEnvelope({ ok: false, tool: tool.name, error: String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="tool-form">
      <p className="tool-desc">{tool.description}</p>
      {Object.entries(tool.input_schema.properties ?? {})
        .filter(([name]) => name !== "confirm")
        .map(([name, p]) => (
          <Field
            key={name}
            label={
              <>
                {name}
                <span className="tool-type"> {p.type ?? "string"}</span>
                {required.has(name) && <span className="tool-req"> required</span>}
              </>
            }
          >
            {p.type === "boolean" ? (
              <select value={values[name] ?? "false"} onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}>
                <option value="false">false</option>
                <option value="true">true</option>
              </select>
            ) : LONG.has(name) ? (
              <textarea
                rows={4}
                value={values[name] ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
              />
            ) : (
              <input
                value={values[name] ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))}
              />
            )}
          </Field>
        ))}

      <div className="tool-actions">
        <Button size="sm" variant={tool.destructive ? "danger" : "primary"} onClick={run} loading={busy}>
          Run {tool.name}
        </Button>
      </div>

      {envelope &&
        (envelope.ok ? (
          <div className="tool-result">
            <div className="tool-result-head">
              ok · <Num value={envelope.took_ms / 1000} kind="duration" />
            </div>
            <pre className="num tool-json">{JSON.stringify(envelope.result, null, 2)}</pre>
          </div>
        ) : (
          <ErrorPanel error={envelope} context={`${tool.name} refused`} />
        ))}
    </div>
  );
}

export default function ToolsPanel() {
  const [selected, setSelected] = useState<string>("");
  const q = useQuery({
    queryKey: ["tools"],
    queryFn: async () => {
      const env = await api.get<Envelope<ToolsList>>("/api/tools");
      if (!env.ok) throw env;
      return env.result;
    },
  });

  const bySection = useMemo(() => {
    const out = new Map<string, ToolSpec[]>();
    for (const t of q.data?.tools ?? []) {
      const list = out.get(t.section) ?? [];
      list.push(t);
      out.set(t.section, list);
    }
    return out;
  }, [q.data]);

  const tool = (q.data?.tools ?? []).find((t) => t.name === selected);

  return (
    <div className="panel">
      <p className="panel-sub">
        Everything this interface can do, it does through one of these. There is nothing else to call.
      </p>
      <Async query={q} skeleton="lines" context="listing the tools">
        {() => (
          <>
            {Array.from(bySection.entries()).map(([section, tools]) => (
              <div className="tool-section" key={section}>
                <h4 className="panel-h">{section}</h4>
                <ul className="tool-list">
                  {tools.map((t) => (
                    <li key={t.name}>
                      <button
                        className={cx("tool-row", selected === t.name && "is-active")}
                        onClick={() => setSelected(selected === t.name ? "" : t.name)}
                      >
                        <span className="tool-name">{t.name}</span>
                        {t.destructive ? (
                          <Badge tone="danger">destructive</Badge>
                        ) : t.mutates ? (
                          <Badge tone="warn">mutates</Badge>
                        ) : (
                          <Badge>reads</Badge>
                        )}
                      </button>
                      {selected === t.name && tool ? <ToolForm tool={tool} /> : null}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </>
        )}
      </Async>
    </div>
  );
}
