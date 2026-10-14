#!/usr/bin/env python3
"""``openlab guide | tools | tool``: one implementation for the person and the agent. With a server set it is a thin
client (no engine, no keys, no data) -- an agent's VM gets this file alone and talks to ``openlab-mcp --http``
through its allowlist; a person can point it at ``openlab app``. Without one, the full ``openlab`` runs the same
commands on the local engine. Standard library only, so it can be copied alone:

    openlab guide                          how to work with openlab (read this first)
    openlab tools [SECTION]                the tools' names by section; with SECTION, one line each
    openlab tool NAME --help               one tool's arguments
    openlab tool NAME key=value ...        run it; values are JSON when they parse, text otherwise; @FILE reads a file

The reply prints as lines and tables; `--json` prints the whole envelope as JSON instead.

``$OPENLAB_URL`` (or ``$OPENLAB_APP_URL``) is the server, with ``$OPENLAB_APP_TOKEN`` when it wants one; copied alone,
the default is http://10.0.2.2:8765 (the host as a QEMU guest sees it). Exit code 0 when the reply is ok, 1 when
it is a refusal or an error, 2 when the server cannot be reached."""

import json
import os
import sys
import urllib.error
import urllib.request

URL = (os.environ.get("OPENLAB_URL") or os.environ.get("OPENLAB_APP_URL") or "").rstrip("/")
HEADERS = {"authorization": f"Bearer {os.environ['OPENLAB_APP_TOKEN']}"} if os.environ.get("OPENLAB_APP_TOKEN") else {}


def _get(path: str):
    req = urllib.request.Request(URL + path, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:                          # noqa: S310 -- the user's server
        return r.read().decode()


def _post(path: str, body: dict) -> dict:
    if not URL:                                         # the full openlab, no server: the local engine
        from openlab.app.client import call
        return call(path.rsplit("/", 1)[1], body)
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **HEADERS})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:                     # noqa: S310
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read()
        try:
            return json.loads(text)
        except ValueError:                              # a proxy's or server's plain-text error page
            return {"ok": False, "error": f"HTTP {e.code}", "meaning": text.decode("utf-8", "replace")[:500]}


def _value(v: str):
    if v.startswith("@"):
        with open(v[1:], encoding="utf-8") as f:
            return f.read()
    try:
        return json.loads(v)
    except ValueError:
        return v


WIDTH = 160                                             # a table cell or a line is cut past this
CELL = 40
COLUMNS = 8


def _short(v, n: int) -> str:
    s = "-" if v is None else v if isinstance(v, str) else json.dumps(v, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "\u2026"


def _scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _table(rows: list, indent: str) -> list:
    cols = []
    for r in rows[:20]:
        cols += [k for k, v in r.items() if k not in cols and _scalar(v)]
    cols = cols[:COLUMNS]
    if not cols:
        return [indent + _short(r, WIDTH) for r in rows]
    cells = [[_short("" if r.get(c) is None else r.get(c), CELL) for c in cols] for r in rows]
    w = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda xs: (indent + "  ".join(x.ljust(w[i]) for i, x in enumerate(xs))).rstrip()   # noqa: E731
    return [line(cols)] + [line(row) for row in cells]


def render(v, indent: str = "") -> list:
    """A reply as a person reads it: `key: value` lines, a list of records as a table, nesting indented."""
    if _scalar(v):
        return [indent + _short(v, WIDTH)]
    if isinstance(v, list):
        if not v:
            return [indent + "(none)"]
        if all(isinstance(x, dict) for x in v):
            return _table(v, indent)
        if all(_scalar(x) for x in v):
            return [indent + _short(", ".join(str(x) for x in v), WIDTH)]
        return [line for x in v for line in render(x, indent)]
    out = [] if v else [indent + "(none)"]
    for k, x in v.items():
        if x == {}:
            out.append(f"{indent}{k}: (none)")
        elif _scalar(x) or (isinstance(x, list) and all(_scalar(y) for y in x)):
            out.append(f"{indent}{k}: " + (_short(x, WIDTH) if _scalar(x) else
                                            _short(", ".join(str(y) for y in x), WIDTH) or "(none)"))
        else:
            out += [f"{indent}{k}:"] + render(x, indent + "  ")
    return out


def render_envelope(env: dict) -> str:
    if not env.get("ok"):
        return "\n".join(f"{k}: {env[k]}" for k in ("error", "meaning", "try") if env.get(k))
    return "\n".join(render(env.get("result")))


def _args_line(schema: dict) -> str:
    req = schema.get("required", [])
    return ", ".join(f"{k}:{v.get('type', '')}" if k in req else f"{k}={json.dumps(v.get('default'))}"
                     for k, v in schema.get("properties", {}).items()) or "none"


def local_guide() -> str:
    """The agent instructions, as the MCP gives them, with the shell form first."""
    from openlab.agent_guide import instructions             # the MCP's text, without importing the mcp SDK
    return ("From the shell: `openlab tool NAME key=value ...`, NAME being the tool name below without `openlab_` "
            "(openlab tools lists them).\n\n" + instructions())


def _tools() -> list:
    env = json.loads(_get("/api/tools")) if URL else _post("/api/tool/tools", {})
    return env["result"]["tools"]


def main(argv=None) -> int:
    global URL
    argv = sys.argv[1:] if argv is None else argv
    if not URL and __name__ == "__main__":              # copied alone: the agent's default server
        URL = "http://10.0.2.2:8765"
    try:
        if not argv or argv[0] in ("-h", "--help"):
            print(__doc__)
            return 0
        if argv[0] == "guide":
            print(_get("/api/guide") if URL else local_guide())
            return 0
        as_json = "--json" in argv
        argv = [a for a in argv if a != "--json"]
        tools = _tools() if argv[0] == "tools" or "--help" in argv or argv == ["tool"] else []
        tools = [t for t in tools if not t.get("human")]
        if argv[0] == "tools" or argv == ["tool"]:
            section = argv[1] if len(argv) > 1 and argv[0] == "tools" else ""
            if as_json:
                print(json.dumps([t for t in tools if not section or t["section"] == section], indent=1))
            elif section:
                for t in tools:
                    if t["section"] == section:
                        print(f"{t['name']:<22} {_short(t['description'].split('. ')[0], 110)}")
            else:
                by: dict = {}
                for t in tools:
                    by.setdefault(t["section"], []).append(t["name"])
                import textwrap
                for sec, names in by.items():
                    print(textwrap.fill(f"{sec} ({len(names)}): {', '.join(names)}", 118, subsequent_indent="  "))
                print("openlab tools SECTION: one line each; openlab tool NAME --help: its arguments")
            return 0
        if argv[0] != "tool" or len(argv) < 2:
            print("usage: openlab tool NAME key=value ... [--json]   (openlab --help)", file=sys.stderr)
            return 1
        if "--help" in argv:
            t = next((t for t in tools if t["name"] == argv[1]), None)
            if t is None:
                print(f"no tool {argv[1]!r}: openlab tools")
            else:
                print(json.dumps(t, indent=1) if as_json else
                      f"{t['name']} ({t['section']}): {t['description']}\narguments: {_args_line(t['input_schema'])}")
            return 0 if t else 1
        args = {}
        for kv in argv[2:]:
            k, _, v = kv.partition("=")
            args[k] = _value(v)
        env = _post(f"/api/tool/{argv[1]}", args)
    except (urllib.error.URLError, OSError) as e:
        print(f"openlab: cannot reach {URL}: {e}", file=sys.stderr)
        return 2
    print(json.dumps(env, indent=1, default=str) if as_json else render_envelope(env))
    return 0 if env.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
