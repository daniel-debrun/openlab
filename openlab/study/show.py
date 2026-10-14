"""A record node in one line: what the CLI prints and the tools return instead of its full payload."""

from __future__ import annotations

WIDTH = 160


def _num(v) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def alpha_budget(p: dict) -> str:
    """A test's move on the error budget (alpha-investing), in words: what it spent or earned, and what is left."""
    d = p["wealth_after"] - p["wealth_before"]
    return (f"alpha earned {_num(d)}" if d >= 0 else f"alpha spent {_num(-d)}") + \
        f", alpha budget left: {_num(p['wealth_after'])}"


def selected(n: int) -> str:
    """"selected from N trials": how many research trials a result a claim rests on was chosen among."""
    return f"selected from {n} trial{'s' if n != 1 else ''}"


def _metrics(metrics: dict) -> str:
    return "; ".join(f"{k} {_num(m.get('value'))}" + (f" [{_num(m['lo'])}, {_num(m['hi'])}]" if "lo" in m else "")
                     for k, m in metrics.items())


def _paper(p: dict) -> str:
    """The paper's value in the metric's unit, and as the paper writes it when that differs."""
    return _num(p["paper"]) + (f" ({p['paper_written']} as printed)" if p.get("paper_written") else "")


def _tolerance(p: dict) -> str:
    """The tolerance, in the metric's own unit (a share's is a fraction: 0.02 is two points)."""
    from .metrics import METRICS
    m = METRICS.get(p["metric"])
    return f"tolerance {_num(p['tolerance'])} " + ("as a fraction" if m is not None and m.unit == "fraction"
                                                   else f"in {p['metric']}'s units")


LINES = {
    "plan": lambda p: f"sealed spec {p['sha'][:12]}",
    "amendment": lambda p: f"{'person (signed)' if p['person'] else 'unsigned'} amendment ({', '.join(p['changes'])}): "
                           f"{p['reason']}",
    "dataset": lambda p: f"{p['name']}: {p['rows']} rows, sha {p['sha256'][:12]}",
    "run": lambda p: f"{p['segment']} run{' (cached)' if p['cached'] else ''}, key {p['key'][:12]}, {p['sandbox']}"
                     + (f": {p['note']}" if p.get("note") else ""),
    "result": lambda p: _metrics(p["metrics"]),
    "look": lambda p: f"look {p['n']}: {p['reason']}",
    "model": lambda p: f"{p['name']} v{p['version']}",
    "action": lambda p: f"{p['verb']}: " + ("ok" if p["outcome"].get("ok") else f"refused: {p['outcome'].get('error')}"),
    "forecast": lambda p: f"forecast at {p['t']} by {p['model']['name']} v{p['model']['version']}: {p['rows']} rows, "
                          f"gate {'passed' if p['gate']['passed'] else 'FAILED'}, {p['file']}",
    "outcome": lambda p: f"outcome of {p['forecast']} ({p['n']} rows): {_metrics(p['metrics'])}",
    "query": lambda p: f"{p['source']} `{p['text']}`: " + (f"failed ({p['error']})" if p["error"] else f"{p['n_hits']} hits"),
    "source": lambda p: f"{p['title']} ({p['year']}, {p['version_kind']})" + (" RETRACTED" if p["retracted"] else ""),
    "document": lambda p: f"{p['n_pages']} pages from {p['url']}",
    "coverage": lambda p: f"{p['verdict']}: {len(p['queries'])} sources, {sum(p['queries'].values())} queries, "
                          f"{p['hops']} hops, {p['edges']} citation edges, {p['records']} records -> "
                          f"{p['unique']} unique" + (f", {p['failed']} calls failed" if p["failed"] else ""),
    "screen": lambda p: f"{p['decision']} ({p['criterion']}): {p['rationale']}",
    "evidence": lambda p: f"p.{p['page']} {p['section']} ({p['kind']}): "
                          + (", ".join(f"{k} {_num(v)}{'%' if p.get('units', {}).get(k) == 'percent' else ''}"
                                       for k, v in p["fields"].items()) or p["quote"]),
    "comparison": lambda p: f"{p['metric']} {_num(p['values'][0])} vs {_num(p['values'][1])}: {p['verdict']} ("
                            + ", ".join(f"{k} {v}" for k, v in p["dims"].items()) + ")" + (
                                f"; a - b {_num(p['paired']['diff'])} [{_num(p['paired']['lo'])}, "
                                f"{_num(p['paired']['hi'])}], p {_num(p['paired']['p'])}" if p.get("paired") else ""),
    "recall": lambda p: f"recall {p['recall']:.1%}: {len(p['matched'])} of {len(p['references'])} references of "
                        f"{p['gold']}",
    "paper": lambda p: f"paper, pdf {p['pdf_sha'][:12]}: {len(p['bindings'])} bindings, {len(p['lint'])} waived "
                       f"findings, signed {p['signed']}",
    "claim":lambda p: f"{p['state']}, {p['verdict']}" + (f", {p['level']}" if p["level"] else "")
                       + (f", value {_num(p['value'])}" if p["value"] is not None else "")
                       + (f", {selected(p['selected_from'])}" if p.get("selected_from") else "")
                       + f": {p['statement']}",
    "approval": lambda p: f"{p['decision']}: {p['reason']}",
    "question": lambda p: p["text"] + (f" (out of scope: {p['out_of_scope']})" if p["out_of_scope"] else ""),
    "hypothesis": lambda p: f"{p['statement']} [{', '.join(p['tags'])}]",
    "test": lambda p: f"{p['family']}: p {_num(p['p'])} at alpha {_num(p['alpha_j'])} vs {p['baseline']}, "
                      f"{'rejected' if p['rejected'] else 'not rejected'}; {alpha_budget(p)}",
    "assumption": lambda p: f"{p['statement']} (breaks if {p['breaks_if']})",
    "break": lambda p: f"broken: {p['reason']}",
    "dead_end": lambda p: f"{p['statement']}: {p['reason']}",
    "agent": lambda p: f"agent {p['id']} ({p['kind']}" + (f", role {p['role']}" if p["role"] else "")
                       + (f", parent {p['parent']}" if p["parent"] else "") + ")",
    "contra_search": lambda p: f"contradiction search: {p['state']}",
    "target": lambda p: f"{p['kind']} target, tolerance {_num(p['tolerance'])} (max {_num(p['max_tolerance'])})"
                        + (f", the paper prints shares in {p['paper_unit']}" if p.get("paper_unit") else ""),
    "fidelity": lambda p: f"{'rebuilt' if p['passed'] else 'NOT rebuilt'}: {p['arm']} {p['metric']} "
                          f"{_num(p['ours'])} vs the paper's {_paper(p)}, {_tolerance(p)}",
    "replication": lambda p: f"{p['verdict']}" + (f" ({p['reason']})" if p["reason"] else "") + f": {p['arm']} "
                             f"{p['metric']} {_num(p['ours'])} [{_num(p['lo'])}, {_num(p['hi'])}] vs the paper's "
                             f"{_paper(p)}, baseline {_num(p['reference'])}"
                             + (f"; ours {p['vs_reference']}" if p.get("vs_reference") else ""),
    "reproduction": lambda p: f"reproduction of {p['run']}: "
                              + ("matched" if p["matched"] else f"DIFFERS ({p['differs']})")
                              + (f", max |diff| {_num(p['max_abs_diff'])}" if p["max_abs_diff"] else ""),
}


#: kinds whose line is a computed verdict with its numbers, shown whole
WHOLE = ("fidelity", "replication")


def line(n: dict) -> str:
    text = LINES.get(n["kind"], lambda p: "")(n["payload"])
    return text if len(text) <= WIDTH or n["kind"] in WHOLE else text[:WIDTH - 3] + "..."


def brief(n: dict) -> dict:
    """A node without its payload: id, kind, who, when, and its line."""
    return {"id": n["id"], "kind": n["kind"], "actor": n["actor"], "agent": n.get("agent") or n.get("session", ""),
            "at": n["at"], "summary": line(n)}                              # agent = the node's session
