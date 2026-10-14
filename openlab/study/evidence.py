"""The literature engine, part B: evidence quoted from full text, comparisons between results, and the comparison
lattice with its gaps.

The agent reads the page and types the values; the schema bodies (`Evidence`, `Comparison`) check them against
the page and the sealed protocol, and this module never re-checks them. The lattice is derived, never stored.
"""

from __future__ import annotations

import json
import re

from . import schema
from . import spec as _spec
from .errors import StudyError
from .literature import _source_node, _under, document
from .record import Node
from .schema import Comparison, Evidence


def kind_of(kind: str, section: str) -> str:
    """The evidence kind: as given (`text` is a passage), else from the section ("Table 2" a table, "Figure 3" a
    figure), else a passage of running text."""
    if kind not in ("", "text"):
        return kind                                 # passage, table, figure; anything else the schema refuses
    s = section.strip().lower()
    return "table" if kind == "" and re.match(r"(table|tab\.)\s*\w", s) else \
        "figure" if kind == "" and re.match(r"(figure|fig\.?)\s*\w", s) else "passage"


def extract(study, source: str, page: int, quote: str, fields: dict, kind: str = "",
            section: str = "", units: dict | None = None, header: str = "") -> Node:
    """Record what page `page` of the source's fetched full text says: the quote, and the values read from it
    as fields of the sealed extraction schema. `kind` defaults from the section (see `kind_of`). `units` ({field:
    percent or fraction}) says how the paper writes a number; a field [review.extract] declares with a unit, or
    whose number the quote writes with a % sign, gets it without asking (schema.evidence_units)."""
    kind = kind_of(kind, section)
    rec = study.record
    sealed = _spec.check(study.folder, rec)
    units = schema.evidence_units(quote, fields, dict(units or {}),
                                  schema.extraction_units(sealed.review) if sealed.review else {})
    src = _source_node(rec, source)
    doc = document(rec, src)
    if doc is None:
        raise StudyError(f"{src.id} has no full text on record", "evidence is quoted from a page of the document",
                         "fetch it first (study_fetch_text)")
    return rec.add(Evidence(source=src.id, page=page, section=section, kind=kind, quote=quote, fields=fields,
                            valid_from=src.payload["published"], units=units, header=header),
                   actor="agent", links=[("from", doc.id), ("of", src.id), ("under", _under(rec))])


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _match(needle: str):
    """A test of one flattened passage: a number by value (0.90 finds 0.9, as extract checks it), else the text, case
    and whitespace aside."""
    from decimal import Decimal, InvalidOperation
    nums = schema._numbers(needle)
    flat = schema._flat(needle).casefold()
    try:
        value = Decimal(flat.replace(",", "")) if len(nums) == 1 else None
    except InvalidOperation:
        value = None
    if value is not None:
        return lambda text: value in schema._numbers(text)
    return lambda text: flat in schema._flat(text).casefold()


def find(study, source: str, needle: str, around: int = 0, n: int = 10) -> dict:
    """Where the fetched full text of `source` says `needle` (a phrase, or a number by value): each hit's page and
    the exact sentence(s) around it, in reading order or, for a table row, the laid-out line; each is a quote that
    `extract` accepts for that page (at least MIN_QUOTE characters: a short sentence takes its neighbours)."""
    rec = study.record
    src = _source_node(rec, source)
    doc = document(rec, src)
    if doc is None:
        raise StudyError(f"{src.id} has no full text on record", "find searches the fetched pages",
                         f"fetch it first: openlab study fetch {study.folder.name} {src.id}")
    if len(schema._flat(needle)) < 1:
        raise StudyError("find needs the text or number to look for")
    forms = {k: json.loads(rec.blob(doc.payload[k])) for k in ("pages", "layout") if doc.payload.get(k)}
    hit, out, seen = _match(needle), [], set()
    for i in range(doc.payload["n_pages"]):
        for form, pages in forms.items():
            page = pages[i] if i < len(pages) else ""
            units = ([schema._flat(x) for x in page.splitlines()] if form == "layout" else
                     _SENTENCE.split(schema._flat(page)))
            units = [u for u in units if u]
            for j, u in enumerate(units):
                if not hit(u):
                    continue
                lo, hi = max(0, j - around), min(len(units), j + around + 1)
                while len(" ".join(units[lo:hi])) < schema.MIN_QUOTE and (lo > 0 or hi < len(units)):
                    lo, hi = (lo - 1, hi) if lo > 0 else (lo, hi + 1)      # the line or sentence before first
                quote = " ".join(units[lo:hi])
                key = (i + 1, quote)
                if key in seen or len(quote) < schema.MIN_QUOTE or schema._flat(quote) not in schema._flat(page):
                    continue
                seen.add(key)
                out.append({"page": i + 1, "form": "table row" if form == "layout" else "text", "quote": quote})
    out.sort(key=lambda h: (h["page"], h["form"] != "text"))
    return {"source": src.id, "document": doc.id, "needle": needle, "hits": out[:n], "total": len(out),
            "next": f'openlab study extract {study.folder.name} {src.id} --page <page> --quote "<quote>" --section ...'}


def evidence(study, source: str = "") -> list:
    """The evidence on record, oldest first, or one source's."""
    rec = study.record
    keys: dict = {}
    rows = []
    for e in rec.find("evidence", **({"source": source} if source else {})):
        p = e.payload
        if p["source"] not in keys:
            keys[p["source"]] = schema.source_key(rec.get(p["source"]).payload)
        rows.append({"id": e.id, "source": p["source"], "key": keys[p["source"]], "page": p["page"],
                     "section": p["section"], "kind": p["kind"], "quote": p["quote"], "fields": p["fields"],
                     "valid_from": p["valid_from"], "at": e.at})
    return rows


QUOTE_EXCERPT = 90                                  # characters of a quote in the evidence table
TABLE_COLUMNS = ("cite", "year", "method", "dataset", "metric", "value", "page", "section", "quote", "agent",
                 "evidence", "source")


def cite(p: dict) -> str:
    """A source's short citation: the first author's surname, "et al." past two authors, and the year."""
    names = [str(a).split()[-1] for a in p.get("authors") or [] if str(a).strip()]
    who = (names[0] if len(names) == 1 else f"{names[0]} & {names[1]}" if len(names) == 2
           else f"{names[0]} et al." if names else p["title"][:40])
    return f"{who} {p['year']}" if p.get("year") else who


def table(study) -> list:
    """The evidence matrix, one row per number read: source (short cite), year, method, dataset, metric, value,
    page, section, a quote excerpt and the agent that extracted it. A [review.extract] with `metric` (text) and
    `value` (number) fields gives one row per evidence node; otherwise each number field is a metric of its own.
    `dataset` is the field of that name, else the protocol coordinates."""
    rec = study.record
    sealed = _spec.sealed(rec)
    review = sealed.review if sealed else {}
    types = schema.extraction(review) if review else {}
    coords = review.get("coordinates", []) if review else []
    roles = {n.payload["id"]: n.payload.get("role") or "" for n in rec.find("agent")}
    sources: dict = {}
    rows = []
    for e in rec.find("evidence"):
        p, f = e.payload, e.payload["fields"]
        src = sources.setdefault(p["source"], rec.get(p["source"]).payload)
        if "value" in f and "metric" in f:
            pairs = [(str(f["metric"]), f["value"])]
        else:
            pairs = [(k, v) for k, v in f.items() if (types.get(k, ("",))[0] in ("number", "int") or (
                not types and isinstance(v, (int, float)) and not isinstance(v, bool))) and k not in coords]
        dataset = f.get("dataset") if "dataset" in f else ", ".join(f"{c} {f[c]}" for c in coords if c in f)
        quote = " ".join(p["quote"].split())
        who = e.session + (f" ({roles[e.session]})" if roles.get(e.session) else "")
        for metric, value in pairs or [("", None)]:
            rows.append({"cite": cite(src), "year": src.get("year"), "method": str(f.get("method", "")),
                         "dataset": str(dataset or ""), "metric": metric, "value": value, "page": p["page"],
                         "section": p["section"],
                         "quote": quote if len(quote) <= QUOTE_EXCERPT else quote[:QUOTE_EXCERPT - 3] + "...",
                         "agent": who, "evidence": e.id, "source": schema.source_key(src)})
    return rows


def compare(study, a: str, b: str, metric: str) -> Node:
    """Set two results (evidence or a study's result) side by side on `metric`; refused unless they agree on
    every comparable_on coordinate."""
    rec = study.record
    _spec.check(study.folder, rec)
    sides = [rec.get(a), rec.get(b)]
    return rec.add(Comparison(**schema.comparison(rec, sides, metric)), actor="agent",
                   links=[("compares", a), ("compares", b), ("under", _under(rec))])


def lattice(study, by: str = "method") -> dict:
    """The comparison lattice, derived from the evidence: one row per combination of the review's coordinates
    seen, one column per value of the text field `by`, each cell the number fields read for it (with the evidence
    behind them), and each row's `gaps`: the columns no evidence fills, which is the gap matrix ("not found in the
    extracted sources", never "novel")."""
    rec = study.record
    review = _spec.check(study.folder, rec).review
    fields = schema.extraction(review) if review else {}
    coords = review.get("coordinates", []) if review else []
    if not coords:
        raise StudyError("the sealed review declares no coordinates", "the lattice's rows are protocol coordinates",
                         "add [review] coordinates = [...] naming fields of [review.extract]")
    if fields.get(by, ("",))[0] != "text" or by in coords:
        raise StudyError(f"`{by}` is not a text field of [review.extract] outside the coordinates",
                         "the lattice's columns are the values of one such field, e.g. method")
    values = [k for k, (t, _) in fields.items() if t in ("number", "int") and k not in coords]
    rows: dict = {}
    columns: dict = {}
    for e in rec.find("evidence"):
        f = e.payload["fields"]
        if by not in f:
            continue
        row = rows.setdefault(tuple(schema.coord_key(f.get(c)) for c in coords),
                              {"coordinates": {c: f.get(c) for c in coords}, "cells": {}})
        col = columns.setdefault(schema.coord_key(f[by]), f[by])
        row["cells"].setdefault(col, []).append({"evidence": e.id, "source": e.payload["source"],
                                                 "page": e.payload["page"], **{k: f[k] for k in values if k in f}})
    for row in rows.values():
        row["gaps"] = [c for c in columns.values() if c not in row["cells"]]
    filled = sum(len(r["cells"]) for r in rows.values())
    return {"coordinates": coords, "by": by, "values": values, "columns": list(columns.values()),
            "rows": list(rows.values()), "cells": len(rows) * len(columns), "filled": filled,
            "gaps": len(rows) * len(columns) - filled}
