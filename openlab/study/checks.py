"""A check over a table, as SQL: a forecast's gate predicates over its predictions, a table
named ``t``. ``not_null(col)`` and ``unique(col)`` are shorthands; a failing comparison says what the left side held.
"""

from __future__ import annotations

import re

_CMP = re.compile(r"^(.*?)\s*(<=|>=|!=|<>|=|<|>)\s*(.*)$")


def expand(expr: str) -> str:
    """Shorthands: ``not_null(col)`` and ``unique(col)`` as the SQL they stand for."""
    expr = re.sub(r"\bnot_null\(\s*(\w+)\s*\)", r"(count(\1) = count(*))", expr)
    return re.sub(r"\bunique\(\s*(\w+)\s*\)", r"(count(distinct \1) = count(*))", expr)


# Polars SQL's table functions read any path the process can (a holdout's labels included): frames only
FILE_SQL = re.compile(r"\b(read|scan)_\w+[\s\"'`/*-]*\(", re.I)


def no_files(sql: str) -> None:
    if FILE_SQL.search(sql):
        raise ValueError("a check reads its table only, not files (read_csv/read_json/...)")


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if depth < 0:
            return False
    return depth == 0


def check_frame(df, expr: str) -> dict:
    """One predicate over one frame, as SQL over a table named ``t``; a failing comparison says what the
    left side held."""
    import polars as pl
    rec = {"expect": expr}
    try:
        ctx = pl.SQLContext(t=df)
        sql = expand(expr)
        no_files(sql)
        rec["ok"] = bool(ctx.execute(f"select ({sql}) as ok from t", eager=True).item())
        # a shorthand expands wrapped in parentheses: compare inside them, or the left side is cut mid-call
        bare = sql[1:-1] if sql.startswith("(") and sql.endswith(")") and _balanced(sql[1:-1]) else sql
        m = _CMP.match(bare)
        if m and not rec["ok"] and _balanced(m.group(1)) and _balanced(m.group(3)):
            val = lambda g: str(ctx.execute(f"select ({g}) as v from t", eager=True).item())  # noqa: E731
            rec["actual"] = val(m.group(1))
            if re.search(r"[A-Za-z_]\s*\(", m.group(3)):         # the right side is a computed value too
                rec["actual"] += f", against {val(m.group(3))}"
    except Exception as exc:                                    # noqa: BLE001 -- a bad line is a failed line
        rec.update(ok=False, error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
    return rec
