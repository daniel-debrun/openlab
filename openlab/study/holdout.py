"""Holdout looks: a person's decision to score the holdout, counted against `budget.holdout_looks`. The `look`
node, with the study's spec, code and data hashes, is written before anything is scored; the schema makes the
holdout run that follows carry the same hashes. A look whose run fails stays spent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _later, spec as _spec
from .errors import StudyError
from .fingerprint import fingerprint, staged
from .record import Node
from .schema import Look

if TYPE_CHECKING:
    from . import Study


def look(study: "Study", *, reason: str, person: bool, code_dir=None) -> Node:
    """Record a person's look at the holdout, then run the holdout segment under it; returns the result node.
    `code_dir` holds model.py when it is not the study folder."""
    if not person:
        raise StudyError("only a person can take a holdout look", "an agent cannot decide to spend the holdout",
                         "ask a person to run `openlab study holdout <folder> --reason ...` in their own terminal")
    rec = study.record
    spec = _spec.check(study.folder, rec)                 # sealed and unedited: else the run would refuse a spent look
    if not {"holdout", "fraction", "labels"} & set(spec.split):
        raise StudyError("the spec declares no holdout", "there is nothing to look at; no look was taken",
                         "a person declares split.holdout (temporal) or split.fraction (static) by amendment")
    from .ledger import shared
    sh = shared(study.folder, rec)                        # looks of the studies sharing this one's ledger count too
    used, budget = sh["looks"], spec.budget["holdout_looks"]
    if used >= budget:
        raise StudyError(f"the holdout budget is spent: {used} of {budget} looks taken" + (
                         f" ({', '.join(str(v['looks']) + ' in ' + k for k, v in sh['by'].items())})" if sh["by"] else ""),
                         "every look at the holdout weakens what it can show",
                         "a person can raise budget.holdout_looks by amendment; it shows on the record")
    runner = _later("runner")
    with staged(study, code_dir) as snap:          # the look and its run see the same code
        fp = fingerprint(study, snap)
        from .search import check_look
        check_look(study, fp)                      # with a selection rule: the look is on the run it picked
        under = rec.last("amendment") or rec.last("plan")
        node = rec.add(Look(n=len(rec.find("look")) + 1, reason=reason, spec_sha=fp["spec_sha"], code=fp["code"], data=fp["data"]),
                       actor="person", links=[("under", under.id)])
        try:
            return runner.run(study, actor="person", segment="holdout", look=node.id, code_dir=snap)
        except Exception as e:
            err = e if isinstance(e, StudyError) else StudyError(type(e).__name__, str(e))
            raise StudyError(f"look {node.payload['n']} ({node.id}) is spent, but its holdout run failed: "
                             f"{err.error}", err.meaning, f"{err.fix or 'fix the cause'}; then a person may take a new "
                             "look if the budget allows (this one stays on the record)") from e
