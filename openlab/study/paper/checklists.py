"""The reporting checklists a paper's checklist section reads: TRIPOD+AI (2024) for prediction models, CONSORT
(2025) for randomised trials and the NeurIPS paper checklist for ML results -- the items of each that a study's
record can show, not the full guideline (tripod-statement.org, consort-spirit.org, neurips.cc).

Each item is ``(id, text, what a person states)``: the third is "" for an item the record decides, and otherwise
what only a person can say. `sections.checklist` marks each item ``met``, ``not met``, ``to state`` or ``n/a``.
"""

from __future__ import annotations

MET, NOT_MET, TO_STATE, NA = "met", "not met", "to state", "n/a"

CHECKLISTS = {
    "tripod": ("TRIPOD+AI (2024), the items a study's record can show", [
        ("plan", "analysis plan specified in advance", ""),
        ("data", "source of data", ""),
        ("outcome", "outcome defined", ""),
        ("predictors", "predictors used", ""),
        ("missing", "handling of missing data", ""),
        ("sample", "sample size", ""),
        ("validation", "internal validation", ""),
        ("discrimination", "discrimination reported", ""),
        ("calibration", "calibration reported", ""),
        ("leakage", "no information from the evaluation data in development", ""),
        ("limitations", "limitations discussed", ""),
        ("code", "code available", ""),
    ]),
    "consort": ("CONSORT (2025), the items a study's record can show", [
        ("plan", "statistical analysis plan specified in advance", ""),
        ("primary", "primary outcome pre-specified", ""),
        ("numbers", "numbers analysed in each group", ""),
        ("allocation", "allocation as intended (no sample ratio mismatch)", ""),
        ("effect", "effect estimate with its precision", ""),
        ("multiplicity", "multiple analyses accounted for", ""),
        ("prespecified", "analyses not pre-specified identified", ""),
        ("flow", "participant flow (enrolled, allocated, followed up, analysed)", "state the flow of participants"),
        ("harms", "harms", "state the harms observed, or none"),
        ("sharing", "data sharing statement", "state where the data and code can be had"),
    ]),
    "neurips": ("NeurIPS paper checklist, the items a study's record can show", [
        ("claims", "claims match the results", ""),
        ("limitations", "limitations discussed", ""),
        ("errorbars", "error bars or confidence intervals", ""),
        ("heldout", "evaluated on data not used in training", ""),
        ("significance", "statistical significance accounted for", ""),
        ("leakage", "no leakage between training and evaluation", ""),
        ("code", "code released", ""),
        ("compute", "compute resources", "state the hardware and run time"),
    ]),
}
