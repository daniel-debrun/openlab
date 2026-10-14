"""Guarantees of the paper builder: a clean paper builds to a PDF, a Paper node and a signed bundle that verifies;
a typed number, an unresolved or stale binding, a quote not in its source, an equation that disagrees with its
function and "novel" each refuse the build and write no Paper node; the schema refuses a Paper node whose prose
fails, however it is written; a person's waiver lets one named finding pass and is printed; "significantly" needs a
claim a rejecting test supports, and a causal word a claim a person approved. One study run for the file (about
12 s), then a lint or a build per test (latexmk, a few seconds). No network."""

import json
import shutil

import pytest

import openlab.study as vs
from openlab.sources import papers
from openlab.study.errors import StudyError
from openlab.study.paper import lint
from openlab.study.paper.bindings import mask_comments, parse
from openlab.study.paper.build import BUILD

pytestmark = pytest.mark.skipif(not shutil.which("latexmk"), reason="the paper is built with latexmk")

REVIEW = """
[review]
question = "Which statistical post-processing methods are used for station forecasts?"
sources = ["arxiv"]
queries = ["ensemble postprocessing"]
include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]
exclude = [{id = "E1", text = "not about weather"}]
citation_hops = 0
n = 5
coverage_floor = {queries_per_source = 1, hops = 0}

[audit]
n = 2
"""
SOURCE = papers._paper(title="Neural networks for postprocessing ensemble weather forecasts", arxiv="1805.09091",
                       version="preprint", authors=["Stephan Rasp", "Sebastian Lerch"], date="2018-05-23",
                       abstract="Ensemble weather predictions require statistical post-processing of systematic errors.")
TOML = """title = "Does the signal predict tomorrow?"
authors = ["A. Person"]
abstract = "sections/abstract.tex"
sections = ["introduction", "data", "methods", "literature", "results", "discussion", "limitations",
            "reproducibility", "checklist"]
years = true
"""
PROSE = {
    "abstract": r"We ask whether a daily signal predicts the next day's outcome. The model's mean absolute error is "
                r"\lab{@RES@#metrics.mae}, against \lab{@RES@#baselines.persistence.mae} for persistence.",
    "introduction": r"Statistical post-processing corrects forecasts "
                    r"\labcite{@SRC@}{require statistical post-processing of systematic errors}. The model is linear "
                    "in the signal:\n" r"\labeq{mu}{model.mu}{\mu(x) = 0.5 + 2x}" "\n"
                    r"% a comment may say 42 or \lab{nothing}",
    "results": r"\labtab{@RES@}" "\n\n" r"\labfig{@RES@}",
    "discussion": "Over the research segment of 2024 the model's error interval lies below that of persistence.",
}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A temporal template study with a one-source review, run once; model.py also defines mu(x) = 0.5 + 2x."""
    mp = pytest.MonkeyPatch()
    mp.setattr(papers, "ask", lambda src, q, n, since: [SOURCE])
    mp.setattr(papers, "crossref_updates", lambda dois: {})
    try:
        s = vs.new(tmp_path_factory.mktemp("paper") / "study", "temporal")
        (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + REVIEW)
        model = s.folder / "model.py"
        model.write_text(model.read_text() + "\n\ndef mu(x):\n    return 0.5 + 2 * x\n")
        s.run()
        s.discover()
    finally:
        mp.undo()
    return s.folder


def write(s, **edits) -> None:
    """paper.toml and the prose, with `edits` replacing a section's text."""
    res, src = s.record.last("result").id, s.record.last("source").id
    (s.folder / "paper" / "sections").mkdir(parents=True, exist_ok=True)
    (s.folder / "paper" / "paper.toml").write_text(TOML)
    for name, text in {**PROSE, **edits}.items():
        (s.folder / "paper" / "sections" / f"{name}.tex").write_text(
            text.replace("@RES@", res).replace("@SRC@", src) + "\n")


@pytest.fixture
def study(base, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_SIGNING_KEY", str(tmp_path / "signing.key"))
    shutil.copytree(base, tmp_path / "study")
    s = vs.open(tmp_path / "study")
    write(s)
    return s


def test_a_clean_paper_builds_to_a_signed_pdf_and_a_paper_node(study):     # ~6 s
    rec = study.record
    res, src = rec.last("result").id, rec.last("source").id
    assert study.paper_lint()["ok"]
    node = rec.get(study.paper()["id"])
    p = node.payload
    assert node.kind == "paper" and p["bindings"] == sorted({res, src}) and p["lint"] == []
    assert sorted(d for _, _, d in rec.links(node.id, rel="renders")) == p["bindings"]
    assert set(p["figures"]) == {res} and "spec_curve v1" in p["figures"][res]["generator"]
    build = study.folder / BUILD
    assert (build / "main.pdf").read_bytes()[:5] == b"%PDF-"
    tex = (build / "main.tex").read_text()
    for sec in ("Introduction", "Data", "Methods", "Literature", "Results", "Discussion", "Limitations",
                "Reproducibility", "Checklist", "Waivers"):
        assert f"\\section{{{sec}}}" in tex
    mae = rec.get(res).payload["metrics"]["mae"]
    assert f"{mae['value']:.3g}\\ [{mae['lo']:.3g},\\ {mae['hi']:.3g}]" in tex
    assert f"\\citep{{{src}}}" in tex and src in (build / "references.bib").read_text()
    assert "\\begin{equation}\\label{eq:mu}\\mu(x) = 0.5 + 2x" in tex
    assert rec.blob(p["pdf_sha"]) == (build / "main.pdf").read_bytes()
    v = study.verify_paper()
    assert v["ok"] and v["paper"] == node.id and v["head"] == node.hash, v
    (build / "main.pdf").write_bytes((build / "main.pdf").read_bytes() + b"\n% changed")
    assert not study.verify_paper()["ok"]


def test_a_first_writer_builds_the_paper_its_agent_node_does_not_stale_the_head(study):     # ~6 s
    from openlab.agents.identity import acting_as
    rec = study.record
    with acting_as("person:first"):                   # openlab show at a terminal: never wrote to this study yet
        assert not rec.find("agent", id="person:first")
        node = rec.get(study.paper()["id"])
    assert rec.find("agent", id="person:first") and node.session == "person:first"
    assert node.payload["head"] == node.prev


def test_a_preview_renders_unsaved_prose_with_its_findings_and_records_nothing(study):     # ~4 s
    from openlab.study.paper.build import preview
    rec = study.record
    head, papers_before = rec.head(), len(rec.find("paper"))
    disc = "paper/sections/discussion.tex"
    on_disk = (study.folder / disc).read_text()
    r = preview(study, {disc: "The error fell by 12 percent. \\lab{result:000000000000#metrics.mae}\n"})
    assert r["pdf"][:5] == b"%PDF-" and r["pages"][disc] >= 1 and r["bindings"] >= 3
    rules = {(f["file"], f["rule"]) for f in r["findings"]}
    assert {(disc, "number.typed"), (disc, "binding.unresolved")} <= rules
    assert rec.head() == head and len(rec.find("paper")) == papers_before and not (study.folder / BUILD).exists()
    assert (study.folder / disc).read_text() == on_disk                  # the file is as it was
    r = preview(study, {disc: "fine\n\\undefinedmacro\n"})
    assert r["pdf"] is None and r["error"] == "the paper does not compile"
    assert (r["errors"][0]["file"], r["errors"][0]["line"]) == (disc, 2)


@pytest.mark.parametrize("section, text, rule", [
    ("discussion", "The error fell by 12 percent over the year.", "number.typed"),
    ("discussion", r"The error is \lab{result:000000000000#metrics.mae} here.", "binding.unresolved"),
    ("discussion", "This is a novel way to forecast.", "wording.novelty"),
    ("discussion", "The model is significantly more accurate.", "wording.significance"),
    ("introduction", r"It helps \labcite{@SRC@}{post-processing always removes every error}.", "cite.quote"),
    ("introduction", r"\labeq{mu}{model.mu}{\mu(x) = 0.5 + 3x}", "eq.gate"),
])
def test_each_failure_refuses_the_build_and_writes_no_paper(study, section, text, rule):     # ~3 s each
    write(study, **{section: text})
    report = study.paper_lint()
    assert not report["ok"] and [f["rule"] for f in report["findings"]] == [rule], report["findings"]
    with pytest.raises(StudyError, match="the paper does not build: 1 finding") as e:
        study.paper()
    assert f"[{rule}]" in e.value.meaning and report["findings"][0]["id"] in e.value.meaning
    assert not study.record.find("paper") and not (study.folder / BUILD).exists()
    act = study.record.last("action")
    assert act.payload["verb"] == "paper" and not act.payload["outcome"]["ok"]


def test_a_binding_is_stale_once_model_py_changes(study):
    model = study.folder / "model.py"
    model.write_text(model.read_text() + "\n# edited after the run\n")
    with pytest.raises(StudyError) as e:
        study.paper()
    assert "[binding.stale]" in e.value.meaning and "model.py" in e.value.meaning
    assert "openlab study run" in e.value.meaning and not study.record.find("paper")


def test_an_ablation_binds_an_earlier_variant_and_results_say_how_many_trials(study):     # ~15 s
    """An earlier variant binds as \\lab[ablation]{...} (earlier code, same spec and data); bound plainly it is stale,
    so the headline is the current model. Claims and the results section say how many trials a result was selected
    from."""
    old = study.record.last("result").id
    (study.folder / ".openlab" / "origin").unlink()          # the fixture's copy stands in for the original here
    model = study.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    study.run(note="x only")
    write(study, discussion=rf"Without y\_now the error is \lab{{@RES@#metrics.mae}}; with it, "
                            rf"\lab[ablation]{{{old}#metrics.mae}}.")
    rep = study.paper_lint()
    assert rep["ok"], rep["findings"]
    (abl,) = [b for b in rep["bindings"] if b["ablation"]]
    assert abl["node"] == old and "model.py" in abl["earlier"] and not abl["stale"] and rep["trials"] == 2
    from openlab.study import cli
    assert cli.main(["paper-lint", str(study.folder)]) == 0
    write(study, discussion=rf"With y\_now the error is \lab{{{old}#metrics.mae}}.")
    stale = [f for f in study.paper_lint()["findings"] if f["rule"] == "binding.stale"]
    assert stale and "lab[ablation]" in stale[0]["message"]
    write(study, discussion=rf"With y\_now the error is \lab[ablate]{{{old}#metrics.mae}}.")
    assert any(f["rule"] == "binding.unresolved" and "no option [ablate]" in f["message"]
               for f in study.paper_lint()["findings"])

    c = study.claim("Without y_now the error is lower.", "observed", rests_on=[study.record.last("result").id])
    assert c["payload"]["selected_from"] == 2
    assert study.claims()[-1]["selected_from"] == 2
    write(study)
    study.paper()
    tex = (study.folder / BUILD / "main.tex").read_text()
    assert "Selected from 2 research trials." in tex


def test_the_schema_refuses_a_paper_node_whose_prose_fails(study):           # ~6 s
    rec = study.record
    p = rec.get(study.paper()["id"]).payload
    links = [("renders", i) for i in p["bindings"]]

    def forged(**edit) -> dict:
        sources = {**json.loads(rec.blob(p["sources"])), **edit}
        return {**p, "head": rec.head(), "sources": rec.put_blob(json.dumps(sources).encode())}
    with pytest.raises(StudyError, match="has no person's waiver"):
        rec.add("paper", forged(**{"paper/sections/discussion.tex": "A novel result."}), links=links)
    with pytest.raises(StudyError, match="do not resolve or are stale"):
        rec.add("paper", forged(**{"paper/sections/discussion.tex": r"\lab{result:000000000000#metrics.mae}"}),
                links=links)
    with pytest.raises(StudyError, match="binds exactly the nodes its prose binds"):
        rec.add("paper", {**forged(), "bindings": p["bindings"][:1]}, links=links[:1])
    with pytest.raises(StudyError, match="names the record head it was built at"):
        rec.add("paper", p, links=links)
    assert len(rec.find("paper")) == 1


def test_the_number_lint_passes_math_references_comments_and_allowed_names():
    text = "In $x_2 = 3$ and \\[y = 4\\] (Section 2, Table~\\ref{tab:a1}) we use 2D maps of COVID-19 % and 99\n" \
           "in 2016, 7 times."
    masked = mask_comments(text)
    plain = lint.prose(masked, parse(masked, "f.tex"))
    typed = lambda years: [f.message.split("`")[1] for f in lint.numbers("f.tex", plain, ("COVID-19",), years)]  # noqa: E731
    assert typed(True) == ["7"] and typed(False) == ["2016", "7"]


def test_a_persons_waiver_lets_one_named_finding_pass_and_is_printed(study, person_key):      # ~12 s
    first = study.paper()["id"]
    write(study, discussion="A novel reading of the signal. Another unprecedented turn.")
    ids = [f["id"] for f in study.paper_lint()["findings"]]
    assert len(ids) == 2
    with pytest.raises(StudyError, match="can only be written by person"):          # an agent cannot waive
        study.approve(first, "waive", f"{ids[0]}: the phrase is the cited work's title")
    with person_key.act():
        w = study.approve(first, "waive", f"{ids[0]}: the phrase is the cited work's title", person=True)["id"]
    with pytest.raises(StudyError) as e:                    # the waiver names one finding; the other still fails
        study.paper()
    assert ids[1] in e.value.meaning and ids[0] not in e.value.meaning
    write(study, discussion="A novel reading of the signal.")
    p = study.paper()["payload"]
    assert [(d["id"], d["waived_by"]) for d in p["lint"]] == [(ids[0], w)]
    appendix = (study.folder / BUILD / "main.tex").read_text().split("\\section{Waivers}")[1]
    assert ids[0] in appendix and w in appendix and "the cited work" in appendix


def test_claim_words_follow_the_claims_tests_and_a_persons_approval(study, person_key):       # ~8 s
    res = study.record.last("result").id
    c = study.claim("The model's error is below that of persistence.", "observed", [res])["id"]
    write(study, discussion=f"The error is significantly below chance alone: \\labclaim{{{c}}}.\n\n"
                            f"The signal causes the outcome: \\labclaim{{{c}}}.")
    assert sorted(f["rule"] for f in study.paper_lint()["findings"]) == ["wording.causal", "wording.significance"]
    study.test(c, res, "persistence", 0.04)                 # the paired p rejects, with the model better
    assert [f["rule"] for f in study.paper_lint()["findings"]] == ["wording.causal"]
    with person_key.act():
        study.approve(c, "approve", "the runs vary the signal alone", person=True)
    assert study.paper_lint()["ok"]
    study.paper()
    tex = (study.folder / BUILD / "main.tex").read_text()
    assert "\\emph{The model's error is below that of persistence.} \\textup{[observed; supported; recorded; selected from 1 trial; " \
           "approved by a person]}" in tex


# --- the one-document paper: paper/paper.tex ---------------------------------------------------------------------
def _png(path) -> None:
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 1.5))
    fig.subplots().plot([0, 1, 2], [1, 0, 2])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png")


ONE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{xcolor}
\title{One document}
\author{An agent}
\begin{document}
\maketitle
\begin{abstract}
The model's error is \lab{@RES@#metrics.mae}.
\end{abstract}
\section{Introduction}
\setlength{\parskip}{6pt}
A \textcolor{blue}{coloured} word, and the hypothesis H0 on the series pnl3.
\section{Results}
\labtab{@RES@}
\begin{figure}[htbp]\centering
\includegraphics[width=0.5\linewidth]{figures/plot.png}
\caption{A plot the agent made.}
\end{figure}
\section{Reproducibility}
\labsection{reproducibility}
\end{document}
"""


def one(s, text=ONE) -> str:
    (s.folder / "paper").mkdir(exist_ok=True)
    text = text.replace("@RES@", s.record.last("result").id)
    (s.folder / "paper" / "paper.tex").write_text(text)
    return text


def test_a_one_document_paper_loads_its_packages_includes_its_images_and_records_them(study):     # ~6 s
    _png(study.folder / "paper" / "figures" / "plot.png")
    one(study)
    assert study.paper_lint()["ok"], study.paper_lint()["findings"]       # 11pt, 2cm, 6pt, 0.5\linewidth, H0, pnl3
    rec = study.record
    node = rec.get(study.paper()["id"])
    fig = node.payload["figures"]["file:paper/figures/plot.png"]
    assert fig["generator"] == "file" and rec.blob(fig["sha256"]) == (study.folder / "paper/figures/plot.png").read_bytes()
    tex = (study.folder / BUILD / "main.tex").read_text()
    assert "{media/paper/figures/plot.png}" in tex and "Figures from files" in tex and "\\usepackage{xcolor}" in tex
    assert tex.count("\\section{Introduction}") == 1 and "\\@ifpackageloaded{booktabs}" in tex
    assert "\\section{Data}" not in tex                                   # only the sections the paper writes
    assert study.verify_paper()["ok"]


def test_an_image_it_cannot_include_refuses_the_build_and_shows_as_a_box_in_the_draft(study, tmp_path):
    from openlab.study.paper.build import draft
    _png(tmp_path / "outside.png")
    for path, why in ((str(tmp_path / "outside.png"), "outside the study"), ("figures/none.png", "not a file"),
                      ("../.openlab/record.db", "a paper never includes")):
        one(study, ONE.replace("figures/plot.png", path))
        with pytest.raises(StudyError, match="cannot") as e:
            study.paper()
        assert why in e.value.meaning
    papers_before = len(study.record.find("paper"))
    r = draft(study)
    assert r["compiled"] and (study.folder / BUILD / "draft.pdf").read_bytes()[:5] == b"%PDF-"
    assert len(study.record.find("paper")) == papers_before


def test_the_draft_lists_findings_and_a_latex_error_points_at_its_line_of_paper_tex(study):
    from openlab.study.paper.build import draft, preview
    text = one(study, ONE.replace("A \\textcolor", "The error fell by 12 percent. A \\textcolor"))
    _png(study.folder / "paper" / "figures" / "plot.png")
    r = draft(study)
    assert r["compiled"] and [f["rule"] for f in r["findings"]] == ["number.typed"]
    assert "fix the findings" in r["next"]
    bad = text.replace("\\section{Results}", "\\section{Results}\n\\undefinedcommand")
    line = bad.splitlines().index("\\undefinedcommand") + 1
    p = preview(study, {"paper/paper.tex": bad})
    assert p["pdf"] is None and {"file": "paper/paper.tex", "line": line} == {k: p["errors"][0][k] for k in ("file", "line")}


def test_init_carries_an_older_paper_over_into_one_document_with_each_heading_once(study):     # ~6 s
    from openlab.study.paper.build import init
    r = init(study)
    assert r["carried_over"] and (study.folder / "paper" / "paper.tex").is_file()
    text = (study.folder / "paper" / "paper.tex").read_text()
    assert "\\labsection{data}" in text and "\\labsection{reproducibility}" in text and "post-processing" in text
    with pytest.raises(StudyError, match="exists"):
        init(study)
    study.paper()
    tex = (study.folder / BUILD / "main.tex").read_text()
    assert tex.count("\\section{Introduction}") == 1 and tex.count("\\section{Results}") == 1


def test_a_new_paper_starts_from_the_template_headings(study):
    from openlab.study.paper.build import init
    shutil.rmtree(study.folder / "paper")
    init(study)
    text = (study.folder / "paper" / "paper.tex").read_text()
    for h in ("Introduction", "Data", "Methods", "Results", "Discussion", "Limitations", "Reproducibility"):
        assert f"\\section{{{h}}}" in text
    assert "\\begin{abstract}" in text
    from openlab.study.paper.build import draft
    assert draft(study)["compiled"]
