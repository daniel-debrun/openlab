r"""`openlab study paper`: the research -> paper build.

The paper is one LaTeX document, paper/paper.tex (`study_paper_init` writes it from a template: the abstract and the
usual headings). It is the agent's own: any package it loads (\usepackage), any image of the study's folder it
includes (\includegraphics{figures/x.png}, a path from paper/ or from the study; copied into the build, its sha256
recorded, and listed in the paper as a figure from a file, not drawn by openlab), and `\labsection{<name>}` for the
text of a section openlab generates (below). Only its body is linted. paper/paper.toml, when there is one beside it,
holds only the settings (digits, years, allow, checklist).

The older layout, without paper.tex: paper/paper.toml names the title, authors, the abstract file and the section order. A section is the agent's prose
in paper/sections/<name>.tex, openlab's generated section of that name (data, methods, literature, results,
limitations, reproducibility, checklist), or both: the prose first. The prose binds numbers, tables, figures,
claims, citations and equations to the record with macros (bindings.py); it is linted (lint.py); the rest is
generated (sections.py). The build refuses, listing every finding, while a binding is unresolved or stale or a
finding has no person's waiver. Then latexmk builds paper/build/main.pdf (no shell escape, paranoid file
access), the Paper node is written (its schema analyses the prose again) and the bundle is signed.

Stale bindings: a binding whose run used other code, spec or data than the study has now is refused, so the
headline is always the current model. An ablation table binds earlier variants on purpose with the [ablation]
option, `\lab[ablation]{<result>#metrics.mae}` (also \labtab, \labfig, \labclaim): code may differ, spec and data may
not. Selection: every result the paper prints says how many research trials it was selected from (the study's trial
count at build time), in its table's caption, a bound claim's marks and the generated results section.

    title = "..."
    authors = ["..."]
    abstract = "sections/abstract.tex"              # relative to paper/
    sections = ["introduction", "data", "methods", "results", "discussion", "limitations", "reproducibility"]
    digits = 3                                      # significant digits of a bound value
    years = true                                    # years 1900-2100 may be typed in prose
    allow = ["COVID-19"]                            # tokens with digits that are names, not results
    checklist = "neurips"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10
    import tomli as tomllib

from .. import clock
from .checklists import CHECKLISTS
from .. import spec as _spec
from ..errors import StudyError
from ..graph import retracted_since
from ..record import Node
from ..schema import Paper
from . import bindings, lint, sections

TOML = "paper/paper.toml"
PAPER = "paper/paper.tex"                                    # the one-document paper (the older layout: paper.toml)
MEDIA = (".png", ".jpg", ".jpeg", ".pdf")                    # what \includegraphics may include, as pdflatex reads it
MEDIA_MAX = 20 << 20                                         # bytes of one included file
BUILD = "paper/build"
SIG = "main.pdf.sig.json"
LATEX_TIMEOUT = 180
SHOWN = 20                                                   # findings a refusal names before it counts the rest
_EXAMPLE = 'title = "..."\nauthors = ["..."]\nabstract = "sections/abstract.tex"\nsections = ["introduction", "methods", "results"]'


@dataclass(frozen=True)
class Config:
    title: str
    authors: tuple
    abstract: str
    sections: tuple
    digits: int = 3
    years: bool = False
    allow: tuple = ()
    checklist: str = "neurips"


_KEYS = {"title": str, "authors": list, "abstract": str, "sections": list, "digits": int, "years": bool,
         "allow": list, "checklist": str}


def config(text: str, single: bool = False) -> Config:
    """paper.toml, validated. `single`: beside paper/paper.tex, which holds the title, the abstract and the sections,
    so paper.toml (if any) only sets digits, years, allow and checklist."""
    if single and not text.strip():
        return Config("", (), "", ())
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise StudyError(f"{TOML} does not parse", str(e)) from None
    bad = [k for k in raw if k not in _KEYS] + [k for k, t in _KEYS.items() if k in raw and type(raw[k]) is not t]
    missing = [] if single else [k for k in ("title", "authors", "abstract", "sections") if k not in raw]
    if bad or missing:
        raise StudyError(f"{TOML}: " + "; ".join([f"`{k}` is unknown or of the wrong type" for k in bad]
                                                 + [f"`{k}` is required" for k in missing]),
                         "keys: " + ", ".join(f"{k} ({t.__name__})" for k, t in _KEYS.items()), _EXAMPLE)
    if raw.get("checklist", "neurips") not in CHECKLISTS or not 1 <= raw.get("digits", 3) <= 8:
        raise StudyError(f"{TOML}: digits is 1..8 and checklist one of {', '.join(CHECKLISTS)}")
    if single:
        return Config("", (), "", (), raw.get("digits", 3), raw.get("years", False), tuple(raw.get("allow", ())),
                      raw.get("checklist", "neurips"))
    names = raw["sections"]
    if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and re.fullmatch(r"[a-z][a-z0-9_]*", n)
                                                              for n in names):
        raise StudyError(f"{TOML}: `sections` is a non-empty list of distinct names (a-z, 0-9, _)")
    if ".." in Path(raw["abstract"]).parts or Path(raw["abstract"]).is_absolute():
        raise StudyError(f"{TOML}: `abstract` is a file under paper/")
    return Config(raw["title"], tuple(raw["authors"]), raw["abstract"], tuple(names), raw.get("digits", 3),
                  raw.get("years", False), tuple(raw.get("allow", ())), raw.get("checklist", "neurips"))


def _files(cfg: Config) -> list:
    """(path relative to the study, section name or "") of the abstract and each section's prose."""
    return [(f"paper/{cfg.abstract}", "")] + [(f"paper/sections/{n}.tex", n) for n in cfg.sections]


def single(folder: Path, overrides: Optional[dict] = None) -> bool:
    """Whether the study's paper is the one document paper/paper.tex (else the older layout of paper.toml)."""
    return PAPER in (overrides or {}) or (Path(folder) / PAPER).is_file()


def layout(sources: dict) -> tuple:
    """(config, [(path, section name or "")]) of the files in `sources`: paper.tex alone, or paper.toml's files."""
    if PAPER in sources:
        return config(sources.get(TOML, ""), single=True), [(PAPER, "")]
    cfg = config(sources[TOML])
    return cfg, _files(cfg)


def read_sources(folder: Path, overrides: Optional[dict] = None) -> dict:
    """{path: text} of the paper's files: paper.tex (with paper.toml when there is one), or paper.toml and the prose
    files it names; what the build reads, kept as one blob. `overrides` ({path: text}) stands in for files on disk:
    a draft not saved yet (preview)."""
    overrides = overrides or {}
    if single(folder, overrides):
        out = {}
        for rel in (PAPER, TOML):
            if rel in overrides:
                out[rel] = overrides[rel]
            elif (folder / rel).is_file():
                out[rel] = (folder / rel).read_text(encoding="utf-8")
        return out
    try:
        text = overrides[TOML] if TOML in overrides else (folder / TOML).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise StudyError(f"no {TOML}", "the paper's title, authors, abstract and section order", _EXAMPLE) from None
    out = {TOML: text}
    for rel, name in _files(config(text)):
        if rel in overrides:
            out[rel] = overrides[rel]
        elif (folder / rel).is_file():
            out[rel] = (folder / rel).read_text(encoding="utf-8")
        elif name not in sections.GENERATED:
            raise StudyError(f"{rel} is missing", f"`{name or 'abstract'}` is in {TOML} and openlab generates no "
                             f"section of that name", f"write it, or drop it from {TOML}")
    return out


_LABSECTION = re.compile(r"\\labsection\s*\{([^{}]*)\}")
_GRAPHICS = re.compile(r"\\includegraphics\s*\*?\s*(?:\[[^\]]*\])?\s*\{(?P<path>[^{}]+)\}")
_BEGIN, _END = re.compile(r"\\begin\s*\{document\}"), re.compile(r"\\end\s*\{document\}")
#: the packages openlab's own LaTeX (tables, figures, citations) uses, each loaded only when the paper has not
PREAMBLE = r"""% --- openlab: the packages its tables, figures and citations use, each only if the paper loads it not
\makeatletter
\@ifpackageloaded{amsmath}{}{\usepackage{amsmath}}\@ifpackageloaded{amssymb}{}{\usepackage{amssymb}}
\@ifpackageloaded{array}{}{\usepackage{array}}\@ifpackageloaded{booktabs}{}{\usepackage{booktabs}}
\@ifpackageloaded{graphicx}{}{\usepackage{graphicx}}
\@ifpackageloaded{biblatex}{}{\@ifpackageloaded{natbib}{}{\usepackage[numbers,sort&compress]{natbib}}}
\@ifpackageloaded{hyperref}{}{\usepackage[hidelinks]{hyperref}}
\makeatother
"""


def body(text: str) -> str:
    """paper.tex with all but its document body blanked (offsets kept): the preamble is not prose."""
    b, e = _BEGIN.search(text), _END.search(text)
    start, end = (b.end() if b else 0), (e.start() if e else len(text))
    return _blank(text[:start]) + text[start:end] + _blank(text[end:])


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def media(folder: Path, text: str) -> list:
    """Each \\includegraphics of paper.tex (comments masked): {start, end, path, file (relative to the study), error}.
    A path is read from paper/, then from the study; one with no extension tries each of MEDIA. It must be a regular
    file of the study, outside .openlab/ and .git/, of a kind pdflatex reads, under MEDIA_MAX bytes."""
    root, out = Path(folder).resolve(), []
    for m in _GRAPHICS.finditer(bindings.mask_comments(text)):
        path, file, error = m.group("path").strip(), "", ""
        tries = [path] if Path(path).suffix else [path + x for x in MEDIA]
        found = [c for base in (root / "paper", root) for t in tries for c in [(base / t)] if c.is_file()]
        if not found:
            error = f"{path} is not a file of the study (read from paper/, then from the study)"
        else:
            c = found[0]
            real = c.resolve()
            try:
                rel = real.relative_to(root)
            except ValueError:
                rel = None
            if rel is None or c.is_symlink():
                error = f"{path} is outside the study, or a link"
            elif rel.parts[0] in (".openlab", ".git"):
                error = f"{path} is in {rel.parts[0]}/, which a paper never includes"
            elif real.suffix.lower() not in MEDIA:
                error = f"{path}: a paper includes {', '.join(MEDIA)} files"
            elif real.stat().st_size > MEDIA_MAX:
                error = f"{path} is over {MEDIA_MAX >> 20} MB"
            else:
                file = rel.as_posix()
        out.append({"start": m.start("path"), "end": m.end("path"), "path": path, "file": file, "error": error})
    return out


def _copy_media(ctx, found: list, keep: bool) -> dict:
    """Copy each included file into the build (media/<its path in the study>) and note it in ctx.figures
    ("file:<path>": file, sha256, generator "file"); `keep` puts its bytes in the record's blobs. {path: built name}."""
    names = {}
    for f in found:
        if f["error"] or f["file"] in names:
            continue
        src, name = ctx.study.folder / f["file"], f"media/{f['file']}"
        data = src.read_bytes()
        (ctx.build / name).parent.mkdir(parents=True, exist_ok=True)
        (ctx.build / name).write_bytes(data)
        sha = ctx.rec.put_blob(data) if keep else hashlib.sha256(data).hexdigest()
        ctx.figures[f"file:{f['file']}"] = {"file": name, "sha256": sha, "generator": "file", "source": f["file"]}
        names[f["file"]] = name
    return names


def init(study) -> dict:
    """Write paper/paper.tex from the template (the abstract and the usual headings), titled by the study's question.
    A paper in the older layout (paper.toml and its section files) is carried over into it: its title and authors,
    the abstract, and each section's prose under its heading (the prose's own \\section heading when it opens with
    one), with \\labsection{<name>} after a section openlab generates. Refused when paper.tex exists."""
    from .tex import esc
    folder = study.folder
    if (folder / PAPER).exists():
        raise StudyError(f"{PAPER} exists", "the paper is already one document", "edit it")
    tex = (Path(__file__).parent / "templates" / "paper.tex").read_text(encoding="utf-8")
    title, authors, carried = esc((study.spec.question or study.spec.name).split("\n")[0][:160]), "", []
    if (folder / TOML).is_file():
        cfg = config((folder / TOML).read_text(encoding="utf-8"))
        title, authors = esc(cfg.title), r" \and ".join(esc(a) for a in cfg.authors)
        part = []
        for rel, name in _files(cfg):
            text = (folder / rel).read_text(encoding="utf-8").strip() if (folder / rel).is_file() else ""
            if not name:
                tex = re.sub(r"(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})",
                             lambda m: m.group(1) + (text or "What was asked, what was done, and what was found.") + m.group(2), tex)
                continue
            head = re.match(r"\\section\*?\{([^{}]*)\}\s*(\\label\{[^{}]*\})?\s*", text)
            heading = head.group(1) if head else name.replace("_", " ").capitalize()
            prose = text[head.end():] if head else text
            made = f"\n\\labsection{{{name}}}" if name in sections.GENERATED else ""
            part.append(f"\\section{{{heading}}}\n{prose}{made}\n")
            carried += [rel] if text else []
        tex = re.sub(r"(?s)\\section\{Introduction\}.*?(?=\\end\{document\})", lambda m: "\n".join(part) + "\n", tex)
    tex = tex.replace("<<title>>", title).replace("<<authors>>", authors or "the study's agents")
    (folder / PAPER).parent.mkdir(parents=True, exist_ok=True)
    (folder / PAPER).write_text(tex, encoding="utf-8")
    return {"file": PAPER, "carried_over": carried,
            "next": "write the paper in paper.tex (study_paper draft=true renders it as it stands); paper.toml now "
                    "only sets digits, years, allow and checklist" + ("; the section files are no longer read"
                                                                       if carried else "")}


def _unique(found: list) -> list:
    return list({f.id: f for f in reversed(found)}.values())[::-1]


def analyse(rec, folder: Path, sources: dict) -> tuple:
    """(config, bindings, findings) of the prose in `sources` against the record: every macro resolved and checked
    for staleness, the number and wording lints, the citation gate. Deterministic: the Paper schema re-runs it."""
    cfg, files = layout(sources)
    now = bindings.current(rec, folder)
    bounds, found = [], []
    for rel, _ in files:
        if rel not in sources:
            continue
        masked = bindings.mask_comments(sources[rel])
        if rel == PAPER:
            masked = body(masked)
            found += [lint.finding("section.unknown", rel, masked.count("\n", 0, m.start()) + 1, m.group(1),
                                   f"\\labsection{{{m.group(1)}}}: openlab generates no section of that name (it "
                                   f"generates {', '.join(sections.GENERATED)})", waivable=False)
                      for m in _LABSECTION.finditer(masked) if m.group(1) not in sections.GENERATED]
        macros = bindings.parse(masked, rel)
        bs = [bindings.resolve(rec, m, now) for m in macros]
        plain = lint.prose(masked, macros)
        bounds += bs
        found += [f for b in bs for f in lint.bound(b)]
        found += lint.numbers(rel, plain, cfg.allow, cfg.years) + lint.wording(rec, rel, plain, bs)
        found += lint.math_numbers(rel, masked, macros, cfg.allow)
    return cfg, bounds, _unique(found + lint.cites(rec, bounds) + lint.traced(rec, bounds))


def _gate(study) -> tuple:
    """analyse, then the equation gate: (config, sources, bindings, equations, [(finding, waiver or None)])."""
    rec = study.record
    sources = read_sources(study.folder)
    cfg, bounds, found = analyse(rec, study.folder, sources)
    eqs = bindings.equations(study, [b.macro for b in bounds if b.macro.name == "labeq" and not b.error])
    found += [lint.equation(m, why) for m, why in eqs.items() if why]
    return cfg, sources, bounds, eqs, [(f, lint.waiver(rec, f)) for f in found]


def selected_from(rec) -> int:
    """The research trials the paper's results were selected from: the study's trial count now (with the studies
    sharing its ledger)."""
    from ..ledger import shared
    return shared(rec.folder, rec)["trials"]


def lint_report(study) -> dict:
    """The paper checked without building it: its bindings and every finding, open or waived by a person."""
    rec = study.record
    _, _, bounds, _, found = _gate(study)
    rows = [{**f.to_dict(), "waived_by": w.id if w else None} for f, w in found]
    return {"ok": all(r["waived_by"] for r in rows),
            "bindings": [{"macro": b.macro.text[:160], "at": f"{b.macro.file}:{b.macro.line}", "node": b.node,
                          "error": b.error, "stale": b.stale, "ablation": b.macro.option == "ablation",
                          "earlier": b.earlier} for b in bounds],
            "trials": selected_from(rec),
            "findings": [r for r in rows if not r["waived_by"]], "waived": [r for r in rows if r["waived_by"]]}


def _refuse(open_: list) -> StudyError:
    lines = [f"{f.location} [{f.rule}] {f.message} (id {f.id})" for f in open_[:SHOWN]]
    more = [f"... {len(open_) - SHOWN} more"] if len(open_) > SHOWN else []
    return StudyError(f"the paper does not build: {len(open_)} finding{'s' if len(open_) != 1 else ''}",
                      "\n".join(lines + more), "fix the prose (study_paper_lint lists every finding); a stale binding "
                      "needs `openlab study run`; only a person can waive a finding, by its id")


def _prose(ctx, text: str, rel: str, bounds: list, render=None) -> str:
    """The prose with each macro replaced by its LaTeX, marked with its file for the LaTeX log."""
    render = render or sections.render
    for b in sorted((b for b in bounds if b.macro.file == rel), key=lambda b: -b.macro.start):
        text = text[:b.macro.start] + render(ctx, b) + text[b.macro.end:]
    return f"% --- {rel}\n{text.strip()}\n"


def _tex_single(ctx, cfg: Config, sources: dict, bounds: list, waived: list, render=None, missing=None) -> str:
    """paper.tex as main.tex: each macro rendered, each \\labsection its generated text, each included file's path the
    copy in the build, openlab's packages before \\begin{document}, and before \\end{document} the bibliography (when
    it cites and does not print one), the figures from files and the waivers. A marked line (`% --- paper/paper.tex:N`)
    after each insertion of several lines keeps LaTeX's line numbers traceable to paper.tex. `missing`: how a file
    that cannot be included is drawn (the draft); else it is refused."""
    text = sources[PAPER]
    plain = bindings.mask_comments(text)
    b, e, masked = _BEGIN.search(plain), _END.search(plain), body(plain)
    if not b or not e:
        raise StudyError(f"{PAPER} is not a whole LaTeX document", "it needs \\documentclass, \\begin{document} and "
                         "\\end{document}", "start from the template: openlab study paper --init (study_paper_init)")
    render = render or sections.render
    found = media(ctx.study.folder, text)
    bad = [f for f in found if f["error"]]
    if bad and missing is None:
        raise StudyError(f"the paper includes {len(bad)} file{'s' if len(bad) > 1 else ''} it cannot",
                         "; ".join(f"{PAPER}:{text.count(chr(10), 0, f['start']) + 1}: {f['error']}" for f in bad[:5]),
                         "put the image in the study's folder (e.g. paper/figures/) and include it by that path")
    names = _copy_media(ctx, found, keep=missing is None)
    edits = [(x.macro.start, x.macro.end, render(ctx, x)) for x in bounds if x.macro.file == PAPER]
    edits += [(m.start(), m.end(), sections.GENERATED[m.group(1)](ctx)) for m in _LABSECTION.finditer(masked)
              if m.group(1) in sections.GENERATED]
    for f in found:
        if f["error"]:
            g = _GRAPHICS.match(text, text.rfind("\\includegraphics", 0, f["start"]))
            edits.append((g.start(), g.end(), missing(f)) if g else (f["start"], f["end"], f["path"]))
        else:
            edits.append((f["start"], f["end"], names[f["file"]]))
    edits.append((b.start(), b.start(), PREAMBLE))
    edits.append((e.start(), e.start(), _tail(ctx, text, waived)))
    out = text
    for start, end, new in sorted(edits, key=lambda x: -x[0]):
        if "\n" in new:
            new = f"{new}%\n% --- {PAPER}:{text.count(chr(10), 0, end) + 1}\n"
        out = out[:start] + new + out[end:]
    return f"% --- {PAPER}:1\n" + out


def _tail(ctx, text: str, waived: list) -> str:
    """What openlab adds at the end of a one-document paper: its bibliography, its figures from files, its waivers."""
    from .tex import esc
    parts = []
    if ctx.cited and "\\bibliography{" not in text and "\\printbibliography" not in text:
        parts.append("\\bibliographystyle{plainnat}\n\\bibliography{references}")
    files = [v for k, v in ctx.figures.items() if k.startswith("file:")]
    if files:
        parts.append(r"\paragraph{Figures from files.} These images come from the study's folder, not from the "
                     r"record: openlab did not draw them and checks no number in them. "
                     + "; ".join(rf"\texttt{{{esc(v['source'])}}} (sha256 \texttt{{{v['sha256'][:12]}}})" for v in files) + ".")
    if waived:
        parts.append(r"\appendix" + "\n" + sections.waivers(ctx.rec, waived))
    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


def _tex(ctx, cfg: Config, sources: dict, bounds: list, waived: list, render=None, missing=None) -> str:
    from .tex import esc
    if PAPER in sources:
        return _tex_single(ctx, cfg, sources, bounds, waived, render, missing)
    files = dict((n, rel) for rel, n in _files(cfg))
    body = []
    for name in cfg.sections:
        prose = _prose(ctx, sources[files[name]], files[name], bounds, render) if files[name] in sources else ""
        made = sections.GENERATED[name](ctx) if name in sections.GENERATED else ""
        if prose.strip() or made.strip():
            body.append(rf"\section{{{esc(name.replace('_', ' ').capitalize())}}}\label{{sec:{name}}}" + "\n"
                        + prose + made)
    abstract = f"paper/{cfg.abstract}"
    slots = {"title": esc(cfg.title), "authors": r" \and ".join(esc(a) for a in cfg.authors),
             "date": f"built by openlab on {clock.now()[:10]}",
             "abstract": _prose(ctx, sources[abstract], abstract, bounds, render) if abstract in sources else "",
             "body": "\n\n".join(body),
             "bibliography": "\\bibliographystyle{plainnat}\n\\bibliography{references}" if ctx.cited else "",
             "appendix": sections.waivers(ctx.rec, waived)}
    tex = (Path(__file__).parent / "templates" / "article.tex").read_text(encoding="utf-8")
    return re.sub(r"<<(\w+)>>", lambda m: slots[m.group(1)], tex)


def _latexmk(out: Path) -> bytes:
    """main.tex -> main.pdf: latexmk with no rc files, no shell escape, and TeX reading and writing only in the
    build directory (openin_any/openout_any = p), so prose cannot pull a file from elsewhere into the PDF."""
    exe = shutil.which("latexmk")
    if not exe:
        raise StudyError("latexmk is not installed", fix="install TeX Live (latexmk, pdflatex, bibtex)")
    cmd = [exe, "-norc", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
           "-latexoption=-no-shell-escape", "main.tex"]
    try:
        r = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=LATEX_TIMEOUT,
                           env={**os.environ, "openin_any": "p", "openout_any": "p"})
    except subprocess.TimeoutExpired:
        raise StudyError(f"the LaTeX build took more than {LATEX_TIMEOUT} s") from None
    pdf = out / "main.pdf"
    if r.returncode or not pdf.is_file():
        log = (out / "main.log").read_text(errors="replace") if (out / "main.log").is_file() else r.stdout + r.stderr
        errs = [ln for ln in log.splitlines() if ln.startswith(("./main.tex:", "!"))][:8]
        raise StudyError("the paper does not compile", "\n".join(errs) or log[-1500:],
                         f"see {BUILD}/main.tex at the line given: each prose file is marked `% --- <file>` there")
    return pdf.read_bytes()


def _marked(ctx, b) -> str:
    """A binding in a preview: its value, or, when it does not resolve, a boxed ?? naming the macro."""
    from .tex import esc
    if not b.error:
        try:
            return sections.render(ctx, b)
        except Exception:                                               # noqa: BLE001 -- the preview still builds
            pass
    return rf"\fbox{{\textbf{{??}}~\texttt{{{esc(b.macro.text[:48])}}}}}"


def _where(tex: str, line: int) -> tuple:
    """(prose file, line in it) of line `line` of main.tex, by the `% --- <file>` marks; ("main.tex", line) outside."""
    rows = tex.splitlines()
    for i in range(min(line, len(rows)) - 1, -1, -1):
        if rows[i].startswith("% --- "):
            mark = rows[i][6:].strip()
            n = re.search(r":(\d+)$", mark)
            return (mark[:n.start()], int(n.group(1)) + line - i - 2) if n else (mark, line - i - 1)
    return "main.tex", line


def _missing(f: dict) -> str:
    """A file the draft cannot include, drawn as a box naming it and why."""
    from .tex import esc
    return rf"\fbox{{\parbox{{0.8\linewidth}}{{\textbf{{missing image}}~\texttt{{{esc(f['path'])}}}: {esc(f['error'])}}}}}"


def draft(study) -> dict:
    """The paper rendered as it stands, findings and all, into paper/build/draft.pdf: the draft an agent reads while
    it writes. Nothing is recorded or signed; the findings it lists are what the signed build (`build`) refuses."""
    r = preview(study)
    out = study.folder / BUILD
    if r["pdf"]:
        out.mkdir(parents=True, exist_ok=True)
        (out / "draft.pdf").write_bytes(r["pdf"])
    open_ = [f for f in r["findings"] if not f["waived"]]
    return {"draft": f"{BUILD}/draft.pdf" if r["pdf"] else "", "compiled": bool(r["pdf"]), "error": r["error"],
            "latex_errors": r["errors"][:8], "bindings": r["bindings"], "findings": open_[:SHOWN],
            "more_findings": max(0, len(open_) - SHOWN),
            "next": "fix the findings, then study_paper without draft to build, record and sign it" if open_
            else "no findings: study_paper without draft builds, records and signs it"}


def preview(study, overrides: Optional[dict] = None) -> dict:
    """The paper as it would build now, with `overrides` ({path: text}, a draft not saved yet) in place of the files
    on disk: {pdf: bytes or None, error, errors: [{file, line, message}], findings: [{file, line, rule, message,
    waived}], bindings}. A finding does not stop it and an unresolved binding prints as a boxed ??. Nothing is
    recorded or signed and paper/build is not touched: it builds in a folder of its own, removed after."""
    import tempfile
    rec, folder = study.record, study.folder
    sources = read_sources(folder, overrides)
    cfg, bounds, found = analyse(rec, folder, sources)
    files = layout(sources)[1]
    eqs = bindings.equations(study, [b.macro for b in bounds if b.macro.name == "labeq" and not b.error])
    found += [lint.equation(m, why) for m, why in eqs.items() if why]
    rows = []
    for f in found:
        d = f.to_dict()
        file, _, line = str(d.get("location") or "").partition(":")
        rows.append({"file": file, "line": int(line.split(":")[0]) if line.split(":")[0].isdigit() else 0,
                     "rule": d.get("rule", ""), "message": d.get("message", ""), "id": d.get("id", ""),
                     "waived": bool(lint.waiver(rec, f))})
    out = {"pdf": None, "error": "", "errors": [], "findings": rows, "bindings": len(bounds), "pages": {}}
    with tempfile.TemporaryDirectory(prefix="openlab-preview-") as d:
        tmp = Path(d)
        now = bindings.current(rec, folder)
        ctx = sections.Ctx(study, cfg.digits, tmp, sections.current_results(rec, now), eqs, cfg.checklist)
        tex = _tex(ctx, cfg, sources, bounds, [], _marked, _missing)
        (tmp / "main.tex").write_text(tex, encoding="utf-8")
        if ctx.cited:
            (tmp / "references.bib").write_text(sections.bibtex(ctx), encoding="utf-8")
        try:
            out["pdf"] = _latexmk(tmp)
            aux = (tmp / "main.aux").read_text(errors="replace") if (tmp / "main.aux").is_file() else ""
            at = {m.group(1): int(m.group(2)) for m in re.finditer(r"\\newlabel\{sec:([^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux)}
            out["pages"] = {rel: at.get(name, 1) if name else 1 for rel, name in files}
        except StudyError as e:
            out["error"] = e.error
            for ln in (e.meaning or "").splitlines():
                head, _, msg = ln.partition(": ") if ln.startswith("./main.tex:") else ("", "", ln)
                num = head.rpartition(":")[2]
                file, line = _where(tex, int(num)) if num.isdigit() else ("", 0)
                out["errors"].append({"file": file, "line": line, "message": (msg or ln).strip()})
    return out


def build(study, person: bool = False) -> Node:
    """Build, record and sign the paper; refused, naming the findings, unless every binding resolves and is
    current and every finding is waived by a person."""
    from ... import __version__
    from ..signing import sign_file
    rec, folder = study.record, study.folder
    _spec.check(folder, rec)
    cfg, sources, bounds, eqs, found = _gate(study)
    open_ = [f for f, w in found if w is None]
    if open_:
        raise _refuse(open_)
    out = folder / BUILD
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    now = bindings.current(rec, folder)
    ctx = sections.Ctx(study, cfg.digits, out, sections.current_results(rec, now), eqs, cfg.checklist)
    tex = _tex(ctx, cfg, sources, bounds, found)
    (out / "main.tex").write_text(tex, encoding="utf-8")
    if ctx.cited:
        (out / "references.bib").write_text(sections.bibtex(ctx), encoding="utf-8")
    pdf = _latexmk(out)
    ids = sorted({b.node for b in bounds if b.node})
    figures = {k: {f: v[f] for f in ("file", "sha256", "generator", "source") if f in v} for k, v in ctx.figures.items()}
    blobs = {"tex_sha": rec.put_blob(tex.encode("utf-8")), "pdf_sha": rec.put_blob(pdf),
             "sources": rec.put_blob(json.dumps(sources, sort_keys=True).encode("utf-8"))}
    waived = [{**f.to_dict(), "waived_by": w.id} for f, w in found]
    if not person:                  # a first writer's `agent` node goes on now: written inside add, it moves the head
        from ...agents.identity import current, ensure_agent_node
        ensure_agent_node(rec, current())
    node = rec.add(Paper(**blobs, bindings=ids, lint=waived, head=rec.head(), signed=f"{BUILD}/{SIG}", figures=figures),
                   actor="person" if person else "agent", links=[("renders", i) for i in ids])
    p = node.payload
    sign_file(out / "main.pdf", {"kind": "openlab-paper", "study": study.spec.name, "file": f"{BUILD}/main.pdf",
                                 "paper": node.id, "head": node.hash, "tex_sha": p["tex_sha"], "pdf_sha": p["pdf_sha"],
                                 "spec_sha": now["spec_sha"], "code": now["code"], "data": now["data"],
                                 "figures": figures, "signed_at": clock.now(), "openlab_version": __version__})
    return node


def verify(study) -> dict:
    """The signed paper checked: the signature over its manifest and the PDF's sha256, the record's hash chain,
    and that the record holds the Paper node the manifest names, at its head, with the same PDF and TeX."""
    from ..signing import verify_file
    rec, sig = study.record, study.folder / BUILD / SIG
    if not sig.is_file():
        return {"ok": False, "why": f"no signed paper ({BUILD}/{SIG})", "fix": "build it with `openlab study paper`"}
    r = verify_file(sig)
    meta = json.loads(sig.read_text(encoding="utf-8"))
    out = {"paper": meta.get("paper"), "head": meta.get("head"), "key_id": meta.get("key_id"),
           "signed_at": meta.get("signed_at")}
    if not r["ok"]:
        return {**out, "ok": False, "why": r["why"]}
    chain = rec.verify()
    if not chain["ok"]:
        return {**out, "ok": False, "why": f"the record's hash chain is broken at seq {chain['broken_at']}"}
    try:
        node = rec.get(meta["paper"])
    except StudyError:
        return {**out, "ok": False, "why": f"the record has no {meta['paper']}"}
    tex = study.folder / BUILD / "main.tex"
    why =("the manifest's head is not that paper node's hash" if node.kind != "paper" or node.hash != meta["head"]
           else "the signed PDF is not the PDF the paper node records" if meta["sha256"] != node.payload["pdf_sha"]
           else "main.tex is not the TeX the paper node records" if tex.is_file()
           and hashlib.sha256(tex.read_bytes()).hexdigest() != node.payload["tex_sha"] else "")
    if why:
        return {**out, "ok": False, "why": why}
    gone = retracted_since(study, node)
    if gone:
        return {**out, "ok": False, "retracted": gone, "why": "the paper is stale: " + "; ".join(
            f"{g['source']} was retracted after the build ({g['note']}), and {', '.join(g['bindings'])} rest on it"
            for g in gone) + "; build it again, which prints the citation as retracted"}
    return {**out, "ok": True}
