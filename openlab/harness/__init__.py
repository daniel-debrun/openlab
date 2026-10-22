"""The harness: every text openlab puts before an agent it spawns or serves (the spawn brief, the role prompts, the
workflow briefs of a proposal and a replication, the MCP instructions and the one-line tool descriptions), kept as
files under a folder (the packaged one: openlab/harness/default/) rather than in code, and fingerprinted, so a spawn
records which harness briefed it (``harness_sha`` in its launch.json and spawn event) and an outcome can be traced
to the text that produced it.

A harness is advice only. What enforces (the hooks, the person verbs, the [agents] deny list, the sandbox, budgets,
allowed tools, the holdout) lives in code and openlab.toml, and :func:`validate` refuses a harness folder holding any
file outside :data:`SURFACE`, so no harness can carry a hook or a setting. Its one kind of code is a skill's script
(skills/<name>/script.py), which runs only through the harness_script tool, in the bubblewrap sandbox with no network
(openlab.harness.skills). The caps, the vocabulary it is checked against and the tool list it describes are here,
not in the folder they check; a workspace's LAWS.openlab (the person's) adds rules of its own (:func:`laws`).

A workspace runs the packaged harness until a person makes one of its stored versions active
(openlab.harness.versions); probe signals, harness lessons, the offline evaluator and per-task refinement build on
that (signals, lessons, reflect, evaluate, refine).

A file's text is its contents less one final newline.
"""

from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "default"

#: the roles a spawned agent may be briefed as (roles/<role>.md: a `description:` front matter line, then the body)
ROLES = ("lead", "screener", "replicator", "experimenter", "adversary", "reviewer")

#: each formatted text and the str.format fields openlab renders it with: a harness must use exactly these
FIELDS = {
    "brief.md": {"id", "role", "parent_line", "study_line", "question_line", "budget_line"},
    "workflows/propose_planner.md": {"rel", "roles", "cap_usd", "cap_minutes", "mode", "prompt"},
    "workflows/propose_lead.md": {"rel", "title", "pid", "sha", "question", "root", "hyps", "branches"},
    "workflows/propose_resume.md": {"rel", "why", "finished", "decisions", "usd", "minutes"},
    "workflows/replicate_planner.md": {"folder", "key", "metrics", "cap_usd", "cap_minutes", "cores", "memory", "fit"},
    "workflows/replicate_continue.md": {"folder", "lint"},
    "workflows/replicate_build.md": {"title", "key", "folder", "md", "aid", "sha", "tol", "metric", "max", "file",
                                     "methods", "claims", "data_name", "data_source", "protocol", "proposed", "usd",
                                     "minutes", "trials"},
    "workflows/replicate_resume.md": {"why", "folder", "record", "notes", "usd", "minutes"},
}
#: texts used as written
PLAIN = ("self.md", "common.md", "guide.md")
SHORT_FILE = "tools/short.toml"

#: every file a harness may hold; anything else is refused
SURFACE = (*FIELDS, *PLAIN, SHORT_FILE, *(f"roles/{r}.md" for r in ROLES))
SKILL = re.compile(r"skills/([a-z0-9][a-z0-9-]{0,40})/SKILL\.md")
SCRIPT = re.compile(r"skills/([a-z0-9][a-z0-9-]{0,40})/script\.py")

#: characters: what a spawned agent is always told (brief, its role, the decide-yourself note), on either CLI
ALWAYS_ON_MAX = 6000
COMMON_MAX = 2000           # common.md: in every role file and Codex's AGENTS.md section
GUIDE_MAX = 2000            # guide.md: the MCP instructions (with the tool list, under mcp_server.BUDGET_CHARS)
WORKFLOW_MAX = 8000         # one workflow brief, before its fields are filled
SHORT_MAX = 240             # one tool's description line
DESCRIPTION_MAX = 300       # one role's description line

#: words that would tie a harness to how a benchmark grades it (MLE-bench's competition names are added)
EVAL_WORDS = ("grader", "rubric", "medal", "leaderboard", "mle-bench", "mlebench")


class HarnessError(Exception):
    """A harness folder openlab cannot brief agents from: each problem on its own line."""

    def __init__(self, problems: list):
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Harness:
    path: Path
    files: dict                 # path relative to the folder -> text
    sha: str                    # the fingerprint of the folder's files

    def text(self, rel: str) -> str:
        return self.files[rel]

    @property
    def roles(self) -> dict:
        """role -> (description, body), in ROLES order."""
        return {r: _role(self.files[f"roles/{r}.md"]) for r in ROLES}

    @property
    def short(self) -> dict:
        return _toml(self.files[SHORT_FILE])

    @property
    def skills(self) -> tuple:
        return tuple(sorted(m.group(1) for rel in self.files if (m := SKILL.fullmatch(rel))))

    @property
    def always_on(self) -> str:
        """What every agent briefed by it reads before its first call: the brief and the common text."""
        return self.files["brief.md"] + "\n" + self.files["common.md"]


def _toml(text: str) -> dict:
    try:
        import tomllib
    except ImportError:                                             # Python 3.10
        import tomli as tomllib
    return tomllib.loads(text)


def _role(text: str) -> tuple:
    m = re.match(r"---\ndescription: ([^\n]*)\n---\n\n", text)
    if not m:
        raise HarnessError(["a role file starts with ---, `description: <one line>`, --- and a blank line"])
    return m.group(1), text[m.end():]


def _files(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def fingerprint(files: dict) -> str:
    """sha256 over each file's path and bytes, in path order."""
    h = hashlib.sha256()
    for rel in sorted(files):
        data = files[rel] if isinstance(files[rel], bytes) else (files[rel] + "\n").encode("utf-8")
        h.update(f"{rel}\0{len(data)}\0".encode("utf-8") + data)
    return h.hexdigest()


def read(path=DEFAULT) -> Harness:
    """The harness in `path`, unchecked."""
    path = Path(path).resolve()
    raw = _files(path)
    text = {rel: b.decode("utf-8").removesuffix("\n") for rel, b in raw.items()}
    return Harness(path, text, fingerprint(raw))


@lru_cache(maxsize=None)
def load(path=DEFAULT) -> Harness:
    """The harness in `path` (default: the packaged one), refused with every problem :func:`validate` finds."""
    h = read(path)
    if problems := validate(h):
        raise HarnessError([f"{h.path}: {p}" for p in problems])
    return h


def _eval_words() -> tuple:
    try:
        from ..benchmark.mle import LITE
    except Exception:                                               # noqa: BLE001 -- the fixed words still hold
        LITE = ()
    return EVAL_WORDS + tuple(LITE)


def laws(root) -> dict:
    """The workspace's LAWS.openlab (written by a person; the hooks refuse an agent's edit of it): `forbid: <regular
    expression>` lines no harness text may match, `require: <text>` lines the always-on text must hold word for
    word. {"forbid": [...], "require": [...]}; other lines are comments."""
    out = {"forbid": [], "require": []}
    try:
        text = (Path(root) / "LAWS.openlab").read_text(encoding="utf-8")
    except (OSError, TypeError):
        return out
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in out and value.strip():
            out[key.strip()].append(value.strip())
    return out


def validate(h: Harness, rules: dict | None = None) -> list:
    """What makes `h` unfit to brief an agent: a file outside the surface or one missing, a text whose format fields
    differ from what openlab fills, a role file without its description, a tool list other than openlab's, a text
    over its cap, an unnamed skill, a script that does not compile, benchmark-grading vocabulary, or a break of the
    workspace's laws (`rules`, from :func:`laws`)."""
    out = []
    for rel in h.files:
        if rel not in SURFACE and not SKILL.fullmatch(rel) and not SCRIPT.fullmatch(rel):
            out.append(f"{rel} is not part of a harness: it holds only {', '.join(sorted(SURFACE))}, "
                       "skills/<name>/SKILL.md and skills/<name>/script.py (hooks and settings are openlab's, not a "
                       "harness's)")
        elif SCRIPT.fullmatch(rel) and f"{rel.rsplit('/', 1)[0]}/SKILL.md" not in h.files:
            out.append(f"{rel} has no SKILL.md beside it to say when it runs")
    out += [f"{rel} is missing" for rel in SURFACE if rel not in h.files]
    if out:
        return out
    for rel, want in FIELDS.items():
        try:
            got = {f for _, f, _, _ in string.Formatter().parse(h.files[rel]) if f is not None}
        except ValueError as e:
            out.append(f"{rel}: {e} (a literal brace is written twice: {{{{ }}}})")
            continue
        if got != want:
            out.append(f"{rel} uses the fields {sorted(got)}; openlab fills {sorted(want)}")
    roles = {}
    for r in ROLES:
        try:
            roles[r] = _role(h.files[f"roles/{r}.md"])
        except HarnessError as e:
            out.append(f"roles/{r}.md: {e}")
            continue
        if len(roles[r][0]) > DESCRIPTION_MAX:
            out.append(f"roles/{r}.md: its description is {len(roles[r][0])} characters, over {DESCRIPTION_MAX}")
    try:
        short = h.short
    except Exception as e:                                          # noqa: BLE001 -- a TOML error, reported
        out.append(f"{SHORT_FILE}: {e}")
        short = {}
    for name, line in short.items():
        if not isinstance(line, str) or "\n" in line or len(line) > SHORT_MAX:
            out.append(f"{SHORT_FILE}: {name} is not one line of at most {SHORT_MAX} characters")
    if short and h.path != DEFAULT:
        want = set(read().short)
        if set(short) != want:
            out.append(f"{SHORT_FILE} describes {', '.join(sorted(set(short) ^ want))} differently from openlab's tool "
                       "list: a harness rewords the default tools, it does not choose them")
    longest = max((len(b) for _, b in roles.values()), default=0)
    always = len(h.files["brief.md"]) + longest + len(h.files["self.md"])
    if always > ALWAYS_ON_MAX:
        out.append(f"brief.md, the longest role and self.md come to {always} characters, over {ALWAYS_ON_MAX}")
    for rel, cap in (("common.md", COMMON_MAX), ("guide.md", GUIDE_MAX),
                     *((rel, WORKFLOW_MAX) for rel in FIELDS if rel.startswith("workflows/"))):
        if len(h.files[rel]) > cap:
            out.append(f"{rel} is {len(h.files[rel])} characters, over {cap}")
    named = h.files["brief.md"] + h.files["common.md"]
    out += [f"skills/{s}/SKILL.md: no always-on text (brief.md, common.md) names the skill {s!r}, so no agent "
            "would load it" for s in h.skills if not re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", named)]
    for rel, text in h.files.items():
        if SCRIPT.fullmatch(rel):
            try:
                compile(text, rel, "exec")
            except (SyntaxError, ValueError) as e:
                out.append(f"{rel} does not compile: {e}")
    words = _eval_words()
    for rel, text in h.files.items():
        low = text.lower()
        out += [f"{rel} says {w!r}: a harness says nothing of how a benchmark grades" for w in words
                if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low)]
    for rx in (rules or {}).get("forbid", []):
        for rel, text in h.files.items():
            try:
                if re.search(rx, text):
                    out.append(f"{rel} matches the law `forbid: {rx}` of LAWS.openlab")
            except re.error as e:
                out.append(f"LAWS.openlab: `forbid: {rx}` is not a regular expression ({e})")
                break
    out += [f"the always-on text (brief.md, common.md) lacks the law `require: {req}` of LAWS.openlab"
            for req in (rules or {}).get("require", []) if req not in h.always_on]
    return out


#: [harness] of openlab.toml (the person's): what reflection and refinement may spend, and how lessons are used
SETTINGS = {"reflect": False,              # a reflector after each spawned agent ends (it spends reflector_budget_usd)
            "reflector_budget_usd": 0.5,
            "advice": 3,                   # approved harness lessons put in a brief
            "lesson_ttl_days": 90,         # a lesson not confirmed for this long is stale: no longer put in briefs
            "refine_budget_usd": 5.0,      # one refinement step, all its agents
            "refine_max_fraction": 0.25,   # and at most this share of [agents] cap_usd
            "k_max": 3,
            "probe_minutes": 30.0}


def settings(root) -> dict:
    """[harness] of the workspace's openlab.toml over SETTINGS (a value of the wrong type keeps the default)."""
    out = dict(SETTINGS)
    try:
        given = _toml((Path(root) / "openlab.toml").read_text(encoding="utf-8")).get("harness") or {}
    except Exception:                                               # noqa: BLE001 -- no file, or a broken one
        given = {}
    for k, v in given.items() if isinstance(given, dict) else ():
        d = SETTINGS.get(k)
        if d is not None and (type(v) is type(d) or (isinstance(d, float) and type(v) is int)):
            out[k] = float(v) if isinstance(d, float) else v
    return out
