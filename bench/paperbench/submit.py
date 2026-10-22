"""PaperBench submission, as an openlab instrument: the lead's one way to hand in a replication.

`check` says whether a repository would be accepted; `submit` copies its committed HEAD (a clean clone, history
kept, no remote) to $PAPERBENCH_SUBMISSIONS/<paper>/submission/, the layout PaperBench's PBDirectSubmissionSolver
grades. A replaced submission is kept in the instrument's state. It never grades: the person runs PaperBench's judge
on the submissions folder, and no score comes back through here.

Registered once by the person (docs/benchmark.md, "PaperBench"), with fixed variables:
  PAPERBENCH_WORK_ROOT   repositories must be under it (the jobs folder)
  PAPERBENCH_SUBMISSIONS where submissions go (submit only)
  PAPERBENCH_PAPERS      optional: PaperBench's data/papers, to check the paper id and read its blacklist.txt

Refused (exit 2): a repository outside the work root, not a git repository, no commit, no committed reproduce.sh or
README.md at its root, committed files over 1 GB, an unknown paper. Warned (still accepted): uncommitted or untracked
files (not submitted: only HEAD is), committed text holding the repository's absolute path, or a blacklisted address.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 1 << 30                                     # PaperBench: committed files at most 1 GB
REQUIRED = ("reproduce.sh", "README.md")
SCAN_MAX = 2 << 20                                  # read committed text files up to 2 MB each for the warnings


class Refused(Exception):
    pass


def git(repo, *args, check=True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and p.returncode:
        raise Refused(f"git {' '.join(args)}: {(p.stderr or p.stdout).strip()}")
    return p.stdout


def blacklist(paper: str) -> list:
    papers = os.environ.get("PAPERBENCH_PAPERS")
    if not papers:
        return []
    f = Path(papers) / paper / "blacklist.txt"
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines() if f.exists() else []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#") and ln.strip() != "none"]


def inspect(paper: str, repo: str) -> dict:
    """What a submission of `repo` for `paper` would be, or Refused saying why not."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", paper):
        raise Refused(f"'{paper}' is not a PaperBench paper id (the id in the paper's config.yaml)")
    papers = os.environ.get("PAPERBENCH_PAPERS")
    if papers and not (Path(papers) / paper).is_dir():
        raise Refused(f"no paper '{paper}' in PaperBench's data")
    root = os.environ.get("PAPERBENCH_WORK_ROOT")
    if not root:
        raise Refused("PAPERBENCH_WORK_ROOT is not set: the instrument was registered without it")
    if not os.path.isabs(repo):
        raise Refused(f"'{repo}' is not an absolute path (give the repository's full path)")
    path = Path(repo).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise Refused(f"{path} is outside the jobs folder this instrument accepts ({root})")
    if not (path / ".git").exists():
        raise Refused(f"{path} is not a git repository (git init, then commit your work)")
    head = git(path, "rev-parse", "--verify", "-q", "HEAD^{commit}", check=False).strip()
    if not head:
        raise Refused(f"{path} has no commit: only committed files are submitted")
    sizes = {}
    for line in git(path, "ls-tree", "-r", "-l", "HEAD").splitlines():
        meta, name = line.split("\t", 1)
        size = meta.split()[3]
        sizes[name] = int(size) if size.isdigit() else 0
    missing = [f for f in REQUIRED if f not in sizes]
    if missing:
        raise Refused(f"no committed {' or '.join(missing)} at the repository's root")
    total = sum(sizes.values())
    if total > LIMIT:
        big = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
        raise Refused(f"committed files are {total / 2**30:.2f} GB, over PaperBench's 1 GB; largest: "
                      + ", ".join(f"{n} ({s / 2**20:.0f} MB)" for n, s in big)
                      + ". Commit code; let reproduce.sh make or fetch the heavy files")
    warnings = []
    dirty = [ln[3:] for ln in git(path, "status", "--porcelain", "--untracked-files=all").splitlines()]
    if dirty:
        warnings.append(f"{len(dirty)} uncommitted or untracked file(s) are not submitted (only HEAD is; "
                        f"PaperBench also runs git clean -fd): " + ", ".join(dirty[:8]) + (" ..." if len(dirty) > 8 else ""))
    needles = {str(path): "the repository's absolute path (the grader runs it elsewhere)"}
    needles.update({b: "a blacklisted address (the paper's own code, or a source PaperBench forbids)"
                    for b in blacklist(paper)})
    hits = {}
    for name, size in sizes.items():
        if size > SCAN_MAX:
            continue
        text = git(path, "show", f"HEAD:{name}", check=False)
        for needle, why in needles.items():
            if needle in text:
                hits.setdefault(why, []).append(name)
    warnings += [f"committed files hold {why}: {', '.join(names[:8])}" for why, names in hits.items()]
    return {"paper": paper, "repo": str(path), "commit": head, "tree": git(path, "rev-parse", "HEAD^{tree}").strip(),
            "files": len(sizes), "bytes": total, "warnings": warnings}


def submit(info: dict) -> dict:
    """Copy HEAD (a clone, history kept, no remote) to the submissions folder; keep a replaced one in state."""
    dest_root = os.environ.get("PAPERBENCH_SUBMISSIONS")
    if not dest_root:
        raise Refused("PAPERBENCH_SUBMISSIONS is not set: the instrument was registered without it")
    paper_dir = Path(dest_root) / info["paper"]
    dest = paper_dir / "submission"
    paper_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_root, prefix=".incoming-") as tmp:   # same filesystem: the move is atomic
        clone = Path(tmp) / "submission"
        p = subprocess.run(["git", "clone", "-q", "--no-hardlinks", info["repo"], str(clone)], capture_output=True,
                           text=True)
        if p.returncode:
            raise Refused(f"git clone: {p.stderr.strip()}")
        git(clone, "checkout", "-q", info["commit"])
        git(clone, "remote", "remove", "origin")
        replaced = ""
        if dest.exists():
            replaced = git(dest, "rev-parse", "HEAD", check=False).strip()
            state = os.environ.get("OPENLAB_INSTRUMENT_STATE")
            if state:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                keep = Path(state) / info["paper"] / f"{stamp}-{replaced[:12] or 'unknown'}"
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(keep))
            else:
                shutil.rmtree(dest)
        os.replace(clone, dest)
    return {**info, "submitted": str(dest), "replaced": replaced}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="paperbench-submit", description="Check or hand in a PaperBench replication: "
                                 "a git repository whose committed HEAD holds reproduce.sh and README.md at its root.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help in (("check", "say whether the repository would be accepted; nothing is copied"),
                       ("submit", "hand in the repository's committed HEAD; it replaces an earlier submission")):
        p = sub.add_parser(name, help=help)
        p.add_argument("--paper", required=True, help="the paper's id (the id in its config.yaml)")
        p.add_argument("repo", help="the repository's absolute path")
    a = ap.parse_args(argv)
    out = Path(os.environ.get("OPENLAB_INSTRUMENT_OUT") or ".")
    try:
        info = inspect(a.paper, a.repo)
        res = submit(info) if a.cmd == "submit" else {**info, "submitted": ""}
    except Refused as e:
        (out / "submission.json").write_text(json.dumps({"ok": False, "refused": str(e)}, indent=2) + "\n")
        print(f"refused: {e}", file=sys.stderr)
        return 2
    (out / "submission.json").write_text(json.dumps({"ok": True, **res}, indent=2) + "\n")
    verb = "submitted" if res["submitted"] else "would be accepted"
    print(f"{res['paper']}: commit {res['commit'][:12]} {verb} ({res['files']} files, {res['bytes'] / 2**20:.1f} MB)"
          + (f", replacing {res['replaced'][:12]}" if res.get("replaced") else ""))
    for w in res["warnings"]:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
