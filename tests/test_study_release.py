"""A release's README says what the record holds: plainly nothing when no run was made; else each result beside its
baselines, the paired difference and which is better, tested or not; and the study's dead ends and claims."""

from openlab import study as vs
from openlab.study.release import build


def readme(s) -> str:
    build(s)
    return (s.folder / "release" / "README.md").read_text(encoding="utf-8")


def test_a_release_with_no_runs_says_so_at_the_top(tmp_path):
    s = vs.new(tmp_path / "s", "static")
    s.materialise()
    text = readme(s)
    assert text.splitlines()[2].startswith("**No runs yet: nothing has been tested.**")
    assert "No result is on the record yet" in text


def test_the_result_is_shown_beside_its_baseline_tested_or_not_with_dead_ends_and_claims(tmp_path):
    s = vs.new(tmp_path / "s", "static")
    s.materialise()
    r = s.run(note="first")["id"]
    text = readme(s)
    assert "No runs yet" not in text
    assert f"- result `{r}` (untested)" in text and "  - brier: " in text
    assert "  - baseline majority: auc " in text
    assert "  - model minus majority on auc: " in text and "better: the model" in text
    assert "## Dead ends" not in text and "## Claims" not in text

    q = s.question("Does the model beat the majority class?")["id"]
    h = s.hypothesis("The model beats the majority class.", "higher AUC", "a paired test that does not reject", q,
                     ["auc"])["id"]
    s.test(h, r, "majority", 0.01)
    s.dead_end(q, "the raw sensor feed", "it lags by a day", ["feed"], [r])
    s.claim("The model's AUC beats the majority class's.", "observed", [r])
    (s.folder / "model.py").write_text((s.folder / "model.py").read_text() + "\n# a later variant\n")
    later = s.run(note="later")["id"]
    text = readme(s)
    tested, rest = text.split(f"The latest result, `{later}`, is a later run that no test names")
    assert f"- result `{r}`, tested by `test:" in tested and "against majority" in tested
    assert "baseline majority" in tested and "model minus majority" in tested and "baseline majority" not in rest
    assert "## Dead ends\n- the raw sensor feed: it lags by a day (`dead_end:" in text
    assert "## Claims\n- The model's AUC beats the majority class's. (observed, " in text


def test_the_studys_readme_is_the_release_readme_with_the_records_part_between_its_marks(tmp_path):
    from openlab.study.release import RECORD_END, RECORD_MARK
    s = vs.new(tmp_path / "s", "static")
    s.materialise()
    text = readme(s)                                       # no README.md: the name, the question, the record's part
    assert text.splitlines()[0] == f"# {s.status()['name']}" and RECORD_MARK in text and RECORD_END in text
    assert not (s.folder / "release" / "code" / "README.md").exists()
    (s.folder / "README.md").write_text("# Wine quality\n\nWhy this matters, in **plain** words.\n")
    text = readme(s)                                       # no marks: the record's part at the end
    assert text.splitlines()[0] == "# Wine quality" and text.index("plain") < text.index(RECORD_MARK) < text.index("## Data")
    (s.folder / "README.md").write_text(f"# Wine\n\nTop.\n\n{RECORD_MARK}\n\n## Data\nhand-edited, stale\n\n{RECORD_END}\n\nBottom.\n")
    text = readme(s)                                       # between the marks: rewritten; around them: kept
    assert "hand-edited" not in text and "## Result" in text
    assert text.index("Top.") < text.index("## Data") < text.index(RECORD_END) < text.index("Bottom.")
    assert text.count(RECORD_MARK) == 1
    (s.folder / "README.md").unlink()
    (s.folder / "README.md").symlink_to("/etc/hostname")
    assert readme(s).splitlines()[0] == f"# {s.status()['name']}"
