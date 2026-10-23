"""study.toml written field by field (an amendment's fields): tables and lists of tables read back as written."""
from openlab.study.ledger import write_fields
from openlab.study.spec import tomllib


def test_an_amendment_writes_tables_and_lists_of_tables_that_read_back(tmp_path):
    (tmp_path / "study.toml").write_text('[study]\nname = "s"  # kept\n', encoding="utf-8")
    review = {"question": "How large are day-1 errors?", "n": 25, "citation_hops": 0,
              "coverage_floor": {"hops": 0, "queries_per_source": 1},
              "include": [{"id": "I1", "text": "Reports the \"MAE\" in degrees"}],
              "exclude": [{"id": "E1", "text": "No day-1 forecast"}], "scope": {"a b": True}}
    assert write_fields(tmp_path, {"review": review})[0] == "review.question"
    text = (tmp_path / "study.toml").read_text(encoding="utf-8")
    assert "# kept" in text
    assert tomllib.loads(text) == {"study": {"name": "s"}, "review": review}
