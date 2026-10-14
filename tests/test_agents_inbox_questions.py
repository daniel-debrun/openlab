"""`openlab inbox` and an agent's questions: listed with their numbered choices, answered (unsigned) with `openlab
inbox answer`, never approved or denied, waited for by `inbox wait`, and flagged WAITING in `openlab watch`."""

import pytest

from openlab.agents import cli, hooks
from openlab.agents.cli import frame, main
from openlab.agents.inbox import check_request
from openlab.agents.store import Store


@pytest.fixture
def ws(tmp_path):
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="running", question="does x predict y?")

    def ask(reason="How many seeds?", **args):
        args = {"choices": ["3 seeds", "5 seeds", "10 seeds"], **args}
        return st.request("lead-1", "question", reason, args=check_request(st, "lead-1", "question", args))
    return tmp_path, st, ask


def inbox(root, *argv):
    return main(["inbox", "--root", str(root), *argv])


def test_inbox_lists_a_question_with_its_numbered_choices(ws, capsys):
    root, st, ask = ws
    item = ask()
    assert inbox(root) == 0
    out = capsys.readouterr().out
    assert "question" in out and "How many seeds?" in out and "pick one" in out
    assert "1. 3 seeds" in out and "2. 5 seeds" in out and "3. 10 seeds" in out
    assert f"answer with: openlab inbox answer {item} --pick N [--other TEXT] [-m MESSAGE]" in out
    multi = ask("Which metrics?", multi=True, other=False)
    inbox(root)
    out = capsys.readouterr().out
    assert f"openlab inbox answer {multi} --pick N [--pick N ...] [-m MESSAGE]" in out and "pick any" in out


def test_answer_records_the_picks_unsigned(ws, capsys, monkeypatch):
    root, st, ask = ws
    from openlab.study import personkey
    monkeypatch.setattr(personkey, "current", lambda: pytest.fail("an answer is not signed"))
    item = ask()
    assert inbox(root, "answer", str(item), "--pick", "2", "-m", "then stop") == 0
    out = capsys.readouterr().out
    assert "answered" in out and "answer: 5 seeds" in out
    got = st.item(item)
    assert got["status"] == "answered" and got["result"] == {"picks": ["5 seeds"], "other": ""}
    assert got["decision_reason"] == "then stop" and got["decided_by"].startswith("person:")
    decided = [e for e in st.events("lead-1") if f"#{item}" in e["summary"] and e["kind"] != "request"]
    assert decided and not any((e["payload"] or {}).get("sig") for e in decided)
    inbox(root)
    assert capsys.readouterr().out.strip() == "the inbox is empty"
    inbox(root, "--all")
    out = capsys.readouterr().out
    assert "answer: 5 seeds" in out and "answer with" not in out


def test_answer_takes_several_picks_and_own_words(ws, capsys):
    root, st, ask = ws
    item = ask("Which metrics?", multi=True)
    assert inbox(root, "answer", str(item), "--pick", "1", "--pick", "3", "--pick", "1", "--other", "also crps") == 0
    assert st.item(item)["result"] == {"picks": ["3 seeds", "10 seeds"], "other": "also crps"}
    words = ask("Anything else?")
    assert inbox(root, "answer", str(words), "--other", "use the 2019 split") == 0
    assert st.item(words)["result"] == {"picks": [], "other": "use the 2019 split"}


@pytest.mark.parametrize("argv, error", [
    (["--pick", "4"], "--pick 4 is not one of the choices 1..3"),
    (["--pick", "0"], "is not one of the choices"),
    (["--pick", "two"], "is not one of the choices"),
    (["--pick", "1", "--pick", "2"], "this question takes one choice"),
    ([], "needs an answer"),
    (["-m", "just a message"], "needs an answer"),
    (["--other", "   "], "needs an answer"),
])
def test_answer_refuses(ws, capsys, argv, error):
    root, st, ask = ws
    item = ask()
    assert inbox(root, "answer", str(item), *argv) == 1
    assert error in capsys.readouterr().err
    assert st.item(item)["status"] == "pending"


def test_answer_refuses_own_words_a_question_does_not_take(ws, capsys):
    root, st, ask = ws
    item = ask(other=False)
    assert inbox(root, "answer", str(item), "--other", "7 seeds") == 1
    assert "takes only its choices" in capsys.readouterr().err


def test_answer_refuses_a_non_question_an_unknown_item_and_an_answered_one(ws, capsys):
    root, st, ask = ws
    budget = st.request("lead-1", "budget", "more", args={"usd": 9})
    assert inbox(root, "answer", str(budget), "--pick", "1") == 1
    err = capsys.readouterr().err
    assert "not a question" in err and f"openlab inbox approve|deny {budget}" in err
    assert st.item(budget)["status"] == "pending"
    assert inbox(root, "answer", "99", "--pick", "1") == 1
    assert "no inbox item #99" in capsys.readouterr().err
    item = ask()
    assert inbox(root, "answer", str(item), "--pick", "1") == 0
    capsys.readouterr()
    assert inbox(root, "answer", str(item), "--pick", "2") == 1
    assert "already answered" in capsys.readouterr().err
    assert st.item(item)["result"]["picks"] == ["3 seeds"]


def test_an_agent_cannot_answer(ws, capsys, monkeypatch):
    root, st, ask = ws
    item = ask()
    monkeypatch.setenv("CLAUDECODE", "1")                           # an agent's shell
    assert inbox(root, "answer", str(item), "--pick", "1") == 1
    assert "the person's" in capsys.readouterr().err and st.item(item)["status"] == "pending"
    assert hooks.gate("Bash", {"command": f"openlab inbox answer {item} --pick 1"})


@pytest.mark.parametrize("verb", ["approve", "deny"])
def test_approve_and_deny_refuse_a_question_and_point_to_answer(ws, capsys, monkeypatch, verb):
    root, st, ask = ws
    from openlab.agents import person
    monkeypatch.setattr(person, "act", lambda *a, **k: pytest.fail("no passphrase is asked for a question"))
    item = ask()
    assert inbox(root, verb, str(item), "--reason", "ok") == 1
    err = capsys.readouterr().err
    assert "is a question" in err and f"openlab inbox answer {item} --pick N" in err
    assert st.item(item)["status"] == "pending"


def test_inbox_wait_wakes_on_a_new_question(ws, capsys, monkeypatch):
    root, st, ask = ws
    st.request("lead-1", "budget", "more", args={"usd": 9})      # already pending: not new
    asked = []

    def sleep(_):
        if asked:
            pytest.fail("inbox wait did not wake on the question")
        asked.append(ask("Which split?"))
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert inbox(root, "wait", "--minutes", "1") == 0
    out = capsys.readouterr().out
    assert f"#{asked[0]}" in out and "Which split?" in out and "1. 3 seeds" in out
    assert f"openlab inbox answer {asked[0]}" in out and "budget" not in out


def test_watch_flags_an_agent_with_a_pending_question_waiting(ws, capsys):
    root, st, ask = ws
    st.register("other-2", "spawned", role="worker", status="running")
    item = ask()
    text, rows = frame(st, width=200, studies=[])
    by = {r["agent"]["id"]: r for r in rows}
    assert by["lead-1"]["state"] == "waiting" and "WAITING" in by["lead-1"]["flags"]
    assert "WAITING" not in by["other-2"]["flags"] and "inbox: 1 pending" in text
    inbox(root, "answer", str(item), "--pick", "1")
    text, rows = frame(st, width=200, studies=[])
    lead = next(r for r in rows if r["agent"]["id"] == "lead-1")
    assert "WAITING" not in lead["flags"] and lead["state"] != "waiting"
    assert main(["watch", str(root), "--once"]) == 0
    assert "inbox: 0 pending" in capsys.readouterr().out
