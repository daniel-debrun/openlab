"""openlab show in a real browser: the server on a free port, headless Chrome driven over the DevTools protocol. The
page switches researches, answers an agent's question (Other, with a message) into the control store, verifies the
ledger, and redraws within seconds when an agent writes the control store from another process. No JS errors."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")
CHROME = shutil.which("google-chrome")
pytestmark = pytest.mark.skipif(CHROME is None, reason="google-chrome is not installed")

from openlab.agents.store import Store  # noqa: E402
from openlab.app import ToolContext, dispatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, seconds: float = 30):
    end = time.time() + seconds
    while time.time() < end:
        try:
            return urllib.request.urlopen(url, timeout=2).read()
        except Exception:                                        # noqa: BLE001 -- not up yet
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not answer")


class Page:
    """A minimal CDP client: evaluate JS, collect exceptions and console errors."""

    def __init__(self, ws):
        self.ws, self.n, self.errors = ws, 0, []

    async def call(self, method, **params):
        self.n += 1
        req_id = self.n
        await self.ws.send(json.dumps({"id": req_id, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append(d.get("exception", {}).get("description") or d.get("text", ""))
            elif m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] == "error":
                self.errors.append(str(m["params"]["args"])[:300])
            if m.get("id") == req_id:
                return m

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        res = r.get("result", {})
        if "exceptionDetails" in res:
            raise AssertionError(f"{expr[:80]}: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def until(self, expr, seconds=6.0):
        end = time.time() + seconds
        while time.time() < end:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.2)
        return False


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    for name in ("alpha", "beta"):
        subprocess.run([sys.executable, "-m", "openlab", "init", str(h / name)], check=True, capture_output=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    root = h / "alpha"
    from agents_fake import install
    install(tmp_path, monkeypatch)                     # a fake claude answers a side question (openlab.agents.btw)
    st = Store(root)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 6}, cli="claude",
                session_id="sess-lead-1", cwd=str(root))
    st.register("child-1", "spawned", role="worker", status="running", budget={"usd": 2}, parent="lead-1")
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    ctx = ToolContext(str(root))
    assert dispatch(ctx, "agent_plan", {"nodes": [
        {"id": "data", "title": "Data", "kind": "data", "state": "done"},
        {"id": "base", "title": "Baseline run", "kind": "run", "needs": ["data"], "state": "active"}], "note": "first"})["ok"]
    q = dispatch(ctx, "agent_request", {"kind": "question", "reason": "How many seeds?",
                                        "args": {"choices": ["3", "5"], "other": True}})["result"]["item"]
    return h, ctx, st, q


@pytest.fixture
def server(home):
    h = home[0]
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "openlab", "show", "--root", str(h), "--port", str(port), "--token", "tok"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONPATH": str(REPO), "OPENLAB_APP_TERMINAL": "0"})
    try:
        line = ""
        end = time.time() + 30
        while "&r=" not in line and time.time() < end:
            line = proc.stdout.readline()
        reviewer = line.strip().split("&r=")[1]
        wait_http(f"http://127.0.0.1:{port}/api/health")
        yield f"http://127.0.0.1:{port}/show/?t=tok&r={reviewer}"
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def chrome(tmp_path):
    port = free_port()
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", f"--remote-debugging-port={port}",
                             "--window-size=1400,900", f"--user-data-dir={tmp_path / 'chrome'}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tabs = json.loads(wait_http(f"http://127.0.0.1:{port}/json"))
        yield next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_the_live_page_end_to_end(home, server, chrome):
    h, ctx, st, q = home

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#wsTabs .wst').length === 2", 15)
            assert "w=alpha" in await p.js("location.search")

            # the research tabs switch, and the URL remembers which
            await p.js("document.querySelector('#wsTabs [data-ws=\"beta\"]').click()")
            assert await p.until("location.search.includes('w=beta') && document.querySelector('#wsTabs .wst.on').dataset.ws === 'beta'")
            await p.js("document.querySelector('#wsTabs [data-ws=\"alpha\"]').click()")
            assert await p.until("document.querySelector('#sessNav') ? document.body.innerText.includes('child-1') : document.body.innerText.includes('child-1')")

            # every view draws
            for v in ("roadmap", "experiments", "burn", "ledger", "paper", "settings", "inbox"):
                await p.js(f"document.querySelector('[data-view={v}]').click()")
                await asyncio.sleep(0.3)
                assert await p.js(f"document.querySelector('#v-{v}') ? getComputedStyle(document.querySelector('#v-{v}')).display !== 'none' : true")

            # answer the question with Other and a message: it lands in the control store
            await p.js("""(() => { const f = document.querySelector('.ibx form[data-type=choice]');
                const o = f.querySelector('[data-other]'); o.value = 'seven'; o.dispatchEvent(new Event('input', {bubbles: true}));
                f.querySelector('textarea.msg').value = 'use seven seeds'; f.querySelector('button[type=submit]').click(); })()""")
            assert await p.until("document.querySelector('#inbox').innerText.includes('Other: “seven”')")
            item = next(i for i in st.inbox(None) if i["id"] == q)
            assert item["status"] == "answered"
            got = dispatch(ctx, "agent_inbox_item", {"id": q})["result"]
            assert got["result"]["other"] == "seven" and got["decision_reason"] == "use seven seeds"
            # and the tab's count of what waits for the person follows
            assert await p.until("!document.querySelector('#wsTabs [data-ws=\"alpha\"] .ct')", 6)

            # the ledger verifies its chain
            await p.js("document.querySelector('[data-view=ledger]').click()")
            if await p.until("!!document.querySelector('[data-verify]')", 2):      # a research with a record
                await p.js("document.querySelector('[data-verify]').click()")
                assert await p.until("document.querySelector('#ledger').innerText.includes('chain verified')")

            # an agent writes the control store from another process: the page redraws within about 3 s
            await p.js("document.querySelector('[data-view=inbox]').click()")
            subprocess.run([sys.executable, "-c", (
                "import os; os.environ['OPENLAB_AGENT']='lead-1'\n"
                "from openlab.app import ToolContext, dispatch\n"
                f"print(dispatch(ToolContext({str(h / 'alpha')!r}), 'agent_request', {{'kind': 'question', "
                "'reason': 'Which split?', 'args': {'choices': ['time', 'random']}})['ok'])")],
                check=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(REPO)})
            assert await p.until("document.querySelector('#inbox').innerText.includes('Which split?')", 4)
            assert await p.until("document.querySelector('#wsTabs [data-ws=\"alpha\"] .ct')?.textContent === '1'", 4)
            await p.js("1")
            assert p.errors == []

    asyncio.run(run())


def test_the_person_prompts_a_lead_and_pauses_the_leads(home, server, chrome):
    h, ctx, st, q = home

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#wsTabs .wst').length === 2", 15)
            await p.js("document.querySelector('[data-view=roadmap]').click()")
            # a Prompt button beside the lead's name, none beside the agent under it
            assert await p.until("!!document.querySelector('#list [data-prompt=\"lead-1\"]')", 8)
            assert await p.js("!document.querySelector('[data-prompt=\"child-1\"]')")
            await p.js("document.querySelector('#list [data-prompt=\"lead-1\"]').click()")
            assert await p.until("document.querySelector('#pr').open && document.querySelector('#prT').textContent === 'lead-1'")
            await p.js("document.querySelector('#prIn').value = 'why is this taking so long?'; document.querySelector('#prTell').click()")
            assert await p.until("document.querySelector('#prThread').innerText.includes('why is this taking so long?')", 6)
            asks = [e for e in st.untold("lead-1") if e["payload"].get("ask")]
            assert [e["payload"]["text"] for e in asks] == ["why is this taking so long?"]
            # the lead reads it at its next tool call and answers; the conversation shows it
            from openlab.agents.hooks import delivery
            assert "why is this taking so long?" in delivery(st, st.get("lead-1"))
            assert dispatch(ctx, "agent_reply", {"text": "the baseline is slow; done in 10 min", "to": asks[0]["seq"]})["ok"]
            assert await p.until("document.querySelector('#prThread').innerText.includes('done in 10 min')", 6)
            assert await p.js("document.querySelector('#prThread').innerText.includes('read')")
            # a side question (btw): answered now by a fork of its session, and the lead is not told
            await p.js("document.querySelector('#prIn').value = 'which branch is slow?'; document.querySelector('#prOk').click()")
            assert await p.until("document.querySelector('#prThread').innerText.includes('side answer to: which branch is slow?')", 15)
            assert not st.untold("lead-1")
            await p.js("document.querySelector('#prX').click()")
            # pause every lead from the header; the agents under them are held with them
            assert await p.until("document.querySelector('#leadsAll').textContent.includes('Pause all leads')")
            await p.js("document.querySelector('#leadsAll').click()")
            assert await p.until("document.querySelector('#leadsAll').textContent.includes('Play all leads')", 6)
            assert st.get("lead-1")["control"] == "pause" and st.blocked("child-1")[0] == "lead-1"
            await p.js("1")
            assert p.errors == []

    asyncio.run(run())


def test_an_ended_agents_steps_stop_at_its_end_on_the_chart(home, server, chrome):
    h, ctx, st, q = home
    st.set_status("lead-1", "done")                     # it ended with "Baseline run" still active

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#wsTabs .wst').length === 2", 15)
            await p.js("document.querySelector('[data-view=roadmap]').click()")
            assert await p.until("(LAB.data.plans['lead-1'] || []).length > 0", 8)
            base = await p.js("LAB.data.plans['lead-1'].at(-1).nodes.find((n) => n.id === 'base')")
            assert base["state"] == "dropped" and base["cap"]
            await asyncio.sleep(2.5)
            assert await p.js("!!document.querySelector('#list [data-prompt=\"lead-1\"]') && !document.querySelector('#list [data-hold=\"lead-1\"]')")
            assert p.errors == []

    asyncio.run(run())


@pytest.mark.skipif(not shutil.which("latexmk"), reason="the draft renders with latexmk")
def test_the_person_starts_the_paper_in_the_workspace_and_it_renders_beside_its_latex(home, server, chrome):
    h, ctx, st, q = home
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"})["ok"]
    paper = h / "alpha" / "s" / "paper"

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#wsTabs .wst').length === 2", 15)
            await p.js("document.querySelector('[data-view=paper]').click()")
            assert await p.until("!!document.querySelector('#wTree [data-wpath=s]')", 8)
            await p.js("document.querySelector('#wTree [data-wpath=s]').click()")
            assert await p.until("!!document.querySelector('#wMain [data-wstart=s]')", 8)
            await p.js("document.querySelector('#wMain [data-wstart=s]').click()")
            assert await p.until("(document.querySelector('#wTa') || {}).value?.includes('\\\\begin{abstract}')", 10)
            assert (paper / "paper.tex").is_file()
            assert await p.until("!document.querySelector('#wMsg').textContent.includes('Rendering')", 60)
            assert "paper/paper.tex" in told_text(st)
            await p.js("1")
            assert p.errors == []

    asyncio.run(run())


def told_text(st) -> str:
    return " ".join(e["payload"].get("text", "") for e in st.untold("lead-1"))




def test_flags_are_one_chip_per_kind_and_the_paper_names_its_folder(home, server, chrome):
    h, ctx, st, q = home
    for _ in range(5):
        st.event("child-1", "refusal", tool="Read", summary="Read .openlab/record.jsonl: read the record through the tools")
    st.event("child-1", "bypass_attempt", tool="Bash", summary="Bash: echo x > .openlab/record.jsonl")
    for name in ("fxnow", "wxmodels"):
        assert dispatch(ctx, "study_new", {"folder": name, "template": "static"})["ok"]
    (h / "alpha" / "fxnow" / "paper").mkdir(exist_ok=True)
    (h / "alpha" / "fxnow" / "paper" / "paper.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n")

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#wsTabs .wst').length === 2", 15)
            # an agent's flags are on its Roadmap row, one chip per kind that needs you; the tip says what it means
            # and shows the attempts the hooks refused
            await p.js("document.querySelector('[data-view=roadmap]').click()")
            assert await p.until("!!document.querySelector('#list [data-wid=\"child-1\"] .rdh')", 8)
            chips = await p.js("[...document.querySelectorAll('#list [data-wid=\"child-1\"] .rdh')].map((c) => c.textContent)")
            assert "BYPASS ×1" in chips and "LOOP ×5" in chips
            tip = await p.js("[...document.querySelectorAll('#list [data-wid=\"child-1\"] .rdh')].find((c) => c.textContent.startsWith('BYPASS')).title")
            assert "get around a guard" in tip and "echo x > .openlab" in tip
            # the Workspace: the research's files as a tree; a study folder starts its paper; a file opens beside its
            # rendering; with nothing selected, a comment is on the whole file
            await p.js("document.querySelector('[data-view=paper]').click()")
            assert await p.until("document.querySelector('#dTitle').textContent === 'alpha / Workspace'", 8)
            assert await p.until("!!document.querySelector('#wTree [data-wpath=wxmodels]')", 8)
            await p.js("document.querySelector('#wTree [data-wpath=wxmodels]').click()")
            assert await p.until("!!document.querySelector('#wMain [data-wstart=wxmodels]')", 8)
            await p.js("document.querySelector('#wMain [data-wstart=wxmodels]').click()")
            assert await p.until("(document.querySelector('#wTa') || {}).value?.includes('documentclass')", 10)
            assert (h / "alpha" / "wxmodels" / "paper" / "paper.tex").is_file()
            assert await p.js("document.querySelector('.wpath').textContent === 'wxmodels/paper/paper.tex'")
            await p.js("document.querySelector('#wTree [data-wpath=fxnow]').click()")
            assert await p.until("!!document.querySelector('#wMain [data-wpath=\"fxnow/paper/paper.tex\"]')", 8)
            assert not await p.js("!!document.querySelector('#wMain [data-wstart]')")               # it has a paper
            await p.js("document.querySelector('#wMain [data-wpath=\"fxnow/paper/paper.tex\"]').click()")
            assert await p.until("(document.querySelector('#wTa') || {}).value?.includes('begin{document}')", 10)
            await p.js("document.querySelector('#wPv').click()")
            assert await p.until("document.querySelector('#wBody').classList.contains('noprev')")
            await p.js("document.querySelector('#wCmt').click()")
            assert await p.until("!!document.querySelector('#wNew')")
            assert "whole file" in await p.js("document.querySelector('#wCmts .dcard.new .q').textContent")
            await p.js("""(() => { const t = document.querySelector('#wNew'); t.value = 'Shorter, please.';
                t.dispatchEvent(new Event('input', {bubbles: true})); document.querySelector('[data-wcact=post]').click(); })()""")
            assert await p.until("document.querySelector('#wCmts').innerText.includes('Shorter, please.')", 8)
            from openlab.agents.comments import threads
            c = threads(st)[-1]
            assert (c["study"], c["file"], c["quote"], c["text"]) == ("fxnow", "paper/paper.tex", "", "Shorter, please.")
            await p.js("document.querySelector('[data-whide]').click()")
            assert await p.until("document.querySelector('#wsp').classList.contains('notree')")
            await p.js("document.querySelector('#wMain [data-wshow]').click()")
            assert await p.until("!document.querySelector('#wsp').classList.contains('notree')")
            # the start-the-lead dialog: "Let the agents decide", unchecked by default, sends decide=self when checked
            await p.js("openPal('lead')")
            assert await p.until("!!document.querySelector('#lDecide')")
            assert await p.js("!document.querySelector('#lDecide').checked && document.querySelector('#lDecide').closest('label').title.includes('still come to your inbox')")
            await p.js("""(() => { LAB.startLead = async (n, b) => { window._lead = b; return { ok: false, error: 'held by the test' }; };
                document.querySelector('#lP').value = 'go'; document.querySelector('#fGo').click(); })()""")
            assert await p.until("window._lead && !('decide' in window._lead)")
            await p.js("document.querySelector('#lDecide').checked = true; document.querySelector('#fGo').click()")
            assert await p.until("window._lead.decide === 'self'")
            await p.js("1")
            assert p.errors == []

    asyncio.run(run())
