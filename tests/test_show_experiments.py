"""The Experiments tab of openlab show, in a real browser (headless Chrome over the DevTools protocol, as
test_show_e2e): a running experiment and a concluded one, drawn from the studies' records with their progress, jobs,
runs, end and reaction, the Roadmap step each implements, and the registered instruments below. A run's chip shows its
output, read through instrument_output. No JS errors; a screenshot is kept (SHOW_SHOT_DIR keeps a copy)."""

import asyncio
import base64
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")
from test_show_e2e import CHROME, Page, chrome, free_port, wait_http  # noqa: E402,F401 -- the browser fixture

pytestmark = pytest.mark.skipif(CHROME is None, reason="google-chrome is not installed")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch, person_key):
    import test_experiments as T
    from openlab.agents import plan
    from openlab.agents.identity import acting_as
    from openlab.agents.store import Store
    from openlab.study import experiments as E
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    monkeypatch.setattr(E, "SPAWNER", lambda root, **kw: {"id": "reviewer-0a0b0c"})
    h = tmp_path / "home"
    subprocess.run([sys.executable, "-m", "openlab", "init", str(h / "alpha")], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(REPO)})
    ws = h / "alpha"
    pin = T.register(ws, person_key)
    T.make_study(ws, {"tender": pin})
    st = Store(ws)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 6})
    plan.report(st, "lead-1", [{"id": "paper-exp", "title": "The forward paper trial", "kind": "run", "state": "planned"},
                               {"id": "warm", "title": "A warm-up", "kind": "run", "state": "planned"}])
    folder = ws / "tender"
    T.save(folder, T.spec(stop="passes = 20\nuntil = 2026-12-31"))
    T.save(folder, T.spec(name="warmup", stop="passes = 1").replace('plan_step = "paper-exp"', 'plan_step = "warm"'),
           "warmup")
    for name in ("paper_trial", "warmup"):
        with acting_as("lead-1"):
            req = E.request(ws, "tender", name, f"the {name}")
        with person_key.act():
            E.start(ws, "tender", name, "go", spec_sha=req["sha"])
    for day in (19, 20):
        E.job(ws, "tender", "paper_trial", "scan", at=T.ny(day, 9, 30))
        E.job(ws, "tender", "paper_trial", "export", at=T.ny(day, 23, 30))
    end = E.job(ws, "tender", "warmup", "export", at=T.ny(19, 23, 30))
    assert end["ended"]["state"] == "concluded"
    with acting_as("reviewer-0a0b0c"):
        E.react(ws, "tender", "warmup", "met", "the warm-up exported its rows", [end["pass"]])
    return h, ws, pin


@pytest.fixture
def server(home):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "openlab", "show", "--root", str(home[0]), "--port", str(port),
                             "--token", "tok"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONPATH": str(REPO), "OPENLAB_APP_TERMINAL": "0"})
    try:
        line, end = "", dt.datetime.now().timestamp() + 30
        while "&r=" not in line and dt.datetime.now().timestamp() < end:
            line = proc.stdout.readline()
        reviewer = line.strip().split("&r=")[1]
        wait_http(f"http://127.0.0.1:{port}/api/health")
        yield f"http://127.0.0.1:{port}/show/?t=tok&r={reviewer}#experiments"
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_the_experiments_tab_draws_progress_runs_the_reaction_and_the_instruments(home, server, chrome, tmp_path):
    h, ws, pin = home

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as sock:
            p = Page(sock)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=server)
            assert await p.until("document.querySelectorAll('#experiments .xcard').length === 2", 20)
            text = await p.js("document.querySelector('#experiments').innerText")
            assert "paper_trial" in text and "warmup" in text and pin in text
            assert await p.js("!!document.querySelector('[data-xcard=\"tender/paper_trial\"] .xst.running')")
            assert await p.js("!!document.querySelector('[data-xcard=\"tender/warmup\"] .xst.reacted')")
            card = "document.querySelector('[data-xcard=\"tender/paper_trial\"]')"
            assert "2 of 20 passes" in await p.js(f"{card}.innerText")              # at first glance: how far along
            # its jobs, runs and timing are one click away
            await p.js("document.querySelectorAll('#experiments details.xmore').forEach((d) => { d.open = true; })")
            assert "2 of 20 ok" in await p.js(f"{card}.innerText")
            assert "until" in await p.js(f"{card}.innerText")
            assert "Roadmap: lead-1 · The forward paper trial · active" in await p.js(f"{card}.innerText")
            assert await p.js(f"{card}.querySelectorAll('.xrun.ok').length") == 4
            assert "stop rule met: 1 passes" in await p.js(
                "document.querySelector('[data-xcard=\"tender/warmup\"]').innerText")
            assert "the warm-up exported its rows" in text                      # the verdict's first sentence
            await p.js("document.querySelectorAll('#experiments details.more').forEach((d) => { d.open = true; })")
            assert "reviewer-0a0b0c" in await p.js(              # who concluded it, one click away
                "document.querySelector('[data-xcard=\"tender/warmup\"]').innerText")
            # a run's chip shows its output, read through instrument_output
            await p.js(f"{card}.querySelector('.xrun.ok').click()")
            assert await p.until(f"({card}.querySelector('.xout pre') || {{}}).innerText?.includes('ran ')", 8)
            shot = await p.call("Page.captureScreenshot", format="png")
            png = tmp_path / "experiments-tab.png"
            png.write_bytes(base64.b64decode(shot["result"]["data"]))
            if os.environ.get("SHOW_SHOT_DIR"):
                Path(os.environ["SHOW_SHOT_DIR"]).mkdir(parents=True, exist_ok=True)
                shutil.copy(png, Path(os.environ["SHOW_SHOT_DIR"]) / png.name)
            # its Roadmap step opens on the Roadmap
            await p.js(f"{card}.querySelector('.stepl').click()")
            assert await p.until("getComputedStyle(document.querySelector('#v-roadmap')).display !== 'none'", 4)
            assert p.errors == []

    asyncio.run(run())


def test_the_demo_page_draws_its_experiments(chrome):
    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as sock:
            p = Page(sock)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=(REPO / "openlab" / "app" / "show" / "index.html").as_uri() + "#experiments")
            assert await p.until("document.querySelectorAll('#experiments .xcard').length === 3", 10)
            await p.js("document.querySelector('#experiments [data-xusage]').click()")
            assert await p.until("document.querySelector('#experiments .xout pre').innerText.includes('usage: wx_obs')", 4)
            assert p.errors == []

    asyncio.run(run())
