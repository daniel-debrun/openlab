"""The seal in openlab show, in a real browser (headless Chrome over the DevTools protocol, as test_show_e2e): an
unsealed study is badged Not sealed in the Workspace with what sealing freezes and costs; its Seal button opens a
confirm naming those consequences, and the passphrase signs the study's first research run, whose trial, result id
and primary metric the dialog shows. An experiment of an unsealed study says so on its card and on its inbox item,
with a Seal button. No JS errors; screenshots are kept (SHOW_SHOT_DIR keeps a copy)."""

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

pytestmark = [pytest.mark.skipif(CHROME is None, reason="google-chrome is not installed"),
              pytest.mark.skipif(not shutil.which("bwrap"), reason="a research run is sandboxed under bwrap")]

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch, person_key):
    import openlab.study as vs
    import test_experiments as T
    from openlab.agents.identity import acting_as
    from openlab.agents.store import Store
    from openlab.study import experiments as E
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    h = tmp_path / "home"
    subprocess.run([sys.executable, "-m", "openlab", "init", str(h / "lab")], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(REPO)})
    ws = h / "lab"
    vs.new(ws / "s1", "static").materialise()
    pin = T.register(ws, person_key)
    folder = T.make_study(ws, {"tender": pin})
    T.save(folder, T.spec())
    Store(ws).register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    with acting_as("lead-1"):
        E.request(ws, "tender", "paper_trial", "start this only after the study is sealed")
    return h, ws


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
        yield f"http://127.0.0.1:{port}/show/?t=tok&r={reviewer}"
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _shot(p, tmp_path, name):
    shot = await p.call("Page.captureScreenshot", format="png")
    png = tmp_path / name
    png.write_bytes(base64.b64decode(shot["result"]["data"]))
    if os.environ.get("SHOW_SHOT_DIR"):
        Path(os.environ["SHOW_SHOT_DIR"]).mkdir(parents=True, exist_ok=True)
        shutil.copy(png, Path(os.environ["SHOW_SHOT_DIR"]) / png.name)


def test_an_unsealed_study_is_badged_explained_and_sealed_from_the_page(home, server, chrome, tmp_path):
    from conftest import PASSPHRASE
    from openlab.study import sealing
    ws = home[1]

    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as sock:
            p = Page(sock)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            # the inbox item of kind experiment, and the Experiments card, say the study is not sealed
            await p.call("Page.navigate", url=server + "#inbox")
            assert await p.until("!!document.querySelector('#inbox .sealline')", 20)
            line = await p.js("document.querySelector('#inbox .sealline').innerText")
            assert "The study tender is not sealed" in line and "spends trial 1" in line
            assert "Not yet" in line and "openlab study materialise tender" in line    # its data is not built yet
            await _shot(p, tmp_path, "seal-inbox.png")
            await p.js("go('experiments')")
            assert await p.until("!!document.querySelector('#experiments .xcard .sealline')", 8)
            assert "tender" in await p.js("document.querySelector('#experiments .xcard .sealline').innerText")
            # the Workspace: the research's folder lists the unsealed studies; a study's card explains the seal
            await p.js("go('paper')")
            assert await p.until("!!document.querySelector('#wMain [data-seal=\"s1\"]')", 10)
            assert await p.until("!!document.querySelector('#wTree [data-wpath=\"s1\"] .nseal')", 6)
            await p.js("document.querySelector('#wTree [data-wpath=\"s1\"]').click()")
            assert await p.until("!!document.querySelector('#wMain .sealbox [data-seal=\"s1\"]')", 10)
            box = await p.js("document.querySelector('#wMain .sealbox').innerText")
            for words in ("Not sealed", "the split and holdout", "the primary metric: auc", "trial 1",
                          "holdout is untouched", "openlab study amend", "openlab study seal s1"):
                assert words in box, words
            await _shot(p, tmp_path, "seal-card.png")
            # Seal: the confirm names the consequences, then the passphrase signs the first research run
            await p.js("document.querySelector('#wMain .sealbox [data-seal=\"s1\"]').click()")
            assert await p.until("document.querySelector('#sl').open", 4)
            dlg = await p.js("document.querySelector('#sl').innerText")
            for words in ("Seal s1 now?", "trial 1", "20 trials", "holdout is untouched", "cannot be undone",
                          "passphrase signs the seal"):
                assert words in dlg, words
            await _shot(p, tmp_path, "seal-confirm.png")
            await p.js("document.querySelector('#slOk').click()")             # no passphrase: nothing is sent
            assert not await p.js("!!document.querySelector('#slOut .ok')")
            await p.js(f"document.querySelector('#slP').value = {PASSPHRASE!r}")
            await p.js("document.querySelector('#slOk').click()")
            assert await p.until("!!document.querySelector('#slOut .ok')", 120)
            out = await p.js("document.querySelector('#slOut').innerText")
            assert "Sealed." in out and "trial 1 of 20" in out and "result:" in out and "auc" in out
            await _shot(p, tmp_path, "seal-done.png")
            await p.js("document.querySelector('#slX').click()")
            assert await p.until("!!document.querySelector('#wMain .yseal')", 10)    # the card now says Sealed
            assert not await p.js("!!document.querySelector('#wMain .sealbox')")
            assert p.errors == []

    asyncio.run(run())
    st = sealing.state(ws / "s1")
    assert st["sealed"] and st["by"] == "the person"


def test_the_demo_page_still_draws_without_seal_data(chrome):
    async def run():
        async with websockets.connect(chrome, max_size=50_000_000) as sock:
            p = Page(sock)
            await p.call("Runtime.enable")
            await p.call("Page.enable")
            await p.call("Page.navigate", url=(REPO / "openlab" / "app" / "show" / "index.html").as_uri() + "#inbox")
            assert await p.until("document.querySelectorAll('#inbox .it').length > 0", 10)
            assert not await p.js("!!document.querySelector('.sealline')")
            assert p.errors == []

    asyncio.run(run())
