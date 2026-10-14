"""The README's screenshots of `openlab show`, taken from the demo workspace (scripts/demo_workspace.py) in headless
Chrome over the DevTools protocol.

    python scripts/demo_workspace.py /tmp/openlab-demo
    python scripts/screenshots.py /tmp/openlab-demo docs/images

Needs google-chrome and the `websockets` package (in the `test` extra).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

W, H = 1440, 900
RESEARCH = "station-forecasts"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, seconds: float = 30) -> str:
    end = time.time() + seconds
    while time.time() < end:
        try:
            return urllib.request.urlopen(url, timeout=2).read().decode()
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not answer")


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        n = self.n
        await self.ws.send(json.dumps({"id": n, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == n:
                return m

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("result", {}).get("value")

    async def until(self, expr, seconds=15.0):
        end = time.time() + seconds
        while time.time() < end:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.25)
        raise RuntimeError(f"timed out waiting for {expr}")

    async def click(self, selector):
        await self.js(f"document.querySelector({json.dumps(selector)}).dispatchEvent(new MouseEvent('click', {{bubbles: true}}))")

    async def shot(self, path: Path):
        r = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["result"]["data"]))
        print(path)


async def take(url: str, chrome_ws: str, out: Path):
    import websockets
    async with websockets.connect(chrome_ws, max_size=100_000_000) as ws:
        p = Page(ws)
        await p.call("Page.enable")
        await p.call("Runtime.enable")
        await p.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        await p.call("Page.navigate", url=f"{url}&w={RESEARCH}")
        await p.until("document.body.innerText.includes('lead-4f2a')", 30)
        await asyncio.sleep(1.5)
        await p.shot(out / "activity.png")

        await p.click("[data-view=roadmap]")
        await p.until("!!document.querySelector('.bar[data-node=final]')")
        await p.click(".bar[data-node=final]")
        await p.until("document.querySelector('#peek').classList.contains('on')")
        await asyncio.sleep(0.8)
        await p.shot(out / "roadmap.png")

        await p.click("[data-view=paper]")
        await p.until("!!document.querySelector('#wTree [data-wpath=\"next-day\"]')")
        await p.click("#wTree [data-wpath=\"next-day\"]")
        await p.until("!!document.querySelector('#wTree [data-wpath=\"next-day/paper\"]')")
        await p.click("#wTree [data-wpath=\"next-day/paper\"]")
        await p.until("!!document.querySelector('#wTree [data-wpath=\"next-day/paper/paper.tex\"]')")
        await p.click("#wTree [data-wpath=\"next-day/paper/paper.tex\"]")
        await p.until("(document.querySelector('#wMsg') || {}).textContent?.includes('Rendered')", 90)
        await p.click("#wCmts .dcard")
        await asyncio.sleep(3)
        await p.shot(out / "workspace.png")

        await p.click("[data-view=inbox]")
        await p.until("document.querySelector('#inbox').innerText.includes('holdout')")
        await asyncio.sleep(0.6)
        await p.shot(out / "inbox.png")

        await p.click("[data-view=ledger]")
        await p.until("document.querySelector('#ledger').innerText.includes('result:')")
        await asyncio.sleep(0.6)
        await p.shot(out / "ledger.png")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root, out = Path(argv[0]).resolve(), Path(argv[1] if len(argv) > 1 else "docs/images").resolve()
    out.mkdir(parents=True, exist_ok=True)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        sys.exit("google-chrome is not installed")
    sport, cport = free_port(), free_port()
    server = subprocess.Popen([sys.executable, "-m", "openlab", "show", "--root", str(root), "--port", str(sport),
                               "--token", "demo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    profile = tempfile.mkdtemp(prefix="openlab-shots-")
    browser = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                                f"--remote-debugging-port={cport}", f"--window-size={W},{H}",
                                f"--user-data-dir={profile}", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        line = ""
        end = time.time() + 30
        while "&r=" not in line and time.time() < end:
            line = server.stdout.readline()
        url = line.strip()
        wait_http(f"http://127.0.0.1:{sport}/api/health")
        tabs = json.loads(wait_http(f"http://127.0.0.1:{cport}/json"))
        asyncio.run(take(url, next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page"), out))
    finally:
        for p in (browser, server):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(profile, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
