r"""Call a tool from outside the server: the CLI and the MCP use this.

With ``OPENLAB_APP_URL`` (and ``OPENLAB_APP_TOKEN``) the call goes to the running workspace server, so an agent's
actions run in the sessions the UI has open and show up there as they happen. Without it, a local
:class:`ToolContext` over ``$OPENLAB_WORKSPACE`` (else the working directory) serves the call in this process.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

_LOCAL = None


def _local():
    global _LOCAL
    if _LOCAL is None:
        from . import ToolContext
        _LOCAL = ToolContext(os.environ.get("OPENLAB_WORKSPACE") or ".")
    return _LOCAL


def _agent() -> str:
    from ..agents.identity import current
    return current()


def call(name: str, args: Optional[dict] = None, timeout: float = 900.0) -> dict:
    url = os.environ.get("OPENLAB_APP_URL", "").rstrip("/")
    if not url:
        from . import dispatch
        return dispatch(_local(), name, args or {}, caller="agent")
    req = urllib.request.Request(f"{url}/api/tool/{name}", data=json.dumps(args or {}).encode(), method="POST",
                                 headers={"content-type": "application/json", "x-openlab-caller": "agent",
                                          "x-openlab-agent": _agent(),               # who is acting
                                          "authorization": f"Bearer {os.environ.get('OPENLAB_APP_TOKEN', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:        # noqa: S310 -- the user's own server
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except ValueError:
            return {"ok": False, "tool": name, "error": f"http {e.code}", "meaning": str(e.reason),
                    "try": "check OPENLAB_APP_URL and OPENLAB_APP_TOKEN"}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "tool": name, "error": "unreachable", "meaning": f"{url}: {e}",
                "try": "start it with `openlab app`, or unset OPENLAB_APP_URL to work locally"}
