"""`python -m openlab.agents.hook <Event>`: the fast entry point Claude Code's hooks run (see openlab.agents.hooks)."""

import sys

from .hooks import main

if __name__ == "__main__":
    sys.exit(main())
