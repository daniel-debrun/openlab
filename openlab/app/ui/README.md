# The web UI

React 18 + Vite + TypeScript. The build output is committed to `../static/` and shipped as package data.

    npm install
    npm run build        # tsc --noEmit, then vite build into ../static
    npm run dev          # against a running `openlab app`

The UI computes nothing: every action is a tool call (`POST /api/tool/{name}`), the same dispatch the CLI
and the MCP use. Colours and sizes come from `src/styles/tokens.css`; `tests/test_ui_tokens.py` enforces it.
