#!/bin/bash
# The front door, from nothing: build the wheel, install it with the `study` extra into an empty venv with an empty
# cache, then walk the quickstart in a temp workspace with no model call:
#   openlab doctor; openlab init --claude; study new / materialise / run / run again (cached) / status;
#   openlab agents; openlab watch --once; propose new --draft-with none; propose lint (refuses the empty draft).
#   scripts/fresh_install_check.sh      # prints each step and the seconds it took; exit 0 when every check holds
# Needs uv and network for the install; nothing else leaves the machine.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export UV_CACHE_DIR="$TMP/cache" HOME="$TMP/home"             # an empty home: no person key, no user settings
mkdir -p "$HOME"
unset OPENLAB_AGENT OPENLAB_AGENT_TOKEN OPENLAB_WORKSPACE
fail() { echo "FAIL: $*" >&2; exit 1; }
check() { grep -q -- "$1" "$2" || { cat "$2" >&2; fail "$3"; }; }
t0=$(date +%s)
(cd "$REPO" && uv build --wheel -q -o "$TMP/dist")
t1=$(date +%s)
uv venv -q -p 3.11 "$TMP/v"
WHL="$(ls "$TMP"/dist/*.whl)"
uv pip install -q -p "$TMP/v/bin/python" "openlab-research[study] @ file://$WHL"
t2=$(date +%s)
export PATH="$TMP/v/bin:$PATH"                                 # `openlab` on PATH, as after a pip install
mkdir "$TMP/ws" && cd "$TMP/ws"
O="$TMP/out"

openlab doctor > "$O" || true                                  # the person key and hooks are not there yet
check "extra: study" "$O" "doctor lists the study extra"
echo "doctor: $(tail -1 "$O")"

openlab init --claude > "$O"
check ".claude/settings.json" "$O" "init --claude writes the hooks"
check "openlab hook SessionStart" .claude/settings.json "the hooks call \`openlab hook\` on PATH"
openlab doctor > "$O" || true
check "hooks  *yes" "$O" "doctor sees the hooks init wrote"
echo "init --claude: hooks, roles ($(ls .claude/agents | wc -l) files), .mcp.json, openlab.toml"

openlab study new demo --template temporal > "$O"
openlab study materialise demo > "$O"
check "panel: 2192 rows" "$O" "materialise registers the template data"
s=$(date +%s)
openlab study run demo > "$O"
r1=$(( $(date +%s) - s ))
check "trial 1 of 20" "$O" "the first run says it spent trial 1"
check "audit: clean" "$O" "the template run's audit is clean"
echo "run ($r1 s): $(head -1 "$O") | $(sed -n 2p "$O")"
openlab study run demo > "$O"
check "cached: no trial spent" "$O" "an unchanged rerun is cached"
openlab study status demo > "$O"
check "trials 1 of 20" "$O" "status counts one trial"
echo "status: $(grep '^trials' "$O")"

openlab agents > "$O"
check "no agents" "$O" "openlab agents on an empty workspace"
openlab watch --once > "$O"
check "openlab watch" "$O" "watch --once prints a frame"
echo "agents, watch --once: ok"

openlab propose new lagx "Does yesterday's x beat persistence?" --mode temporal --draft-with none > "$O"
check "proposal.toml" "$O" "propose new scaffolds proposal.toml"
if openlab propose lint lagx > "$O" 2>&1; then fail "lint passed an empty draft"; fi
check "falsifier" "$O" "lint refuses a hypothesis without a falsifier"
echo "propose new, lint: $(tail -1 "$O")"
t3=$(date +%s)
echo "build $((t1-t0)) s · install $((t2-t1)) s · front door $((t3-t2)) s · install-to-result $((t3-t1)) s"
