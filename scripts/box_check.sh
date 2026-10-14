#!/bin/bash
# Phase 2 on the box, end to end. Run it from your own machine: it holds your person key, and you type your
# passphrase for the two signed steps (4 and 6). Everything happens in a scratch workspace on the box (/root/p2check),
# never in a real one.
#
#   HOST=gpu-box scripts/box_check.sh
#
# 1 syncs this checkout to the box and installs it with the W&B extra; 2 puts your W&B key on the box (the mirror
# uploads from there); 3 makes a study that needs one GPU, mirrored to the W&B project openlab-smoke; 4 pins the box's
# workspace to your key, signed here; 5 starts three runs at once (two get a 3090 each, the third waits its turn);
# 6 takes a holdout look on the box, signed here; 7 verifies the record there; 8 shows the mirror's outcome.
set -euo pipefail
HOST=${HOST:-gpu-box}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAB=/root/openlab/.venv/bin/openlab
PY=/root/openlab/.venv/bin/python
WS=/root/p2check
LOCAL=("$REPO/.venv/bin/python" -m openlab)
on_box() { ssh -o ClearAllForwardings=yes -o LogLevel=ERROR "$HOST" "$@"; }
step() { printf '\n== %s\n' "$*"; }

step "1. sync this checkout to $HOST:/root/openlab and install it (with wandb)"
(cd "$REPO" && git ls-files -co --exclude-standard | grep -v '^todo/' | tar -cf - -T -) |
  on_box 'mkdir -p /root/openlab && cd /root/openlab && find . -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} + &&
          tar -xf - && { [ -x .venv/bin/python ] || uv venv -q -p 3.12 .venv; } &&
          uv pip install -q -p .venv/bin/python -e ".[test,study,mirror]" && echo "installed: $(.venv/bin/openlab --version)"'

step "2. your W&B key into $HOST:/root/.netrc (mode 600), if it is not there yet"
grep -A2 '^machine api.wandb.ai' ~/.netrc |
  on_box 'umask 077; touch /root/.netrc; if grep -q api.wandb.ai /root/.netrc; then cat >/dev/null; echo "already there";
          else cat >> /root/.netrc; echo "added"; fi; chmod 600 /root/.netrc'

step "3. a scratch workspace with a study that needs one GPU (W&B project openlab-smoke)"
on_box "rm -rf $WS && mkdir -p $WS && cd $WS && $LAB init >/dev/null &&
        printf '\n[mirror]\nproject = \"openlab-smoke\"\n' >> openlab.toml &&
        $LAB study new gpu --template temporal >/dev/null &&
        sed -i 's/^\[sandbox\]\$/[sandbox]\ngpus = 1/' gpu/study.toml && grep -A1 '^\[sandbox\]' gpu/study.toml &&
        $LAB study materialise gpu | tail -1 && $LAB box && $LAB mirror status gpu"

step "4. pin the box's workspace to YOUR key, signed on this machine (type your passphrase)"
"${LOCAL[@]}" person act "$HOST" --openlab "$LAB" --cd "$WS" -- person pin

step "5. three runs at once (seeds 1-3): two hold a 3090 each, the third waits for one"
on_box "cd $WS && for s in 1 2 3; do $LAB study run gpu --seed \$s --note seed\$s > run\$s.log 2>&1 & done;
        sleep 5; $LAB box queue; wait; echo; grep -h -E 'queued|^trial' run*.log;
        echo; $PY -c 'from openlab import study; s = study.open(\"gpu\")
for r in s.record.find(\"run\"): print(r.id, r.payload[\"seed\"], r.payload[\"sandbox\"], r.payload[\"box\"])'"

step "6. a holdout look on the box, signed on this machine (type your passphrase)"
"${LOCAL[@]}" person act "$HOST" --openlab "$LAB" --cd "$WS" -- study holdout gpu --reason "phase 2 check"

step "7. the record on the box verifies, against your key"
on_box "cd $WS && $LAB study verify gpu | head -6 || true; $LAB person show 2>/dev/null | head -3 || true"
echo "your key here: $("${LOCAL[@]}" person show 2>/dev/null | grep -m1 -o 'ed25519:[0-9a-f:]*' || true)"

step "8. the W&B mirror (it uploads in the background; give it half a minute)"
sleep 30
on_box "cd $WS && $LAB mirror status gpu; tail -n 4 gpu/.openlab/mirror.jsonl 2>/dev/null | cut -c1-200 || true"
echo
echo "done. The scratch workspace stays at $HOST:$WS; remove it with: ssh $HOST rm -rf $WS"
