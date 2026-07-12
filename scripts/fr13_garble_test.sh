#!/usr/bin/env bash
# ============================================================================
# CANONICAL FR13 garble test harness (2026-07-12). ONE reliable path so the
# garble hunt stops re-tripping on: greedy-gate false-negatives, background-task
# kills, OOM-wedge reboots, GPU_UTIL=0.88 guard-graze, ad-hoc health waits.
#
# Usage:  scripts/fr13_garble_test.sh [KEY=VAL ...]    # extra env armed at boot
#   e.g.  scripts/fr13_garble_test.sh FR13_INPROJ_BA_BMM=1 ENFORCE_EAGER=1
#   e.g.  scripts/fr13_garble_test.sh BATCH_INVARIANT=1
#   REUSE=1 scripts/fr13_garble_test.sh   # skip boot, gate the running server
#
# Gate = temp-0.6 matrix_build (+ wcs_slice, token_ledger) undefined-name rate,
# TREE vs the committed NATIVE reference (native = 0/15 on all prompts). The
# greedy _rows/idx-65 gate is DEPRECATED (false negatives, see
# reference_garble_gate_greedy_false_negative). FIXED => matrix_build ~0/N.
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
N="${GARBLE_N:-20}"
PORT="${PORT:-9950}"
EP="http://127.0.0.1:$PORT"

if [[ "${REUSE:-0}" != "1" ]]; then
  echo "[boot] kill + recover-wedge + launch (GPU_UTIL=0.78, cache OFF, arm: $*)"
  docker rm -f fr13-lad >/dev/null 2>&1 || true
  # ModelServer host-memory recovery (SIGKILL bypasses graceful stop -> ~100GiB wedge)
  # shellcheck disable=SC1091
  source "$REPO/.lumo.local.env" 2>/dev/null || true
  printf '%s\n' "${LUMO_SUDO_PASSWORD:-}" | sudo -S bash -lc \
    'sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true' >/dev/null 2>&1 || true
  echo "[boot] mem free after recovery: $(free -g | awk 'NR==2{print $4}')GiB"
  env CONTAINER=fr13-lad PORT="$PORT" FR13_ENABLE_APC="${FR13_ENABLE_APC:-0}" GPU_UTIL="${GPU_UTIL:-0.78}" "$@" \
    bash "$HERE/fr13_launch_locked.sh" > /tmp/fr13_garble_boot.log 2>&1
  echo "[boot] launcher rc=$? ; waiting for health (bounded, foreground)..."
fi

# Foreground bounded health wait (curl --retry; no 'sleep' Bash-tool block, no bg-task kill)
if ! curl -s --retry 90 --retry-delay 6 --retry-all-errors -o /dev/null \
       -w '[health] %{http_code} after retries\n' --max-time 560 "$EP/health" 2>/dev/null | grep -q 200; then
  # one more direct check (server may have come up right at the deadline)
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$EP/health" 2>/dev/null || echo 000)
  if [[ "$code" != "200" ]]; then
    echo "[FATAL] NEVER HEALTHY (http=$code). boot tail:"; docker logs --tail 15 fr13-lad 2>&1 | tail -15
    exit 2
  fi
fi
echo "[health] 200 OK"

# The reliable gate: temp-0.6 tree-vs-native undefined-name rate (fr13_garble_gate.py).
OUT="$REPO/output/fr13_garble_gate_test.jsonl"
python3 "$HERE/fr13_garble_gate.py" run --endpoint "$EP/v1" --model qwen3.6-27b \
    --arm test --n "$N" --concurrency 4 --out "$OUT" 2>&1 | grep -E "undefined-name-rate|syntax|wrote" || true

echo "=== PER-PROMPT (TREE) vs NATIVE(0/N) -- the reliable garble verdict ==="
python3 - "$OUT" <<'PY'
import json,ast,builtins,sys
from collections import defaultdict
BI=set(dir(builtins))|{"self","cls","np","torch","os","sys","re","math"}
def code(t):
    if "```python" in t: t=t.split("```python",1)[1]
    if "```" in t: t=t.split("```",1)[0]
    return t.strip()
def undef(c):
    try: tr=ast.parse(c)
    except: return None
    L={x.id for x in ast.walk(tr) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    S={x.id for x in ast.walk(tr) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
    A=set()
    for fn in ast.walk(tr):
        if isinstance(fn,(ast.FunctionDef,ast.Lambda)):
            for a in fn.args.args+fn.args.posonlyargs+fn.args.kwonlyargs: A.add(a.arg)
    return L-S-A-BI
per=defaultdict(lambda:[0,0,0])  # garbled, syntaxbad, total
for l in open(sys.argv[1]):
    r=json.loads(l); c=code(r["text"]); u=undef(c)
    per[r["prompt"]][2]+=1
    if u is None: per[r["prompt"]][1]+=1
    elif u: per[r["prompt"]][0]+=1
for p,(g,sb,n) in sorted(per.items()):
    flag=" <-- garble" if p=="matrix_build" and g>1 else (" CLEAN" if p=="matrix_build" and g<=1 else "")
    print("  %-14s garbled=%d/%d  syntax_bad=%d%s"%(p,g,n,sb,flag))
mb=per.get("matrix_build",[0,0,0])
print("VERDICT: matrix_build garbled=%d/%d (native=0/%d). ~0 = FIXED. syntax_bad>~5 => degenerate, not a clean read."%(mb[0],mb[2],mb[2]))
PY
