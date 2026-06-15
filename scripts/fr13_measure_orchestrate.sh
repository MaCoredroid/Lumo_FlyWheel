#!/usr/bin/env bash
# FR13 CANONICAL MEASURE ORCHESTRATOR (2026-06-15)
# Serialized GPU boots driving scripts/fr13_measure.py in the canonical regime.
# GPU SERIALIZED -> ONE boot at a time on PORT 9950; recover_host_memory +
# assert MemAvailable>=95GiB + docker-empty BEFORE each boot; teardown after.
#
# THE REGIME (baked, no re-roll): prompts_swe4.json, seed 1313, max_tokens 128,
# FR10_METRICS=0, BATCH_INVARIANT=0, /v1/completions raw prompt, ONE raw
# self-warm (NOT chat template -- that was the phase0 confound).
#
# INSTRUMENT ON/OFF: every arm gets a CLEAN-OFF speed pass (no logprobs) AND a
# separate ON pass (capture-q top-K logprobs). SPEED is read ONLY from the OFF
# pass; the ON pass feeds lossless/drift + the diag-residue tax.
#
# USAGE:
#   scripts/fr13_measure_orchestrate.sh <command> [args]
#     native <eN>                  -> OFF speed B=1 + B=4 + ON capture-q for native MTP-N
#     tree   <name> "<TREE>"       -> same for a TREE shape (e.g. cat9)
#     reconcile                    -> reduce all OFF speed records vs banked
# The launchers: native = fr10_launch_speed_server.sh (naive_mtp, FLASH);
#                tree   = fr13_launch_locked.sh (cat9) or
#                         fr13_launch_forked_fa2_tree_server.sh (arbitrary TREE).
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel
cd "$REPO"
OUT=${OUT:-output/fr13_measure}
mkdir -p "$OUT"
PROMPTS="$REPO/output/fr13_acceptance_ladder/prompts_swe4.json"
SEED=1313
PORT=9950
ENDPOINT="http://127.0.0.1:$PORT"
MODEL=qwen3.6-27b
MEASURE="$REPO/scripts/fr13_measure.py"

recover() { PYTHONPATH="$REPO/src" python3 -c 'from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()' >/dev/null 2>&1 || true; }
assert_free() {
  python3 - <<'PY'
from pathlib import Path
f = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    k, v = line.split(":", 1); f[k] = int(v.strip().split()[0])
avail = f.get("MemAvailable", 0) / 1024 / 1024
swap = (f.get("SwapTotal", 0) - f.get("SwapFree", 0)) / 1024 / 1024
if avail < 95 or swap != 0:
    raise SystemExit(f"HYGIENE FAIL MemAvailable={avail:.1f}GiB(<95) swap_used={swap:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap=0 OK")
PY
}
wait_health() {
  local cont="$1" deadline; deadline=$(( $(date +%s)+900 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local st; st="$(docker inspect -f '{{.State.Status}}' "$cont" 2>/dev/null || echo absent)"
    if [ "$st" = exited ] || [ "$st" = dead ]; then
      echo "[boot] $cont $st before health; log tail:"; docker logs --tail 60 "$cont" 2>&1 | tail -60; return 1
    fi
    curl -fsS --max-time 5 "$ENDPOINT/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "[boot] $cont health TIMEOUT"; return 1
}
teardown() { docker rm -f "$1" >/dev/null 2>&1 || true; recover; }
empty_docker() { docker rm -f fr10-speed-start fr13-forked-fa2-tree >/dev/null 2>&1 || true; }

# measure_arm <arm> <tree-or-empty> <container>  (server booted+healthy)
# OFF speed B=1 + B=4, then ON capture-q, then diag-residue.
measure_arm() {
  local arm="$1" tree="$2" cont="$3"
  local treearg=(); [ -n "$tree" ] && treearg=(--tree "$tree")
  echo "===== measure $arm (instrument OFF: speed B=1, B=4) ====="
  # OFF B=1 sequential greedy speed
  python3 "$MEASURE" speed --arm "$arm" "${treearg[@]}" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --batch-size 1 --temperature 0.0 --top-p 1.0 --seed "$SEED" --max-tokens 128 \
    --out "$OUT/${arm}_speed_b1_off.json" || { echo "[$arm] speed B=1 FAIL"; return 1; }
  # OFF B=4 co-resident greedy speed
  python3 "$MEASURE" speed --arm "$arm" "${treearg[@]}" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --batch-size 4 --temperature 0.0 --top-p 1.0 --seed "$SEED" --max-tokens 128 \
    --out "$OUT/${arm}_speed_b4_off.json" || echo "[$arm] speed B=4 nonzero (continuing)"
  echo "===== measure $arm (instrument ON: capture-q temp 0.6/top_p 0.95) ====="
  python3 "$MEASURE" capture-q --arm "$arm" "${treearg[@]}" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --seed "$SEED" --max-tokens 128 \
    --out "$OUT/${arm}_q_temp06_on.json" || echo "[$arm] capture-q nonzero (continuing)"
  # diag-residue (OFF vs ON s/fwd tax)
  if [ -f "$OUT/${arm}_speed_b1_off.json" ] && [ -f "$OUT/${arm}_q_temp06_on.json" ]; then
    python3 "$MEASURE" diag-residue --off "$OUT/${arm}_speed_b1_off.json" \
      --on "$OUT/${arm}_q_temp06_on.json" --out "$OUT/${arm}_diag_residue.json" \
      || echo "[$arm] diag-residue nonzero (continuing)"
  fi
  echo "[$arm] done: $(ls "$OUT"/${arm}_*.json 2>/dev/null | tr '\n' ' ')"
}

boot_native() {  # eN
  local arm="native_$1" N="${1#e}"
  echo "######## boot $arm (native MTP-$N, FLASH) ########"
  recover; assert_free || return 1; empty_docker
  NUM_SPECULATIVE_TOKENS="$N" \
  SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$N}" \
  CONTAINER=fr10-speed-start PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
    bash "$REPO/scripts/fr10_launch_speed_server.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  sleep 8
  if wait_health fr10-speed-start; then measure_arm "$arm" "" fr10-speed-start; fi
  teardown fr10-speed-start
}

boot_tree() {  # name TREE
  local arm="$1" tree="$2"
  echo "######## boot $arm (tree_mtp, TREE_ATTN) ########"
  recover; assert_free || return 1; empty_docker
  if [ "$arm" = "cat9" ]; then
    CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_locked.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  else
    TREE="$tree" CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  fi
  sleep 8
  if wait_health fr13-forked-fa2-tree; then measure_arm "$arm" "$tree" fr13-forked-fa2-tree; fi
  teardown fr13-forked-fa2-tree
}

CMD="${1:-}"; shift || true
case "$CMD" in
  native) boot_native "$1" ;;
  tree)   boot_tree "$1" "$2" ;;
  reconcile)
    python3 "$MEASURE" reconcile --speed "$OUT"/*_speed_b1_off.json \
      --out "$OUT/reconcile.json" ;;
  *)
    echo "usage: $0 {native eN | tree NAME 'TREE' | reconcile}" >&2; exit 2 ;;
esac
