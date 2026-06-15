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

# measure_arm_b1 <arm> <tree-or-empty> <container>  (gold regime, MAX_NUM_SEQS=1)
# OFF speed B=1 (the deployment number, reproduces banked), then ON capture-q,
# then diag-residue. B=4 is a SEPARATE boot (measure_arm_b4) so it does not
# perturb the B=1 trajectory.
measure_arm_b1() {
  local arm="$1" tree="$2" cont="$3"
  local treearg=(); [ -n "$tree" ] && treearg=(--tree "$tree")
  echo "===== measure $arm (instrument OFF: speed B=1 SEQUENTIAL gold regime) ====="
  # OFF B=1 sequential greedy speed (dump streams to bind the fork point)
  python3 "$MEASURE" speed --arm "$arm" "${treearg[@]}" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --batch-size 1 --temperature 0.0 --top-p 1.0 --seed "$SEED" --max-tokens 128 \
    --dump-streams \
    --out "$OUT/${arm}_speed_b1_off.json" || { echo "[$arm] speed B=1 FAIL"; return 1; }
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
  echo "[$arm] B=1 done: $(ls "$OUT"/${arm}_*.json 2>/dev/null | tr '\n' ' ')"
}

# measure_arm_b4 <arm> <tree-or-empty> <container>  (co-residency, MAX_NUM_SEQS=4)
measure_arm_b4() {
  local arm="$1" tree="$2" cont="$3"
  local treearg=(); [ -n "$tree" ] && treearg=(--tree "$tree")
  echo "===== measure $arm (instrument OFF: speed B=4 CO-RESIDENT) ====="
  python3 "$MEASURE" speed --arm "$arm" "${treearg[@]}" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --batch-size 4 --temperature 0.0 --top-p 1.0 --seed "$SEED" --max-tokens 128 \
    --out "$OUT/${arm}_speed_b4_off.json" || echo "[$arm] speed B=4 nonzero (continuing)"
  echo "[$arm] B=4 done: $(ls "$OUT"/${arm}_speed_b4_off.json 2>/dev/null)"
}

# REGIME FIX (2026-06-15): the banked native E5 3.161290 was captured with
# MAX_NUM_SEQS=1 (FR13_B1_CURRENT_GATE_BIND.md L33). The launcher DEFAULT is
# MAX_NUM_SEQS=4, which is exactly diagnosed amplifier #3 (FR13_SPEED_MEASURE_
# INFRA.md §1): it perturbs the per-step bf16/fp8 realization and forks native
# E5 onto a DEGENERATE accept-~1.5 trajectory (measured this session: a forked
# boot gave accept 1.5 / fp 02f1e63b vs gold 3.16). So the B=1 gold regime is
# MAX_NUM_SEQS=1. B=4 co-residency NEEDS MAX_NUM_SEQS>=4, so it is a SEPARATE
# boot (boot_native_b4) -- its accept is the genuinely co-residency-degraded
# number and is labelled batch_size=4. The two MUST NOT share one boot (mixing
# is what forked native).
boot_native() {  # eN  -> B=1 gold regime (MAX_NUM_SEQS=1), reproduces banked 3.161
  local arm="native_$1" N="${1#e}"
  echo "######## boot $arm (native MTP-$N, FLASH, MAX_NUM_SEQS=1 gold regime) ########"
  recover; assert_free || return 1; empty_docker
  MAX_NUM_SEQS=1 NUM_SPECULATIVE_TOKENS="$N" \
  SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$N}" \
  CONTAINER=fr10-speed-start PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
    bash "$REPO/scripts/fr10_launch_speed_server.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  sleep 8
  if wait_health fr10-speed-start; then measure_arm_b1 "$arm" "" fr10-speed-start; fi
  teardown fr10-speed-start
}

boot_native_b4() {  # eN  -> B=4 co-residency smoke (MAX_NUM_SEQS=4)
  local arm="native_$1" N="${1#e}"
  echo "######## boot $arm B=4 (native MTP-$N, FLASH, MAX_NUM_SEQS=4 co-resident) ########"
  recover; assert_free || return 1; empty_docker
  MAX_NUM_SEQS=4 NUM_SPECULATIVE_TOKENS="$N" \
  SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$N}" \
  CONTAINER=fr10-speed-start PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
    bash "$REPO/scripts/fr10_launch_speed_server.sh" > "$OUT/${arm}_b4_boot.log" 2>&1 &
  sleep 8
  if wait_health fr10-speed-start; then measure_arm_b4 "$arm" "" fr10-speed-start; fi
  teardown fr10-speed-start
}

boot_tree() {  # name TREE  -> B=1 gold regime (MAX_NUM_SEQS=1, reproduces cat9 3.18)
  local arm="$1" tree="$2"
  echo "######## boot $arm (tree_mtp, TREE_ATTN, MAX_NUM_SEQS=1 gold regime) ########"
  recover; assert_free || return 1; empty_docker
  if [ "$arm" = "cat9" ]; then
    MAX_NUM_SEQS=1 CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_locked.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  else
    MAX_NUM_SEQS=1 TREE="$tree" CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh" > "$OUT/${arm}_boot.log" 2>&1 &
  fi
  sleep 8
  if wait_health fr13-forked-fa2-tree; then measure_arm_b1 "$arm" "$tree" fr13-forked-fa2-tree; fi
  teardown fr13-forked-fa2-tree
}

boot_tree_b4() {  # name TREE  -> B=4 co-residency smoke (MAX_NUM_SEQS=4)
  local arm="$1" tree="$2"
  echo "######## boot $arm B=4 (tree_mtp, TREE_ATTN, MAX_NUM_SEQS=4 co-resident) ########"
  recover; assert_free || return 1; empty_docker
  if [ "$arm" = "cat9" ]; then
    MAX_NUM_SEQS=4 CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_locked.sh" > "$OUT/${arm}_b4_boot.log" 2>&1 &
  else
    MAX_NUM_SEQS=4 TREE="$tree" CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 BATCH_INVARIANT=0 \
      bash "$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh" > "$OUT/${arm}_b4_boot.log" 2>&1 &
  fi
  sleep 8
  if wait_health fr13-forked-fa2-tree; then measure_arm_b4 "$arm" "$tree" fr13-forked-fa2-tree; fi
  teardown fr13-forked-fa2-tree
}

CMD="${1:-}"; shift || true
case "$CMD" in
  native)    boot_native "$1" ;;       # B=1 gold regime (MAX_NUM_SEQS=1)
  native-b4) boot_native_b4 "$1" ;;    # B=4 co-residency smoke (MAX_NUM_SEQS=4)
  tree)      boot_tree "$1" "$2" ;;     # B=1 gold regime
  tree-b4)   boot_tree_b4 "$1" "$2" ;;  # B=4 co-residency smoke
  reconcile)
    python3 "$MEASURE" reconcile --speed "$OUT"/*_speed_b1_off.json \
      --out "$OUT/reconcile.json" ;;
  *)
    echo "usage: $0 {native eN | native-b4 eN | tree NAME 'TREE' | tree-b4 NAME 'TREE' | reconcile}" >&2
    exit 2 ;;
esac
