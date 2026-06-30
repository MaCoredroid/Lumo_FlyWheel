#!/usr/bin/env bash
# FR13 NO-SPEC + cache SWE solve-rate arm — the DISCRIMINATOR runner.
# =================================================================================
# BINDING QUESTION: "does no-spec + cache SOLVE astropy-12907?"  This isolates
# whether the cache-ON solve-rate carrier is:
#   (A) SPEC-specific  = the drafter/committer x cache interaction (tree node-bank
#       staleness, captured-graph row indexing under the spec path), OR
#   (B) COMMON         = recurrent-decode vs chunked-prefill GDN realization at the
#       cache block boundaries (kernel differs ~0.0078 >> 1 bf16 ULP), independent
#       of spec.  This is the cache-ON-vs-OFF gap the launcher comments root-cause.
# If no-spec + cache-ON FAILS (and no-spec + cache-OFF resolves), the carrier is
# COMMON (B): the cached chunked checkpoint poisons a plain recurrent decode too.
# If no-spec + cache-ON RESOLVES, the carrier is SPEC-specific (A).
#
# This runner is the NO-SPEC analog of the spec arms in scripts/fr13_bigN_solve_spine_tree.sh
# (tree=cat6root cache-ON, spine=chain5 cache-ON, cacheoff=cache-OFF). It reuses the EXACT
# SWE orchestrator + verdict-extraction + proxy scaffolding; the ONLY change vs those arms
# is the BOOT: base (non-forked) vLLM with NO --speculative-config instead of the forked-spec
# launcher.
#
# BOOT = scripts/fr10_launch_speed_server.sh (the SAME base launcher the no-spec "oracle" arm
# in scripts/fr13_apc_multiturn_one_arm.sh uses) with FR12_NO_SPECULATIVE_CONFIG=1, which
# makes that launcher OMIT --speculative-config entirely (fr10_launch_speed_server.sh:300-302:
# `if [[ "$FR12_NO_SPECULATIVE_CONFIG" != "1" ]]; then SPEC_ARGS=(--speculative-config ...)`).
# NO drafter, NO tree fork, NO MTP.  Same model /models/qwen3.6-27b-fp8 as all other arms.
#
# Two arms per repeat (the cache A/B WITHIN no-spec):
#   nospecon  = no-spec + cache ON  (FR13_ENABLE_APC=1, matched to tree/spine arms)
#   nospecoff = no-spec + cache OFF (FR13_ENABLE_APC=0, the within-no-spec control)
#
# Usage:  N=3 bash scripts/fr13_nospec_cache_swe.sh
#   N (env, default 3)         repeats per arm (temp-0.6 agentic solve is flaky -> RATE)
#   ONLY (env, optional)       "on" or "off" to run a single arm
#   OFFLOAD_CODEX (env, def 0) 1 = codex/proxy on alienware (matches bigN default 0 = on-GB10)
#
# NO GPU run is performed by writing this script; it is a RUNNER (boots vLLM when invoked).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

N=${N:-3}
ONLY=${ONLY:-}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_nospec_cache/run_$TS
mkdir -p "$ROOT"
export RUNROOT="$ROOT"

# ---- deployment regime: MATCH the tree/spine arms (fr13_bigN_solve_spine_tree.sh) ----
# Solve-rate test (not a speed test) -> B=1, single concurrent codex task.
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_CODEX=${OFFLOAD_CODEX:-0} \
       DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# constrained decoding PAUSED -> LUMO_PROXY_TOOL_STRICT unset (no grammar), same as bigN.

echo "NO-SPEC cache discriminator: nospecon(cache-ON) vs nospecoff(cache-OFF), astropy-12907, N=$N -> $ROOT" \
  | tee "$ROOT/RESULTS.txt"
echo "  boot=fr10_launch_speed_server.sh FR12_NO_SPECULATIVE_CONFIG=1 (NO --speculative-config)" \
  | tee -a "$ROOT/RESULTS.txt"

run_one() {  # arm i   (arm in {nospecon, nospecoff})
  local arm=$1 i=$2 A="${1}_r${2}"
  PYTHONPATH="$PWD/src" .venv/bin/python -c \
    "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" \
    >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 \
    && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] arm=$arm APC=${FR13_ENABLE_APC:-0} $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  ARM="$A" bash scripts/fr13_nospec_serve_one.sh "$A" "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local v rr
  # SAME extraction as fr13_bigN_solve_spine_tree.sh:39-41 (apples-to-apples).
  v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  rr=$(grep -oE "resolved_rate=[0-9.]+" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v $rr" | tee -a "$ROOT/RESULTS.txt"
}

# INTERLEAVED nospecon (cache-ON) <-> nospecoff (cache-OFF) per i, so the binding
# rate(nospecon) vs rate(nospecoff) builds PAIRWISE. The give-up flake floors both;
# the cache signal is the DIFFERENCE.
for i in $(seq 1 "$N"); do
  if [[ -z "$ONLY" || "$ONLY" == "on" ]]; then
    export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
    run_one nospecon "$i"
    unset FR13_ENABLE_APC MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE CUDAGRAPH_MODE 2>/dev/null || true
  fi
  if [[ -z "$ONLY" || "$ONLY" == "off" ]]; then
    # cache OFF = FR13_ENABLE_APC=0. The base launcher then emits a serve command with
    # NO --enable-prefix-caching / --enable-chunked-prefill / --mamba-* / cudagraph_mode.
    export FR13_ENABLE_APC=0
    unset MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE CUDAGRAPH_MODE 2>/dev/null || true
    run_one nospecoff "$i"
    unset FR13_ENABLE_APC 2>/dev/null || true
  fi
done

echo "=== SOLVE RATES (no-spec) ===" | tee -a "$ROOT/RESULTS.txt"
for arm in nospecon nospecoff; do
  res=$(grep -cE "^${arm}_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
  tot=$(grep -cE "^${arm}_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
  echo "$arm: ${res}/${tot} resolved" | tee -a "$ROOT/RESULTS.txt"
done
echo "=== DONE -> $ROOT/RESULTS.txt ==="
