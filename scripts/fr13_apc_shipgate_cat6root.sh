#!/usr/bin/env bash
# FR13 APC 4-TASK SHIP GATE — cat6root arm, cache-OFF vs cache-ON (both APC fixes baked)
# =================================================================================
# WRITE-ONLY DRIVER (validated with `bash -n`). DO NOT RUN here. The orchestrator
# runs the GPU stage GPU-SOLO / serialized on GB10 (never a third concurrent
# workflow). This driver runs the TWO arms SEQUENTIALLY (one vLLM boot each,
# recover_host_memory between) and leaves the offline lossless rescore + the
# consolidation to a SEPARATE GPU phase (documented at the bottom; NOT run here).
#
# WHAT IT IS
# ----------
# The ship gate for enabling vLLM Automatic Prefix Caching (APC) on the
# Qwen3-Next-27B fp8 GDN-hybrid + cat6root tree-spec-decode (FR13) deployment.
# Runs the 4 SWE-Verified tasks (astropy 12907/13033/13236/13398) at temp 0.6,
# B=1, on the cat6root tree shape, comparing:
#   ARM 1  cache-OFF  (no APC: FR13_ENABLE_APC unset)              -> the bar
#   ARM 2  cache-ON   (APC, both fixes baked: max_num_batched=block_size
#                      + FR13_APC_HIT_RECURRENT_SUFFIX=1)
# capturing the FOUR ship metrics (see SHIP METRICS below).
#
# Both arms reuse the canonical x86-offload SWE harness
# (scripts/fr13_bigdenom_swe_serve_variant.sh KIND=cat6root): vLLM ALONE on the
# GB10, the SWE agent (qwen-code-runner:v1) + inference_proxy OFFLOADED to
# alienware, pair-dumps rsynced back. This is the SANCTIONED offload path (the
# agent container on alienware), NOT the legacy on-GB10 8022 path / OOM-guard. METRICS OFF on both arms
# (FR10_METRICS=0, BATCH_INVARIANT=0, no FR13_*_DIAG / STATE_PROBE / argmax-gate /
# SFWD_GPU_TIMER) so the deploy-speed read is the clean /metrics decode bracket.
#
# THE ONLY DIFFERENCE between the two arms is the APC env (FR13_ENABLE_APC + the
# baked sub-fixes). The cat6root TREE shape, the locked cat9 pipeline pins, the
# temp/seed, the subset, and the offload regime are IDENTICAL -> apples-to-apples.
#
# APC PLUMBING (verified)
# -----------------------
# serve_variant's forked-arm boot (KIND=cat6root) calls
# scripts/fr13_launch_forked_fa2_tree_server.sh, which reads FR13_ENABLE_APC /
# MAMBA_BLOCK_SIZE / MAMBA_SSM_CACHE_DTYPE / APC_MAX_NUM_BATCHED_TOKENS /
# FR13_APC_HIT_RECURRENT_SUFFIX from its SHELL ENVIRONMENT (host-side; the
# APC_FLAGS string is built host-side and appended to the vllm serve command, and
# the FR13_APC_* sub-fix flags are exported into the container -e). So EXPORTING
# them in this driver before invoking serve_variant is sufficient and complete.
#   * cache-OFF arm: FR13_ENABLE_APC unset -> APC_FLAGS empty -> serve command is
#     BYTE-IDENTICAL to the locked cat6root (non-APC) path. The launcher's
#     `${FR13_ENABLE_APC:-0}` guard makes the non-APC arm provably unaffected.
#   * cache-ON arm: FR13_ENABLE_APC=1, MAMBA_BLOCK_SIZE=1024 (=>
#     APC_MAX_NUM_BATCHED_TOKENS defaults to 1024 = the #45238 fix),
#     MAMBA_SSM_CACHE_DTYPE=float32, FR13_APC_HIT_RECURRENT_SUFFIX=1 (the
#     first-suffix-chunk recurrent recompute closing the cache-ON residual). The
#     other baked APC sub-fixes (FR13_APC_CONV_FIX / CONV_SNAPSHOT / SSM_SNAPSHOT /
#     SSM_WRITE_THROUGH / BLOCK_ALIGN_45477) default-ON inside the launcher's
#     APC block; FR13_APC_DROP_FINAL_BLOCK stays default-0 (REFUTED carrier).
#
# SHIP METRICS (the 4 axes; see bottom for the offline rescore/consolidate)
# ------------------------------------------------------------------------
#   1. decode-TPS / s-per-fwd  : CLEAN from GB10 /metrics (decode bracket, local
#      on the GB10 — never across-wire). cache-OFF target ~18.51 TPS / 0.138 s/fwd.
#   2. TTFT / task-wall        : APC prefill win (cache-ON TTFT << cache-OFF).
#   3. lossless                : SWE pair-dumps -> oracle src -> recurrent rescore
#      -> clear-margin flip-rate per arm; GATE cache-ON clear-margin <= cache-OFF.
#   4. coding-quality          : x86 SWE-Verified pass/fail per task per arm;
#      GATE cache-ON matches cache-OFF.
#
# SERIALIZATION / GPU-SOLO
# ------------------------
# The two arms boot ONE vLLM at a time on PORT 9950. serve_variant does the
# pre-boot hygiene (recover_host_memory + assert MemAvailable>=95GiB + swap=0 +
# docker-empty) and a clean teardown (docker rm -f + recover) on EXIT, so this
# driver just runs them back-to-back with an extra recover between. During each
# agent run the GB10 is vLLM-ONLY (agent+proxy on alienware).
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

# ---- pinned inputs ----
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_four.json}   # 12907/13033/13236/13398
KIND=cat6root
EXPECT_TOK_PER_DRAFT=6                                          # cat6root EXPECT_RATIO=6
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)                 # UTC-millisecond run key (no cross-run clobber)
RUNROOT=output/fr13_apc_shipgate/$TS             # per-run subfolder; serve_variant honors RUNROOT (ARMDIR=$RUNROOT/$ARM)
export RUNROOT
GATEDIR=$RUNROOT
mkdir -p "$GATEDIR"
echo "[run-key] artifacts -> $RUNROOT (UTC-ms keyed; cache-ON runs FIRST)"

# ---- arm names (serve_variant writes to $RUNROOT/<arm>/) ----
ARM_OFF=${ARM_OFF:-cat6root_apcoff_b1}
ARM_ON=${ARM_ON:-cat6root_apcon_b1}

# ---- B=1, temp 0.6 deployment regime (serve_variant defaults; forced temp 0.6 is
#      pinned by the offload proxy DEPLOY_FORCE_TEMP=0.6 = the real deployment temp).
export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}   # B=1
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}      # single-stream, no co-residency confound
export OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}          # agent+proxy on alienware (SANCTIONED), GB10 vLLM-only
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
# seed: the proxy/orchestrator pin SEED 1313 is the canonical deployment seed; the
# offload proxy forces temp 0.6. (run_swe_bench_q36_a temperature is governed by the
# proxy bundle; seed flows through the codex/q36-a request path.)
export SEED=${SEED:-1313}
# AGENT_WALL_S: leave UNSET for a full ship run (real codex task wall). Set e.g.
# AGENT_WALL_S=600 only for a DEV screen.
# AGENT_WALL_S=${AGENT_WALL_S:-}

# ---- METRICS OFF on BOTH arms (clean deploy-speed). These are serve_variant /
#      launcher defaults; we pin them explicitly + assert no diag flag is set. ----
export FR10_METRICS=0
export BATCH_INVARIANT=0
# Hard-assert no diagnostic / state-probe / argmax-gate / sfwd-timer flag is live
# in THIS shell (any would contaminate the clean speed read or add eager syncs).
_DIAG_FLAGS=(FR13_SFWD_GPU_TIMER FR13_COMMIT_ARGMAX_GATE FR13_FORK_MARGIN_DUMP \
  FR13_CHASE_DIAG FR13_FIX1_SELFCHECK \
  FR13_REPLAY_BOUNDARY_LOG FR13_GDN_SUBOP_MAB FR13_FA2_MAB FR13_REPLAY_DURABLE_AB \
  FR13_TCF_SELFCHECK)
for f in "${_DIAG_FLAGS[@]}"; do
  v="${!f:-0}"
  if [[ "$v" != "0" && -n "$v" ]]; then
    echo "FAIL: diagnostic flag $f=$v is set — metrics-OFF ship gate requires it OFF"; exit 2
  fi
done
echo "[metrics-off] all diag/state-probe/argmax-gate/sfwd-timer flags OFF — clean speed read"

echo "=== FR13 APC SHIP GATE (cat6root) ts=$TS subset=$SUBSET ==="
echo "    ARM_OFF=$ARM_OFF (cache-OFF / no APC)   ARM_ON=$ARM_ON (cache-ON / APC fixes baked)"
git rev-parse HEAD 2>/dev/null | tee "$GATEDIR/git_head_$TS.txt" || true

# =================================================================================
# ARM 1 — cache-ON (APC, BOTH fixes baked). RUN FIRST so any regression shows EARLY.
#   max_num_batched=block_size (#45238 fix; launcher default when APC_MAX_NUM_BATCHED
#     unset + MAMBA_BLOCK_SIZE=1024) + FR13_APC_HIT_RECURRENT_SUFFIX=1 (residual fix).
# =================================================================================
echo
echo "########## ARM 1: cache-ON (APC fixes baked) — run FIRST to surface issues early ##########"
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
# The first-suffix-chunk recurrent recompute residual fix. The launcher defaults
# this OFF (:=0) so the locked cat9/cat6root non-APC serve stays byte-identical;
# the shipgate must EXPLICITLY arm it for the cache-ON arm (ported into the live
# patcher 2026-06-26 e67eaf39; before that this export was inert). CAP unset=>64.
export FR13_APC_HIT_RECURRENT_SUFFIX=1
export FR13_APC_HIT_SUFFIX_CAP=${FR13_APC_HIT_SUFFIX_CAP:-64}
# CUDA-graph carrier fix (re-rooted 2026-06-21, launcher L273-283): default FULL graphs
# read GDN recurrent state via capture-time-baked indexing, so after an APC cache-hit
# re-prefill the captured graph reads the WRONG block-pool row -> garbage/runaway tool
# calls (#43559). PIECEWISE keeps graph capture for dense GEMMs/norms/MLP (decode TPS
# preserved) but runs the GDN/mamba scan EAGER so it reads the live restored row. This
# is the coding-quality carrier; the recompute above is the per-token lossless residual.
export CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-PIECEWISE}
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true   # launcher defaults to block_size = the #45238 fix

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_ON" "$KIND" "$SUBSET"
ON_RC=$?
echo "ARM_ON rc=$ON_RC (armdir=$RUNROOT/$ARM_ON)"

# recover between arms (serve_variant teardown already recovers; belt-and-suspenders).
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
sleep 3

# =================================================================================
# ARM 2 — cache-OFF (no APC / baseline). FR13_ENABLE_APC UNSET => byte-identical locked cat6root.
# =================================================================================
echo
echo "########## ARM 2: cache-OFF (no APC / baseline) ##########"
# Defensive: ensure no APC env leaks into the cache-OFF boot.
unset FR13_ENABLE_APC MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE APC_MAX_NUM_BATCHED_TOKENS \
      FR13_APC_CONV_FIX FR13_APC_CONV_SNAPSHOT \
      FR13_APC_HIT_RECURRENT_SUFFIX FR13_APC_HIT_SUFFIX_CAP CUDAGRAPH_MODE \
      FR13_APC_BLOCK_ALIGN_45477 2>/dev/null || true

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_OFF" "$KIND" "$SUBSET"
OFF_RC=$?
echo "ARM_OFF rc=$OFF_RC (armdir=$RUNROOT/$ARM_OFF)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

# =================================================================================
# SHIP-METRIC SUMMARY (the bits computable on-host, post-run; lossless rescore is
# a SEPARATE GPU phase below). Writes a single per-arm summary json.
# =================================================================================
echo
echo "=== SHIP-METRIC SUMMARY (per arm) ==="
# METRIC 1 (decode-TPS / s-per-fwd) via fr13_measure.py over the per-task
# /metrics brackets (--expected-tok-per-draft 6 for cat6root). Run for each arm.
for ARM in "$ARM_OFF" "$ARM_ON"; do
  ARMDIR="$RUNROOT/$ARM"
  echo "--- $ARM ---"
  # METRIC 1: CANONICAL deployment s/fwd + accept/event + derived TPS from the
  # PER-TASK /metrics brackets (vllm_metrics_pre/post.txt) run_swe_bench_q36_a
  # writes under swe_out/*/per_task/*/. Basis = decode_seconds (raw-counter delta,
  # local on GB10, never across-wire). cmd_deploy_speed asserts class-9 engagement
  # (expected-tok-per-draft=6 for cat6root) so a vacuous arm fails loud.
  if [[ -d "$ARMDIR/swe_out" ]]; then
    .venv/bin/python scripts/fr13_measure.py deploy-speed \
      --arm "$ARM" --out-root "$ARMDIR/swe_out" \
      --expected-tok-per-draft "$EXPECT_TOK_PER_DRAFT" \
      --batch-size 1 --basis decode_seconds \
      --out "$GATEDIR/${ARM}_speed.json" \
      > "$GATEDIR/${ARM}_speed.log" 2>&1 \
      && cat "$GATEDIR/${ARM}_speed.json" \
      || { echo "  WARN: fr13_measure.py deploy-speed failed (see ${ARM}_speed.log)."; \
           echo "  Manual fallback: s/fwd = d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)"; \
           echo "  over per-task vllm_metrics_pre/post.txt (or the overall metrics_before_swe/after_swe brackets)."; }
  else
    echo "  WARN: $ARM missing swe_out/ (no per-task brackets to reduce)"
  fi
  # METRIC 2: task-wall (proxy for TTFT win) from serve_variant's health.json +
  # swe_window_wall_s. The true APC prefill TTFT win is read from the proxy
  # per-request rows (offload_request_metrics.jsonl) — first-token latency per turn.
  [[ -f "$ARMDIR/health.json" ]] && { echo "  health/wall:"; cat "$ARMDIR/health.json"; }
  # METRIC 4: coding-quality verdicts per task (x86 SWE-Verified).
  echo "  verdicts:"
  for m in "$ARMDIR"/swe_out/*/per_task/*/runner_metadata.json; do
    [[ -f "$m" ]] || continue
    .venv/bin/python - "$m" <<'PY'
import json, sys
m = json.loads(open(sys.argv[1]).read())
er = (m.get("eval_report") or {})
print(f"    {m.get('instance_id')}: {er.get('verdict','missing')}")
PY
  done
  echo "  pair_dumps: $(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)"
done

echo
echo "=== SHIP GATE arms complete (off_rc=$OFF_RC on_rc=$ON_RC) ==="
echo
# =================================================================================
# OFFLINE LOSSLESS RESCORE -> METRIC 3 (SEPARATE GPU boots; run AFTER this driver,
# GPU-solo). The flip reference is the NO-SPEC RECURRENT decode oracle
# (deployment-correct), NOT streamed logprobs / backend-name. APPLES-TO-APPLES:
# cat6root has its own inherent spec-decode flip rate; APC is lossless iff cache-ON
# does NOT push the clear-margin flip-rate above cache-OFF (the real LARGE
# denominator = all 4 tasks' pair-dumps, not the 1-prompt precheck).
#
#   # per arm: pair-dumps -> oracle src (CPU, in container) -> recurrent rescore (GPU)
#   for ARM in cat6root_apcoff_b1 cat6root_apcon_b1; do
#     DUMP="output/fr13_bigdenom_swe/$ARM/proxy_pair_dumps"
#     SRC="output/fr13_apc_shipgate_cat6root/${ARM}_src.json"
#     docker run --rm -v "$PWD:/workspace" -v /models:/models -e PYTHONPATH=/workspace/src \
#       --entrypoint bash \
#       vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
#       -lc "cd /workspace; python3 scripts/fr13_swe_stream_to_oracle_src.py \
#              --dump-dir /workspace/${DUMP} --out /workspace/${SRC}"
#     scripts/fr13_recur_rescore_in_container.sh "$ARM" "$SRC" \
#       "output/fr13_apc_shipgate_cat6root/${ARM}_rescore.json"
#   done
#
#   # consolidate -> clear-margin flip rate + Wilson-95 CI per arm. The consolidator
#   # takes a --cat9-* / --native-* PAIR; here pass cache-ON as --cat9-* and cache-OFF
#   # as --native-* so it reports CI separation (PASS = cache-ON NOT above cache-OFF,
#   # i.e. ci_separated_cat9_above_native == false AND cat9.wilson95_hi <=
#   # cache-OFF.wilson95_hi):
#   scripts/fr13_bigdenom_rescore_consolidate.py \
#     --cat9-rescore   output/fr13_apc_shipgate_cat6root/cat6root_apcon_b1_rescore.json \
#     --cat9-src       output/fr13_apc_shipgate_cat6root/cat6root_apcon_b1_src.json \
#     --native-rescore output/fr13_apc_shipgate_cat6root/cat6root_apcoff_b1_rescore.json \
#     --native-src     output/fr13_apc_shipgate_cat6root/cat6root_apcoff_b1_src.json \
#     --out            output/fr13_apc_shipgate_cat6root/lossless_consolidated.json
#
#   # LOSSLESS GATE (apples-to-apples, large denominator):
#   #   cache-ON clear_margin_rate (and wilson95_hi) <= cache-OFF
#   #   AND cache-ON within the absolute E5 floor 12.90% (native E5 b1 = 630/4883).
# =================================================================================
echo "NEXT (offline, SEPARATE GPU boots — DO NOT run in this driver):"
echo "  1) per arm: fr13_swe_stream_to_oracle_src.py (pair-dumps -> src) then"
echo "     fr13_recur_rescore_in_container.sh (recurrent-oracle rescore)"
echo "  2) fr13_bigdenom_rescore_consolidate.py (cache-ON as --cat9-*, cache-OFF as --native-*)"
echo "  3) LOSSLESS PASS iff cache-ON clear-margin <= cache-OFF AND within 12.90% E5 floor"
echo "  (the cmd templates are in the comment block above)"

OVERALL=0
[[ "$OFF_RC" == "0" && "$ON_RC" == "0" ]] || OVERALL=1
echo "DRIVER_DONE overall=$OVERALL (off_rc=$OFF_RC on_rc=$ON_RC); lossless METRIC 3 pending offline rescore"
exit $OVERALL
