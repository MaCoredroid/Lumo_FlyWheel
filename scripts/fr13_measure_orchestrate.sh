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

IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
ORACLE="$REPO/scripts/fr13_recurrent_decode_oracle.py"
TOKENIZER=/models/qwen3.6-27b-fp8

# GAP-1: recurrent oracle p over the FULL served stream of an arm's capture-q,
# with full top-K on EVERY position, then the real id-aligned temp06-drift TV.
# The oracle runs in-process (offline LLM, no server) in a fresh container so the
# GPU is clean. recover_host_memory + assert>=95GiB before the boot.
drift_arm() {  # arm  (uses $OUT/<arm>_q_temp06_on.json captured by native/tree)
  local arm="$1"
  local q="$OUT/${arm}_q_temp06_on.json"
  local p="$OUT/${arm}_recurrent_p.json"
  [ -f "$q" ] || { echo "[$arm] drift: missing capture-q $q (run native/tree first)"; return 1; }
  echo "######## GAP-1 drift $arm: recurrent oracle p (full-stream top-K) ########"
  recover; assert_free || return 1; empty_docker
  docker run --rm --name "fr13-recur-oracle-${arm}" --gpus all --ipc=host \
    --entrypoint bash --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO:/workspace" -v /models:/models \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    -e FR12_NO_SPECULATIVE_CONFIG=1 -e PYTHONPATH=/workspace/src -e HF_HUB_OFFLINE=1 \
    "$IMAGE" -lc "set -e; cd /workspace; python3 scripts/fr13_recurrent_decode_oracle.py rescore \
      --arm ${arm} --src /workspace/output/fr13_measure/${arm}_q_temp06_on.json \
      --out /workspace/output/fr13_measure/${arm}_recurrent_p.json \
      --seed ${SEED} --top-k 20 --threshold 1.0 --full-topk-all-positions \
      --attn-backend FLASH_ATTN --gpu-util 0.88 2>&1" \
    > "$OUT/${arm}_recurrent_p_boot.log" 2>&1
  recover
  [ -f "$p" ] || { echo "[$arm] drift: oracle p not produced (see ${arm}_recurrent_p_boot.log)"; return 1; }
  echo "===== GAP-1 temp06-drift $arm (id-aligned TV q vs p) ====="
  python3 "$MEASURE" temp06-drift --q "$q" --p "$p" --temp 0.6 \
    --per-position-floor "${PER_POS_FLOOR:-0.05}" --tokenizer "$TOKENIZER" \
    --dump-positions --out "$OUT/${arm}_temp06_drift.json"
}

# GAP-2: fork-immune paired teacher-forced accept. reference = the no-spec
# recurrent oracle GREEDY stream (deployment-correct ground truth); BOTH arms
# scored on it. The reference is produced once (native arm's recurrent p whose
# served stream IS the greedy ground truth when captured at temp 0). Pass the
# arms' capture-q artifacts; structural (CPU) reduce here, force-mode is the GPU
# ground-truth hook documented in FR13_MEASURE_INFRA_GAPS_CLOSED.md.
paired_accept() {  # reference_arm arm_q1 [arm_q2 ...]
  local refarm="$1"; shift
  local ref="$OUT/${refarm}_recurrent_p.json"
  [ -f "$ref" ] || { echo "paired: missing reference oracle $ref (run drift ${refarm} first)"; return 1; }
  local qargs=()
  for a in "$@"; do qargs+=("$OUT/${a}_q_temp06_on.json"); done
  echo "######## GAP-2 paired-accept (reference=${refarm}) arms: $* ########"
  python3 "$MEASURE" paired-accept --reference "$ref" --arm-q "${qargs[@]}" \
    --out "$OUT/paired_accept_ref_${refarm}.json"
}

# =========================================================================== #
# DEPLOYMENT REGIME (the CANONICAL path) — real SWE-Verified + codex            #
# =========================================================================== #
# fr13_measure ORCHESTRATES the proven big-denom machinery (it does NOT re-
# invent it). The chain, GPU-SERIALIZED (each stage waits + recovers):
#   1. fr13_bigdenom_swe_serve.sh <arm> <cat9|native> <subset> : boots the arm
#      (cat9 via fr13_launch_locked / native via fr10_launch_speed num_spec=5) +
#      runs scripts/run_swe_bench_q36_a.py (the codex agent loop) with the proxy
#      pair-dump ON + raw-/metrics spec-engagement asserts. run_swe_bench_q36_a
#      ALREADY brackets /metrics per task -> vllm_metrics_pre/post.txt.
#   2. cmd_deploy_speed reduces those per-task brackets -> deployment s/fwd +
#      accept/event + committed + derived TPS (SAME truthful basis, real codex
#      trajectory, no degenerate fork). [OFF, instrument clean]
#   3. fr13_bigdenom_phase3_rescore.sh : proxy pair-dump -> fr13_swe_stream_to_
#      oracle_src (byte-exact detok src) -> fr13_recurrent_decode_oracle rescore
#      (no-spec RECURRENT decode = the lossless flip) -> consolidate (Wilson CI).
#   4. cmd_deploy_lossless reads the consolidation -> within-floor lossless
#      verdict (cat9 vs native-E5, each vs its OWN no-spec oracle). [ON]
# DEV-iteration = a CHEAP deployment-faithful proxy (a SHORT codex run / few
# turns via SUBSET, NOT raw prompts). FINAL judgment = real SWE-Verified + codex,
# B=4 + CUDA-captured + 4 tasks + ~30min, lossless re-confirmed at B=4.
BIGDENOM_RUNROOT="$REPO/output/fr13_bigdenom_swe"
BIGDENOM_RESCORE="$REPO/output/fr13_bigdenom_rescore"
DEPLOY_OUT="$OUT/deployment"

# deploy_serve <arm_dir> <cat9|native> <subset.json> : ONE deployment SWE arm
# (boots + codex loop + pair-dump + per-task /metrics brackets). The serialized
# GPU stage; serve.sh owns its own pre-boot hygiene + teardown + recover.
deploy_serve() {  # arm_dir kind subset
  local armdir="$1" kind="$2" subset="${3:-subset_astropy12907.json}"
  echo "######## DEPLOY serve $armdir (kind=$kind subset=$subset) ########"
  bash "$REPO/scripts/fr13_bigdenom_swe_serve.sh" "$armdir" "$kind" "$subset"
}

# deploy_speed <arm_label> <arm_dir> <expected_tok_per_draft> [batch_size]
# Reduce the per-task /metrics brackets from a finished deployment arm -> the
# CANONICAL deployment s/fwd + accept/event. CPU-only (no GPU).
deploy_speed() {  # arm_label arm_dir expected_ratio [batch]
  local label="$1" armdir="$2" ratio="$3" batch="${4:-1}"
  mkdir -p "$DEPLOY_OUT"
  echo "===== DEPLOY speed $label (B=$batch, real codex trajectory) ====="
  python3 "$MEASURE" deploy-speed --arm "$label" \
    --out-root "$BIGDENOM_RUNROOT/$armdir/swe_out" \
    --expected-tok-per-draft "$ratio" --batch-size "$batch" \
    --out "$DEPLOY_OUT/${label}_deploy_speed_b${batch}.json"
}

# deploy_rescore : run the big-denom Phase-3 rescore + consolidate (GPU), then
# the CANONICAL deployment lossless verdict. Assumes both arms' deployment serve
# runs finished (cat9_a + native_a pair-dumps present). GPU-SERIALIZED.
deploy_rescore() {
  mkdir -p "$DEPLOY_OUT"
  echo "######## DEPLOY rescore (big-denom Phase-3: oracle flips + Wilson CI) ########"
  bash "$REPO/scripts/fr13_bigdenom_phase3_rescore.sh"
  echo "===== DEPLOY lossless verdict (within-floor) ====="
  python3 "$MEASURE" deploy-lossless \
    --consolidated "$BIGDENOM_RESCORE/consolidated.json" \
    --out "$DEPLOY_OUT/deploy_lossless.json"
}

# deploy_full : the FINAL-judgment chain end-to-end (serialized). cat9 + native
# serve arms, deploy-speed each, then rescore+lossless. For DEV iteration use a
# SHORT subset; for FINAL judgment use the 4-task SWE-Verified subset at B=4.
deploy_full() {  # subset [batch]
  local subset="${1:-subset_astropy12907.json}" batch="${2:-1}"
  deploy_serve native_a native "$subset"
  deploy_speed native_e5 native_a 5 "$batch"
  deploy_serve cat9_a cat9 "$subset"
  deploy_speed cat9 cat9_a 9 "$batch"
  deploy_rescore
  echo "######## DEPLOY full chain done -> $DEPLOY_OUT ########"
}

CMD="${1:-}"; shift || true
case "$CMD" in
  # --- CANONICAL: deployment regime (real SWE-Verified + codex) ---
  deploy-serve)    deploy_serve "$1" "$2" "${3:-}" ;;     # ONE arm: boot+codex+pairdump+brackets
  deploy-speed)    deploy_speed "$1" "$2" "$3" "${4:-1}" ;; # reduce /metrics brackets -> s/fwd+accept (CPU)
  deploy-rescore)  deploy_rescore ;;                        # big-denom Phase-3 + lossless verdict (GPU)
  deploy-full)     deploy_full "${1:-}" "${2:-1}" ;;        # end-to-end serialized chain
  # --- DEPRECATED: raw-/v1/completions regime (off-distribution, cautionary) ---
  native)    boot_native "$1" ;;       # B=1 gold regime (MAX_NUM_SEQS=1)
  native-b4) boot_native_b4 "$1" ;;    # B=4 co-residency smoke (MAX_NUM_SEQS=4)
  tree)      boot_tree "$1" "$2" ;;     # B=1 gold regime
  tree-b4)   boot_tree_b4 "$1" "$2" ;;  # B=4 co-residency smoke
  drift)     drift_arm "$1" ;;          # GAP-1: oracle p (full-stream) + id-aligned TV
  paired)    paired_accept "$@" ;;      # GAP-2: paired teacher-forced accept (CPU structural)
  reconcile)
    python3 "$MEASURE" reconcile --speed "$OUT"/*_speed_b1_off.json \
      --out "$OUT/reconcile.json" ;;
  *)
    echo "usage: $0 <command>" >&2
    echo "  CANONICAL (deployment = real SWE-Verified + codex):" >&2
    echo "    deploy-serve ARM_DIR {cat9|native} [SUBSET]   boot+codex+pairdump+brackets (GPU)" >&2
    echo "    deploy-speed LABEL ARM_DIR EXPECT_RATIO [B]    reduce /metrics brackets -> s/fwd+accept (CPU)" >&2
    echo "    deploy-rescore                                 big-denom Phase-3 + lossless verdict (GPU)" >&2
    echo "    deploy-full [SUBSET] [B]                        end-to-end serialized chain" >&2
    echo "  DEPRECATED (raw-/v1/completions, off-distribution cautionary):" >&2
    echo "    native eN | native-b4 eN | tree NAME 'TREE' | tree-b4 NAME 'TREE'" >&2
    echo "    drift ARM | paired REF_ARM ARM [ARM...] | reconcile" >&2
    exit 2 ;;
esac
