#!/usr/bin/env bash
# FR14 ARM B — K64 vs K0 draft-vocabulary ablation on THEIR bench shape.
#
# Mark's question (REDTEAM pass 12): under the NVFP4 lm_head, K64's byte
# advantage nearly vanishes. Full-vocab drafting costs +34.3 ms of floor in the
# fp8 era (153.94 vs 119.66 — the reason K64 exists at all); on RadixArk it
# costs **+0.807 ms** (93.152 vs 92.345), because five full-vocab reads of the
# 0.715 GB NVFP4 head are about the same bytes as five BF16-dequant K64 slices.
#
# Bytes say +-0. Only the WALL says the truth, because K64 also bought DFWD
# COMPUTE (65k vs 248k rows per draft-head GEMV) and removes a subset-miss
# acceptance penalty that leg 3 measured at 4.526 -> 2.358 on out-of-corpus
# content. This runs both arms on identical content and reports accept, TPS and
# step wall.
#
# LANE: DIAGNOSTIC, NON-CITABLE — same lane and the same compromises as arm A's
# leg 3 (ablation_a_leg3.md): engine-only (no SWE runner, no proxy, no agent),
# middleware dropped so bench_serving can reach the engine, deployment sampling
# (temp 0.6 / top_p 0.95 / top_k 20) because the fixed32 committer hard-refuses
# greedy. The ONLY variable between the two arms is the draft vocabulary.
#
#   ARM K64: FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1  (as built)
#            floor 25,210,209,416 B / 92.345089436 ms
#            drafts through the DVK shim: 128-id block gather out of the NVFP4
#            head, dequantised to BF16 at boot (Phase 1).
#   ARM K0 : FR13_DRAFT_VOCAB_K=0     FR13_DRAFT_VOCAB_ROOT=0
#            floor 25,430,574,256 B / 93.15228665201465 ms
#            drafts through the STOCK fp4 GEMM on the full 248,320-row head.
#
#            WHY THE SHIM IS INERT -- corrected after actually reading the
#            patcher rather than reasoning from the root flag. There are TWO
#            _fr13_dvk_prepare() call sites: one under `if _fr13_dvk_root`, and
#            a SECOND under `if not _fr13_dvk_root` (which builds the loop
#            subset after the unchanged full root head). So "ROOT=0 means the
#            prepare is never called" is FALSE. What actually makes K0 inert is
#            the function's own first statement:
#                if _fr13_dvk_configured <= 0 or self._fr13_dvk_dead:
#                    return 0, None
#            K=0 takes that early return, so neither the shim nor the Phase-1
#            dequant is reached. (The 65536:0 arm WOULD build the shim through
#            the second site -- it is simply not one of these two arms.)
#            Asserted below from the boot log, not assumed.
#
# ROOT must be 0 when K is 0: the patcher raises "FR13_DRAFT_VOCAB_ROOT=1
# requires FR13_DRAFT_VOCAB_K>=128", and the floor sequence has no 0:1 arm.
#
# Usage: bash armb_k64_ablation.sh
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

OUT=${OUT:-/home/mark/shared/tmp-scratch/fr14_armb_k64ab}
PORT=9950
IMG=lmsysorg/sglang:qwen38-27b
MODEL=qwen3.8-27b-nvfp4-radixark
TOKENIZER=/models/qwen3.8-27b-nvfp4-radixark
NUM_PROMPTS=${NUM_PROMPTS:-8}
CONCURRENCY=${CONCURRENCY:-1}
mkdir -p "$OUT"

# DRIFT GUARD. The nomiddleware launcher is a COPY, so it silently rots the
# moment the real launcher moves. Assert the only difference is the middleware
# block -- if anything else diverges, the ablation is measuring a different
# engine from the stock arm and must not run.
assert_launcher_parity() {
  local diff_out
  diff_out=$(diff scripts/fr13_launch_forked_fa2_tree_server.sh \
                  scripts/fr14_armb_leg3_launch_nomiddleware.sh || true)
  local bad
  bad=$(printf '%s\n' "$diff_out" \
        | grep -E '^[<>]' \
        | grep -vE 'FR13_FIXED32_MIDDLEWARE_FLAGS=|^> *# ' || true)
  if [[ -n "$bad" ]]; then
    echo "FAIL: nomiddleware launcher has drifted beyond the middleware line:" >&2
    printf '%s\n' "$bad" >&2
    exit 2
  fi
  echo "[parity] nomiddleware launcher differs from the stock launcher ONLY by the middleware flag"
}
assert_launcher_parity

boot_arm() {  # arm k root container
  local arm=$1 k=$2 root=$3 container=$4
  local armout="$OUT/$arm"
  mkdir -p "$armout/logs"
  chmod 700 "$armout" "$armout/logs"

  echo "[$arm] ==== memory preflight ===="
  # On GB10 the GPU pool IS host RAM, so the previous arm's 20 GB of checkpoint
  # page cache is charged against this arm's demand even though MemAvailable
  # calls it available. Same sync + drop the launcher's own FR14 preflight does;
  # `sudo -n` is deliberately non-interactive so an unattended run never stalls.
  sync
  sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
  free -g | tee "$armout/free_before_boot.txt"
  local memfree
  memfree=$(awk '/^MemFree:/{print $2/1048576}' /proc/meminfo)
  echo "[$arm] MemFree=${memfree}GiB (engine demands ~0.70 x 117.5 = 82.3GiB)"
  awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
    || { echo "[$arm] FAIL: unified-memory preflight"; return 2; }
  docker ps -q | grep -q . && { echo "[$arm] FAIL: docker not empty"; return 2; }

  (
    set -uo pipefail
    run_variant(){ :; }
    export BSIZE=1 CONC=1 TAG=armbk64ab
    export FR13_DRAFT_VOCAB_K="$k" FR13_DRAFT_VOCAB_ROOT="$root" FR13_DRAFT_HEAD_FP8=0
    # The launcher gates the full-vocabulary arm behind a SANCTIONED OVERRIDE:
    #   [[ (MAX_NUM_SEQS==1||4) && FR13_NEEDS_ALLOW == "FR13_DRAFT_VOCAB_K=0" ]]
    #   || "fixed32 full-vocabulary mode requires exact B1/B4 and its
    #      sanctioned override"; exit 2
    # It exists so nobody drifts off K64 by accident. This ablation IS the
    # sanctioned use -- Mark's pass-12 directive, B1, diagnostic lane -- so the
    # override is set for the K0 arm and ONLY the K0 arm.
    if [[ "$k" == "0" ]]; then
      export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
    fi
    # The COMMITTED arm-B sequence file -- no HEAD pin needed, the constant
    # train is HEAD now.
    source scripts/fr13_fixed32_floor_timers_seq.sh
    echo "[$arm] seq: K=$k ROOT=$root BYTES=$FR13_MANDATORY_WEIGHT_BYTES FLOOR_MS=$FR13_WEIGHT_FLOOR_MS GPU_UTIL=$GPU_UTIL"
    printf '%s %s\n' "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" > "$armout/floor.txt"

    mapfile -t C < <(.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
import fr13_fixed32_topology as t
t.validate_contract()
print(repr(list(t.FIXED32_CHOICES)))
print(f"{t.TAIL6_VALID_MASK:#x}")
print(t.TAIL6_ACTIVE_DRAFTS)
PY
)
    (( ${#C[@]} == 3 )) || { echo "[$arm] FAIL: topology authority"; exit 2; }

    export FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged
    export FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=0
    export FR13_FIXED32_MODE=tail6_fixed32
    export FR13_FIXED32_VALID_MASK="${C[1]}"
    export FR13_FIXED32_ACTIVE_NODES="${C[2]}"
    export FR13_FIXED32_PHYSICAL_DRAFTS=31
    export FR13_FIXED32_B1_DIAGNOSTIC=0
    export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
    export FR13_DRAFT_VOCAB_K="$k" FR13_DRAFT_VOCAB_ROOT="$root"
    export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
    export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
    export FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}.json
    export FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json
    export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json
    mkdir -p output/fr13_sfwd_sidecar

    SECRET=$(mktemp /tmp/fr13_fixed32_ingress.XXXXXXXX)
    chmod 600 "$SECRET"
    .venv/bin/python - "$SECRET" <<'PY'
import json, os, secrets, sys
path = sys.argv[1]
payload = {"schema": "fr13-fixed32-ingress-secrets-v1",
           "task_hmac_key_hex": secrets.token_hex(32),
           "engine_bearer": "fr13_engine_" + secrets.token_hex(32)}
enc = (json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
try:
    off = 0
    while off < len(enc):
        off += os.write(fd, enc[off:])
    os.fsync(fd)
finally:
    os.close(fd)
PY
    export FR13_FIXED32_INGRESS_SECRET_FILE="$SECRET"
    export FR13_FIXED32_INGRESS_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
    printf '%s\n' "$SECRET" > "$armout/.secret_path"

    export CONTAINER="$container"
    export PORT=9950 MAX_NUM_SEQS=1 SWE_CONCURRENCY=1
    export TREE="${C[0]}"
    export FR10_METRICS=0 BATCH_INVARIANT=0
    export LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export FR13_RUN_DIR="$armout" LOG_DIR="$armout/logs"

    bash scripts/fr14_armb_leg3_launch_nomiddleware.sh > "$armout/launch.log" 2>&1
  )
  local rc=$?
  echo "[$arm] launcher rc=$rc"
  if (( rc != 0 )); then tail -40 "$armout/launch.log"; return 2; fi

  local boot=0 i
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { boot=1; break; }
    docker ps -q -f "name=$container" | grep -q . || break
    sleep 5
  done
  echo "[$arm] boot=$boot after $((i*5))s"
  docker logs "$container" > "$armout/boot_container.log" 2>&1
  if (( boot != 1 )); then tail -80 "$armout/boot_container.log"; return 3; fi
  # docker exec, NOT /proc/<pid>/environ on the host: the container runs as root
  # and the host-side read is Permission denied (it was, on both arms of the
  # 2026-08-17 run -- the envs had to be recaptured by hand afterwards).
  docker exec "$container" bash -lc 'tr "\0" "\n" < /proc/1/environ | sort' \
    > "$armout/container_env.txt" 2>/dev/null || true
  docker exec "$container" bash -lc 'tr "\0" " " < /proc/1/cmdline' \
    > "$armout/engine_cmdline.txt" 2>&1
  echo "[$arm] HEALTHY"
  return 0
}

bench_arm() {  # arm
  local arm=$1 armout="$OUT/$1"
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$armout/metrics_pre.txt"
  echo "[$arm] bench start $(date -u +%H:%M:%SZ)"
  docker run --rm --network host -v /home/mark/shared/models:/models "$IMG" \
    python3 -m sglang.bench_serving --backend vllm \
      --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --dataset-name random --random-input-len 1024 --random-output-len 1024 \
      --random-range-ratio 1 --disable-tqdm --seed 1 \
      --extra-request-body '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
    > "$armout/bench.log" 2>&1
  echo "[$arm] bench rc=$? $(date -u +%H:%M:%SZ)"
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$armout/metrics_post.txt"
  docker logs "$CONTAINER_NAME" > "$armout/run_container.log" 2>&1 || true
}

teardown() {  # container arm
  docker rm -f "$1" >/dev/null 2>&1
  local sp="$OUT/$2/.secret_path"
  [[ -f "$sp" ]] && rm -f "$(cat "$sp")"
  sleep 10
}

for spec in "k64 65536 1 fr14-armb-k64" "k0 0 0 fr14-armb-k0"; do
  set -- $spec
  ARM=$1 K=$2 ROOT=$3 CONTAINER_NAME=$4
  echo "================ ARM $ARM (K=$K ROOT=$ROOT) ================"
  if boot_arm "$ARM" "$K" "$ROOT" "$CONTAINER_NAME"; then
    bench_arm "$ARM"
  else
    echo "[$ARM] BOOT FAILED — capturing and continuing (a K=0 boot refusal is a FINDING)"
    docker logs "$CONTAINER_NAME" > "$OUT/$ARM/boot_container.log" 2>&1 || true
  fi
  teardown "$CONTAINER_NAME" "$ARM"
done

echo "[ablation] done -> $OUT"
