#!/usr/bin/env bash
# FR14 ABLATION ARM A / LEG 3 (diagnostic, non-citable): boot OUR fixed32 stock
# engine exactly as the successful fr14_b1_stock_20260816T204931Z/tail6 arm did,
# but WITHOUT any SWE runner and WITHOUT the fixed32 ASGI ingress middleware, so
# sglang's bench_serving can drive it on THEIR 1024/1024 bench shape.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"
OUT=${OUT:-/home/mark/shared/tmp-scratch/fr14_ablation_a/leg3}
mkdir -p "$OUT/logs"
chmod 700 "$OUT" "$OUT/logs"

# ---- 1. campaign env: the EXACT sequence file the stock B1 arm ran under ----
run_variant(){ :; }
export BSIZE=1 CONC=1 TAG=fr14leg3
export FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_HEAD_FP8=0
# shellcheck disable=SC1091
# HEAD-PINNED: another agent is live-editing scripts/ on this branch (floor re-derivation).
# Leg 3 must reproduce the committed b1-stock arm, so source the HEAD copy.
source /home/mark/shared/tmp-scratch/fr14_ablation_a/head_pin/fr13_fixed32_floor_timers_seq.HEAD.sh
echo "[leg3] seq env sourced: GPU_UTIL=$GPU_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN FLOOR_MS=$FR13_WEIGHT_FLOOR_MS"

# ---- 2. fixed32 topology authority (same source of truth as the variant) ----
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
(( ${#C[@]} == 3 )) || { echo "FAIL: topology authority"; exit 2; }
FIXED32_TREE=${C[0]}; FIXED32_TAIL_MASK=${C[1]}; FIXED32_TAIL_ACTIVE=${C[2]}
echo "[leg3] fixed32 mask=$FIXED32_TAIL_MASK active=$FIXED32_TAIL_ACTIVE"

# ---- 3. the tail6_fixed32 arm XFLAGS (variant lines 430-442) ----
export FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged
export FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=0
export FR13_FIXED32_MODE=tail6_fixed32
export FR13_FIXED32_VALID_MASK="$FIXED32_TAIL_MASK"
export FR13_FIXED32_ACTIVE_NODES="$FIXED32_TAIL_ACTIVE"
export FR13_FIXED32_PHYSICAL_DRAFTS=31
export FR13_FIXED32_B1_DIAGNOSTIC=0
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1

# per-arm timer sidecars (same shape as the stock arm's)
export FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_leg3.json
export FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_leg3_dfwd.json
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_leg3_cfwd.json
mkdir -p output/fr13_sfwd_sidecar

# ---- 4. fixed32 ingress secret (launcher still validates it even with the
#         middleware off; identical generation code to the variant) ----
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

# ---- 5. launch (patched copy: middleware dropped, nothing else) ----
export CONTAINER=fr14-leg3-tail6-fixed32
export PORT=9950
export MAX_NUM_SEQS=1
export SWE_CONCURRENCY=1
export TREE="$FIXED32_TREE"
export FR10_METRICS=0
export BATCH_INVARIANT=0
export LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
export FR13_RUN_DIR="$OUT"
export LOG_DIR="$OUT/logs"

echo "[leg3] free before boot:"; free -g | tee "$OUT/free_before_boot.txt"
docker ps -q | grep -q . && { echo "FAIL: docker not empty"; exit 2; }
bash scripts/fr14_leg3_launch_nomiddleware.sh > "$OUT/launch.log" 2>&1
RC=$?
echo "[leg3] launcher rc=$RC"
if (( RC != 0 )); then tail -40 "$OUT/launch.log"; rm -f "$SECRET"; exit 2; fi
printf '%s\n' "$SECRET" > "$OUT/.secret_path"
echo "[leg3] container launched; waiting for health on :$PORT"
BOOT=0
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { BOOT=1; break; }
  docker ps -q -f "name=$CONTAINER" | grep -q . || break
  sleep 5
done
echo "[leg3] boot=$BOOT after $((i*5))s"
docker logs "$CONTAINER" > "$OUT/boot_container.log" 2>&1
if (( BOOT != 1 )); then tail -60 "$OUT/boot_container.log"; exit 3; fi
tr '\0' '\n' < /proc/$(docker inspect -f '{{.State.Pid}}' "$CONTAINER")/environ \
  | sort > "$OUT/container_env.txt" 2>/dev/null || true
docker exec "$CONTAINER" bash -lc 'tr "\0" " " < /proc/1/cmdline' > "$OUT/engine_cmdline.txt" 2>&1
echo "[leg3] HEALTHY"
