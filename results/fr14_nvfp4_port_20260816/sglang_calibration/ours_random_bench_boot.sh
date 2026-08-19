#!/usr/bin/env bash
# FR14 APPLES-TO-APPLES MEASUREMENT 1 -- OUR PROMOTED STACK on THEIR WORKLOAD.
#
# Boots the PROMOTED PRODUCTION STACK (hydra27_fixed32 + K0 full-vocab drafting
# + split-K tier-B armed BY DEFAULT + fused draft top-k default) with NO SWE
# runner and NO fixed32 ASGI ingress middleware, so sglang's bench_serving can
# drive it on THEIR random-1024/1024 bs=1 shape.
#
# Vehicle = results/fr14_nvfp4_port_20260816/ablation_a_leg3_boot.sh, retargeted
# from tail6/K64 to hydra27/K0 and to the RadixArk NVFP4 checkpoint.
#
# THE B1 FA2 ARM IS DELIBERATELY UNNAMED: the promoted split-K default must arm
# itself from the launcher's own staged credential. That default path is what is
# under test here.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"
OUT=${OUT:-/home/mark/shared/tmp-scratch/nvfp4_port/ours_random_bench}
mkdir -p "$OUT/logs"
chmod 700 "$OUT" "$OUT/logs"

# ---- 1. campaign env: the K0 floor contract the promoted stack runs under ----
export BSIZE=1 CONC=1 WALL=9000
export FR13_CAMPAIGN_TASK_BUDGET_S=9000
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 FR13_DRAFT_HEAD_FP8=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab
export FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE=full_vocab
export FR13_FLOOR_ORDER=TH
# shellcheck disable=SC1091
source scripts/fr13_canonical_env.sh
run_variant() { :; }
# shellcheck disable=SC1091
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_K" == "0" && "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_FIXED32_TAW_WALK_CAP" == "12" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "25430574256" \
   && "$FR13_WEIGHT_FLOOR_MS" == "93.15228665201465" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "FAIL: K0 floor contract drifted (bytes=$FR13_MANDATORY_WEIGHT_BYTES floor=$FR13_WEIGHT_FLOOR_MS cap=$FR13_FIXED32_TAW_WALK_CAP)"; exit 2; }
echo "[ours] seq env: GPU_UTIL=$GPU_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN FLOOR_MS=$FR13_WEIGHT_FLOOR_MS"

# ---- 2. fixed32 topology authority (same source of truth as the variant) ----
mapfile -t C < <(.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
import fr13_fixed32_topology as t
t.validate_contract()
t.validate_gate_contract()
print(repr(list(t.FIXED32_CHOICES)))
print(f"{t.HYDRA27_VALID_MASK:#x}")
print(t.HYDRA27_ACTIVE_DRAFTS)
print(t.PHYSICAL_DRAFTS)
print(t.profile(t.PROFILE_HYDRA27)["walk_cap"])
PY
)
(( ${#C[@]} == 5 )) || { echo "FAIL: topology authority"; exit 2; }
FIXED32_TREE=${C[0]}; HYDRA_MASK=${C[1]}; HYDRA_ACTIVE=${C[2]}
PHYS_DRAFTS=${C[3]}; HYDRA_WALK_CAP=${C[4]}
[[ "$PHYS_DRAFTS" == "31" && "$HYDRA_WALK_CAP" == "12" ]] \
  || { echo "FAIL: hydra27 contract drifted"; exit 2; }
echo "[ours] hydra27 mask=$HYDRA_MASK active=$HYDRA_ACTIVE walk_cap=$HYDRA_WALK_CAP"

# ---- 3. the hydra27_fixed32 arm XFLAGS (variant lines 476-487) ----
export FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged
export FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=0
export FR13_FIXED32_MODE=hydra27_fixed32
export FR13_FIXED32_TAW_WALK_CAP="$HYDRA_WALK_CAP"
export FR13_FIXED32_VALID_MASK="$HYDRA_MASK"
export FR13_FIXED32_ACTIVE_NODES="$HYDRA_ACTIVE"
export FR13_FIXED32_PHYSICAL_DRAFTS="$PHYS_DRAFTS"
export FR13_FIXED32_B1_DIAGNOSTIC=0
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export ENFORCE_EAGER=0
export CUDAGRAPH_MODE=FULL_AND_PIECEWISE
# the canonical exact4 identity the tier-B serve block requires
export FR13_FA2_QROW32_B1_EXACT4_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
export FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
# device multidraft, as the promoted serves run it
export FR13_DEVICE_MULTIDRAFT=1
export FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py

# OPTIONAL PROVENANCE. Left EMPTY on the first attempt on purpose: the promoted
# split-K default is supposed to arm itself. If the launcher refuses because its
# B1 SELECTOR provenance gate wants a commit the default never sets, that is a
# finding about the default boot path, and RERUN_WITH_PROVENANCE=1 re-attempts
# with the minimal operator input the default omits.
if [[ "${RERUN_WITH_PROVENANCE:-0}" == "1" ]]; then
  export FR13_FA2_QROW32_B1_SOURCE_COMMIT="$(git rev-parse HEAD)"
  export FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | cut -d' ' -f1)"
  echo "[ours] provenance supplied: commit=$FR13_FA2_QROW32_B1_SOURCE_COMMIT"
fi

# per-arm timer sidecars
export FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1
export FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_ours_random.json
export FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_ours_random_dfwd.json
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/fr14_ours_random_cfwd.json
mkdir -p output/fr13_sfwd_sidecar

# ---- 4. fixed32 ingress secret (launcher validates it even with the
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
export CONTAINER=fr14-ours-random-hydra27
export PORT=9950
export MAX_NUM_SEQS=1
export SWE_CONCURRENCY=1
export TREE="$FIXED32_TREE"
export FR10_METRICS=0
export BATCH_INVARIANT=0
export LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
export FR13_RUN_DIR="$OUT"
export LOG_DIR="$OUT/logs"

echo "[ours] free before boot:"; free -g | tee "$OUT/free_before_boot.txt"
[[ -z "$(docker ps -aq)" ]] || { echo "FAIL: docker not empty"; rm -f "$SECRET"; exit 2; }
REPO="$REPO" bash "$OUT/launch_nomiddleware.sh" > "$OUT/launch.log" 2>&1
RC=$?
echo "[ours] launcher rc=$RC"
if (( RC != 0 )); then tail -40 "$OUT/launch.log"; rm -f "$SECRET"; exit 2; fi
printf '%s\n' "$SECRET" > "$OUT/.secret_path"
echo "[ours] container launched; waiting for health on :$PORT"
BOOT=0
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { BOOT=1; break; }
  docker ps -q -f "name=$CONTAINER" | grep -q . || break
  sleep 5
done
echo "[ours] boot=$BOOT after $((i*5))s"
docker logs "$CONTAINER" > "$OUT/boot_container.log" 2>&1
if (( BOOT != 1 )); then tail -80 "$OUT/boot_container.log"; exit 3; fi
tr '\0' '\n' < /proc/$(docker inspect -f '{{.State.Pid}}' "$CONTAINER")/environ \
  | sort > "$OUT/container_env.txt" 2>/dev/null || true
docker exec "$CONTAINER" bash -lc 'tr "\0" " " < /proc/1/cmdline' > "$OUT/engine_cmdline.txt" 2>&1
echo "[ours] HEALTHY"
