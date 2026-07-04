#!/usr/bin/env bash
# FR13 BIG-DENOMINATOR SWEServe VARIANT arm — generalizes fr13_bigdenom_swe_serve.sh
# to the speed-campaign candidate arms (the cat-shape swap + the two levers), while
# preserving the canonical hygiene / health / spec-engagement / pair-dump machinery.
#
# It is the EXACT deployment regime of fr13_bigdenom_swe_serve.sh (real SWE-Verified +
# codex agent loop, per-task /metrics brackets -> deploy-speed) with ONE thing varied:
# the boot config (which the deploy-speed reducer is agnostic to — it only needs the
# per-task vllm_metrics_pre/post.txt brackets + the correct --expected-tok-per-draft).
#
# ARMS (KIND):
#   cat9        = locked cat9 (== fr13_bigdenom_swe_serve.sh cat9; here for same-wall A/B)
#   cat9-opta   = locked cat9 + OPT-A (FR13_GB10_FP8_GEMV_CFG=1, lossless byte-identical)
#   cat9-opt1   = locked cat9 + OPT-1 (FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1)
#   cat6root    = R4 shape via the forked launcher TREE override (depth-5, 6 nodes, EXPECT 6)
#   cat10       = cat10 shape via the forked launcher TREE override (depth-5, 10 nodes, EXPECT 10)
#
# Each boots the LOCKED cat9 pipeline (FIX-1/2/3/A, REPLAY_ROUTE, FA2_TREE_BIAS,
# CONV_COMMITTED_PATH, LUMO_FB_KERNEL_ROWS=1 + PROJ_PAD_ROWS=16, BATCH_INVARIANT=0)
# except the one varied lever/shape. K1 (FR13_SCAN_ALIGN) NOT baked (asserted off).
#
# Usage: fr13_bigdenom_swe_serve_variant.sh <arm_name> <KIND> <subset.json>
#   AGENT_WALL_S (env): bound the codex wall for a DEV-iteration screen (e.g. 600).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

ARM=${1:?usage: fr13_bigdenom_swe_serve_variant.sh <arm> <KIND> <subset.json>}
KIND=${2:?cat9|cat9-opta|cat9-opt1|nativemtp5|nativemtp5apc|nativemtp5_exseed|cat6root|cat10}
SUBSET=${3:?subset json}

RUNROOT=${RUNROOT:-output/fr13_bigdenom_swe}
ARMDIR="$RUNROOT/$ARM"
[[ -f "$SUBSET" ]] || SUBSET="output/fr13_b1_gold_swe/$SUBSET"
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
CONTAINER="fr13-bigdenom-$ARM"
PORT=9950
PROXY_PORT=8022
EVAL_HOST=${EVAL_HOST:-alienware}
# --- FR13 CODEX-OFFLOAD (unified-memory contamination fix; see fr13_bigdenom_swe_serve.sh) ---
# OFFLOAD_CODEX=1 (default): proxy+codex on alienware, GB10 stays vLLM-only =
# clean deploy-speed (s/fwd read GB10-locally, stall-immune). =0 = legacy.
OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
OFFLOAD_HOST=${OFFLOAD_HOST:-alienware}
OFFLOAD_PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
GB10_TS_IP=${GB10_TS_IP:-100.103.10.122}
OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh
OFFLOAD_LINK_DOWN_MAX_S=${OFFLOAD_LINK_DOWN_MAX_S:-300}
mkdir -p "$ARMDIR/logs"

# Arm dispatch: launcher + TREE + extra flags + expected draft-token ratio.
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
# chain5 = the linear 5-spine (cat6root MINUS the (1,) sibling branch). Apples-to-
# apples spine vs cat6root tree on the SAME forked launcher + APC; the spine is
# lossless w.r.t. APC without the recompute (snapshot matches native block layout).
CHAIN5_TREE="[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
CAT10_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]"
# cat8 = spine + rank-2 sibling at ONLY the first 3 depths (where acceptance reaches
# ~3.5) -- the middle ground between cat6root (sibling@depth-1 only) and cat10 (siblings
# @all 5 depths, but depths 4-5 are past the accept frontier = dead weight). 8 nodes.
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"
# 3-3-3 (depth-3, 3 candidates/depth: spine top-1 + rank-1 top-2 + rank-2 top-3 at
# the root and both interior spine depths). EXPECT 9 (num_speculative_tokens=9).
THREETHREE_TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2)]"
# FR13_RESHAPE_WIDE: width-5 shapes for the wider-not-deeper sweep. Both are
# sorted (len, path) caterpillar trees consumed by the GENERAL width-N drafter
# (top-5 runner-up reads off the spine logits; lossless by the drafter-agnostic
# committer). Widths = the digit pattern per spine depth.
#   cat555  = [5,5,5]      depth-3, 15 nodes (pad16). vs native E3.
#   cat55222= [5,5,2,2,2]  depth-5, 16 nodes (== N_PAD=16). vs native E5.
CAT555_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4)]"
# cat55222 [5,5,2,2,2]=16 nodes OVERFLOWS the GDN tree-verifier warm cap:
# n_pad=next_pow2(nodes+1) must be <=16, but 16 nodes -> n17 -> pad32 -> raise.
# Kept for reference only; it FAILS engine-init. Use cat55221 (15 nodes) instead.
CAT55222_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]"
# cat55221 [5,5,2,2,1]=15 nodes, depth-5: the FITTING depth-5 wide arm (n16->pad16).
# Wide front (5,5) intact; deepest spine node carries no leaf (width-1). vs E5.
CAT55221_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0)]"
case "$KIND" in
  cat9)      LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=() ;;
  cat9-opta) LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=(FR13_GB10_FP8_GEMV_CFG=1) ;;
  cat9-opt1) LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=(FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1) ;;
  # nativemtp5 = the fr9 decode path: STOCK vLLM native MTP-5 (qwen3_5_mtp,
  # num_speculative_tokens=5, NO tree). The drafter emits 5 linear draft tokens
  # per step, so draft_tokens/drafts == 5 (EXPECT_RATIO=5). No forked-fa2, no
  # tree_attn, no APC, no in-container patcher — the apples half of the
  # chain5(forked)-vs-native-MTP A/B for the char-8 tool-call regression.
  nativemtp5) LAUNCHER=native; TREEARG="";            EXPECT_RATIO=5;  declare -a XFLAGS=() ;;
  nativemtp5apc) LAUNCHER=native; TREEARG="";         EXPECT_RATIO=5;  declare -a XFLAGS=(NATIVE_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32) ;;
  # nativemtp5_exseed = NATIVE MTP-5 decode (clean native kernel: FLASH_ATTN, native
  # qwen3_5_mtp spec, NO speculative_token_tree, NO forked-fa2 tree-attn) + the
  # patcher's LOSSLESS deployment cache FR13_APC_EXACT_SEED=1. It uses LAUNCHER=forked
  # ONLY to run the in-container patcher (fr10_phase4_patch_vllm_tree_gdn.py, line 590)
  # that ships EXACT_SEED — but overrides the launcher's TREE defaults to a native
  # config via XFLAGS. Draft is linear MTP-5 => draft_tokens/drafts==5 (EXPECT 5),
  # launcher-agnostic (proves the MTP drafter ran). EXACT_SEED is tree-independent:
  # its non-spec row map publishes on EVERY _prepare_inputs step incl. pure prefill
  # (patcher ~L11720), ckpt0 capture + restore key off block hashes + non-spec prefill
  # rows, never tree parents. The forked launcher's tree-conv path self-gates off
  # (use_fr10_tree_conv requires decode_mode==tree_mtp AND fr10_tree_parent != None;
  # naive_mtp + no tree => both False => native causal_conv1d path). XFLAGS:
  #   ATTENTION_BACKEND=FLASH_ATTN            -> clean native kernel (no TREE_ATTN)
  #   SPEC_CONFIG=native qwen3_5_mtp,5 tokens -> native MTP-5, NO speculative_token_tree
  #   FR10_DECODE_MODE_DEFAULT=naive_mtp      -> linear MTP path (NOT tree_mtp)
  #   FR13_REPLAY_ROUTE/FA2_TREE_BIAS/FA2_PREFILL_NATIVE/TREE_SAMPLE_ROW/
  #   CONV_COMMITTED_PATH=0                   -> un-leak the forked launcher tree env
  #   FR13_ENABLE_APC=1 + MAMBA/APC block flags-> triggers forked APC_FLAGS branch (L224)
  #   FR13_APC_EXACT_SEED=1                   -> the lossless deployment cache under test
  nativemtp5_exseed) LAUNCHER=forked; TREEARG=""; EXPECT_RATIO=5; declare -a XFLAGS=(
        ATTENTION_BACKEND=FLASH_ATTN
        'SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}'
        FR10_DECODE_MODE_DEFAULT=naive_mtp
        FR13_REPLAY_ROUTE=0 FR13_FA2_TREE_BIAS=0 FR13_FA2_PREFILL_NATIVE=0
        FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0
        FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1
        MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
        FR13_APC_REQUIRE_SNAP_FIX=1) ;;
  # flash_ns5_nocache = CARRIER-LOCATOR cell A0: forked launcher (=> the forked FA2 .so IS mounted +
  # copied over stock at launch, L590) + clean FLASH_ATTN naive-MTP ns5, but CACHE OFF (no APC/EXACT_SEED).
  # vs stock-native N (LAUNCHER=native, stock .so) isolates the FORKED FA2 .so; vs tree D0 isolates the
  # TREE_ATTN kernel; vs cache-on A isolates the cache effect OFF the tree path.
  flash_ns5_nocache) LAUNCHER=forked; TREEARG=""; EXPECT_RATIO=5; declare -a XFLAGS=(
        ATTENTION_BACKEND=FLASH_ATTN
        'SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":5}'
        FR10_DECODE_MODE_DEFAULT=naive_mtp
        FR13_REPLAY_ROUTE=0 FR13_FA2_TREE_BIAS=0 FR13_FA2_PREFILL_NATIVE=0
        FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0) ;;
  # flash_ns8_exseed = CARRIER-LOCATOR cell C: identical to nativemtp5_exseed (clean FLASH_ATTN
  # naive-MTP kernel, EXACT_SEED cache) but num_speculative_tokens=8 -> isolates draft DEPTH (5->8)
  # under the clean kernel. If this stays clean while cat8 corrupts, depth is not the carrier.
  flash_ns8_exseed) LAUNCHER=forked; TREEARG=""; EXPECT_RATIO=8; declare -a XFLAGS=(
        ATTENTION_BACKEND=FLASH_ATTN
        'SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":8}'
        FR10_DECODE_MODE_DEFAULT=naive_mtp
        FR13_REPLAY_ROUTE=0 FR13_FA2_TREE_BIAS=0 FR13_FA2_PREFILL_NATIVE=0
        FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0
        FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1
        MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
        FR13_APC_REQUIRE_SNAP_FIX=1) ;;
  cat6root)  LAUNCHER=forked; TREEARG="$CAT6ROOT_TREE"; EXPECT_RATIO=6;  declare -a XFLAGS=() ;;
  chain5)    LAUNCHER=forked; TREEARG="$CHAIN5_TREE";   EXPECT_RATIO=5;  declare -a XFLAGS=() ;;
  cat10)     LAUNCHER=forked; TREEARG="$CAT10_TREE";    EXPECT_RATIO=10; declare -a XFLAGS=() ;;
  cat8)      LAUNCHER=forked; TREEARG="$CAT8_TREE";     EXPECT_RATIO=8;  declare -a XFLAGS=() ;;
  333)       LAUNCHER=forked; TREEARG="$THREETHREE_TREE"; EXPECT_RATIO=9; declare -a XFLAGS=() ;;
  cat555)    LAUNCHER=forked; TREEARG="$CAT555_TREE";    EXPECT_RATIO=15; declare -a XFLAGS=() ;;
  cat55222)  LAUNCHER=forked; TREEARG="$CAT55222_TREE";  EXPECT_RATIO=16; declare -a XFLAGS=() ;;
  cat55221)  LAUNCHER=forked; TREEARG="$CAT55221_TREE";  EXPECT_RATIO=15; declare -a XFLAGS=() ;;
  *) echo "FAIL: unknown KIND=$KIND"; exit 2 ;;
esac
# NATIVE_DECODE: arms whose DECODE is the native/linear MTP path (no tree), so the
# tree-flag class-9 pins + the tree-only worker environ needle DO NOT APPLY and the
# native-appropriate asserts run instead. Covers LAUNCHER=native (nativemtp5[apc], no
# patcher) AND nativemtp5_exseed (LAUNCHER=forked ONLY to run the patcher for
# EXACT_SEED, but a native FLASH_ATTN+qwen3_5_mtp config with no speculative_token_tree
# and FR10_DECODE_MODE_DEFAULT=naive_mtp). Both prove the drafter ran via the launcher-
# agnostic draft_tokens/drafts==5 spec-ratio assert, NOT via TREE_ATTN/tree env.
NATIVE_DECODE=0
if [[ "$LAUNCHER" == "native" || "$KIND" == "nativemtp5_exseed" || "$KIND" == "flash_ns8_exseed" || "$KIND" == "flash_ns5_nocache" ]]; then NATIVE_DECODE=1; fi
# Per-request decode-mode xarg for the warmup probe: naive_mtp for native-decode arms
# (tree_mtp would ask the patched resolver for a tree that does not exist), tree_mtp
# for the tree/reshape arms. The native launcher ignores the xarg (no patcher), but
# nativemtp5_exseed DOES load the patcher, so it must ask for the mode it actually runs.
if (( NATIVE_DECODE == 1 )); then PROBE_MODE=naive_mtp; else PROBE_MODE=tree_mtp; fi
# B=4 co-residency knobs (default B=1 = byte-identical to the prior runs).
# MAX_NUM_SEQS_OVR + SWE_CONCURRENCY env override the boot co-residency + the
# number of concurrent codex tasks. For genuine B=4 co-residency pass a 4-task
# subset + MAX_NUM_SEQS_OVR=4 + SWE_CONCURRENCY=4.
MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}

echo "=== BIGDENOM-VARIANT SWEServe ARM $ARM kind=$KIND launcher=$LAUNCHER expect=$EXPECT_RATIO xflags=[${XFLAGS[*]:-none}] subset=$SUBSET ==="
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD | tee "$ARMDIR/git_head.txt"

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

echo "[hygiene] recover_host_memory + assert free"
recover_host || true
.venv/bin/python - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=f.get("SwapTotal",0)-f.get("SwapFree",0)
if avail<95 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
(( $? == 0 )) || { echo "FAIL: pre-boot hygiene"; exit 2; }
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi
free -g | tee "$ARMDIR/free_before_boot.txt"

OLDPID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
[[ -n "$OLDPID" ]] && kill "$OLDPID" 2>/dev/null
pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
sleep 1

teardown(){
  echo "[teardown] kill proxy + docker rm -f $CONTAINER + recover_host_memory"
  kill "$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null)" 2>/dev/null
  pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
  [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null
  if [[ "$OFFLOAD_CODEX" == "1" ]]; then
    LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" stop "$OFFLOAD_HOST" >> "$ARMDIR/offload_teardown.log" 2>&1 || true
  fi
  docker logs "$CONTAINER" > "$ARMDIR/docker_full.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  recover_host || true
  sleep 2
  free -g | tee "$ARMDIR/free_after_teardown.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_ended_at.txt"
}
trap teardown EXIT

# ---- boot server (class 11: everything pinned except the arm lever/shape) ----
# extra flags exported into THIS shell so the launcher's docker -e picks them up.
for kv in "${XFLAGS[@]:-}"; do [[ -n "$kv" ]] && export "$kv"; done
if [[ "$LAUNCHER" == "locked" ]]; then
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_locked.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
elif [[ "$LAUNCHER" == "native" ]]; then
  # NATIVE MTP-5 = the fr9 decode path: STOCK vLLM MTP, no forked-fa2 / tree /
  # APC / patcher. The launcher boots the model's own MTP head via the stock
  # qwen3_5_mtp spec method. It carries NONE of the FR13 tree/APC/FIX env, so we
  # deliberately do NOT export the locked cat9 flags here (they are tree-only and
  # inert on the native path). Same CONTAINER/PORT/RUN_DIR wiring as the others so
  # the offload proxy / health / warmup / teardown machinery is identical.
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_native_mtp_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
else
  # Reshape arms boot the LOCKED cat9 pipeline flags (FIX-1/2/3/A, REPLAY_ROUTE,
  # FA2_TREE_BIAS, CONV_COMMITTED_PATH) + the BAKED in_proj_ba pad
  # (LUMO_FB_KERNEL_ROWS=1, LUMO_FB_PROJ_PAD_ROWS=16) so the only difference vs
  # cat9 is the TREE shape. The forked launcher defaults these flags ON except
  # the LUMO_FB pad, which we pin here for a like-for-like vs the deployed cat9.
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  TREE="$TREEARG" FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
fi
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -30 "$ARMDIR/launch.log"; exit 2; fi

T0=$(date +%s)
HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL: health not up in 1200s"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 2; }
echo "healthy after $(( $(date +%s) - T0 ))s"

# ---- class 9: flag state in container env ----
# The FR13 tree/APC machinery is DELIBERATELY ABSENT on the native MTP path
# (LAUNCHER=native = the fr9 decode path). So for nativemtp5 the tree-flag pins +
# the FR10_DECODE_MODE_DEFAULT worker needle DO NOT APPLY; we assert the stock
# native config instead (the spec method + no tree env leaked in). All other
# (locked/forked) arms keep the identical class-9 gate.
docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
if (( NATIVE_DECODE == 1 )); then
  # Native MTP: assert the stock spec method is live (native qwen3_5_mtp, NO
  # speculative_token_tree) and NO tree env leaked in. Applies to LAUNCHER=native AND
  # nativemtp5_exseed (LAUNCHER=forked with the tree env overridden to native via XFLAGS).
  grep -q "^SPEC_CONFIG=.*qwen3_5_mtp" "$ARMDIR/container_env.txt" \
    || { echo "FAIL: native SPEC_CONFIG (qwen3_5_mtp) not in container env"; exit 3; }
  grep -q "^SPEC_CONFIG=.*speculative_token_tree" "$ARMDIR/container_env.txt" \
    && { echo "FAIL: native arm SPEC_CONFIG carries speculative_token_tree (not the native MTP path)"; exit 3; }
  for forbidden in "FR10_DECODE_MODE_DEFAULT=tree_mtp" "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1"; do
    grep -q "^$forbidden$" "$ARMDIR/container_env.txt" \
      && { echo "FAIL: native arm leaked tree env: $forbidden (not the fr9 path)"; exit 3; }
  done
  # nativemtp5_exseed: also assert the FLASH_ATTN native kernel + the EXACT_SEED /
  # APC prefix-caching flags are LIVE in the container env (the whole point of the arm).
  # The forked launcher only emits APC_FLAGS when FR13_ENABLE_APC=1, and EXACT_SEED
  # requires SNAP_FIX=1 (baked ON when APC is on); verify both reached the container.
  if [[ "$KIND" == "nativemtp5_exseed" ]]; then
    grep -q "^ATTENTION_BACKEND=FLASH_ATTN$" "$ARMDIR/container_env.txt" \
      || { echo "FAIL: nativemtp5_exseed ATTENTION_BACKEND != FLASH_ATTN (native kernel)"; exit 3; }
    grep -q "^FR13_APC_EXACT_SEED=1$" "$ARMDIR/container_env.txt" \
      || { echo "FAIL: nativemtp5_exseed FR13_APC_EXACT_SEED=1 not in container env"; exit 3; }
    grep -q "^FR13_APC_SNAP_FIX=1$" "$ARMDIR/container_env.txt" \
      || { echo "FAIL: nativemtp5_exseed FR13_APC_SNAP_FIX=1 not live (EXACT_SEED requires it)"; exit 3; }
  fi
  echo "container env OK (native MTP: qwen3_5_mtp, no tree env; KIND=$KIND)"
else
  NEEDS=("FR13_DRAFTER_SINGLE_LOGITS=1" "FR13_EAGER_PACK=1" "FR13_TREE_CONV_FUSED=1" \
         "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" \
         "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1" "FR13_CONV_COMMITTED_PATH=1" \
         "FR10_DECODE_MODE_DEFAULT=tree_mtp" \
         "LUMO_FB_KERNEL_ROWS=1" "LUMO_FB_PROJ_PAD_ROWS=16")
  if grep -q "^FR13_SCAN_ALIGN=1$" "$ARMDIR/container_env.txt" && [ "${FR13_ALLOW_SCAN_ALIGN:-0}" != "1" ]; then
    echo "FAIL: FR13_SCAN_ALIGN=1 present — K1 must NOT be baked (set FR13_ALLOW_SCAN_ALIGN=1 for the recompute diagnostic)"; exit 3
  fi
  for need in "${NEEDS[@]}"; do
    grep -q "^$need$" "$ARMDIR/container_env.txt" || { echo "FAIL: env pin missing: $need"; exit 3; }
  done
  # arm-specific OPT flag must be LIVE in container env
  for kv in "${XFLAGS[@]:-}"; do
    [[ -n "$kv" ]] && { grep -q "^$kv$" "$ARMDIR/container_env.txt" || { echo "FAIL: OPT flag not live: $kv"; exit 3; }; }
  done
  echo "container env OK ($KIND)"
fi

# ---- worker /proc environ bridge-needle (tree path only) ----
# Keyed on FR10_DECODE_MODE_DEFAULT=tree_mtp, the TREE decode path. The native-decode
# arms either do not set it (LAUNCHER=native) or set it to naive_mtp
# (nativemtp5_exseed); either way the tree-only needle DOES NOT APPLY. The equivalent
# proof (the MTP drafter actually ran) is the spec-engagement ratio assert below, so
# skip this needle for all native-decode arms.
if (( NATIVE_DECODE == 0 )); then
  docker exec "$CONTAINER" bash -lc '
hit=""; pids=""
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  e="/proc/$p/environ"; [ -r "$e" ] || continue
  cmd="$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)"
  case "$cmd" in *vllm*|*VllmWorker*|*EngineCore*|*python*) ;; *) continue ;; esac
  dm="$(tr "\0" "\n" < "$e" | grep -E "^FR10_DECODE_MODE_DEFAULT=" || true)"
  if [ -n "$dm" ]; then
    pids="$pids $p"
    echo "PID $p cmd=[$cmd]"
    tr "\0" "\n" < "$e" | grep -E "^(FR10_DECODE_MODE_DEFAULT|LUMO_FB_KERNEL_ROWS|FR13_GB10_FP8_GEMV_CFG|FR13_GPU_COMMITTER|FR13_COMMITTER_SYNCKILL|SPEC_CONFIG|FR13_REPLAY_ROUTE)=" | sed "s/^/   /"
    hit="1"
  fi
done
echo "SUMMARY worker_env_seen=[$hit] pids=[$pids]"
' | tee "$ARMDIR/worker_environ_needle.txt"
  grep -q "worker_env_seen=\[1\]" "$ARMDIR/worker_environ_needle.txt" \
    || { echo "FAIL: no vLLM worker with FR10_DECODE_MODE_DEFAULT in /proc environ"; exit 3; }
  echo "worker environ needle OK ($KIND)"
else
  echo "worker environ needle SKIPPED (native-decode arm: no tree_mtp decode mode; spec-ratio assert covers drafter-ran)"
fi

# ---- FULL CUDA capture needle (skip under --enforce-eager: no graphs to capture) ----
docker logs "$CONTAINER" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  echo "[capture needle] SKIPPED (ENFORCE_EAGER=1: eager mode has no CUDA graph capture)"
else
  grep -m1 "Graph capturing finished" "$ARMDIR/boot_log_snapshot.txt" \
    || { echo "FAIL: no 'Graph capturing finished' in boot log"; exit 3; }
fi

# ---- OPT engagement needle in boot/serve log (OPT-1 logs once; OPT-A patcher prints) ----
case "$KIND" in
  cat9-opt1)
    # the synckill needle fires on first committed event during warmup/codex; check post-warmup
    echo "[needle] OPT-1 synckill engagement checked after warmup (see docker_full.log)";;
  cat9-opta)
    grep -m1 "FR13_GB10_FP8_GEMV_CFG" "$ARMDIR/boot_log_snapshot.txt" >/dev/null 2>&1 \
      && echo "[needle] OPT-A patch present in boot log" || echo "[needle] OPT-A patch line not in boot log (override engages at forward time on GB10+small-M)";;
esac

# ---- warmup probe (fires spec engagement) ----
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_warmup.txt"
.venv/bin/python scripts/fr10_quick_decode_tps_probe.py \
  --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
  --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
  --samples-per-prompt 1 --batch-size 1 --seed 1313 --top-p 1.0 \
  --wait-health 60 --request-timeout 900 --warmup-samples 0 \
  --modes "$PROBE_MODE" \
  --prompt-limit 1 --max-tokens 16 --temperature 0.0 \
  --out "$ARMDIR/warmup_probe.json" \
  --request-metrics-out "$ARMDIR/warmup_request_metrics.jsonl" \
  > "$ARMDIR/warmup_probe_stdout.log" 2>&1
RC=$?
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_warmup.txt"
if (( RC != 0 )); then
  echo "FAIL: warmup probe rc=$RC"
  tail -20 "$ARMDIR/warmup_probe_stdout.log"; docker logs "$CONTAINER" 2>&1 | tail -30; exit 4
fi
.venv/bin/python - "$ARMDIR" "$EXPECT_RATIO" <<'PY'
import sys
from pathlib import Path
armdir, expect = Path(sys.argv[1]), float(sys.argv[2])
def val(path, name):
    for line in (armdir / path).read_text().splitlines():
        if line.startswith(name):
            return float(line.rsplit(None, 1)[-1])
    return 0.0
d  = val("metrics_after_warmup.txt", "vllm:spec_decode_num_drafts_total") \
   - val("metrics_before_warmup.txt", "vllm:spec_decode_num_drafts_total")
dt = val("metrics_after_warmup.txt", "vllm:spec_decode_num_draft_tokens_total") \
   - val("metrics_before_warmup.txt", "vllm:spec_decode_num_draft_tokens_total")
assert d > 0, f"spec engagement FAIL (class 9): spec_drafts delta={d}"
ratio = dt / d
assert abs(ratio - expect) < 1e-9, \
    f"draft-shape FAIL (class 9): draft_tokens/drafts={ratio} expected {expect}"
print(f"spec engagement OK: drafts delta={d} draft_tokens/drafts={ratio}")
PY
(( $? == 0 )) || { echo "FAIL: spec engagement raw-counter assert"; exit 4; }

# ---- FR13 APC worker-env engagement assert (only for APC arms) ----
# The APC fixes read os.environ.get(FR13_APC_*) INSIDE the EngineCore worker (pid 176).
# That dict is intact in the worker (spawn inherits the full parent os.environ; the
# "dropped" symptom was a /proc/PID/environ artifact from setproctitle, NOT a real
# drop — verified 2026-06-27). The gdn module-import bridge in the worker writes
# /logs/fr13_apc_bridge_loaded.flag recording the LIVE os.environ values the gates
# read. Grep it for the masters this arm REQUIRES before recording any APC verdict, so
# a vacuous (defaults-only) run can never masquerade as engaged. Default-OFF arms set
# no FR13_APC_REQUIRE_* and this whole block is skipped -> locked non-APC path unchanged.
if [[ -n "${FR13_APC_REQUIRE_SNAP_FIX:-}${FR13_APC_REQUIRE_HIT_SUFFIX_CAP:-}${FR13_APC_REQUIRE_SHADOW:-}" ]]; then
  APC_MARKER="${FR13_APC_BRIDGE_MARKER_FILE:-/logs/fr13_apc_bridge_loaded.flag}"
  # copy the worker-written marker out of the container's /logs to the arm dir
  docker exec "$CONTAINER" sh -c "cat '$APC_MARKER' 2>/dev/null" > "$ARMDIR/apc_bridge_marker.txt" 2>/dev/null
  # any bridge error file is a hard fail (a swallowed read must never look like engagement)
  if docker exec "$CONTAINER" sh -c "test -f /logs/fr13_apc_bridge_error.flag" 2>/dev/null; then
    echo "FAIL: APC bridge error flag present in worker:"; docker exec "$CONTAINER" cat /logs/fr13_apc_bridge_error.flag 2>/dev/null; exit 4
  fi
  APC_MARKER_TXT="$(cat "$ARMDIR/apc_bridge_marker.txt" 2>/dev/null)"
  [[ -n "$APC_MARKER_TXT" ]] || { echo "FAIL: APC engagement marker $APC_MARKER absent/empty in worker (gdn bridge did not run)"; docker logs "$CONTAINER" 2>&1 | tail -30; exit 4; }
  if [[ -n "${FR13_APC_REQUIRE_SNAP_FIX:-}" ]]; then
    grep -q "SNAP_FIX=${FR13_APC_REQUIRE_SNAP_FIX}\b" "$ARMDIR/apc_bridge_marker.txt" \
      || { echo "FAIL: APC worker SNAP_FIX != required ${FR13_APC_REQUIRE_SNAP_FIX} (vacuous gate). marker: $APC_MARKER_TXT"; exit 4; }
  fi
  if [[ -n "${FR13_APC_REQUIRE_HIT_SUFFIX_CAP:-}" ]]; then
    grep -q "HIT_SUFFIX_CAP=${FR13_APC_REQUIRE_HIT_SUFFIX_CAP}\b" "$ARMDIR/apc_bridge_marker.txt" \
      || { echo "FAIL: APC worker HIT_SUFFIX_CAP != required ${FR13_APC_REQUIRE_HIT_SUFFIX_CAP} (cap defaulted to 64 = vacuous). marker: $APC_MARKER_TXT"; exit 4; }
  fi
  if [[ -n "${FR13_APC_REQUIRE_SHADOW:-}" ]]; then
    grep -q "SHADOW=${FR13_APC_REQUIRE_SHADOW}\b" "$ARMDIR/apc_bridge_marker.txt" \
      || { echo "FAIL: APC worker SHADOW != required ${FR13_APC_REQUIRE_SHADOW} (shadow-log gate vacuous: env not bridged into worker). marker: $APC_MARKER_TXT"; exit 4; }
  fi
  echo "APC worker-env engagement OK: $APC_MARKER_TXT"
fi

curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
  > "$ARMDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"

# ---- CAPTURE_ONLY: G1 kernel-confirm capture (boot + warmup fired the tree-decode payload dump, exit) ----
# The warmup above ran a tree_mtp decode; with FR10_TREE_GDN_CAPTURE_PAYLOAD + FR12_SUBKERNEL_CAPTURE set
# (and FR13_REPLAY_ROUTE=0 so the per-node tree_state scratch survives), the tree GDN kernel saved the
# per-node payload + subkernel gdn_scan_out to /logs (mapped to $ARMDIR/logs). Verify + exit before SWE.
if [[ "${CAPTURE_ONLY:-0}" == "1" ]]; then
  ls -la "$ARMDIR/logs"/*.pt 2>/dev/null | tee "$ARMDIR/g1_capture_manifest.txt" || true
  N=$(ls "$ARMDIR/logs"/*payload*.pt "$ARMDIR/logs"/*subkernel*.pt 2>/dev/null | wc -l)
  echo "[capture-only] SCAN_ALIGN_MODE=${FR13_SCAN_ALIGN_MODE:-body} captured $N .pt -> $ARMDIR/logs"
  if [[ "$N" -ge 1 ]]; then echo "CAPTURE_ONLY_OK"; exit 0; else
    echo "FAIL: no capture .pt produced (warmup tree-decode did not fire the layer-0 capture)"; exit 6; fi
fi

# ---- PROBE_ONLY: generation-level carrier locator (skip offload proxy + SWE) ----
# The server is booted, healthy, spec-engaged and APC-verified above. PROBE_ONLY replays banked
# tool-call-boundary prompts N times DIRECTLY against local vLLM (temp 0.6 lives in the probe body)
# and measures the malformed-markup emission rate — no agent, no nudge net, no offload tunnel = the
# clean de-confounder for {FLASH,TREE}x{ns5,ns8}. Additive: PROBE_ONLY unset => original SWE path.
if [[ "${PROBE_ONLY:-0}" == "1" ]]; then
  : "${PROBE_PROMPTS:?PROBE_ONLY needs PROBE_PROMPTS=<dir of chatreq_*.json>}"
  echo "[probe-only] carrier locator: cell=$ARM prompts=$PROBE_PROMPTS samples=${PROBE_SAMPLES:-20} limit=${PROBE_PROMPT_LIMIT:-20}"
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/fr13_carrier_locator_probe.py \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" --model qwen3.6-27b \
    --cell "$ARM" --prompts "$PROBE_PROMPTS" \
    --prompt-limit "${PROBE_PROMPT_LIMIT:-20}" --samples "${PROBE_SAMPLES:-20}" \
    ${PROBE_LOGPROBS:+--logprobs} \
    --out "$ARMDIR/carrier_probe.json" > "$ARMDIR/carrier_probe_stdout.log" 2>&1
  PRC=$?
  echo "[probe-only] rc=$PRC -> $ARMDIR/carrier_probe.json"; tail -30 "$ARMDIR/carrier_probe_stdout.log" 2>/dev/null || true
  exit $PRC
fi

# ---- launch canonical proxy (forced temp 0.0 + pair dumps), OFFLOAD-gated ----
mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
CODEX_ARGS=()
if [[ "$OFFLOAD_CODEX" == "1" ]]; then
  echo "[offload] OFFLOAD_CODEX=1 — proxy+codex on $OFFLOAD_HOST, GB10 stays vLLM-only"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$OFFLOAD_HOST" \
        "curl -fsS -m 6 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo ok" \
        2>/dev/null | grep -q ok; then
    echo "FAIL: alienware cannot reach GB10 vLLM at http://$GB10_TS_IP:$PORT/health (set OFFLOAD_CODEX=0 to fall back)"
    exit 5
  fi
  echo "[offload] alienware -> GB10 vLLM $GB10_TS_IP:$PORT/health OK"
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" sync "$OFFLOAD_HOST" > "$ARMDIR/offload_sync.log" 2>&1 \
    || { echo "FAIL: offload proxy sync"; cat "$ARMDIR/offload_sync.log"; exit 5; }
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" start "$OFFLOAD_HOST" "$GB10_TS_IP" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_start.log" 2>&1 \
    || { echo "FAIL: offload proxy start"; cat "$ARMDIR/offload_start.log"; exit 5; }
  cat "$ARMDIR/offload_start.log"
  cp "$ARMDIR/offload_proxy_env.txt" "$ARMDIR/proxy_env.txt" 2>/dev/null || true
  CODEX_ARGS=(--codex-host "$OFFLOAD_HOST" \
              --codex-endpoint "http://127.0.0.1:$OFFLOAD_PROXY_PORT/v1")
  echo "proxy OK (OFFLOADED to $OFFLOAD_HOST:$OFFLOAD_PROXY_PORT)"
else
  LUMO_PROXY_FORCE_TEMPERATURE="${DEPLOY_FORCE_TEMP:-0.6}" \
  LUMO_PROXY_REQUEST_DUMP_DIR="$PWD/$ARMDIR/proxy_request_dumps" \
  LUMO_PROXY_PAIR_DUMP_DIR="$PWD/$ARMDIR/proxy_pair_dumps" \
  LUMO_PROXY_LOG_PATH="$PWD/$ARMDIR/proxy.log" \
  LUMO_PROXY_NOHUP_PATH="$PWD/$ARMDIR/proxy.nohup" \
  LUMO_PROXY_STATE_ROOT="/tmp/fr13_bigdenom_proxy_state_${ARM}" \
  bash scripts/swe_x86_helpers/relaunch_proxy.sh > "$ARMDIR/proxy_launch.log" 2>&1
  sleep 3
  P0=$(date +%s); PROXY_OK=0
  while (( $(date +%s) < P0 + 60 )); do
    CODE=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PROXY_PORT/v1/models" 2>/dev/null)
    if [[ -n "$CODE" && "$CODE" != "000" ]]; then PROXY_OK=1; break; fi
    sleep 2
  done
  (( PROXY_OK == 1 )) || { echo "FAIL: proxy not healthy"; tail -20 "$ARMDIR/proxy.nohup" 2>/dev/null; exit 5; }
  # Capture the proxy env AFTER it is confirmed healthy: the pid file / /proc/PID/environ lag the
  # health endpoint, so capturing at the bare `sleep 3` above RACED startup -> empty proxy_env.txt
  # -> spurious "proxy temp pin missing" NORESULT even though the proxy WAS launched with the temp.
  # Retry the read briefly until the forced-temp line appears.
  for _i in $(seq 1 12); do
    PROXY_PID=$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null || true)
    if [[ -n "$PROXY_PID" ]] && [[ -r "/proc/$PROXY_PID/environ" ]]; then
      tr '\0' '\n' < "/proc/$PROXY_PID/environ" | grep -E "^LUMO_(PROXY|TRACK_B)" | sort > "$ARMDIR/proxy_env.txt"
    fi
    grep -q "LUMO_PROXY_FORCE_TEMPERATURE=" "$ARMDIR/proxy_env.txt" 2>/dev/null && break
    sleep 1
  done
  grep -q "LUMO_PROXY_FORCE_TEMPERATURE=${DEPLOY_FORCE_TEMP:-0.6}" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy temp pin missing (expected ${DEPLOY_FORCE_TEMP:-0.6})"; exit 5; }
  grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/proxy_env.txt" || { echo "FAIL: proxy pair-dump pin missing"; exit 5; }
  echo "proxy OK"
fi

# ---- eval offload pre-flight ----
EVAL_ARGS=()
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$EVAL_HOST" \
     "test -f ~/swe_eval_offload/swe_eval_x86_worker.py && echo ok" 2>/dev/null | grep -q ok; then
  EVAL_ARGS=(--eval-host "$EVAL_HOST")
  echo "eval offload: $EVAL_HOST reachable" | tee "$ARMDIR/eval_offload_preflight.txt"
else
  echo "WARN: eval offload $EVAL_HOST UNREACHABLE — eval attempted locally" | tee "$ARMDIR/eval_offload_preflight.txt"
fi

# ---- SWE window: /metrics bracketing + codex loop ----
WALL_ARGS=()
[[ -n "${AGENT_WALL_S:-}" ]] && WALL_ARGS=(--agent-wall-s "$AGENT_WALL_S")

# NETWORK-LINK WATCHDOG (OFFLOAD only; req #4/#5) — see fr13_bigdenom_swe_serve.sh
LINKLOG="$ARMDIR/offload_link_state.log"
LINK_DEAD_MARKER="$ARMDIR/offload_link_dead.flag"
WATCHDOG_PID=""
if [[ "$OFFLOAD_CODEX" == "1" ]]; then
  rm -f "$LINK_DEAD_MARKER"
  ( down_since=0
    while true; do
      ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      if ssh -o BatchMode=yes -o ConnectTimeout=8 "$OFFLOAD_HOST" \
           "curl -fsS -m 5 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo up" \
           2>/dev/null | grep -q up; then
        echo "[$ts] LINK up" >> "$LINKLOG"; down_since=0
      else
        now=$(date +%s); [[ "$down_since" == "0" ]] && down_since=$now
        contig=$(( now - down_since ))
        echo "[$ts] LINK DOWN contig=${contig}s (CLASSIFIED network-drop, not a model fork)" >> "$LINKLOG"
        if (( contig > OFFLOAD_LINK_DOWN_MAX_S )); then
          echo "[$ts] LINK DEAD > ${OFFLOAD_LINK_DOWN_MAX_S}s — watchdog FAIL LOUD" | tee -a "$LINKLOG"
          echo "$ts contig=${contig}s" > "$LINK_DEAD_MARKER"; break
        fi
      fi
      sleep 10
    done ) >> "$ARMDIR/offload_watchdog.log" 2>&1 &
  WATCHDOG_PID=$!
  echo "[offload] link watchdog pid=$WATCHDOG_PID (threshold ${OFFLOAD_LINK_DOWN_MAX_S}s)"
fi

curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_swe.txt"
S0=$(date +%s)
.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$SUBSET" \
  --out-root "$ARMDIR/swe_out" \
  --concurrency "$SWE_CONCURRENCY" \
  --eval-timeout-s "${EVAL_TIMEOUT_S:-1800}" \
  "${WALL_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${CODEX_ARGS[@]}" \
  > "$ARMDIR/swe_orchestrator.log" 2>&1
SWERC=$?
S1=$(date +%s)
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_swe.txt"
SWE_WALL=$((S1-S0))
echo "swe orchestrator rc=$SWERC wall=${SWE_WALL}s"
tail -5 "$ARMDIR/swe_orchestrator.log"

[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null && WATCHDOG_PID=""

# ---- OFFLOAD: fetch alienware pair-dumps back (resilient rsync, req #3) ----
if [[ "$OFFLOAD_CODEX" == "1" ]]; then
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$PWD/$ARMDIR" \
    > "$ARMDIR/offload_fetch.log" 2>&1 || echo "WARN: offload fetch errors (see offload_fetch.log)"
  cat "$ARMDIR/offload_fetch.log"
  if [[ -f "$LINK_DEAD_MARKER" ]]; then
    echo "OFFLOAD_LINK_DEAD: $(cat "$LINK_DEAD_MARKER") — deploy-speed window for $ARM DISCARDED" \
      | tee "$ARMDIR/DEPLOY_SPEED_DISCARDED.flag"
    SWERC=12
  fi
fi

# ---- OPT-1 post-run engagement needle ----
if [[ "$KIND" == "cat9-opt1" ]]; then
  docker logs "$CONTAINER" 2>&1 | grep -m1 "FR13_COMMITTER_SYNCKILL engaged" \
    | tee "$ARMDIR/opt1_synckill_needle.txt" \
    || { echo "FAIL: OPT-1 synckill never engaged (vacuous arm)"; SWERC=9; }
fi

# ---- health rule + pair-dump non-vacuity ----
.venv/bin/python - "$ARMDIR" "$SWE_WALL" "$SWERC" <<'PY'
import json, sys
from pathlib import Path
armdir, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
metas = sorted(armdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
health = {"swe_orchestrator_rc": rc, "swe_window_wall_s": wall, "tasks": []}
for m in metas:
    meta = json.loads(m.read_text())
    codex = meta.get("codex") or {}
    health["tasks"].append({"instance_id": meta.get("instance_id"),
        "codex_elapsed_s": codex.get("elapsed_s"),
        "codex_timed_out": codex.get("timed_out"),
        "patch_bytes": meta.get("patch_bytes"),
        "verdict": (meta.get("eval_report") or {}).get("verdict", "missing")})
(armdir / "health.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
PY

NPAIR=$(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)
echo "pair dumps captured: $NPAIR"

echo "ARM_DONE $ARM kind=$KIND swerc=$SWERC"
exit $SWERC
