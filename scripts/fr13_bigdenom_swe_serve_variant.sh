#!/usr/bin/env bash
# FR13 BIG-DENOMINATOR SWEServe VARIANT arm — generalizes fr13_bigdenom_swe_serve.sh
# to the speed-campaign candidate arms (the cat-shape swap + the two levers), while
# preserving the canonical hygiene / health / spec-engagement / proxy capture machinery.
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
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO=$(cd "$REPO" && pwd)
cd "$REPO"
# shellcheck source=fr13_required_tree_flags.sh
source "$SCRIPT_DIR/fr13_required_tree_flags.sh"

ARM=${1:?usage: fr13_bigdenom_swe_serve_variant.sh <arm> [KIND=tail6] [subset.json]}
# KIND defaults to tail6 = the CANONICAL deployed config (cleanup+bake 2026-07-27,
# FR13_CLEANUP_BAKE_PLAN.md): cat33333 head + 6-node Arctic tail, 21 nodes, BV=8.
KIND=${2:-tail6}
SUBSET=${3:?subset json}
FR13_FIXED32_B1_DIAGNOSTIC=${FR13_FIXED32_B1_DIAGNOSTIC:-0}
FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=${FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB-0}
FR13_FIXED32_COMMITTER_LAYER_BATCH=${FR13_FIXED32_COMMITTER_LAYER_BATCH:-0}
case "$FR13_FIXED32_B1_DIAGNOSTIC" in
  0|1) ;;
  *) echo "FAIL: FR13_FIXED32_B1_DIAGNOSTIC must be exactly 0 or 1"; exit 2 ;;
esac
case "$FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB" in
  0|1) ;;
  *) echo "FAIL: FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB must be exactly 0 or 1"; exit 2 ;;
esac
case "$FR13_FIXED32_COMMITTER_LAYER_BATCH" in
  0|1) ;;
  *) echo "FAIL: FR13_FIXED32_COMMITTER_LAYER_BATCH must be exactly 0 or 1"; exit 2 ;;
esac
export FR13_FIXED32_B1_DIAGNOSTIC FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB \
  FR13_FIXED32_COMMITTER_LAYER_BATCH

RUNROOT=${RUNROOT:-output/fr13_bigdenom_swe}
ARMDIR="$RUNROOT/$ARM"
ARMDIR_ABS=$(realpath -m "$ARMDIR")
[[ -f "$SUBSET" ]] || SUBSET="output/fr13_b1_gold_swe/$SUBSET"
[[ -f "$SUBSET" ]] || { echo "FAIL: subset not found: $SUBSET"; exit 2; }
CONTAINER="fr13-bigdenom-$ARM"
CONTAINER_RUNTIME_REF="$CONTAINER"
FIXED32_CIDFILE=""
FIXED32_CONTAINER_STARTED_AT=""
FIXED32_CONTAINER_INIT_PID=""
FIXED32_CONTAINER_RESTART_COUNT=""
FIXED32_CONTAINER_CLEANUP_OUTCOME=""
PORT=9950
PROXY_PORT=8022
EVAL_HOST=${EVAL_HOST:-alienware}
# --- FR13 AGENT-OFFLOAD (unified-memory contamination fix; see fr13_bigdenom_swe_serve.sh) ---
# OFFLOAD_AGENT=1 (default): proxy+codex on alienware, GB10 stays vLLM-only =
# clean deploy-speed (s/fwd read GB10-locally, stall-immune). =0 = legacy.
OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}
OFFLOAD_HOST=${OFFLOAD_HOST:-alienware}
OFFLOAD_PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
GB10_TS_IP=${GB10_TS_IP:-100.103.10.122}
OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh
OFFLOAD_LINK_DOWN_MAX_S=${OFFLOAD_LINK_DOWN_MAX_S:-300}
if [[ "$KIND" == "tail6_fixed32" || "$KIND" == "hydra27_fixed32" ]]; then
  [[ "$OFFLOAD_AGENT" == "1" ]] \
    || { echo "FAIL: fixed32 requires OFFLOAD_AGENT=1"; exit 2; }
  [[ ! -e "$ARMDIR" ]] \
    || { echo "FAIL: fixed32 arm directory already exists: $ARMDIR"; exit 2; }
fi
mkdir -p "$ARMDIR/logs"
if [[ "$KIND" == "tail6_fixed32" || "$KIND" == "hydra27_fixed32" ]]; then
  chmod 700 "$ARMDIR" "$ARMDIR/logs" \
    || { echo "FAIL: fixed32 arm directories could not be made private"; exit 2; }
fi
if [[ "$FR13_FIXED32_COMMITTER_LAYER_BATCH" == "1" ]]; then
  [[ "$KIND" == "tail6_fixed32" || "$KIND" == "hydra27_fixed32" ]] \
    || { echo "FAIL: committer layer-batch arm requires a fixed32 kind"; exit 2; }
  install -m 600 /dev/null \
    "$ARMDIR/logs/fr13_fixed32_committer_layer_batch.arm" \
    || { echo "FAIL: could not create committer layer-batch arm sidecar"; exit 2; }
fi
FIXED32_MODE=""
FIXED32_PRODUCER_PID=""
FIXED32_REQUEST_PATH="$ARMDIR_ABS/logs/fr13_fixed32_flush_request.json"
FIXED32_ACK_PATH="$ARMDIR_ABS/logs/fr13_fixed32_flush_ack.json"
FIXED32_PID_PATH="$ARMDIR_ABS/logs/fr13_fixed32_engine_pid"
FIXED32_BOUNDARY_SNAPSHOT_PATH="$ARMDIR_ABS/logs/fr13_fixed32_boundary_snapshot"
FIXED32_TAW_REAL_EVENT_ARM_PATH="$ARMDIR_ABS/logs/fr13_fixed32_taw_native_precompute.real_event.arm"
FIXED32_BM8_REAL_EVENT_ARM_PATH="$ARMDIR_ABS/logs/fr13_dfwd_unified_bm8.real_event.arm"
FIXED32_CUTLASS_REAL_EVENT_ARM_PATH="$ARMDIR_ABS/logs/fr13_fixed32_cutlass_streamk.real_event.arm"
FIXED32_INGRESS_SECRET_FILE=""
FR13_FIXED32_INGRESS_TASK_IDS=""

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
# 33333 = widths [3,3,3,3,3]: depth-5 spine (MTP-5-comparable) + 2 branches at EVERY depth (15 choices)
# 55555 = widths [5,5,5,5,5]: depth-5 spine + 4 branches/depth (25 choices, 26 rows) — tree-scaling probe past 16
T55555_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,0,4),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,3),(0,0,0,4),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,3),(0,0,0,0,4)]"
T33333_TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2)]"
# accept>5 TAIL: cat33333 head (15 nodes, depths 0-4, MTP-drafted == t33333 baseline) + a 6-node deep
# Arctic-filled spine CHAIN (depths 6-11) = 21 nodes -> n_pad=32, wide_D=11. Head byte-identical baseline;
# the tail ADDS only (never-regress via the committer). == tail_tree_order(tail_len=6) from fr13_mtp_suffix_assembly.
TAIL6_TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]"
# HYDRA23: tail6 with only the human-depth-5 MTP siblings traded for one
# four-node Arctic continuation below the root rank-1 candidate. The main
# depth-11 spine and both siblings through human depth 4 are unchanged.
HYDRA23_TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,0),(0,0,0),(0,0,1),(0,0,2),(1,0,0),(0,0,0,0),(0,0,0,1),(0,0,0,2),(1,0,0,0),(0,0,0,0,0),(1,0,0,0,0),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]"
# Fixed-32 has one topology authority. Do not duplicate the 31 paths or masks
# in shell: a source/topology drift must fail before the server can launch.
mapfile -t _FIXED32_CONTRACT < <(
  .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
import fr13_fixed32_topology as t
t.validate_contract()
print(repr(list(t.FIXED32_CHOICES)))
print(f"{t.TAIL6_VALID_MASK:#x}")
print(f"{t.HYDRA27_VALID_MASK:#x}")
print(t.TAIL6_ACTIVE_DRAFTS)
print(t.HYDRA27_ACTIVE_DRAFTS)
PY
)
(( ${#_FIXED32_CONTRACT[@]} == 5 )) \
  || { echo "FAIL: fixed32 topology authority did not emit five fields"; exit 2; }
FIXED32_TREE=${_FIXED32_CONTRACT[0]}
FIXED32_TAIL_MASK=${_FIXED32_CONTRACT[1]}
FIXED32_HYDRA_MASK=${_FIXED32_CONTRACT[2]}
FIXED32_TAIL_ACTIVE=${_FIXED32_CONTRACT[3]}
FIXED32_HYDRA_ACTIVE=${_FIXED32_CONTRACT[4]}
unset _FIXED32_CONTRACT
# 333 = widths [3,3,3]: depth-3 spine (spine_steps=2) + 2 branches/depth (9 choices). SAME width
# as t33333, differs ONLY in depth -> dfwd(t33333)-dfwd(t333) = 2 spine-steps' drafter cost.
T333_TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2)]"
CAT55222_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]"
# cat55221 [5,5,2,2,1]=15 nodes, depth-5: the FITTING depth-5 wide arm (n16->pad16).
# Wide front (5,5) intact; deepest spine node carries no leaf (width-1). vs E5.
CAT55221_TREE="[(0,),(1,),(2,),(3,),(4,),(0,0),(0,1),(0,2),(0,3),(0,4),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0)]"
case "$KIND" in
  cat9)      LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=() ;;
  cat9-opta) LAUNCHER=locked; TREEARG="";             EXPECT_RATIO=9;  declare -a XFLAGS=(FR13_GB10_FP8_GEMV_CFG=1) ;;
  # nativemtp5 = the fr9 decode path: STOCK vLLM native MTP-5 (qwen3_5_mtp,
  # num_speculative_tokens=5, NO tree). The drafter emits 5 linear draft tokens
  # per step, so draft_tokens/drafts == 5 (EXPECT_RATIO=5). No forked-fa2, no
  # tree_attn, no APC, no in-container patcher — the apples half of the
  # chain5(forked)-vs-native-MTP A/B for the char-8 tool-call regression.
  nativemtp5) LAUNCHER=native; TREEARG="";            EXPECT_RATIO=5;  declare -a XFLAGS=() ;;
  nativemtp5apc) LAUNCHER=native; TREEARG="";         EXPECT_RATIO=5;  declare -a XFLAGS=(NATIVE_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32) ;;
  # nativemtp11apc = DEPTH-MATCHED native baseline (user 2026-07-19): stock MTP
  # recursion at tail6's depth 11 — decomposes the tree's accept edge into
  # depth-budget vs branching vs mixed-drafter contributions. Expect the deep
  # MTP conditionals to decay (self-fed embeddings) + 11 sequential drafter
  # passes; native's own optimum stays MTP-5.
  nativemtp11apc) LAUNCHER=native; TREEARG="";        EXPECT_RATIO=11; declare -a XFLAGS=(NUM_SPECULATIVE_TOKENS=11 NATIVE_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32) ;;
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
        FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=0
        MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
        ) ;;
  # nativemtp11_exseed = DEPTH-MATCHED twin of nativemtp5_exseed (user 2026-07-22):
  # identical forked-launcher/patcher pathway (so FR13_SFWD/DFWD/CFWD_GPU_TIMER
  # instrumentation works, unlike nativemtp11apc which is LAUNCHER=native and never
  # loads the patcher) but num_speculative_tokens=11 to match tail6's depth. Isolates
  # draft DEPTH (5->11) on the clean native MTP kernel, matching the depth-matched
  # comparison nativemtp11apc was built for but with GPU-timer visibility.
  nativemtp11_exseed) LAUNCHER=forked; TREEARG=""; EXPECT_RATIO=11; declare -a XFLAGS=(
        ATTENTION_BACKEND=FLASH_ATTN
        'SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":11}'
        FR10_DECODE_MODE_DEFAULT=naive_mtp
        FR13_REPLAY_ROUTE=0 FR13_FA2_TREE_BIAS=0 FR13_FA2_PREFILL_NATIVE=0
        FR13_TREE_SAMPLE_ROW=0 FR13_CONV_COMMITTED_PATH=0
        FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=0
        MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
        ) ;;
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
        FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=0
        MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
        ) ;;
  cat6root)  LAUNCHER=forked; TREEARG="$CAT6ROOT_TREE"; EXPECT_RATIO=6;  declare -a XFLAGS=() ;;
  chain5)    LAUNCHER=forked; TREEARG="$CHAIN5_TREE";   EXPECT_RATIO=5;  declare -a XFLAGS=() ;;
  cat10)     LAUNCHER=forked; TREEARG="$CAT10_TREE";    EXPECT_RATIO=10; declare -a XFLAGS=() ;;
  cat8)      LAUNCHER=forked; TREEARG="$CAT8_TREE";     EXPECT_RATIO=8;  declare -a XFLAGS=() ;;
  333)       LAUNCHER=forked; TREEARG="$THREETHREE_TREE"; EXPECT_RATIO=9; declare -a XFLAGS=() ;;
  cat555)    LAUNCHER=forked; TREEARG="$CAT555_TREE";    EXPECT_RATIO=15; declare -a XFLAGS=() ;;
  t333)      LAUNCHER=forked; TREEARG="$T333_TREE";     EXPECT_RATIO=9;  declare -a XFLAGS=() ;;
  t33333)    LAUNCHER=forked; TREEARG="$T33333_TREE";   EXPECT_RATIO=15; declare -a XFLAGS=() ;;
  # accept>5 TAIL arm: cat33333 head + 6-node Arctic chain (21 nodes, n_pad=32, wide_D=11). Needs
  # FR13_TAIL_MODE=1 (caps native spine_steps=4 + appends Arctic tail) + FR13_DRAFT_SOURCE=merged (the
  # Arctic cache lifecycle) + BV=8 (n_pad=32 register budget). EXPECT_RATIO=21 = the tree node count.
  tail6)     LAUNCHER=forked; TREEARG="$TAIL6_TREE";    EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8) ;;
  # Root-rescue trade: exact 23-node topology, same four native MTP head
  # forwards and main Arctic tail as tail6, plus one host trie walk.
  hydra23)   LAUNCHER=forked; TREEARG="$HYDRA23_TREE"; EXPECT_RATIO=23; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=1) ;;
  # Same 31 physical drafts and verifier rows in both arms. Only the active
  # sampler mask differs; every other fixed-work route is shared.
  tail6_fixed32)
    LAUNCHER=forked
    TREEARG="$FIXED32_TREE"
    EXPECT_RATIO=31
    FIXED32_MODE=tail6_fixed32
    declare -a XFLAGS=(
      FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged
      FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=0
      FR13_FIXED32_MODE=tail6_fixed32
      "FR13_FIXED32_VALID_MASK=$FIXED32_TAIL_MASK"
      "FR13_FIXED32_ACTIVE_NODES=$FIXED32_TAIL_ACTIVE"
      FR13_FIXED32_PHYSICAL_DRAFTS=31
    )
    ;;
  hydra27_fixed32)
    LAUNCHER=forked
    TREEARG="$FIXED32_TREE"
    EXPECT_RATIO=31
    FIXED32_MODE=hydra27_fixed32
    declare -a XFLAGS=(
      FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged
      FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_HYDRA23=0
      FR13_FIXED32_MODE=hydra27_fixed32
      "FR13_FIXED32_VALID_MASK=$FIXED32_HYDRA_MASK"
      "FR13_FIXED32_ACTIVE_NODES=$FIXED32_HYDRA_ACTIVE"
      FR13_FIXED32_PHYSICAL_DRAFTS=31
    )
    ;;
  # REPLAY-TIMER probe: tail6 + FR13_REPLAY_GPU_TIMER=1 -> coarse GPU-time of the accepted-path GDN replay
  # per step (sidecar). Settles the 94ms-committer split: replay ~=80ms => the replay is the bottleneck
  # (re-compute, reducible via stateless-tree gather = real lever); replay ~=11ms => the cost is sync-wait/
  # DtoH not the replay (native wins). Diagnostic (synchronize inflates other spans -- read REPLAY only).
  tail6_rt)  LAUNCHER=forked; TREEARG="$TAIL6_TREE";    EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_REPLAY_GPU_TIMER=1 FR13_REPLAY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_rt_replay.json) ;;
  # MULTISTREAM A/B (FR13_REPLAY_MULTISTREAM_DESIGN.md): overlap the 48 write-independent per-layer replays
  # across N streams. tail6_cf2 = baseline (multistream OFF) + whole-committer GPU timer (CF2, one sync/step
  # => captures overlap, unlike per-launch REPLAY_GPU_TIMER whose per-launch synchronize serializes streams).
  # tail6_ms = same + FR13_REPLAY_MULTISTREAM=1. Compare committer gpu_s; accept/output MUST be identical.
  tail6_cf2) LAUNCHER=forked; TREEARG="$TAIL6_TREE";    EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_COMMIT_FULL_GPU_TIMER=1 FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_cf2_commit.json) ;;
  tail6_ms)  LAUNCHER=forked; TREEARG="$TAIL6_TREE";    EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_REPLAY_MULTISTREAM=1 FR13_COMMIT_FULL_GPU_TIMER=1 FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_ms_commit.json) ;;
  # NATIVE-KERNEL committer replay (the cheap attack): route the accepted-path GDN state advance through
  # native fused_sigmoid_gating_delta_rule_update (0.15ms/layer) instead of the custom Triton replay
  # (1.386ms/layer). Offline bit-exact to no-spec (1.19e-7). tail6_ncom vs tail6_cf2 (custom) = committer A/B.
  tail6_ncom) LAUNCHER=forked; TREEARG="$TAIL6_TREE";   EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_COMMITTER_NATIVE=1 FR13_COMMIT_FULL_GPU_TIMER=1 FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_ncom_commit.json) ;;
  # MULTIDRAFT-COMMITTER decomposition (the REAL temp-0.6 committer, per user's catch): tail6 +
  # FR13_MULTIDRAFT_GPU_TIMER=1 -> per-step GPU-time of fr13_device_multidraft_commit (the deployed temp>0
  # rejection committer). Settles what the 94ms committer_gpu span IS: multidraft_ms high => the rejection
  # committer is the cost (optimize it); low => the 94ms is result-DtoH + verify-wait (pipeline). This is
  # the committer I SHOULD have decomposed (FR13_GPU_COMMITTER was the greedy LCP, off the temp-0.6 path).
  tail6_mt)  LAUNCHER=forked; TREEARG="$TAIL6_TREE";    EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_MULTIDRAFT_GPU_TIMER=1 FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_mt_md.json) ;;
  # Direction-2 tail-DEPTH lever: identical config to tail6 (NO drift) but a deeper spine tail (25 nodes,
  # depth-15, tail_len=10 derived from wide_D). Tests whether the RISING deep-tail conditional (d7-11:
  # 0.848->0.950) keeps paying past d11. tail_len auto-derives from the tree; no code change.
  tailx10)   LAUNCHER=forked; TREEARG="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)]"; EXPECT_RATIO=25; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8) ;;
  # Direction-2 d6-BRANCH lever: identical config to tail6 (NO drift) but the tail is BRANCHED at its first
  # 2 depths (d6/d7) with arctic runner-ups (FR13_TAIL_BRANCHES=2 FR13_TAIL_BRANCH_DEPTHS=2) -> 25 nodes,
  # depth-11 (SAME depth as tail6, +4 branch nodes). Targets the measured weakest link (handoff cond 0.666)
  # WITHOUT adding depth cost. Monotone-lossless (branches only ADD candidates). Compare accept to tail6 ~5.1.
  tail6b)    LAUNCHER=forked; TREEARG="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,0,0),(0,0,0,0,0,0,1),(0,0,0,0,0,0,2),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]"; EXPECT_RATIO=25; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_TAIL_BRANCHES=2 FR13_TAIL_BRANCH_DEPTHS=2) ;;
  # Direction-2 seam-CONCENTRATE lever: identical config to tail6b (NO drift, SAME 25 nodes / n_pad=32 /
  # tps) but all 4 seam branches at d6 ONLY (FR13_TAIL_BRANCHES=4 FR13_TAIL_BRANCH_DEPTHS=1) instead of
  # 2+2 across d6/d7. d6 is the biggest single leak (handoff cond 0.666 vs d7 0.848), so concentrating
  # arctic width at THE seam tests concentrate-vs-spread. Needs the fr13_merged_fill width fix (width =
  # max(3, tail_branches+1)) so ranks 3,4 aren't dropped. Monotone-lossless. Run AFTER tail6b confirms
  # arctic seam branches help at all (else concentrate is moot). d6 = spine + 4 arctic runner-ups.
  tail6c)    LAUNCHER=forked; TREEARG="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,3),(0,0,0,0,0,4)]"; EXPECT_RATIO=25; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_TAIL_BRANCHES=4 FR13_TAIL_BRANCH_DEPTHS=1) ;;
  # Direction-2 seam-WIDEN lever: scale the PROVEN tail6b geometry -- 3 branches at d6 AND d7 (BRANCHES=3
  # DEPTHS=2) = 27 nodes, still n_pad=32 / same tps. b7 interim implies HIGH recovery (~35% of the miss),
  # so more seam width (not concentrate) is the natural "scale what works" lever: adds d6 rank-3 + d7 rank-3
  # on top of tail6b's 2+2. Needs fill width=max(3,tail_branches+1)=4 at pp=5,6. Monotone-lossless / no drift.
  tail6e)    LAUNCHER=forked; TREEARG="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,3),(0,0,0,0,0,0,1),(0,0,0,0,0,0,2),(0,0,0,0,0,0,3)]"; EXPECT_RATIO=27; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_TAIL_BRANCHES=3 FR13_TAIL_BRANCH_DEPTHS=2) ;;
  # Direction-2 REALLOCATION lever (zero-node-cost): SAME 21 nodes as tail6 but the 2 deepest tail nodes
  # (d10,d11 -- conditional already 0.90-0.95, low marginal value) REALLOCATED to 2 d6 branches (the 0.334
  # handoff leak). tail_len auto-derives to 4 (wide_D=9 from the shallower tree, patcher:14220) => NO new
  # plumbing/flag => no config drift; only the tree + FR13_TAIL_BRANCHES=2/DEPTHS=1 differ from tail6. SAME
  # node count => SAME s_fwd => the accept delta IS the tps delta (the one possibly-tps-POSITIVE dir-2 lever;
  # adding nodes is measured net-negative at 0.046 accept/node < 0.138 break-even). NON-MONOTONE (drops
  # d10/d11 => can regress on long-repeat spans) => DIAGNOSTIC: does concentrating the fixed 21-node budget
  # at the leak beat reaching deeper? fill width=max(3,2+1)=3, ranks 1,2<3 => boot-clean. Gate vs tail6.
  tail6realloc) LAUNCHER=forked; TREEARG="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0)]"; EXPECT_RATIO=21; declare -a XFLAGS=(FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 FR13_TAIL_BRANCHES=2 FR13_TAIL_BRANCH_DEPTHS=1) ;;
  # accept>5 control: cat33333 (15-node) filled PURELY from Arctic suffix decoding (FLAVOR=always -> run
  # only the root forward, Arctic fills deep spine+branches, MTP deep forwards SKIPPED). == the closed
  # Front-2 config (arctic-only deep drafter, prev -17% B=4) -- re-run on the FIXED pipeline as the control
  # isolating "arctic alone" vs tail6 (MTP-head+arctic-tail) vs t33333 (MTP-only). NO tail mode.
  # suffonly KIND REMOVED 2026-07-27: it exercised the head-merge decide_and_fill path
  # (FR13_MERGED_FLAVOR), deleted in cleanup+bake (Front-2 control; closed no-go).
  t55555)    LAUNCHER=forked; TREEARG="$T55555_TREE";   EXPECT_RATIO=25; declare -a XFLAGS=() ;;
  cat55222)  LAUNCHER=forked; TREEARG="$CAT55222_TREE";  EXPECT_RATIO=16; declare -a XFLAGS=() ;;
  cat55221)  LAUNCHER=forked; TREEARG="$CAT55221_TREE";  EXPECT_RATIO=15; declare -a XFLAGS=() ;;
  cat9f)     LAUNCHER=forked; TREEARG="[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]"; EXPECT_RATIO=9;  declare -a XFLAGS=() ;;
  *) echo "FAIL: unknown KIND=$KIND"; exit 2 ;;
esac

if [[ "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" && -z "$FIXED32_MODE" ]]; then
  echo "FAIL: FR13_FIXED32_B1_DIAGNOSTIC=1 requires a fixed32 arm"
  exit 2
fi
if [[ "$FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB" == "1" \
      && -z "$FIXED32_MODE" ]]; then
  echo "FAIL: FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=1 requires a fixed32 arm"
  exit 2
fi

if [[ -n "$FIXED32_MODE" ]]; then
  # Fixed32 must never operate on a reusable Docker name. The launcher-created
  # cidfile promotes this to the immutable container ID after a successful run.
  CONTAINER_RUNTIME_REF=""
  FIXED32_CIDFILE="$ARMDIR_ABS/logs/fr13_fixed32_container.cid"
  if [[ "${CAPTURE_ONLY:-0}" == "1" \
     || "${ACCEPT_SPEED_PROBE:-0}" == "1" \
     || "${PROBE_ONLY:-0}" == "1" ]]; then
    echo "FAIL: fixed32 campaigns forbid all positive-token probe modes"
    exit 2
  fi
  # Sidecar readers glob per-PID files. A reused arm name must not silently
  # aggregate a prior producer into this campaign.
  for _timer_path in \
    "${FR13_SFWD_GPU_TIMER_JSON:-}" \
    "${FR13_DFWD_GPU_TIMER_JSON:-}" \
    "${FR13_CFWD_GPU_TIMER_JSON:-}"; do
    [[ -n "$_timer_path" ]] || continue
    [[ "$_timer_path" == /workspace/* ]] \
      && _timer_path="$PWD/${_timer_path#/workspace/}"
    rm -f "$_timer_path" "$_timer_path".*
  done
  unset _timer_path
fi
# NATIVE_DECODE: arms whose DECODE is the native/linear MTP path (no tree), so the
# tree-flag class-9 pins + the tree-only worker environ needle DO NOT APPLY and the
# native-appropriate asserts run instead. Covers LAUNCHER=native (nativemtp5[apc], no
# patcher) AND nativemtp5_exseed (LAUNCHER=forked ONLY to run the patcher for
# EXACT_SEED, but a native FLASH_ATTN+qwen3_5_mtp config with no speculative_token_tree
# and FR10_DECODE_MODE_DEFAULT=naive_mtp). Both prove the drafter ran via the launcher-
# agnostic draft_tokens/drafts==5 spec-ratio assert, NOT via TREE_ATTN/tree env.
NATIVE_DECODE=0
if [[ "$LAUNCHER" == "native" || "$KIND" == "nativemtp5_exseed" || "$KIND" == "nativemtp11_exseed" || "$KIND" == "flash_ns8_exseed" || "$KIND" == "flash_ns5_nocache" ]]; then NATIVE_DECODE=1; fi
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
if [[ -n "$FIXED32_MODE" ]]; then
  [[ "$MAX_NUM_SEQS_OVR" == "$SWE_CONCURRENCY" ]] \
    || {
      echo "FAIL: fixed32 server capacity and SWE concurrency must match"
      exit 2
    }
  [[ "$SWE_CONCURRENCY" == "1" || "$SWE_CONCURRENCY" == "4" ]] \
    || { echo "FAIL: fixed32 concurrency must be exactly 1 or 4"; exit 2; }
  if [[ "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" ]]; then
    [[ "$MAX_NUM_SEQS_OVR" == "1" && "$SWE_CONCURRENCY" == "1" ]] \
      || {
        echo "FAIL: fixed32 B1 diagnostic requires MAX_NUM_SEQS_OVR=1 and SWE_CONCURRENCY=1"
        exit 2
      }
  fi
  if [[ "$FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB" == "1" ]]; then
    [[ "$MAX_NUM_SEQS_OVR" == "4" && "$SWE_CONCURRENCY" == "4" ]] \
      || {
        echo "FAIL: fixed32 graph byte diagnostic requires MAX_NUM_SEQS_OVR=4 and SWE_CONCURRENCY=4"
        exit 2
      }
  fi
  mapfile -t _fixed32_subset_binding < <(
    .venv/bin/python - \
      "$SUBSET" \
      "$FR13_FIXED32_B1_DIAGNOSTIC" \
      "$ARMDIR" \
      "$MAX_NUM_SEQS_OVR" \
      "$SWE_CONCURRENCY" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from fr13_floor_gate import validate_fixed32_run_subset


subset_path = Path(sys.argv[1]).resolve()
diagnostic = sys.argv[2] == "1"
arm_dir = Path(sys.argv[3]).resolve()
max_num_seqs = int(sys.argv[4])
concurrency = int(sys.argv[5])
binding = validate_fixed32_run_subset(
    subset_path,
    b1_diagnostic=diagnostic,
)
if diagnostic:
    output_path = arm_dir / "fixed32_b1_diagnostic.json"
    payload = {
        "schema": "fr13-fixed32-b1-diagnostic-v1",
        "run_classification": "b1_diagnostic",
        "gate_eligible": False,
        "floor_acceptance_eligible": False,
        "max_num_seqs": max_num_seqs,
        "swe_concurrency": concurrency,
        "subset_path": str(subset_path),
        "subset_sha256": binding["sha256"],
        "task_ids": binding["task_ids"],
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(output_path)
print(binding["task_count"])
print(",".join(binding["task_ids"]))
PY
  )
  (( $? == 0 && ${#_fixed32_subset_binding[@]} == 2 )) \
    || { echo "FAIL: fixed32 canonical task-set binding"; exit 2; }
  if [[ "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" ]]; then
    [[ "${_fixed32_subset_binding[0]}" == "1" ]] \
      || { echo "FAIL: fixed32 B1 diagnostic task count is invalid"; exit 2; }
  else
    [[ "${_fixed32_subset_binding[0]}" == "4" \
       || "${_fixed32_subset_binding[0]}" == "16" ]] \
      || { echo "FAIL: fixed32 canonical task count is invalid"; exit 2; }
  fi
  FR13_FIXED32_INGRESS_TASK_IDS=${_fixed32_subset_binding[1]}
  export FR13_FIXED32_INGRESS_TASK_IDS
  echo "fixed32 SWE-Verified subset OK: tasks=${_fixed32_subset_binding[0]} diagnostic=$FR13_FIXED32_B1_DIAGNOSTIC"
  unset _fixed32_subset_binding
fi

echo "=== BIGDENOM-VARIANT SWEServe ARM $ARM kind=$KIND launcher=$LAUNCHER expect=$EXPECT_RATIO xflags=[${XFLAGS[*]:-none}] subset=$SUBSET ==="
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_started_at.txt"
git rev-parse HEAD | tee "$ARMDIR/git_head.txt"

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

fixed32_engine_ingress_control(){
  local action=${1:?preflight, begin, or finalize}
  local output_path=${2:?output path}
  .venv/bin/python - \
    "$FIXED32_INGRESS_SECRET_FILE" \
    "$FR13_FIXED32_INGRESS_TASK_IDS" \
    "$PORT" \
    "$action" \
    "$output_path" \
    "$FR13_FIXED32_B1_DIAGNOSTIC" <<'PY'
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from pathlib import Path

import requests


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


secret_path = Path(sys.argv[1])
task_ids = sys.argv[2].split(",")
port = int(sys.argv[3])
action = sys.argv[4]
output_path = Path(sys.argv[5])
diagnostic_text = sys.argv[6]
if diagnostic_text not in {"0", "1"}:
    raise SystemExit("fixed32 B1 diagnostic selector is invalid")
diagnostic = diagnostic_text == "1"
secret_info = os.lstat(secret_path)
if (
    not stat.S_ISREG(secret_info.st_mode)
    or stat.S_IMODE(secret_info.st_mode) != 0o600
):
    raise SystemExit("fixed32 engine secret changed before control request")
with secret_path.open(encoding="ascii") as handle:
    secret = json.load(
        handle,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
if (
    not isinstance(secret, dict)
    or set(secret) != {"schema", "task_hmac_key_hex", "engine_bearer"}
    or secret.get("schema") != "fr13-fixed32-ingress-secrets-v1"
    or not isinstance(secret.get("task_hmac_key_hex"), str)
    or re.fullmatch(r"[0-9a-f]{64}", secret["task_hmac_key_hex"]) is None
    or not isinstance(secret.get("engine_bearer"), str)
    or len(secret["engine_bearer"]) < 32
    or any(ord(char) < 33 or ord(char) > 126 for char in secret["engine_bearer"])
):
    raise SystemExit("fixed32 engine secret contract mismatch")
if (
    len(task_ids) not in ((1,) if diagnostic else (4, 16))
    or len(set(task_ids)) != len(task_ids)
    or (diagnostic and task_ids != ["astropy__astropy-12907"])
    or any(
        re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", task_id) is None
        for task_id in task_ids
    )
):
    raise SystemExit(
        "fixed32 engine task set does not match its formal/diagnostic run class"
    )
canonical_task_set_sha256 = hashlib.sha256(
    json.dumps(
        sorted(task_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
base_url = f"http://127.0.0.1:{port}"
if action == "preflight":
    requests_seen = []
    wrong_bearer = "fr13_wrong_" + secrets.token_hex(32)
    for route in ("/v1/chat/completions", "/v1/responses"):
        for auth_case, headers in (
            ("missing_bearer", {}),
            ("wrong_bearer", {"Authorization": f"Bearer {wrong_bearer}"}),
        ):
            response = requests.post(
                base_url + route,
                headers=headers,
                json={},
                timeout=30,
            )
            if response.status_code != 401:
                raise SystemExit(
                    "fixed32 engine preflight did not reject "
                    f"{route} {auth_case}: HTTP {response.status_code}"
                )
            requests_seen.append(
                {
                    "route": route,
                    "auth_case": auth_case,
                    "status_code": response.status_code,
                }
            )
    denied_alternate_routes = []
    for route in ("/v1/completions", "/reset_prefix_cache"):
        response = requests.post(
            base_url + route,
            headers={"Authorization": f"Bearer {secret['engine_bearer']}"},
            json={},
            timeout=30,
        )
        if response.status_code != 403:
            raise SystemExit(
                "fixed32 engine preflight did not deny alternate POST "
                f"{route}: HTTP {response.status_code}"
            )
        denied_alternate_routes.append(
            {"method": "POST", "route": route, "status_code": 403}
        )
    bypass_seen = []
    for route, headers in (
        ("/health", {}),
        ("/metrics", {}),
        (
            "/v1/models",
            {"Authorization": f"Bearer {secret['engine_bearer']}"},
        ),
    ):
        response = requests.get(base_url + route, headers=headers, timeout=30)
        if response.status_code != 200:
            raise SystemExit(
                f"fixed32 engine preflight bypass {route} returned "
                f"HTTP {response.status_code}"
            )
        bypass_seen.append({"route": route, "status_code": response.status_code})
    payload = {
        "schema": "fr13-fixed32-ingress-auth-preflight-v1",
        "role": "engine",
        "rejected_requests": len(requests_seen),
        "accepted_requests": 0,
        "requests": requests_seen,
        "denied_alternate_routes": denied_alternate_routes,
        "non_inference_bypass": bypass_seen,
    }
elif action in {"begin", "finalize"}:
    if action == "begin":
        route = "/fr13/fixed32/ingress/begin"
        body = {
            "schema": "fr13-fixed32-ingress-begin-v1",
            "canonical_task_count": len(task_ids),
            "canonical_task_set_sha256": canonical_task_set_sha256,
        }
        expected_schema = "fr13-fixed32-engine-ingress-begin-v1"
    else:
        route = "/fr13/fixed32/ingress/finalize"
        body = {"schema": "fr13-fixed32-ingress-finalize-v1"}
        expected_schema = "fr13-fixed32-engine-ingress-finalize-v1"
    response = requests.post(
        base_url + route,
        headers={"Authorization": f"Bearer {secret['engine_bearer']}"},
        json=body,
        timeout=30,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"fixed32 engine {action} failed with HTTP {response.status_code}"
        )
    payload = json.loads(
        response.text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise SystemExit(f"fixed32 engine {action} response schema mismatch")
else:
    raise SystemExit(f"unsupported fixed32 engine control action: {action!r}")
temporary = output_path.with_name(output_path.name + ".tmp")
temporary.write_text(
    json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n",
    encoding="ascii",
)
temporary.replace(output_path)
PY
}

write_fixed32_chat_traffic_audit(){
  .venv/bin/python - \
    "$SUBSET" \
    "$ARMDIR" \
    "$FIXED32_MODE" \
    "$ARMDIR/fixed32_chat_traffic_audit.json" \
    "$FR13_FIXED32_B1_DIAGNOSTIC" \
    "$SWE_CONCURRENCY" <<'PY'
import json
import sys
from pathlib import Path


repo = Path.cwd().resolve()
sys.path.insert(0, str(repo / "scripts"))
from fr13_floor_gate import (  # noqa: E402
    build_fixed32_chat_traffic_audit,
    pinned_dataset_record_digests,
    validate_fixed32_run_subset,
)

subset_path = Path(sys.argv[1]).resolve()
arm_dir = Path(sys.argv[2]).resolve()
mode = sys.argv[3]
output_path = Path(sys.argv[4]).resolve()
diagnostic_text = sys.argv[5]
concurrency_text = sys.argv[6]
if diagnostic_text not in {"0", "1"}:
    raise SystemExit("fixed32 B1 diagnostic selector is invalid")
if concurrency_text not in {"1", "4"}:
    raise SystemExit("fixed32 concurrency selector is invalid")
subset = validate_fixed32_run_subset(
    subset_path,
    b1_diagnostic=diagnostic_text == "1",
)
task_ids = subset["task_ids"]
audit = build_fixed32_chat_traffic_audit(
    arm_dir,
    mode=mode,
    subset=subset,
    dataset_record_digests=pinned_dataset_record_digests(str(repo)),
    concurrency=int(concurrency_text),
)
temporary = output_path.with_name(output_path.name + ".tmp")
temporary.write_text(
    json.dumps(
        audit,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n",
    encoding="ascii",
)
temporary.replace(output_path)
print(
    "fixed32 authenticated chat-task audit OK: "
    f"tasks={len(task_ids)} "
    f"complete_steps={audit['complete_stream']['pure_decode_forward_steps']}"
)
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

_fixed32_container_identity_matches() {
  local container_ref=$1
  local identity=""
  local actual_id=""
  local actual_name=""
  local extra=""
  [[ "$container_ref" =~ ^[0-9a-f]{64}$ ]] || return 1
  identity=$(
    docker inspect --format '{{.Id}} {{.Name}}' "$container_ref" 2>/dev/null
  ) || return 1
  read -r actual_id actual_name extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$container_ref" ]] \
    && [[ "$actual_name" == "/$CONTAINER" ]]
}

_capture_fixed32_container_incarnation() {
  local container_ref=$1
  local identity=""
  local actual_id=""
  local actual_name=""
  local actual_status=""
  local actual_running=""
  local actual_paused=""
  local actual_started_at=""
  local actual_pid=""
  local actual_restart_count=""
  local extra=""

  [[ "$container_ref" =~ ^[0-9a-f]{64}$ ]] || return 1
  identity=$(
    docker inspect --format \
      '{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.Paused}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}}' \
      "$container_ref" 2>/dev/null
  ) || return 1
  read -r actual_id actual_name actual_status actual_running actual_paused \
    actual_started_at actual_pid actual_restart_count extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$container_ref" ]] \
    && [[ "$actual_name" == "/$CONTAINER" ]] \
    && [[ "$actual_status" == "running" ]] \
    && [[ "$actual_running" == "true" ]] \
    && [[ "$actual_paused" == "false" ]] \
    && [[ "$actual_started_at" != "0001-01-01T00:00:00Z" ]] \
    && [[ "$actual_started_at" =~ ^[^[:space:]]+$ ]] \
    && [[ "$actual_pid" =~ ^[1-9][0-9]*$ ]] \
    && [[ "$actual_restart_count" == "0" ]] \
    || return 1
  FIXED32_CONTAINER_STARTED_AT=$actual_started_at
  FIXED32_CONTAINER_INIT_PID=$actual_pid
  FIXED32_CONTAINER_RESTART_COUNT=$actual_restart_count
}

_fixed32_container_incarnation_matches() {
  local container_ref=$1
  local identity=""
  local actual_id=""
  local actual_name=""
  local actual_status=""
  local actual_running=""
  local actual_paused=""
  local actual_started_at=""
  local actual_pid=""
  local actual_restart_count=""
  local extra=""

  [[ "$container_ref" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -n "$FIXED32_CONTAINER_STARTED_AT" ]] \
    && [[ "$FIXED32_CONTAINER_INIT_PID" =~ ^[1-9][0-9]*$ ]] \
    && [[ "$FIXED32_CONTAINER_RESTART_COUNT" == "0" ]] \
    || return 1
  identity=$(
    docker inspect --format \
      '{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.Paused}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}}' \
      "$container_ref" 2>/dev/null
  ) || return 1
  read -r actual_id actual_name actual_status actual_running actual_paused \
    actual_started_at actual_pid actual_restart_count extra <<< "$identity"
  [[ -z "$extra" ]] \
    && [[ "$actual_id" == "$container_ref" ]] \
    && [[ "$actual_name" == "/$CONTAINER" ]] \
    && [[ "$actual_status" == "running" ]] \
    && [[ "$actual_running" == "true" ]] \
    && [[ "$actual_paused" == "false" ]] \
    && [[ "$actual_started_at" == "$FIXED32_CONTAINER_STARTED_AT" ]] \
    && [[ "$actual_pid" == "$FIXED32_CONTAINER_INIT_PID" ]] \
    && [[ "$actual_restart_count" == "$FIXED32_CONTAINER_RESTART_COUNT" ]]
}

_promote_fixed32_container_from_cidfile() {
  local -a container_ids=()
  local candidate=""
  [[ -f "$FIXED32_CIDFILE" && ! -L "$FIXED32_CIDFILE" ]] || return 1
  mapfile -t container_ids < "$FIXED32_CIDFILE"
  (( ${#container_ids[@]} == 1 )) \
    && [[ "${container_ids[0]}" =~ ^[0-9a-f]{64}$ ]] || return 1
  candidate=${container_ids[0]}
  _fixed32_container_identity_matches "$candidate" || return 1
  _capture_fixed32_container_incarnation "$candidate" || return 1
  CONTAINER_RUNTIME_REF=$candidate
  echo \
    "fixed32 container identity promoted: id=$CONTAINER_RUNTIME_REF name=$CONTAINER started_at=$FIXED32_CONTAINER_STARTED_AT pid=$FIXED32_CONTAINER_INIT_PID restart_count=$FIXED32_CONTAINER_RESTART_COUNT"
}

_fixed32_classify_container_state() {
  local container_ref=$1
  local identity=""
  local actual_id=""
  local actual_name=""
  local actual_status=""
  local actual_running=""
  local actual_paused=""
  local actual_started_at=""
  local actual_pid=""
  local actual_restart_count=""
  local extra=""
  local container_list=""
  local -a container_ids=()

  FIXED32_CONTAINER_CLEANUP_OUTCOME="removal_unproven"
  identity=$(
    docker inspect --format \
      '{{.Id}} {{.Name}} {{.State.Status}} {{.State.Running}} {{.State.Paused}} {{.State.StartedAt}} {{.State.Pid}} {{.RestartCount}}' \
      "$container_ref" 2>/dev/null
  )
  if (( $? == 0 )); then
    read -r actual_id actual_name actual_status actual_running actual_paused \
      actual_started_at actual_pid actual_restart_count extra <<< "$identity"
    if [[ -z "$extra" \
       && "$actual_id" == "$container_ref" \
       && "$actual_name" =~ ^/[^[:space:]]+$ \
       && "$actual_status" =~ ^[a-z]+$ \
       && "$actual_running" =~ ^(true|false)$ \
       && "$actual_paused" =~ ^(true|false)$ \
       && "$actual_started_at" =~ ^[^[:space:]]+$ \
       && "$actual_pid" =~ ^[0-9]+$ \
       && "$actual_restart_count" =~ ^[0-9]+$ ]]; then
      FIXED32_CONTAINER_CLEANUP_OUTCOME="drifted"
      if [[ "$actual_name" == "/$CONTAINER" \
         && "$actual_status" == "running" \
         && "$actual_running" == "true" \
         && "$actual_paused" == "false" \
         && "$actual_started_at" == "$FIXED32_CONTAINER_STARTED_AT" \
         && "$actual_pid" == "$FIXED32_CONTAINER_INIT_PID" \
         && "$actual_restart_count" == "$FIXED32_CONTAINER_RESTART_COUNT" ]]; then
        FIXED32_CONTAINER_CLEANUP_OUTCOME="exact_preserved"
      fi
    fi
    return 0
  fi
  container_list=$(docker ps -aq --no-trunc 2>/dev/null)
  if (( $? != 0 )); then
    return 0
  fi
  [[ -n "$container_list" ]] && mapfile -t container_ids <<< "$container_list"
  for actual_id in "${container_ids[@]}"; do
    [[ "$actual_id" =~ ^[0-9a-f]{64}$ ]] || return 0
  done
  if printf '%s\n' "${container_ids[@]}" | grep -Fxq "$container_ref"; then
    FIXED32_CONTAINER_CLEANUP_OUTCOME="removal_unproven"
  else
    FIXED32_CONTAINER_CLEANUP_OUTCOME="absent"
  fi
}

_fixed32_record_container_cleanup_failure() {
  local container_ref=$1
  local context=$2
  local message=""

  case "$FIXED32_CONTAINER_CLEANUP_OUTCOME" in
    exact_preserved)
      message="fixed32 exact container preserved after $context: $container_ref"
      printf '%s\n' "$message" \
        | tee "$ARMDIR/fixed32_container_preserved.txt" >&2
      ;;
    absent)
      message="fixed32 container absent after $context: $container_ref"
      printf '%s\n' "$message" \
        | tee "$ARMDIR/fixed32_container_cleanup_failure.txt" >&2
      ;;
    drifted)
      message="fixed32 container incarnation drifted after $context: $container_ref"
      printf '%s\n' "$message" \
        | tee "$ARMDIR/fixed32_container_cleanup_failure.txt" >&2
      ;;
    *)
      FIXED32_CONTAINER_CLEANUP_OUTCOME="removal_unproven"
      message="fixed32 container removal/preservation unproven after $context: $container_ref"
      printf '%s\n' "$message" \
        | tee "$ARMDIR/fixed32_container_cleanup_failure.txt" >&2
      ;;
  esac
}

_fixed32_remove_attested_container() {
  local container_ref=$1
  local log_output=$2
  FIXED32_CONTAINER_CLEANUP_OUTCOME="removal_unproven"
  if ! _fixed32_container_incarnation_matches "$container_ref"; then
    echo "fixed32 container logs/removal skipped without exact incarnation re-attestation" >&2
    _fixed32_classify_container_state "$container_ref"
    return 1
  fi
  docker logs "$container_ref" > "$log_output" 2>&1 || true
  if ! _fixed32_container_incarnation_matches "$container_ref"; then
    echo "fixed32 container removal skipped after incarnation drift" >&2
    _fixed32_classify_container_state "$container_ref"
    return 1
  fi
  if docker rm -f "$container_ref" >/dev/null 2>&1; then
    FIXED32_CONTAINER_CLEANUP_OUTCOME="removed"
    return 0
  fi
  _fixed32_classify_container_state "$container_ref"
  echo \
    "fixed32 container removal failed: outcome=$FIXED32_CONTAINER_CLEANUP_OUTCOME" \
    >&2
  return 1
}

_fixed32_snapshot_engine_ingress_ledger() {
  local container_ref=$1
  local destination="$ARMDIR_ABS/logs/fr13_fixed32_engine_ingress.jsonl"
  local snapshot_dir=""
  local snapshot_tmp=""
  local expected_owner=""

  if ! _fixed32_container_incarnation_matches "$container_ref"; then
    echo "fixed32 engine-ledger snapshot refused without exact incarnation re-attestation" >&2
    return 1
  fi
  if [[ ! -d "$ARMDIR_ABS" || -L "$ARMDIR_ABS" \
     || ! -d "$ARMDIR_ABS/logs" || -L "$ARMDIR_ABS/logs" \
     || ! -f "$destination" || -L "$destination" \
     || ! -s "$ARMDIR_ABS/fixed32_engine_ingress_begin.json" \
     || -L "$ARMDIR_ABS/fixed32_engine_ingress_begin.json" \
     || ! -s "$ARMDIR_ABS/fixed32_engine_ingress_finalize.json" \
     || -L "$ARMDIR_ABS/fixed32_engine_ingress_finalize.json" ]]; then
    echo "fixed32 engine-ledger snapshot destination is not a regular run artifact" >&2
    return 1
  fi
  expected_owner=$(id -u) || {
    echo "fixed32 engine-ledger snapshot could not identify the host owner" >&2
    return 1
  }
  snapshot_dir=$(
    umask 077
    mktemp -d "$ARMDIR_ABS/.fixed32_engine_ingress_snapshot.XXXXXXXX"
  ) || {
    echo "fixed32 engine-ledger private snapshot directory could not be created" >&2
    return 1
  }
  if [[ ! -d "$snapshot_dir" || -L "$snapshot_dir" ]] \
      || [[ "$(stat -c '%u:%a' "$snapshot_dir" 2>/dev/null)" \
        != "$expected_owner:700" ]]; then
    echo "fixed32 engine-ledger snapshot directory is not private and host-owned" >&2
    [[ -d "$snapshot_dir" && ! -L "$snapshot_dir" ]] && rmdir -- "$snapshot_dir"
    return 1
  fi
  snapshot_tmp="$snapshot_dir/fr13_fixed32_engine_ingress.jsonl"
  if ! docker cp \
      "${container_ref}:/logs/fr13_fixed32_engine_ingress.jsonl" \
      "$snapshot_tmp"; then
    echo "fixed32 engine-ledger snapshot docker cp failed" >&2
    rm -f -- "$snapshot_tmp" 2>/dev/null || true
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if ! _fixed32_container_incarnation_matches "$container_ref"; then
    echo "fixed32 engine-ledger container incarnation drifted during snapshot" >&2
    rm -f -- "$snapshot_tmp" 2>/dev/null || true
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if [[ ! -f "$snapshot_tmp" || -L "$snapshot_tmp" \
     || ! -s "$snapshot_tmp" || ! -r "$snapshot_tmp" ]] \
      || [[ "$(stat -c '%u:%h' "$snapshot_tmp" 2>/dev/null)" \
        != "$expected_owner:1" ]]; then
    echo "fixed32 copied engine ledger is not a host-owned readable nonempty regular file" >&2
    rm -f -- "$snapshot_tmp" 2>/dev/null || true
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if ! chmod 600 "$snapshot_tmp"; then
    echo "fixed32 engine-ledger snapshot privacy mode could not be locked" >&2
    rm -f -- "$snapshot_tmp"
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if ! .venv/bin/python - \
      "$snapshot_tmp" "$destination" "$SUBSET" "$ARMDIR_ABS" \
      "$expected_owner" "${FR13_FIXED32_B1_DIAGNOSTIC:-0}" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path


repo = Path.cwd().resolve()
sys.path.insert(0, str(repo / "scripts"))
from fr13_floor_gate import (  # noqa: E402
    _fixed32_ingress_task_counts,
    _validate_fixed32_ingress_reports,
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    load_fixed32_ingress_ledger,
    validate_fixed32_run_subset,
)

snapshot_path = Path(sys.argv[1])
destination = Path(sys.argv[2])
subset_path = Path(sys.argv[3]).resolve()
arm_dir = Path(sys.argv[4]).resolve()
expected_owner = int(sys.argv[5])
diagnostic_text = sys.argv[6]
if diagnostic_text not in {"0", "1"}:
    raise SystemExit("fixed32 B1 diagnostic selector is invalid")
if destination.parent != arm_dir / "logs":
    raise SystemExit("fixed32 engine-ledger destination escaped the arm logs directory")


class ImmutableLedgerPath:
    def __init__(self, raw, label):
        self._raw = raw
        self._label = label

    def is_file(self):
        return True

    def is_symlink(self):
        return False

    def read_bytes(self):
        return self._raw

    def __str__(self):
        return self._label


def read_fd(fd):
    chunks = []
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def file_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_nlink,
        info.st_size,
        info.st_uid,
        stat.S_IMODE(info.st_mode),
    )


def validate_ledger_bytes(raw, label):
    immutable_path = ImmutableLedgerPath(raw, label)
    rows, ledger_identity = load_fixed32_ingress_ledger(
        immutable_path,
        role="engine",
        canonical_task_keys=canonical_task_keys,
        canonical_task_set_sha256=canonical_task_set_sha256,
    )
    task_counts = _fixed32_ingress_task_counts(
        rows,
        role="engine",
        canonical_task_keys=canonical_task_keys,
    )
    _validate_fixed32_ingress_reports(
        arm_dir,
        role="engine",
        task_ids=task_ids,
        task_counts=task_counts,
        rows=rows,
        ledger_identity=ledger_identity,
        canonical_task_set_sha256=canonical_task_set_sha256,
    )
    return hashlib.sha256(raw).hexdigest()


subset = validate_fixed32_run_subset(
    subset_path,
    b1_diagnostic=diagnostic_text == "1",
)
task_ids = subset["task_ids"]
canonical_task_keys = {fixed32_task_key_id(task_id) for task_id in task_ids}
canonical_task_set_sha256 = fixed32_canonical_task_set_sha256(task_ids)
snapshot_parent_fd = os.open(
    snapshot_path.parent,
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
)
logs_fd = os.open(
    destination.parent,
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
)
try:
    parent_info = os.fstat(snapshot_parent_fd)
    logs_info = os.fstat(logs_fd)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent_info.st_uid != expected_owner
        or not stat.S_ISDIR(logs_info.st_mode)
        or logs_info.st_uid != expected_owner
        or stat.S_IMODE(logs_info.st_mode) & 0o022
    ):
        raise SystemExit("fixed32 engine-ledger publication directories are unsafe")
    snapshot_fd = os.open(
        snapshot_path.name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=snapshot_parent_fd,
    )
    try:
        snapshot_info = os.fstat(snapshot_fd)
        if (
            not stat.S_ISREG(snapshot_info.st_mode)
            or stat.S_IMODE(snapshot_info.st_mode) != 0o600
            or snapshot_info.st_uid != expected_owner
            or snapshot_info.st_nlink != 1
            or snapshot_info.st_size <= 0
        ):
            raise SystemExit("fixed32 copied engine ledger identity is unsafe")
        source_identity = file_identity(snapshot_info)
        validated_raw = read_fd(snapshot_fd)
        if (
            len(validated_raw) != snapshot_info.st_size
            or file_identity(os.fstat(snapshot_fd)) != source_identity
        ):
            raise SystemExit("fixed32 copied engine ledger changed during capture")
        validated_sha256 = validate_ledger_bytes(
            validated_raw,
            str(snapshot_path),
        )

        destination_info = os.stat(
            destination.name,
            dir_fd=logs_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(destination_info.st_mode)
            or destination_info.st_nlink != 1
            or destination_info.st_size <= 0
        ):
            raise SystemExit("fixed32 original engine ledger identity is unsafe")

        source_before_publish = os.fstat(snapshot_fd)
        if (
            file_identity(source_before_publish) != source_identity
            or hashlib.sha256(read_fd(snapshot_fd)).hexdigest()
            != validated_sha256
        ):
            raise SystemExit(
                "fixed32 validated engine ledger bytes changed before publication"
            )
        os.replace(
            snapshot_path.name,
            destination.name,
            src_dir_fd=snapshot_parent_fd,
            dst_dir_fd=logs_fd,
        )
        os.fsync(logs_fd)
        published_fd = os.open(
            destination.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=logs_fd,
        )
        try:
            published_info = os.fstat(published_fd)
            if file_identity(published_info) != source_identity:
                raise SystemExit("fixed32 published engine ledger identity changed")
            published_raw = read_fd(published_fd)
            if hashlib.sha256(published_raw).hexdigest() != validated_sha256:
                raise SystemExit("fixed32 published engine ledger digest changed")
            published_sha256 = validate_ledger_bytes(
                published_raw,
                str(destination),
            )
            if (
                published_sha256 != validated_sha256
                or hashlib.sha256(read_fd(published_fd)).hexdigest()
                != validated_sha256
            ):
                raise SystemExit(
                    "fixed32 published engine ledger changed during revalidation"
                )
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
    finally:
        os.close(snapshot_fd)
finally:
    os.close(logs_fd)
    os.close(snapshot_parent_fd)
PY
  then
    echo "fixed32 finalized engine-ledger validation/publication failed" >&2
    rm -f -- "$snapshot_tmp" 2>/dev/null || true
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if ! rmdir -- "$snapshot_dir"; then
    echo "fixed32 engine-ledger private snapshot directory did not empty" >&2
    rm -f -- "$snapshot_tmp"
    rmdir -- "$snapshot_dir" 2>/dev/null || true
    return 1
  fi
  if ! _fixed32_container_incarnation_matches "$container_ref"; then
    echo "fixed32 engine-ledger container incarnation drifted during publication" >&2
    return 1
  fi
  echo "fixed32 terminal engine ledger snapshotted from immutable container"
}

teardown(){
  local rc=$?
  local flush_rc=0
  local ledger_snapshot_rc=0
  local audit_rc=0
  local container_cleanup_rc=0
  local fixed32_container_attested=1
  trap - EXIT
  set +e
  echo "[teardown] kill proxy + attest/remove run container + recover_host_memory"
  # Stop every ingress path before asking EngineCore for its terminal snapshot.
  kill "$(cat /tmp/track_b_e2e_proxy_${PROXY_PORT}.pid 2>/dev/null)" 2>/dev/null
  pkill -f "lumo_flywheel_serving.inference_proxy" 2>/dev/null
  [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null
  if [[ "$OFFLOAD_AGENT" == "1" ]]; then
    LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" stop "$OFFLOAD_HOST" >> "$ARMDIR/offload_teardown.log" 2>&1 || true
  fi
  if [[ -n "$FIXED32_MODE" ]]; then
    if [[ -z "$CONTAINER_RUNTIME_REF" ]]; then
      _promote_fixed32_container_from_cidfile >/dev/null 2>&1 || true
    fi
    if ! _fixed32_container_identity_matches "$CONTAINER_RUNTIME_REF" \
        || ! _fixed32_container_incarnation_matches "$CONTAINER_RUNTIME_REF"; then
      fixed32_container_attested=0
      echo "fixed32 teardown skipped container operations: immutable incarnation attestation failed" \
        > "$ARMDIR/fixed32_final_flush.stderr"
    elif [[ -n "$FIXED32_PRODUCER_PID" && -f "$FIXED32_ACK_PATH" ]]; then
      .venv/bin/python scripts/fr13_fixed32_flush_protocol.py \
        --container "$CONTAINER_RUNTIME_REF" \
        --producer-pid "$FIXED32_PRODUCER_PID" \
        --mode "$FIXED32_MODE" \
        --request "$FIXED32_REQUEST_PATH" \
        --ack "$FIXED32_ACK_PATH" \
        --action final \
        > "$ARMDIR/fixed32_final_flush.json" \
        2> "$ARMDIR/fixed32_final_flush.stderr"
      flush_rc=$?
    else
      echo "fixed32 final flush missing PID or ready ack" \
        > "$ARMDIR/fixed32_final_flush.stderr"
      flush_rc=1
    fi
    (( fixed32_container_attested == 1 )) || flush_rc=1
    if (( flush_rc == 0 && fixed32_container_attested == 1 )); then
      _fixed32_snapshot_engine_ingress_ledger "$CONTAINER_RUNTIME_REF" \
        > "$ARMDIR/fixed32_engine_ingress_snapshot.log" 2>&1
      ledger_snapshot_rc=$?
    else
      ledger_snapshot_rc=1
    fi
    if (( flush_rc != 0 )); then
      echo "FAIL: fixed32 terminal flush rc=$flush_rc" >&2
      (( rc == 0 )) && rc=13
    elif (( ledger_snapshot_rc != 0 )); then
      echo "FAIL: fixed32 terminal engine-ledger snapshot rc=$ledger_snapshot_rc" >&2
      tail -20 "$ARMDIR/fixed32_engine_ingress_snapshot.log" >&2 || true
      (( rc == 0 )) && rc=16
    else
      write_fixed32_chat_traffic_audit \
        > "$ARMDIR/fixed32_chat_traffic_audit.log" 2>&1
      audit_rc=$?
      if (( audit_rc != 0 )); then
        echo "FAIL: fixed32 chat-task traffic audit rc=$audit_rc" >&2
        tail -20 "$ARMDIR/fixed32_chat_traffic_audit.log" >&2 || true
        (( rc == 0 )) && rc=16
      fi
    fi
  fi
  if [[ -n "$FIXED32_MODE" ]]; then
    if (( ledger_snapshot_rc == 0 )); then
      if [[ -n "$CONTAINER_RUNTIME_REF" ]]; then
        _fixed32_remove_attested_container \
          "$CONTAINER_RUNTIME_REF" "$ARMDIR/docker_full.log"
        container_cleanup_rc=$?
      else
        container_cleanup_rc=1
      fi
      if (( container_cleanup_rc != 0 )); then
        _fixed32_record_container_cleanup_failure \
          "${CONTAINER_RUNTIME_REF:-unavailable}" \
          "cleanup attestation/removal failure"
        (( rc == 0 )) && rc=17
      fi
    else
      if [[ -n "$CONTAINER_RUNTIME_REF" ]]; then
        _fixed32_classify_container_state "$CONTAINER_RUNTIME_REF"
      else
        FIXED32_CONTAINER_CLEANUP_OUTCOME="removal_unproven"
      fi
      _fixed32_record_container_cleanup_failure \
        "${CONTAINER_RUNTIME_REF:-unavailable}" \
        "engine-ledger materialization failure"
    fi
  elif [[ -n "$CONTAINER_RUNTIME_REF" ]]; then
    docker logs "$CONTAINER_RUNTIME_REF" > "$ARMDIR/docker_full.log" 2>&1 || true
    docker rm -f "$CONTAINER_RUNTIME_REF" >/dev/null 2>&1 || true
  fi
  [[ -n "$FIXED32_INGRESS_SECRET_FILE" ]] \
    && rm -f "$FIXED32_INGRESS_SECRET_FILE"
  recover_host || true
  sleep 2
  free -g | tee "$ARMDIR/free_after_teardown.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$ARMDIR/arm_ended_at.txt"
  exit "$rc"
}
trap teardown EXIT

if [[ -n "$FIXED32_MODE" ]]; then
  FIXED32_INGRESS_SECRET_FILE=$(mktemp /tmp/fr13_fixed32_ingress.XXXXXXXX)
  chmod 600 "$FIXED32_INGRESS_SECRET_FILE"
  .venv/bin/python - "$FIXED32_INGRESS_SECRET_FILE" <<'PY'
import json
import os
import secrets
import sys

path = sys.argv[1]
payload = {
    "schema": "fr13-fixed32-ingress-secrets-v1",
    "task_hmac_key_hex": secrets.token_hex(32),
    "engine_bearer": "fr13_engine_" + secrets.token_hex(32),
}
encoded = (
    json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n"
).encode("ascii")
descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
try:
    offset = 0
    while offset < len(encoded):
        written = os.write(descriptor, encoded[offset:])
        if written <= 0:
            raise OSError("short write of fixed32 ingress secret")
        offset += written
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
  (( $? == 0 )) \
    || { echo "FAIL: fixed32 ingress secret generation"; exit 2; }
  export FR13_FIXED32_INGRESS_SECRET_FILE="$FIXED32_INGRESS_SECRET_FILE"
fi

# ---- boot server (class 11: everything pinned except the arm lever/shape) ----
# extra flags exported into THIS shell so the launcher's docker -e picks them up.
for kv in "${XFLAGS[@]:-}"; do [[ -n "$kv" ]] && export "$kv"; done
case "${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" in
  0) ;;
  1)
    [[ -n "$FIXED32_MODE" && "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" ]] \
      || {
        echo "FAIL: fixed32 TAW native real-task arm is B1 diagnostic only"
        exit 2
      }
    ;;
  *)
    echo "FAIL: FR13_FIXED32_TAW_NATIVE_PRECOMPUTE must be exactly 0 or 1"
    exit 2
    ;;
esac
case "${FR13_DFWD_UNIFIED_BM8_LIVE_AB:-0}" in
  0) ;;
  1)
    [[ -n "$FIXED32_MODE" && "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" ]] \
      || {
        echo "FAIL: fixed32 BM8 real-task arm is B1 diagnostic only"
        exit 2
      }
    [[ "${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" == "0" ]] \
      || {
        echo "FAIL: fixed32 BM8 and TAW real-task diagnostics are exclusive"
        exit 2
      }
    ;;
  *)
    echo "FAIL: FR13_DFWD_UNIFIED_BM8_LIVE_AB must be exactly 0 or 1"
    exit 2
    ;;
esac
if [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_coop128_byte_ab" \
      || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_force_wide256_byte_ab" ]]; then
  [[ -n "$FIXED32_MODE" && "$FR13_FIXED32_B1_DIAGNOSTIC" == "1" ]] \
    || {
      echo "FAIL: fixed32 CUTLASS Stream-K real-task arm is B1 diagnostic only"
      exit 2
    }
  [[ "${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" == "0" \
     && "${FR13_DFWD_UNIFIED_BM8_LIVE_AB:-0}" == "0" ]] \
    || {
      echo "FAIL: fixed32 CUTLASS, TAW, and BM8 real-task diagnostics are exclusive"
      exit 2
    }
fi
if [[ "$LAUNCHER" == "locked" ]]; then
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR13_RUN_DIR="$ARMDIR_ABS" LOG_DIR="$ARMDIR_ABS/logs" \
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
  FR13_RUN_DIR="$ARMDIR_ABS" LOG_DIR="$ARMDIR_ABS/logs" \
  scripts/fr13_launch_native_mtp_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
else
  # Reshape arms boot the LOCKED cat9 pipeline flags (FIX-1/2/3/A, REPLAY_ROUTE,
  # FA2_TREE_BIAS, CONV_COMMITTED_PATH) + the BAKED in_proj_ba pad
  # (LUMO_FB_KERNEL_ROWS=1, LUMO_FB_PROJ_PAD_ROWS=16) so the only difference vs
  # cat9 is the TREE shape. The forked launcher now source-defaults
  # FR13_REQUIRED_TREE_FLAGS itself (fr13_required_tree_flags.sh); no need to
  # pin them again here -- pinning was the pre-2026-07-22 workaround for the gap
  # where those flags silently defaulted OFF. Exported (not spliced into the
  # env-prefix chain below -- bash's prefix-assignment parsing needs LITERAL
  # NAME=value words at parse time, so "${arr[@]}" there gets misparsed as the
  # command itself; caught live 2026-07-22 when it broke both native control
  # arms with "FR13_ATTN_KV_REMAP=1: command not found", rc=2, before boot).
  for _fr13_req in "${FR13_REQUIRED_TREE_FLAGS[@]}"; do
    _fr13_req_k="${_fr13_req%%=*}"; _fr13_req_v="${_fr13_req#*=}"
    export "${_fr13_req_k}=${!_fr13_req_k:-$_fr13_req_v}"
  done
  unset _fr13_req _fr13_req_k _fr13_req_v
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL="${GPU_UTIL:-0.78}" MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  TREE="$TREEARG" FR10_METRICS="${FR10_METRICS:-0}" BATCH_INVARIANT="${BATCH_INVARIANT:-0}" \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_RUN_DIR="$ARMDIR_ABS" LOG_DIR="$ARMDIR_ABS/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
  RC=$?
fi
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -30 "$ARMDIR/launch.log"; exit 2; fi
if [[ -n "$FIXED32_MODE" ]]; then
  _promote_fixed32_container_from_cidfile \
    || { echo "FAIL: fixed32 launcher cidfile identity mismatch"; exit 2; }
fi

T0=$(date +%s)
HEALTHY=0
# HEALTH_TIMEOUT_S overridable: first boot of a NEW tree geometry pays a
# one-time Triton compile (tail6_pb 29-col/N_PAD=32 blew the 1200s window).
while (( $(date +%s) < T0 + ${HEALTH_TIMEOUT_S:-1200} )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER_RUNTIME_REF" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health"; docker logs "$CONTAINER_RUNTIME_REF" 2>&1 | tail -40; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL: health not up in 1200s"; docker logs "$CONTAINER_RUNTIME_REF" 2>&1 | tail -40; exit 2; }
echo "healthy after $(( $(date +%s) - T0 ))s"

if [[ -n "$FIXED32_MODE" ]]; then
  mapfile -t _fixed32_pid_lines < <(sed '/^[[:space:]]*$/d' "$FIXED32_PID_PATH")
  (( ${#_fixed32_pid_lines[@]} == 1 )) \
    || { echo "FAIL: fixed32 EngineCore PID file must contain exactly one line"; exit 3; }
  FIXED32_PRODUCER_PID=${_fixed32_pid_lines[0]}
  [[ "$FIXED32_PRODUCER_PID" =~ ^[1-9][0-9]*$ ]] \
    || { echo "FAIL: invalid fixed32 EngineCore PID: $FIXED32_PRODUCER_PID"; exit 3; }
  unset _fixed32_pid_lines
  docker exec "$CONTAINER_RUNTIME_REF" test -r "/proc/$FIXED32_PRODUCER_PID/cmdline" \
    || { echo "FAIL: fixed32 EngineCore PID is not live"; exit 3; }
  docker exec "$CONTAINER_RUNTIME_REF" sh -c \
    "tr '\\0' ' ' < /proc/$FIXED32_PRODUCER_PID/cmdline" \
    > "$ARMDIR/fixed32_engine_cmdline.txt"
  grep -q "EngineCore" "$ARMDIR/fixed32_engine_cmdline.txt" \
    || { echo "FAIL: fixed32 producer PID is not EngineCore"; exit 3; }
  # docker exec does not attach stdin unless -i is explicit. Without it,
  # `python3 -` exits zero after reading an empty stream and leaves an empty
  # identity artifact. Publish only a complete capture so the profiler cannot
  # bind an Nsight session from a partially written process identity.
  PROCESS_IDENTITY_TMP="$ARMDIR/.fixed32_process_identity.json.tmp.$$"
  rm -f "$PROCESS_IDENTITY_TMP"
  docker exec -i "$CONTAINER_RUNTIME_REF" python3 - "$FIXED32_PRODUCER_PID" \
    > "$PROCESS_IDENTITY_TMP" <<'PY'
import json
import sys
from pathlib import Path


def process_record(pid):
    root = Path("/proc") / str(pid)
    argv = [
        value.decode("utf-8", errors="strict")
        for value in root.joinpath("cmdline").read_bytes().split(b"\0")
        if value
    ]
    environ = [
        value.decode("utf-8", errors="strict")
        for value in root.joinpath("environ").read_bytes().split(b"\0")
        if value
    ]
    if any("=" not in value for value in environ):
        raise SystemExit(f"PID {pid} contains a malformed environment entry")
    names = [value.split("=", 1)[0] for value in environ]
    if len(names) != len(set(names)):
        raise SystemExit(f"PID {pid} contains duplicate environment names")
    maps = [
        line
        for line in root.joinpath("maps").read_text().splitlines()
        if "_vllm_fa2_C.abi3.so" in line
    ]
    return {
        "pid": int(pid),
        "argv": argv,
        "environ": sorted(environ),
        "forked_fa2_maps": maps,
    }


engine_pid = int(sys.argv[1])
payload = {
    "schema": "fr13-fixed32-process-identity-v1",
    "pid1": process_record(1),
    "engine_core": process_record(engine_pid),
}
print(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
PY
  if (( $? != 0 )); then
    rm -f "$PROCESS_IDENTITY_TMP"
    echo "FAIL: fixed32 process identity capture"
    exit 3
  fi
  if [[ ! -s "$PROCESS_IDENTITY_TMP" || -L "$PROCESS_IDENTITY_TMP" ]]; then
    rm -f "$PROCESS_IDENTITY_TMP"
    echo "FAIL: fixed32 process identity capture was empty or aliased"
    exit 3
  fi
  mv -T "$PROCESS_IDENTITY_TMP" "$ARMDIR/fixed32_process_identity.json" \
    || {
      rm -f "$PROCESS_IDENTITY_TMP"
      echo "FAIL: fixed32 process identity publish"
      exit 3
    }
  .venv/bin/python - \
    "$ARMDIR/logs/fr13_fixed32_runtime_attestation.json" \
    "$ARMDIR/fixed32_process_identity.json" \
    "$MAX_NUM_SEQS_OVR" \
    "$CONTAINER_RUNTIME_REF" \
    "$CONTAINER" \
    "$ARMDIR/fixed32_container_identity.json" \
    "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" \
    "${FR13_FIXED32_BATCH_GDN_BYTE_AB:-0}" \
    "${FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB:-0}" \
    "${FR13_FIXED32_CUTLASS_WAVE:-stock}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_fixed32_contract as contract

runtime_path = Path(sys.argv[1])
process_path = Path(sys.argv[2])
concurrency = int(sys.argv[3])
container_ref = sys.argv[4]
container_name = sys.argv[5]
container_identity_path = Path(sys.argv[6])
attribution_only_text = sys.argv[7]
batch_gdn_byte_ab_text = sys.argv[8]
batch_gdn_graph_byte_ab_text = sys.argv[9]
cutlass_wave = sys.argv[10]
runtime = contract.validate_runtime_attestation(
    json.loads(runtime_path.read_text(encoding="utf-8"))
)
process = json.loads(process_path.read_text(encoding="utf-8"))
if process.get("schema") != "fr13-fixed32-process-identity-v1":
    raise SystemExit("fixed32 process identity schema mismatch")
pid1 = process.get("pid1")
engine = process.get("engine_core")
if not isinstance(pid1, dict) or pid1.get("pid") != 1:
    raise SystemExit("fixed32 PID1 identity is malformed")
if attribution_only_text not in {"0", "1"}:
    raise SystemExit(
        "fixed32 attribution-only selector must be exactly 0 or 1"
    )
if batch_gdn_byte_ab_text not in {"0", "1"}:
    raise SystemExit(
        "fixed32 batch-GDN byte diagnostic selector must be exactly 0 or 1"
    )
if batch_gdn_graph_byte_ab_text not in {"0", "1"}:
    raise SystemExit(
        "fixed32 batch-GDN graph byte diagnostic selector must be exactly 0 or 1"
    )
if cutlass_wave not in {
    "stock",
    "streamk_coop128_byte_ab",
    "streamk_coop128",
    "streamk_force_wide256_byte_ab",
    "streamk_force_wide256",
}:
    raise SystemExit("fixed32 CUTLASS wave selector is invalid")
try:
    contract.validate_process_pid1_argv(
        pid1.get("argv"),
        concurrency,
        attribution_only=attribution_only_text == "1",
        eager_diagnostic=batch_gdn_byte_ab_text == "1",
        graph_diagnostic=batch_gdn_graph_byte_ab_text == "1",
        streamk_eager_diagnostic=(
            cutlass_wave
            in ("streamk_coop128_byte_ab", "streamk_force_wide256_byte_ab")
        ),
    )
except contract.ContractError as error:
    raise SystemExit(str(error)) from error
if not isinstance(engine, dict):
    raise SystemExit("fixed32 EngineCore identity is malformed")
fa2_path = str(contract.CONTAINER_FA2_DESTINATION)
if not any(fa2_path in line for line in engine.get("forked_fa2_maps", [])):
    raise SystemExit("fixed32 EngineCore has not mapped the pinned FA2 binary")
inspect = subprocess.run(
    ["docker", "inspect", container_ref],
    check=True,
    capture_output=True,
    text=True,
)
rows = json.loads(inspect.stdout)
if len(rows) != 1 or not isinstance(rows[0], dict):
    raise SystemExit("fixed32 container inspect did not return one object")
row = rows[0]
container_identity = {
    "schema": "fr13-fixed32-container-identity-v1",
    "name": row.get("Name"),
    "image_id": row.get("Image"),
    "configured_image": (row.get("Config") or {}).get("Image"),
    "platform": row.get("Platform"),
    "running": (row.get("State") or {}).get("Running"),
}
expected_container_identity = {
    "schema": "fr13-fixed32-container-identity-v1",
    "name": f"/{container_name}",
    "image_id": contract.IMAGE_ID,
    "configured_image": contract.IMAGE_REFERENCE,
    "platform": contract.IMAGE_OS,
    "running": True,
}
if container_identity != expected_container_identity:
    raise SystemExit(
        "fixed32 running container identity mismatch: "
        f"{container_identity!r}"
    )
contract.atomic_write_json(container_identity_path, container_identity)
print(
    "fixed32 runtime identity OK: "
    f"runtime={runtime['overall_canonical_sha256']} "
    f"pid1_args={len(pid1['argv'])}"
)
PY
  (( $? == 0 )) || { echo "FAIL: fixed32 runtime/process attestation"; exit 3; }
  .venv/bin/python - \
    "$CONTAINER_RUNTIME_REF" "$FIXED32_PRODUCER_PID" "$FIXED32_MODE" \
    "$FIXED32_REQUEST_PATH" "$FIXED32_ACK_PATH" \
    > "$ARMDIR/fixed32_ready_ack.json" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from fr13_fixed32_flush_protocol import Fixed32FlushClient

container, pid, mode, request, ack = sys.argv[1:]
client = Fixed32FlushClient(
    container=container,
    producer_pid=int(pid),
    mode=mode,
    request_path=Path(request),
    ack_path=Path(ack),
)
print(json.dumps(client.connect().as_dict(), sort_keys=True))
PY
  (( $? == 0 )) || { echo "FAIL: fixed32 generation-zero ready ack"; exit 3; }
  echo "fixed32 runtime ready: mode=$FIXED32_MODE producer_pid=$FIXED32_PRODUCER_PID"
fi

# ---- class 9: flag state in container env ----
# The FR13 tree/APC machinery is DELIBERATELY ABSENT on the native MTP path
# (LAUNCHER=native = the fr9 decode path). So for nativemtp5 the tree-flag pins +
# the FR10_DECODE_MODE_DEFAULT worker needle DO NOT APPLY; we assert the stock
# native config instead (the spec method + no tree env leaked in). All other
# (locked/forked) arms keep the identical class-9 gate.
docker exec "$CONTAINER_RUNTIME_REF" env | sort > "$ARMDIR/container_env.txt"
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
  if [[ "$KIND" == "nativemtp5_exseed" || "$KIND" == "nativemtp11_exseed" ]]; then
    grep -q "^ATTENTION_BACKEND=FLASH_ATTN$" "$ARMDIR/container_env.txt" \
      || { echo "FAIL: $KIND ATTENTION_BACKEND != FLASH_ATTN (native kernel)"; exit 3; }
    grep -q "^FR13_APC_EXACT_SEED=0$" "$ARMDIR/container_env.txt" \
      || { echo "FAIL: FR13_APC_EXACT_SEED must be DEPRECATED-OFF (=0) — leaking es_ckpt store"; exit 3; }
  fi
  echo "container env OK (native MTP: qwen3_5_mtp, no tree env; KIND=$KIND)"
else
  NEEDS=("FR13_DRAFTER_SINGLE_LOGITS=1" "FR13_EAGER_PACK=1" "FR13_TREE_CONV_FUSED=1" \
         "VLLM_BATCH_INVARIANT=0" "LUMO_BATCH_INVARIANT_VLLM=0" \
         "FR13_REPLAY_ROUTE=1" "FR13_FA2_TREE_BIAS=1" "FR13_CONV_COMMITTED_PATH=1" \
         "FR10_DECODE_MODE_DEFAULT=tree_mtp" \
         "LUMO_FB_KERNEL_ROWS=1" "LUMO_FB_PROJ_PAD_ROWS=16" \
         "${FR13_REQUIRED_TREE_FLAGS[@]}")
  # ^ FR13_REQUIRED_TREE_FLAGS (fr13_required_tree_flags.sh): proven+"BAKED" fixes
  # (garble + accept M-dependence) that silently sat OFF for every tree-kind
  # campaign through this script until 2026-07-22 — now source-defaulted ON in the
  # forked launcher AND asserted here from the SAME registry, so a future default
  # regression (e.g. someone re-adding an explicit =0 override upstream, or a new
  # required flag never getting added to the assertion) fails loud instead of
  # silently degrading. Add new required flags to the registry file ONLY.
  if grep -q "^FR13_SCAN_ALIGN=1$" "$ARMDIR/container_env.txt" && [ "${FR13_ALLOW_SCAN_ALIGN:-0}" != "1" ]; then
    echo "FAIL: FR13_SCAN_ALIGN=1 present — K1 must NOT be baked (set FR13_ALLOW_SCAN_ALIGN=1 for the recompute diagnostic)"; exit 3
  fi
  # FR13_NEEDS_ALLOW (2026-07-27, sanctioned diagnostic override): a
  # space-separated list of KEY=VALUE pairs that REPLACE the registry
  # expectation for THIS arm only — so experiment arms that must flip one
  # pinned flag (e.g. FR13_SUBTREE_PARALLEL=0 for the capture-topology A/B)
  # run through the FULL live-SWE pipeline instead of fleeing to probe
  # boots (user call: even kernel measurements use SWE tasks). LOUD: every
  # override is printed; drift protection stays for everything unlisted.
  for need in "${NEEDS[@]}"; do
    _ov=""
    for kv in ${FR13_NEEDS_ALLOW:-}; do
      [[ "${kv%%=*}" == "${need%%=*}" ]] && _ov="$kv"
    done
    if [[ -n "$_ov" ]]; then
      echo "NEEDS-OVERRIDE (diagnostic arm): expecting $_ov instead of $need"
      grep -q "^$_ov$" "$ARMDIR/container_env.txt" || { echo "FAIL: override pin not live: $_ov"; exit 3; }
      continue
    fi
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
  docker exec "$CONTAINER_RUNTIME_REF" bash -lc '
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
docker logs "$CONTAINER_RUNTIME_REF" > "$ARMDIR/boot_log_snapshot.txt" 2>&1
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

# ---- warmup probe (legacy arms only; fixed32 permits canonical SWE traffic only) ----
if [[ -z "$FIXED32_MODE" ]]; then
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before_warmup.txt"
  WARMUP_TEMPERATURE=0.0
  .venv/bin/python scripts/fr10_quick_decode_tps_probe.py \
    --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
    --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
    --samples-per-prompt 1 --batch-size 1 --seed 1313 --top-p 1.0 \
    --wait-health 60 --request-timeout 900 --warmup-samples 0 \
    --modes "$PROBE_MODE" \
    --prompt-limit "${WARMUP_PROMPT_LIMIT:-1}" --max-tokens "${WARMUP_MAX_TOKENS:-16}" \
    --temperature "$WARMUP_TEMPERATURE" \
    --out "$ARMDIR/warmup_probe.json" \
    --request-metrics-out "$ARMDIR/warmup_request_metrics.jsonl" \
    > "$ARMDIR/warmup_probe_stdout.log" 2>&1
  RC=$?
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_warmup.txt"
  if (( RC != 0 )); then
    echo "FAIL: warmup probe rc=$RC"
    tail -20 "$ARMDIR/warmup_probe_stdout.log"; docker logs "$CONTAINER_RUNTIME_REF" 2>&1 | tail -30; exit 4
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

  # Container env is not sufficient: these one-time kernel messages prove the
  # route executed after the legacy warmup decode.
  if (( NATIVE_DECODE == 0 )); then
    docker logs "$CONTAINER_RUNTIME_REF" > "$ARMDIR/docker_after_warmup.log" 2>&1
    if grep -q "^FR13_SUBTREE_PARALLEL=1$" "$ARMDIR/container_env.txt"; then
      grep -F -m1 "[FR13_SUBTREE_PARALLEL ENGAGED]" \
        "$ARMDIR/docker_after_warmup.log" >/dev/null \
        || { echo "FAIL: subtree path route did not emit its runtime engagement needle"; exit 4; }
      echo "subtree runtime engagement OK ($KIND)"
    else
      echo "subtree runtime engagement needle SKIPPED (route explicitly disabled)"
    fi
    if grep -q "^FR13_SUBTREE_PARALLEL_SELFCHECK=1$" "$ARMDIR/container_env.txt"; then
      grep -F -m1 "[FR13_SUBTREE_PARALLEL SELFCHECK PASS]" \
        "$ARMDIR/docker_after_warmup.log" >/dev/null \
        || { echo "FAIL: subtree selfcheck was armed but emitted no byte-equal PASS"; exit 4; }
      echo "subtree byte selfcheck OK ($KIND)"
    fi
    if [[ "$KIND" == "hydra23" ]]; then
      grep -F -m1 \
        "FR13_HYDRA23 exact topology engaged: nodes=23 rescue_chains=rank1:4" \
        "$ARMDIR/docker_after_warmup.log" >/dev/null \
        || { echo "FAIL: exact Hydra23 topology runtime needle missing"; exit 4; }
      _hydra_floor_line="$(
        grep -F -m1 "[FR13_SUBTREE_PARALLEL] preseeded: n=24 schedule=hydra23_floor" \
          "$ARMDIR/docker_after_warmup.log" || true
      )"
      [[ "$_hydra_floor_line" == *"levels=[1, 9]"* \
         && "$_hydra_floor_line" == *"lens=[4, 8]"* \
         && "$_hydra_floor_line" == *"critical=12"* \
         && "$_hydra_floor_line" == *"route_armed=1"* ]] \
        || { echo "FAIL: Hydra23 hardware-floor schedule/worker arm needle missing: $_hydra_floor_line"; exit 4; }
      echo "Hydra23 floor schedule OK: critical=12"
    fi
  fi
else
  echo "fixed32 pretask generation warmup SKIPPED: canonical SWE traffic only"
fi

# ---- FR13 APC worker-env engagement assert (only for APC arms) ----
# The APC fixes read os.environ.get(FR13_APC_*) INSIDE the EngineCore worker (pid 176).
# That dict is intact in the worker (spawn inherits the full parent os.environ; the
# "dropped" symptom was a /proc/PID/environ artifact from setproctitle, NOT a real
# drop — verified 2026-06-27). The gdn module-import bridge in the worker writes
# /logs/fr13_apc_bridge_loaded.flag recording the LIVE os.environ values the gates
# read. Grep it for the masters this arm REQUIRES before recording any APC verdict, so
# a vacuous (defaults-only) run can never masquerade as engaged. Default-OFF arms set
# no FR13_APC_REQUIRE_* and this whole block is skipped -> locked non-APC path unchanged.
if [[ -n "${FR13_APC_REQUIRE_SNAP_FIX:-}${FR13_APC_REQUIRE_HIT_SUFFIX_CAP:-}" ]]; then
  APC_MARKER="${FR13_APC_BRIDGE_MARKER_FILE:-/logs/fr13_apc_bridge_loaded.flag}"
  # copy the worker-written marker out of the container's /logs to the arm dir
  docker exec "$CONTAINER_RUNTIME_REF" sh -c "cat '$APC_MARKER' 2>/dev/null" > "$ARMDIR/apc_bridge_marker.txt" 2>/dev/null
  # any bridge error file is a hard fail (a swallowed read must never look like engagement)
  if docker exec "$CONTAINER_RUNTIME_REF" sh -c "test -f /logs/fr13_apc_bridge_error.flag" 2>/dev/null; then
    echo "FAIL: APC bridge error flag present in worker:"; docker exec "$CONTAINER_RUNTIME_REF" cat /logs/fr13_apc_bridge_error.flag 2>/dev/null; exit 4
  fi
  APC_MARKER_TXT="$(cat "$ARMDIR/apc_bridge_marker.txt" 2>/dev/null)"
  [[ -n "$APC_MARKER_TXT" ]] || { echo "FAIL: APC engagement marker $APC_MARKER absent/empty in worker (gdn bridge did not run)"; docker logs "$CONTAINER_RUNTIME_REF" 2>&1 | tail -30; exit 4; }
  if [[ -n "${FR13_APC_REQUIRE_SNAP_FIX:-}" ]]; then
    grep -q "SNAP_FIX=${FR13_APC_REQUIRE_SNAP_FIX}\b" "$ARMDIR/apc_bridge_marker.txt" \
      || { echo "FAIL: APC worker SNAP_FIX != required ${FR13_APC_REQUIRE_SNAP_FIX} (vacuous gate). marker: $APC_MARKER_TXT"; exit 4; }
  fi
  if [[ -n "${FR13_APC_REQUIRE_HIT_SUFFIX_CAP:-}" ]]; then
    grep -q "HIT_SUFFIX_CAP=${FR13_APC_REQUIRE_HIT_SUFFIX_CAP}\b" "$ARMDIR/apc_bridge_marker.txt" \
      || { echo "FAIL: APC worker HIT_SUFFIX_CAP != required ${FR13_APC_REQUIRE_HIT_SUFFIX_CAP} (cap defaulted to 64 = vacuous). marker: $APC_MARKER_TXT"; exit 4; }
  fi
  echo "APC worker-env engagement OK: $APC_MARKER_TXT"
fi

if [[ -z "$FIXED32_MODE" ]]; then
  curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
    > "$ARMDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"
else
  echo "fixed32 pretask prefix-cache reset SKIPPED: fresh container"
fi

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

# ---- ACCEPT_SPEED_PROBE: superset-bound (greedy accept/fwd) + real wall-TPS (temp06), local vLLM ----
# Reuses the proven boot above (server healthy + spec-engaged). Fixed prompt, B=1 single request => clean
# decode_tps_wall (no co-residency, prefill excluded). Additive: default off => original SWE path unaffected.
# See FR13_CAT6_CAT8_ACCEPT_INVESTIGATION.md.
if [[ "${ACCEPT_SPEED_PROBE:-0}" == "1" ]]; then
  PF=${PROBE_PROMPT_FILE:-scripts/fr13_probe_prompt.txt}
  BRHIST="$ARMDIR/branch_hist.json"   # host side of FR13_BRANCH_ACCEPT_JSON=/workspace/$ARMDIR/branch_hist.json
  # PROBE_CHAT_MESSAGES=<json {messages:[...]}> => real SWE-Verified chat prompt (ship-config); else fixed prompt file
  if [[ -n "${PROBE_CHAT_MESSAGES:-}" ]]; then PROMPTARG=(--chat-messages "$PROBE_CHAT_MESSAGES"); else PROMPTARG=(--prompt-file "$PF"); fi
  echo "[accept-speed] arm=$ARM kind=$KIND prompt=${PROBE_CHAT_MESSAGES:-$PF} n=${PROBE_N:-512} modes=${PROBE_MODES:-greedy temp06 temp10}"
  for _mode in ${PROBE_MODES:-greedy temp06 temp10}; do
    PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/fr13_accept_speed_probe.py \
      --mode "$_mode" "${PROMPTARG[@]}" --n "${PROBE_N:-512}" \
      --base-url "http://127.0.0.1:$PORT" \
      --out "$ARMDIR/accept_speed_${_mode}.json" >> "$ARMDIR/accept_speed.log" 2>&1 \
      && echo "[accept-speed] $_mode -> $(cat "$ARMDIR/accept_speed_${_mode}.json" | tr -d '\n')" \
      || { echo "[accept-speed] $_mode FAILED"; tail -20 "$ARMDIR/accept_speed.log"; }
    # snapshot the cumulative branch-rescue hist AFTER this mode (per-mode = consecutive snapshot delta)
    [[ -f "$BRHIST" ]] && cp "$BRHIST" "$ARMDIR/brhist_${_mode}.json" 2>/dev/null || true
  done
  exit 0
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

# ---- launch canonical proxy (forced temperature, OFFLOAD-gated) ----
if [[ -z "$FIXED32_MODE" ]]; then
  mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
fi
AGENT_ARGS=()
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  echo "[offload] OFFLOAD_AGENT=1 — proxy+agent on $OFFLOAD_HOST, GB10 stays vLLM-only"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$OFFLOAD_HOST" \
        "curl -fsS -m 6 http://$GB10_TS_IP:$PORT/health >/dev/null 2>&1 && echo ok" \
        2>/dev/null | grep -q ok; then
    echo "FAIL: alienware cannot reach GB10 vLLM at http://$GB10_TS_IP:$PORT/health (set OFFLOAD_AGENT=0 to fall back)"
    exit 5
  fi
  echo "[offload] alienware -> GB10 vLLM $GB10_TS_IP:$PORT/health OK"
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" sync "$OFFLOAD_HOST" > "$ARMDIR/offload_sync.log" 2>&1 \
    || { echo "FAIL: offload proxy sync"; cat "$ARMDIR/offload_sync.log"; exit 5; }
  LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
    bash "$OFFLOAD_HELPER" start "$OFFLOAD_HOST" "$GB10_TS_IP" "$ARMDIR_ABS" \
    > "$ARMDIR/offload_start.log" 2>&1 \
    || { echo "FAIL: offload proxy start"; cat "$ARMDIR/offload_start.log"; exit 5; }
  cat "$ARMDIR/offload_start.log"
  cp "$ARMDIR/offload_proxy_env.txt" "$ARMDIR/proxy_env.txt" 2>/dev/null || true
  AGENT_ARGS=(--agent-host "$OFFLOAD_HOST" \
              --agent-endpoint "http://127.0.0.1:$OFFLOAD_PROXY_PORT/v1")
  echo "proxy OK (OFFLOADED to $OFFLOAD_HOST:$OFFLOAD_PROXY_PORT)"
else
  LUMO_PROXY_FORCE_TEMPERATURE="${DEPLOY_FORCE_TEMP:-0.6}" \
  LUMO_PROXY_REQUEST_DUMP_DIR="$ARMDIR_ABS/proxy_request_dumps" \
  LUMO_PROXY_PAIR_DUMP_DIR="$ARMDIR_ABS/proxy_pair_dumps" \
  LUMO_PROXY_LOG_PATH="$ARMDIR_ABS/proxy.log" \
  LUMO_PROXY_NOHUP_PATH="$ARMDIR_ABS/proxy.nohup" \
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
  if [[ -z "$FIXED32_MODE" ]]; then
    grep -q "LUMO_PROXY_PAIR_DUMP_DIR=" "$ARMDIR/proxy_env.txt" \
      || { echo "FAIL: proxy pair-dump pin missing"; exit 5; }
  fi
  echo "proxy OK"
fi

if [[ -n "$FIXED32_MODE" ]]; then
  fixed32_engine_ingress_control \
    preflight "$ARMDIR/fixed32_engine_ingress_preflight.json" \
    > "$ARMDIR/fixed32_engine_ingress_preflight.log" 2>&1 \
    || {
      echo "FAIL: fixed32 engine ingress auth preflight"
      tail -20 "$ARMDIR/fixed32_engine_ingress_preflight.log" 2>/dev/null || true
      exit 5
    }
  _fixed32_proxy_preflight_tmp="$ARMDIR/fixed32_proxy_ingress_preflight.json.tmp"
  if LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" control "$OFFLOAD_HOST" preflight \
      > "$_fixed32_proxy_preflight_tmp" \
      2> "$ARMDIR/fixed32_proxy_ingress_preflight.log"; then
    mv \
      "$_fixed32_proxy_preflight_tmp" \
      "$ARMDIR/fixed32_proxy_ingress_preflight.json"
  else
    rm -f "$_fixed32_proxy_preflight_tmp"
    echo "FAIL: fixed32 proxy ingress auth preflight"
    tail -20 "$ARMDIR/fixed32_proxy_ingress_preflight.log" 2>/dev/null || true
    exit 5
  fi
  unset _fixed32_proxy_preflight_tmp
  echo "fixed32 ingress auth preflight OK: engine=4 rejects proxy=4 rejects"
fi

# ---- eval offload pre-flight ----
EVAL_ARGS=()
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$EVAL_HOST" \
     "test -f ~/swe_eval_offload/swe_eval_x86_worker.py && echo ok" 2>/dev/null | grep -q ok; then
  EVAL_ARGS=(--eval-host "$EVAL_HOST")
  if [[ -n "$FIXED32_MODE" ]]; then
    echo "eval offload: configured evaluator reachable" \
      | tee "$ARMDIR/eval_offload_preflight.txt"
  else
    echo "eval offload: $EVAL_HOST reachable" \
      | tee "$ARMDIR/eval_offload_preflight.txt"
  fi
else
  if [[ -n "$FIXED32_MODE" ]]; then
    echo "FAIL: fixed32 evaluator offload $EVAL_HOST is unavailable" \
      | tee "$ARMDIR/eval_offload_preflight.txt"
    exit 5
  fi
  echo "WARN: eval offload $EVAL_HOST UNREACHABLE — eval attempted locally" \
    | tee "$ARMDIR/eval_offload_preflight.txt"
fi

# ---- SWE window: /metrics bracketing + codex loop ----
WALL_ARGS=()
[[ -n "${AGENT_WALL_S:-}" ]] && WALL_ARGS=(--agent-wall-s "$AGENT_WALL_S")
FIXED32_RUNNER_ARGS=()
if [[ -n "$FIXED32_MODE" ]]; then
  FIXED32_RUNNER_ARGS=(
    --fixed32-container "$CONTAINER_RUNTIME_REF"
    --fixed32-producer-pid "$FIXED32_PRODUCER_PID"
    --fixed32-mode "$FIXED32_MODE"
    --fixed32-flush-request "$FIXED32_REQUEST_PATH"
    --fixed32-flush-ack "$FIXED32_ACK_PATH"
    --fixed32-boundary-snapshot "$FIXED32_BOUNDARY_SNAPSHOT_PATH"
  )
  if [[ "${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" == "1" ]]; then
    FIXED32_RUNNER_ARGS+=(
      --fixed32-taw-real-event-arm "$FIXED32_TAW_REAL_EVENT_ARM_PATH"
    )
  fi
  if [[ "${FR13_DFWD_UNIFIED_BM8_LIVE_AB:-0}" == "1" ]]; then
    FIXED32_RUNNER_ARGS+=(
      --fixed32-bm8-real-event-arm "$FIXED32_BM8_REAL_EVENT_ARM_PATH"
    )
  fi
  if [[ "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_coop128_byte_ab" \
        || "${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "streamk_force_wide256_byte_ab" ]]; then
    FIXED32_RUNNER_ARGS+=(
      --fixed32-cutlass-real-event-arm "$FIXED32_CUTLASS_REAL_EVENT_ARM_PATH"
    )
  fi
fi

# NETWORK-LINK WATCHDOG (OFFLOAD only; req #4/#5) — see fr13_bigdenom_swe_serve.sh
LINKLOG="$ARMDIR/offload_link_state.log"
LINK_DEAD_MARKER="$ARMDIR/offload_link_dead.flag"
WATCHDOG_PID=""
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
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
if [[ -n "$FIXED32_MODE" ]]; then
  .venv/bin/python - \
    "$ARMDIR/metrics_before_swe.txt" \
    "$ARMDIR/logs/fr13_fixed32_work_census.jsonl" \
    "$ARMDIR/fixed32_ready_ack.json" \
    "$FIXED32_MODE" \
    "$ARMDIR/fixed32_pretask_zero_traffic.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path


metrics_path = Path(sys.argv[1])
census_path = Path(sys.argv[2])
ready_path = Path(sys.argv[3])
mode = sys.argv[4]
output_path = Path(sys.argv[5])


def counter(name, *, required):
    values = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        metric_name = line.split(None, 1)[0].split("{", 1)[0]
        if metric_name == name:
            values.append(float(line.rsplit(None, 1)[-1]))
    if not values and not required:
        return None
    if len(values) != 1:
        raise SystemExit(
            f"expected one pretask metric {name}, got {len(values)}"
        )
    value = values[0]
    if not math.isfinite(value) or value != 0:
        raise SystemExit(f"pretask metric {name} is nonzero: {value!r}")
    return int(value)


optional_worker_metrics = (
    "vllm:fr13_decode_forward_gpu_seconds_total",
    "vllm:fr13_decode_forward_gpu_steps_total",
    "vllm:fr13_decode_forward_gpu_drafts_total",
    "vllm:fr13_decode_step_wall_seconds_total",
    "vllm:fr13_decode_step_wall_drafts_total",
    "vllm:fr13_decode_step_wall_steps_total",
    "vllm:fr13_decode_step_wall_attempts_total",
    "vllm:fr13_decode_step_wall_rejected_total",
)
required_pretask_metrics = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
)
pretask_values = {
    name: counter(name, required=True) for name in required_pretask_metrics
}
for name in optional_worker_metrics:
    counter(name, required=False)


ready = json.loads(ready_path.read_text(encoding="utf-8"))
ready_counters = ready.get("counters")
if (
    ready.get("mode") != mode
    or ready.get("generation") != 0
    or ready.get("action") != "ready"
    or ready.get("status") != "ok"
    or not isinstance(ready_counters, dict)
    or ready_counters.get("pure_decode_forward_steps") != 0
    or ready_counters.get("complete_work_census_events") != 0
    or ready_counters.get("work_census_first_forward_step") is not None
    or ready_counters.get("work_census_last_forward_step") is not None
    or ready_counters.get("sfwd_pending") != 0
    or ready_counters.get("dfwd_pending") != 0
    or ready_counters.get("cfwd_pending") != 0
):
    raise SystemExit("fixed32 ready ack is not a zero-work baseline")

census_bytes = census_path.read_bytes() if census_path.exists() else b""
if census_bytes.strip():
    raise SystemExit("fixed32 work census is nonempty before canonical tasks")
metrics_bytes = metrics_path.read_bytes()
payload = {
    "schema": "fr13-fixed32-pretask-zero-traffic-v1",
    "mode": mode,
    "no_positive_probe": True,
    "generation_probe_commands_executed": 0,
    "metrics": {
        "path": str(metrics_path.resolve()),
        "sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "spec_drafts": pretask_values["vllm:spec_decode_num_drafts_total"],
        "spec_tokens": pretask_values[
            "vllm:spec_decode_num_draft_tokens_total"
        ],
    },
    "work_census": {
        "path": str(census_path.resolve()),
        "exists": census_path.exists(),
        "bytes": len(census_bytes),
        "sha256": hashlib.sha256(census_bytes).hexdigest(),
    },
    "ready_ack": {
        "path": str(ready_path.resolve()),
        "sha256": hashlib.sha256(ready_path.read_bytes()).hexdigest(),
        "generation": ready["generation"],
    },
}
temporary = output_path.with_name(output_path.name + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="utf-8",
)
temporary.replace(output_path)
print("fixed32 pretask zero-traffic baseline OK")
PY
  (( $? == 0 )) || { echo "FAIL: fixed32 pretask zero-traffic baseline"; exit 5; }
  fixed32_engine_ingress_control \
    begin "$ARMDIR/fixed32_engine_ingress_begin.json" \
    > "$ARMDIR/fixed32_engine_ingress_begin.log" 2>&1 \
    || {
      echo "FAIL: fixed32 engine ingress campaign begin"
      tail -20 "$ARMDIR/fixed32_engine_ingress_begin.log" 2>/dev/null || true
      exit 5
    }
  _fixed32_proxy_begin_tmp="$ARMDIR/fixed32_proxy_ingress_begin.json.tmp"
  if LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" control "$OFFLOAD_HOST" begin \
      > "$_fixed32_proxy_begin_tmp" \
      2> "$ARMDIR/fixed32_proxy_ingress_begin.log"; then
    mv \
      "$_fixed32_proxy_begin_tmp" \
      "$ARMDIR/fixed32_proxy_ingress_begin.json"
  else
    rm -f "$_fixed32_proxy_begin_tmp"
    echo "FAIL: fixed32 proxy ingress campaign begin"
    tail -20 "$ARMDIR/fixed32_proxy_ingress_begin.log" 2>/dev/null || true
    exit 5
  fi
  unset _fixed32_proxy_begin_tmp
  echo "fixed32 ingress campaign begun after zero-work baseline"
fi
S0=$(date +%s)
.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$SUBSET" \
  --out-root "$ARMDIR/swe_out" \
  --concurrency "$SWE_CONCURRENCY" \
  --eval-timeout-s "${EVAL_TIMEOUT_S:-1800}" \
  "${WALL_ARGS[@]}" \
  "${FIXED32_RUNNER_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${AGENT_ARGS[@]}" \
  > "$ARMDIR/swe_orchestrator.log" 2>&1
SWERC=$?
S1=$(date +%s)
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after_swe.txt"
SWE_WALL=$((S1-S0))
echo "swe orchestrator rc=$SWERC wall=${SWE_WALL}s"
tail -5 "$ARMDIR/swe_orchestrator.log"

[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null && WATCHDOG_PID=""

if [[ -n "$FIXED32_MODE" ]]; then
  _fixed32_proxy_finalize_tmp="$ARMDIR/fixed32_proxy_ingress_finalize.json.tmp"
  if LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" control "$OFFLOAD_HOST" finalize \
      > "$_fixed32_proxy_finalize_tmp" \
      2> "$ARMDIR/fixed32_proxy_ingress_finalize.log"; then
    mv \
      "$_fixed32_proxy_finalize_tmp" \
      "$ARMDIR/fixed32_proxy_ingress_finalize.json"
  else
    rm -f "$_fixed32_proxy_finalize_tmp"
    echo "FAIL: fixed32 proxy ingress campaign finalize"
    tail -20 "$ARMDIR/fixed32_proxy_ingress_finalize.log" 2>/dev/null || true
    SWERC=13
  fi
  unset _fixed32_proxy_finalize_tmp
  if ! fixed32_engine_ingress_control \
      finalize "$ARMDIR/fixed32_engine_ingress_finalize.json" \
      > "$ARMDIR/fixed32_engine_ingress_finalize.log" 2>&1; then
    echo "FAIL: fixed32 engine ingress campaign finalize"
    tail -20 "$ARMDIR/fixed32_engine_ingress_finalize.log" 2>/dev/null || true
    SWERC=13
  fi
fi

# ---- OFFLOAD: fetch alienware proxy/agent artifacts back (resilient rsync) ----
if [[ "$OFFLOAD_AGENT" == "1" ]]; then
  if LUMO_OFFLOAD_PROXY_PORT="$OFFLOAD_PROXY_PORT" \
      bash "$OFFLOAD_HELPER" fetch "$OFFLOAD_HOST" "$ARMDIR_ABS" \
      > "$ARMDIR/offload_fetch.log" 2>&1; then
    printf 'ok\n' > "$ARMDIR/offload_fetch_status.txt"
  elif [[ -n "$FIXED32_MODE" ]]; then
    printf 'error\n' > "$ARMDIR/offload_fetch_status.txt"
    echo "FAIL: fixed32 offload artifact fetch failed"
    SWERC=13
  else
    echo "WARN: offload fetch errors (see offload_fetch.log)"
  fi
  cat "$ARMDIR/offload_fetch.log"
  if [[ -f "$LINK_DEAD_MARKER" ]]; then
    echo "OFFLOAD_LINK_DEAD: $(cat "$LINK_DEAD_MARKER") — deploy-speed window for $ARM DISCARDED" \
      | tee "$ARMDIR/DEPLOY_SPEED_DISCARDED.flag"
    SWERC=12
  fi
fi

if [[ -n "$FIXED32_MODE" ]]; then
  if ! .venv/bin/python - \
      "$ARMDIR/metrics_before_swe.txt" \
      "$ARMDIR/metrics_after_swe.txt" \
      "$EXPECT_RATIO" <<'PY'
import math
import sys
from pathlib import Path


before_path, after_path = Path(sys.argv[1]), Path(sys.argv[2])
expected_ratio = float(sys.argv[3])


def value(path, name):
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        metric_name = line.split(None, 1)[0].split("{", 1)[0]
        if metric_name == name:
            values.append(float(line.rsplit(None, 1)[-1]))
    if len(values) != 1:
        raise SystemExit(
            f"{path}: expected one metric {name}, got {len(values)}"
        )
    result = values[0]
    if not math.isfinite(result) or result < 0:
        raise SystemExit(f"{path}: invalid metric {name}: {result!r}")
    return result


draft_name = "vllm:spec_decode_num_drafts_total"
token_name = "vllm:spec_decode_num_draft_tokens_total"
drafts = value(after_path, draft_name) - value(before_path, draft_name)
tokens = value(after_path, token_name) - value(before_path, token_name)
if drafts <= 0:
    raise SystemExit(f"canonical-task spec engagement is empty: drafts={drafts}")
ratio = tokens / drafts
if abs(ratio - expected_ratio) >= 1e-9:
    raise SystemExit(
        "canonical-task draft shape mismatch: "
        f"draft_tokens/drafts={ratio} expected={expected_ratio}"
    )
print(
    "canonical-task spec engagement OK: "
    f"drafts delta={drafts} draft_tokens/drafts={ratio}"
)
PY
  then
    echo "FAIL: fixed32 canonical-task spec engagement"
    SWERC=15
  fi

  docker logs "$CONTAINER_RUNTIME_REF" > "$ARMDIR/docker_after_tasks.log" 2>&1
  if grep -q "^FR13_SUBTREE_PARALLEL=1$" "$ARMDIR/container_env.txt"; then
    grep -F -m1 "[FR13_SUBTREE_PARALLEL ENGAGED]" \
      "$ARMDIR/docker_after_tasks.log" >/dev/null \
      || { echo "FAIL: fixed32 canonical tasks emitted no subtree runtime needle"; SWERC=15; }
  fi
  grep -F -m1 \
    "[FR13_FIXED32] topology engaged: mode=$FIXED32_MODE" \
    "$ARMDIR/docker_after_tasks.log" >/dev/null \
    || { echo "FAIL: fixed32 canonical tasks emitted no topology needle"; SWERC=15; }
  grep -F -m1 "[FR13_FIXED32_WORK] engaged:" \
    "$ARMDIR/docker_after_tasks.log" >/dev/null \
    || { echo "FAIL: fixed32 canonical tasks emitted no work needle"; SWERC=15; }
fi

# ---- OPT-1 post-run engagement needle ----
if [[ "$KIND" == "cat9-opt1" ]]; then
  docker logs "$CONTAINER_RUNTIME_REF" 2>&1 | grep -m1 "FR13_COMMITTER_SYNCKILL engaged" \
    | tee "$ARMDIR/opt1_synckill_needle.txt" \
    || { echo "FAIL: OPT-1 synckill never engaged (vacuous arm)"; SWERC=9; }
fi

# ---- health rule + legacy pair-dump accounting ----
.venv/bin/python - "$ARMDIR" "$SWE_WALL" "$SWERC" <<'PY'
import json, sys
from pathlib import Path
armdir, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
metas = sorted(armdir.glob("swe_out/*/per_task/*/runner_metadata.json"))
health = {"swe_orchestrator_rc": rc, "swe_window_wall_s": wall, "tasks": []}
for m in metas:
    meta = json.loads(m.read_text())
    codex = meta.get("agent") or meta.get("codex") or {}
    health["tasks"].append({"instance_id": meta.get("instance_id"),
        "codex_elapsed_s": codex.get("elapsed_s"),
        "codex_timed_out": codex.get("timed_out"),
        "patch_bytes": meta.get("patch_bytes"),
        "verdict": (meta.get("eval_report") or {}).get("verdict", "missing")})
(armdir / "health.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
PY

if [[ -n "$FIXED32_MODE" ]]; then
  echo "raw proxy dumps: disabled"
else
  NPAIR=$(find "$ARMDIR/proxy_pair_dumps" -maxdepth 1 -type f 2>/dev/null | wc -l)
  echo "pair dumps captured: $NPAIR"
fi

echo "ARM_DONE $ARM kind=$KIND swerc=$SWERC"
exit $SWERC
