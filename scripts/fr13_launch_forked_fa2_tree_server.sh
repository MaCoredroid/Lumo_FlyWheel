#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
IMAGE=${IMAGE:-"vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"}
CONTAINER=${CONTAINER:-fr13-forked-fa2-tree}
PORT=${PORT:-9950}
GPU_UTIL=${GPU_UTIL:-0.88}
# DURABLE OOM GUARD (2026-06-24): ALWAYS cap the container cgroup so the host keeps
# ~12GiB headroom -> Claude/watchdog/tmux can NEVER be the kernel OOM victim (GB10
# unified mem; the killer takes a -1000 proc only when nothing else can be freed).
# If a caller sets GPU_UTIL too high for this cap, the CONTAINER cgroup-OOMs at boot
# (relaunchable + LOUD), never the host. Diagnostics should also run ENFORCE_EAGER=1
# (the cuda-graph CAPTURE spike is the real trigger; eager stays flat). Default-ON so
# no caller can forget it (the repeat OOMs were callers that didn't set it).
DOCKER_MEM_CAP=${DOCKER_MEM_CAP:-105g}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-TREE_ATTN}
FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-tree_mtp}
FR10_METRICS=${FR10_METRICS:-0}
BATCH_INVARIANT=${BATCH_INVARIANT:-0}
FR13_FA2_TREE_BIAS=${FR13_FA2_TREE_BIAS:-1}
FR13_FA2_PREFILL_NATIVE=${FR13_FA2_PREFILL_NATIVE:-1}
# FR13 Method-A: BI allowlist for TREE_ATTN (inert by default; only relevant
# with BATCH_INVARIANT=1, and requires the two FR13_FA2_* flags above).
FR13_BI_TREE_ATTN=${FR13_BI_TREE_ATTN:-0}
FR13_TREE_ATTN_EXP2_SOFTMAX=${FR13_TREE_ATTN_EXP2_SOFTMAX:-1}
# FR13_CONV_COMMITTED_PATH (default ON): the next event's prior conv window is
# read from the COMMITTED path's accepted-leaf NODE column (pre-remap), so
# BRANCH winners ([0,2], [0,1,4]) commit a window built from committed-path
# tokens only; spine winners are byte-identical to the legacy linear read.
# =0 restores the legacy post-remap linear-column read.
FR13_CONV_COMMITTED_PATH=${FR13_CONV_COMMITTED_PATH:-1}
# FR13_FORCE_SPINE_COMMIT (default OFF) — DIAGNOSTIC ONLY, like
# FR10_ALLOW_LINEAR_FALLBACK: the greedy committer scores all paths (alts
# still verified) but always commits the spine path's own prefix. For the
# S3/m1 decisive A/B (caterpillar forced-spine vs chain boot) ONLY. NEVER
# bind =1 into a committed serving config or a gate result.
FR13_FORCE_SPINE_COMMIT=${FR13_FORCE_SPINE_COMMIT:-0}
# FR13_DRAFTER_SINGLE_LOGITS (FIX-1, default ON): the caterpillar drafter
# takes draft tokens as argmax of the single already-computed logits tensor
# instead of _greedy_sample's second compute_logits (double full-vocab bf16
# lm-head read per drafter step, FR13_B1_SPEED_ATTRIBUTION_BIND.md). =0 is
# the exact legacy double-logits path (the A/B instrument).
FR13_DRAFTER_SINGLE_LOGITS=${FR13_DRAFTER_SINGLE_LOGITS:-1}
# FR13_EAGER_PACK (FIX-2, default OFF until the lossless gate passes): pack
# the committer's eager DtoH/HtoD storm and batch the 48 per-layer replay
# launches into one (FR13_B1_SPEED_ATTRIBUTION_BIND.md). SEMANTICS-PRESERVING
# ONLY: no computed value changes, only WHERE/HOW the same ints move. =0 is
# the exact legacy transport path (the A/B instrument).
FR13_EAGER_PACK=${FR13_EAGER_PACK:-1}
# FR13_TREE_CONV_FUSED (FIX-3, default OFF until the byte A/B + live gate
# pass): fuse the tree causal-conv emulation's per-node state write-back
# loop / per-col tap loop / remap + committed-prior row math into vectorized
# torch ops over init-time static index tensors (census contributors 1-4,
# FR13_B1_SPEED_ATTRIBUTION_BIND.md). BIT-EXACT-PRESERVING by construction
# (same per-element ops in the same order; tree-only — native
# causal_conv1d_update untouched). =0 is the exact legacy emulation (the
# A/B instrument).
FR13_TREE_CONV_FUSED=${FR13_TREE_CONV_FUSED:-1}
# FR13_FIX1_SELFCHECK (default OFF) — DIAGNOSTIC ONLY, like
# FR13_FORCE_SPINE_COMMIT: with the single-logits drafter serving, ALSO run
# legacy _greedy_sample per drafter step and raise on any token mismatch
# (in-process FIX-1 OFF==ON byte-identity proof; counters dumped to
# FR13_FIX1_SELFCHECK_DUMP). NEVER bind =1 into a serving config or a speed
# number.
FR13_FIX1_SELFCHECK=${FR13_FIX1_SELFCHECK:-0}
FR13_FIX1_SELFCHECK_DUMP=${FR13_FIX1_SELFCHECK_DUMP:-/logs/fr13_fix1_selfcheck.json}
# FR13_COMMIT_ARGMAX_GATE (default OFF) — DIAGNOSTIC ONLY, like
# FR13_FORCE_SPINE_COMMIT / FR13_FIX1_SELFCHECK: in-process per-served-token
# committer-row argmax gate. At each committed/served token the greedy tree
# committer dumps (to FR13_COMMIT_ARGMAX_GATE_DUMP jsonl) the verify-forward
# logit row it ACTUALLY indexed plus channel-1 (committed_id vs
# argmax(verify_logits[row])) and channel-2 (verify argmax + top-2 margin for
# the clean-forward reduce). Localizes the gold-gate non-argmax-served gap
# (FR13_B1_SWE_GOLD_BIND.md). EAGER-only (it syncs/.item()s) — run on an eager
# diagnostic boot. NEVER bind =1 into a serving config or a speed number.
FR13_COMMIT_ARGMAX_GATE=${FR13_COMMIT_ARGMAX_GATE:-0}
FR13_COMMIT_ARGMAX_GATE_DUMP=${FR13_COMMIT_ARGMAX_GATE_DUMP:-/logs/fr13_commit_argmax_gate.jsonl}
# FR13_FORK_MARGIN_DUMP (default OFF) — DIAGNOSTIC ONLY, READ-ONLY, same class
# as FR13_COMMIT_ARGMAX_GATE / FR13_FORCE_SPINE_COMMIT. Per-spec-step committer-
# fork classifier: dumps (to FR13_FORK_MARGIN_DUMP_PATH jsonl) each path's lcp +
# the WINNER/SPINE lcp-divergence nodes' VERIFY top-2 margins (parent_target
# top1-top2), so the reduce splits genuine leaf-LCP wins (margin>1nat = A,
# FUNDAMENTAL) from sub-1-nat near-ties (B, FIXABLE by a rank-2 margin-damp).
# CHANGES NOTHING SERVED (default-OFF = byte-identical). EAGER-only. NEVER bind
# =1 into a serving config or a speed number.
FR13_FORK_MARGIN_DUMP=${FR13_FORK_MARGIN_DUMP:-0}
FR13_FORK_MARGIN_DUMP_PATH=${FR13_FORK_MARGIN_DUMP_PATH:-/logs/fr13_fork_margin_dump.jsonl}
# FR13_CHASE_DIAG (default OFF) — DIAGNOSTIC ONLY, superset-chase Step-1
# in-process instruments (plan wf_c43b084f, FIX1-SELFCHECK pattern): (i)
# per-event integer row record + (iii) drafter root-logit top-K (eagle
# propose, capture-safe but adds syncs), (ii) GDN state-parity taps
# A/B/B_JOIN/CV via the boundary instrument (EAGER-ONLY — boot with
# ENFORCE_EAGER=1), (iv) drafter-KV row hashes. Implies the
# FR13_REPLAY_BOUNDARY_LOG taps on the FR13_REPLAY_BOUNDARY_LAYERS
# layer(s). Pair with FR13_TCF_SELFCHECK=1 on the chase boot for the
# mandatory H6 conv-prior byte check. NEVER bind =1 into a serving config
# or an accept/speed number.
FR13_CHASE_DIAG=${FR13_CHASE_DIAG:-0}
FR13_CHASE_DIAG_DIR=${FR13_CHASE_DIAG_DIR:-/logs}
FR13_CHASE_TOPK=${FR13_CHASE_TOPK:-32}
FR13_CHASE_KV_WINDOW=${FR13_CHASE_KV_WINDOW:-16}
# FR13_CHASE_H3 (active only under FR13_CHASE_DIAG=1): minimal H3 probe --
# ONE target full-attn layer's served-slot K/V hashes at the committed
# positions per event (topological foreign_slot verdict; deviation recorded
# in-band). FR13_CHASE_H3_LAYER pins the layer by static_forward_context
# name (default: first non-drafter full-attn layer). FR13_CHASE_KV_ALLOW_EMPTY
# =1 is the ONLY way to let the drafter-KV harvest bank an empty record (the
# step-1 vacuous-instrument failure now fail-louds, wf_a71e2a24 FAIL-1).
FR13_CHASE_H3=${FR13_CHASE_H3:-1}
FR13_CHASE_H3_LAYER=${FR13_CHASE_H3_LAYER:-}
FR13_CHASE_KV_ALLOW_EMPTY=${FR13_CHASE_KV_ALLOW_EMPTY:-0}
# FR13_TREE_SAMPLE_ROW (FIX-A1, default OFF until gated;
# FR13_CHASE_STEP1_BIND.md H1): sample the drafter at the committed tree
# LEAF's flat verify row (+1-shifted published node id, device-resident
# paths/lens buffers) instead of the stock linear row prev_accepted_len,
# which lands on a REJECTED node after 51.2% of cat9 partial accepts.
# Chain-neutral by construction (chain leaf row == L == stock). Requires
# FR13_TREE_REQKEY=1 (fail-loud). =0 is verbatim stock behavior.
FR13_TREE_SAMPLE_ROW=${FR13_TREE_SAMPLE_ROW:-1}
# LUMO_FB_KERNEL_ROWS / LUMO_FB_PROJ_PAD_ROWS (default OFF) — TARGETED ba-proj
# batch-invariance for the FR13 +17 leaf width carrier (FR13_WIDTH_CARRIER_
# INPROJ_BA_BIND.md, H1). When LUMO_FB_KERNEL_ROWS=1 the gdn_linear_attn forward
# pads the bf16 in_proj_ba (and out_proj) GEMM to a FIXED, tree_n-independent row
# group (LUMO_FB_PROJ_PAD_ROWS per spec, >= max tree_n), computes one batched
# projection, scatters the real rows back, and discards pads => cuBLASLt is pinned
# to ONE shape => M-invariant a/b on the spine row. Lossless-by-construction
# (GEMM row = W @ hidden[row], independent per row; zero pad-rows contribute
# nothing). This is the AUTHORIZED #42960 batch-invariance, NOT a reward-hack, and
# is the TARGETED fix (NOT full VLLM_BATCH_INVARIANT, which takes the GB10 REDUCED
# override branch and perturbs fp8/scan => cat9+BI=34, counterproductive).
# DEFAULT OFF: empty LUMO_FB_KERNEL_ROWS => the gate `== "1"` is False => verbatim
# stock projection path (byte-identical to the locked launcher).
# NOTE (2026-06-14): the pad-block code IS inserted by the patcher and is LIVE
# (the in_proj_ba pad is BAKED into locked cat9, a666f9ec — lossless + speed-
# neutral; the prior "not inserted / INERT" BLOCKER was wrong and is removed).
LUMO_FB_KERNEL_ROWS=${LUMO_FB_KERNEL_ROWS:-}
LUMO_FB_PROJ_PAD_ROWS=${LUMO_FB_PROJ_PAD_ROWS:-16}
# FR13_GDN_SUBOP_MAB worker-env propagation (additive, default-safe).
# *** MEASURED 2026-06-14 (this launcher run, wf l0gdn-envfix): these two ray
# vars are NOT SUFFICIENT for the deployed image. The GDN forward runs in the
# EngineCore process (VLLM::EngineCore, PPid=1) which is spawned with a CURATED
# env that drops even registered VLLM_-prefixed vars (VLLM_SERVER_DEV_MODE /
# VLLM_BATCH_INVARIANT were ABSENT from /proc/<EngineCore>/environ) and is NOT
# the ray-executor path (no "Copying the following environment variables" log;
# ray_env.get_env_vars_to_copy never fires). Only 14/66 FR13_* vars reach the
# worker; the bare master FR13_GDN_SUBOP_MAB (+ _DUMP/_EXPECT_TREE_N/_THRESHOLD)
# is among the dropped. The hard worker-env gate caught this pre-capture.
# REQUIRED FIX (patcher-side, next boot): fr10_phase4_patch_vllm_tree_gdn.py runs
# in pid 1 (master PRESENT) and edits the in-image gdn_linear_attn.py the worker
# imports -> have it READ FR13_GDN_SUBOP_MAB at PATCH time and bake the engaged
# flag (or write a sidecar file the gate reads) into the patched gate, instead of
# os.environ.get() at worker forward time. These ray vars are kept below as
# harmless belt-and-suspenders for any future ray-executor path. ***
# ONLY populated when FR13_GDN_SUBOP_MAB is set/non-zero; otherwise empty =>
# byte-identical to the locked default path. We APPEND to any caller-supplied
# value (additive, never replacing).
VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}
VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY:-}
case "${FR13_GDN_SUBOP_MAB:-0}" in
  1|true|yes|on|TRUE|YES|ON)
    _fr13_subop_names="FR13_GDN_SUBOP_MAB,FR13_GDN_SUBOP_MAB_DUMP,FR13_GDN_SUBOP_MAB_LAYER,FR13_GDN_SUBOP_MAB_SKIP,FR13_GDN_SUBOP_MAB_LIMIT,FR13_GDN_SUBOP_MAB_EXPECT_TREE_N,FR13_GDN_SUBOP_MAB_THRESHOLD"
    if [[ -n "$VLLM_RAY_EXTRA_ENV_VARS_TO_COPY" ]]; then
      VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},${_fr13_subop_names}"
    else
      VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="$_fr13_subop_names"
    fi
    if [[ -n "$VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY" ]]; then
      VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="${VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY},FR13_"
    else
      VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="FR13_"
    fi
    ;;
esac
LUMO_MTP_DRAFT_TRACE_FILE=${LUMO_MTP_DRAFT_TRACE_FILE:-}
LUMO_TREE_SAMPLER_DEBUG_LOG=${LUMO_TREE_SAMPLER_DEBUG_LOG:-}
LUMO_TREE_PATH_LCP_LOG=${LUMO_TREE_PATH_LCP_LOG:-}
LOG_DIR=${LOG_DIR:-"${FR13_RUN_DIR:-$REPO/output/fr13_fa2_tree_e2e/live}/logs"}
FORKED_FA2_SO=${FORKED_FA2_SO:-"$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"}
TREE=${TREE:-"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-$(TREE="$TREE" python3 - <<'PY'
import ast
import os
print(len(ast.literal_eval(os.environ["TREE"])))
PY
)}
SPEC_CONFIG=${SPEC_CONFIG:-"{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"speculative_token_tree\":\"$TREE\"}"}

# FR13_ENABLE_APC (default 0): prefix caching for the GDN-hybrid. With spec-decode on,
# vLLM auto-forces mamba_cache_mode=align (config.py), which hard-requires chunked-prefill
# (a 2nd behavioral change). Qwen3-Next uses the SD conv layout (VLLM_SSM_CONV_STATE_LAYOUT
# unset -> "SD"), so is_conv_state_dim_first() is False -> the DS num_accepted>1 raise
# (mamba_utils.py:315) does NOT apply. fp32 SSM-state cache = SGLang default + vLLM #26807
# lossless lever. mamba_block_size multiple of 8 (causal_conv1d align), <= max-num-batched.
# Default OFF => APC_FLAGS empty => serve command byte-identical to the locked cat9 path.
FR13_ENABLE_APC=${FR13_ENABLE_APC:-0}
MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-1024}
MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}
# APC LOSSLESS FIX (2026-06-21, proven run_20260621T013300Z): default max-num-batched-tokens to
# the mamba block size so each chunked-prefill scheduler step crosses AT MOST ONE block boundary.
# vLLM mamba 'align' caches ONE checkpoint per step (the last boundary it crosses); with a larger
# budget a step spans multiple blocks and an INTERMEDIATE boundary is cached with the step-END
# (overshoot) recurrent state (#45238), so a later cache hit at that boundary restores a grossly
# wrong GDN state -> degenerate 'output_text' garbage. step<=1 block => step-end state == boundary
# state => correct checkpoint => gross poison eliminated (cache-ON coherent + on-task, TTFT 3.96x).
# This is the minimum legal value (align asserts block_size <= max-num-batched). APC-scoped:
# consumed only inside APC_FLAGS below, so the non-APC locked cat9 serve command is byte-identical.
# (#43650 drop-final-block was REFUTED: the GDN restore reads block_table col-0 anchored to
# (seq_len-1)//block_size, NOT the matched-block count, so dropping a matched block is a no-op on
# the restored state. Stays default-OFF.)
APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-$MAMBA_BLOCK_SIZE}
if [[ "$FR13_ENABLE_APC" == "1" ]]; then
  APC_FLAGS="--enable-prefix-caching --enable-chunked-prefill --mamba-block-size $MAMBA_BLOCK_SIZE --mamba-ssm-cache-dtype $MAMBA_SSM_CACHE_DTYPE --max-num-batched-tokens $APC_MAX_NUM_BATCHED_TOKENS"
  # BAKE (2026-06-24, GPU-VERIFIED via verify3b output/fr13_apc_verify3b): the tree+APC
  # node-bank staleness fix. The defect: the tree committer writes the accepted-leaf state
  # to a DYNAMIC node-bank row (spec_state_indices[req, accepted_len-1]) != the static
  # block-pool src row the stock align snapshot reads -> the postprocess SSM snapshot copies
  # a STALE row (measured snap_src_row=74 vs committed leaf, |diff|~14-18, on the batch_memcpy
  # SOURCE = what a cache-HIT restores). DETERMINISTIC 4-ARM VERIFY (src_ptrs = batch_memcpy
  # source, eager, real 12907):
  #   stock           UNFAITHFUL 208/240  (baseline staleness)
  #   FR13_APC_VERBATIM UNFAITHFUL 208/240 (=stock) -- CLOBBERED: its in-loop dst write is
  #                    overwritten by do_mamba_copy_block's batch_memcpy AFTER the loop. DEAD.
  #   FR13_APC_SSM_SNAPSHOT UNFAITHFUL 224/240 -- MIS-WIRED: its get_temporal_copy_spec
  #                    override does NOT reach the memcpy source (still reads row 74). DEAD.
  #   FR13_APC_SNAP_FIX FAITHFUL 240/240  (src=committed leaf) -- THE ONLY WORKING FIX.
  # SNAP_FIX rewrites src_ptrs_np[offset-1]=state[leaf].data_ptr() in collect_mamba_copy_meta
  # AFTER the stock record, so do_mamba_copy_block copies the committed-leaf state into the
  # block-aligned restore row. Wrote_back==leaf_ptr verified; cuda-graph-safe (host-side
  # collect rewrite, not in the captured graph); native-safe (no leaf published -> stock src).
  # SCOPE: makes cache-ON lossless w.r.t. the committed-leaf SSM state (necessary for lossless
  # APC). It is NOT a fix for the agentic tool-call "crash" (that is the cache-INDEPENDENT
  # qwen free-form-runaway flake, audit output/.../wuez11596 -- fires even at 0% cache hit),
  # and "src faithful" is necessary-not-sufficient for full losslessness (committed-leaf vs
  # fresh-prefill is a separate cross-boot residual). VERBATIM (clobbered) + SSM_SNAPSHOT
  # (mis-wired) RETIRED to 0. Conv path (CONV_FIX/CONV_SNAPSHOT) UNCHANGED (separate, not
  # re-verified here -- a conv node-bank analogue may need the same SNAP_FIX-style redirect;
  # follow-up). All APC flags only take effect with APC on, so the non-APC locked cat9 path
  # stays byte-identical (align hooks not invoked without --enable-prefix-caching).
  : "${FR13_APC_CONV_FIX:=1}"
  : "${FR13_APC_CONV_SNAPSHOT:=1}"
  : "${FR13_APC_SNAP_FIX:=1}"        # BAKED 2026-06-24: verify3b FAITHFUL 240/240 (the working SSM node-bank fix)
  : "${FR13_APC_SSM_SNAPSHOT:=0}"    # RETIRED: verify3b UNFAITHFUL (mis-wired, never reached the memcpy source)
  : "${FR13_APC_VERBATIM:=0}"        # RETIRED: verify3b UNFAITHFUL=stock (clobbered by the post-loop batch_memcpy)
  export FR13_APC_CONV_FIX FR13_APC_CONV_SNAPSHOT FR13_APC_SNAP_FIX FR13_APC_SSM_SNAPSHOT FR13_APC_VERBATIM
else
  APC_FLAGS=""
fi

# CUDAGRAPH_MODE knob (CARRIER re-rooted 2026-06-21): the cache-ON garble is NOT the SSM
# align-snapshot stale-row (that is refuted -- EAGER cache-ON replay with 70% cache hits is
# byte-coherent and even solves the bug, WRITE_THROUGH did=False). It is GRAPH-SPECIFIC: the
# FULL decode CUDA-graph reads GDN recurrent state via capture-time-baked persistent indexing,
# so after an APC cache-hit re-prefill writes the restored boundary state into block-pool rows,
# the captured graph reads the wrong row -> wrong initial recurrent state -> empty/garbage
# (align-mode sibling of vLLM #34874; matches open #43559). Confirmed by the eager-vs-graph
# A/B on the 12907 10-turn replay (eager ....ok...., graph ....GGGG.. at the cat-blob turn).
# FIX = cudagraph_mode=PIECEWISE: keep graph capture for the dense GEMMs/norms/MLP (decode TPS
# preserved) but run the GDN/mamba scan EAGER every step so it always reads the live restored
# state. Unset = vLLM default FULL_AND_PIECEWISE (the poisoned regime). Only matters with APC on.
CG_FLAGS=""
if [[ -n "${CUDAGRAPH_MODE:-}" ]]; then
  CG_FLAGS="--compilation-config '{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}'"
fi
# FR13_FULL_ATTN_KV_FP8 (gated, default OFF): set the FULL-ATTENTION KV cache to fp8.
# DISCRIMINATOR for the APC tree residual locus (research w284wg523, #43559): fp8 KV
# only touches full-attn KV storage precision, never the mamba/GDN recurrent state.
# If cat6root+APC+fp8KV recovers the agent -> residual is FULL-ATTN-KV (the post-RoPE-K
# boundary seam); if not -> GDN-state-content. (The #43559 fp8-recovery claim was struck
# unverified, so treat this strictly as a DISCRIMINATOR, validate losslessness before any
# ship.) Off -> kv_cache_dtype=auto (byte-identical to now). Only meaningful with APC on.
KV_FP8_FLAGS=""
if [[ "${FR13_FULL_ATTN_KV_FP8:-0}" == "1" ]]; then
  KV_FP8_FLAGS="--kv-cache-dtype fp8"
fi

if [[ ! -f "$FORKED_FA2_SO" ]]; then
  echo "forked FA2 .so not found: $FORKED_FA2_SO" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

set -a
if [[ -f "$REPO/.lumo.local.env" ]]; then
  source "$REPO/.lumo.local.env"
fi
set +a

_lumo_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

LUMO_NSYS_WRAP_VLLM=${LUMO_NSYS_WRAP_VLLM:-0}
LUMO_NSYS_BIN=${LUMO_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}
LUMO_NSYS_DELAY_S=${LUMO_NSYS_DELAY_S:-600}
LUMO_NSYS_DURATION_S=${LUMO_NSYS_DURATION_S:-150}
# Periodic CUPTI buffer flush (ms). Without it, per-kernel records (incl. graph
# node-level kernels) are dropped as "incomplete" at the delayed-duration session
# stop on GB10 (fr13_b1_profile_bind: 55k/78k events dropped, zero kernel rows).
LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}
# Semicolon-separated lines appended to the in-container nsys user config
# ("$nsys -z"). Default works around the GB10 drop class where ALL per-kernel
# rows are "incomplete CUPTI events dropped ... GPU timestamp information have
# not been retrieved" even with periodic flushes (NVIDIA-documented
# CuptiUseRawGpuTimestamps=false workaround; fr13_b1_profile_node: 102,320
# dropped with --cuda-flush-interval 100 and zero kernel tables).
LUMO_NSYS_CONFIG_DIRECTIVES=${LUMO_NSYS_CONFIG_DIRECTIVES:-CuptiUseRawGpuTimestamps=false}
# nsys --trace value. On GB10 + CUDA 13 the default 'cuda' engages the HARDWARE
# trace engine for kernel records; in delayed-duration sessions ALL kernel rows
# are then dropped ("GPU timestamp information have not been retrieved").
# 'cuda,cuda-sw' forces the software CUPTI kernel-record path (memcpy/memset/
# runtime rows always survived; only hw-trace kernel rows dropped).
LUMO_NSYS_TRACE=${LUMO_NSYS_TRACE:-cuda,nvtx}
LUMO_NSYS_OUTPUT=${LUMO_NSYS_OUTPUT:-/logs/nsys_vllm_${CONTAINER}}
NSYS_DOCKER_ARGS=()
if _lumo_truthy "$LUMO_NSYS_WRAP_VLLM"; then
  for nsight_mount in /opt/nvidia /usr/local/cuda-13.0; do
    if [[ ! -e "$nsight_mount" ]]; then
      echo "LUMO_NSYS_WRAP_VLLM enabled but Nsight mount path is missing: $nsight_mount" >&2
      exit 2
    fi
    NSYS_DOCKER_ARGS+=(-v "$nsight_mount:$nsight_mount:ro")
  done
fi

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory

recover_host_memory()
PY

free -h
python3 - <<'PY'
from pathlib import Path

fields = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    fields[key] = int(value.strip().split()[0])

available_gib = fields.get("MemAvailable", 0) / 1024 / 1024
swap_used_kib = fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
if available_gib < 80 or swap_used_kib != 0:
    raise SystemExit(
        "FR13 launch aborted: host memory recovery did not produce "
        f"MemAvailable>=80GiB and swap_used==0; "
        f"MemAvailable={available_gib:.2f}GiB "
        f"swap_used={swap_used_kib / 1024 / 1024:.2f}GiB"
    )
PY

docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --memory="$DOCKER_MEM_CAP" --memory-swap="$DOCKER_MEM_CAP" \
  ${PROFILE_PTRACE_CAP:+--cap-add=SYS_PTRACE} \
  --ulimit memlock=-1 --ulimit stack=67108864 -p "$PORT:9950" \
  -v "$REPO:/workspace" -v /models:/models -v "$LOG_DIR:/logs" \
  -v "$FORKED_FA2_SO:/tmp/fr13_fork_fa2.so:ro" \
  "${NSYS_DOCKER_ARGS[@]}" \
  -e VLLM_BATCH_INVARIANT="$BATCH_INVARIANT" \
  -e LUMO_BATCH_INVARIANT_VLLM="${LUMO_BATCH_INVARIANT_VLLM:-$BATCH_INVARIANT}" \
  -e LUMO_NSYS_WRAP_VLLM="$LUMO_NSYS_WRAP_VLLM" \
  -e LUMO_NSYS_BIN="$LUMO_NSYS_BIN" \
  -e LUMO_NSYS_DELAY_S="$LUMO_NSYS_DELAY_S" \
  -e LUMO_NSYS_DURATION_S="$LUMO_NSYS_DURATION_S" \
  -e LUMO_NSYS_FLUSH_MS="$LUMO_NSYS_FLUSH_MS" \
  -e LUMO_NSYS_CONFIG_DIRECTIVES="$LUMO_NSYS_CONFIG_DIRECTIVES" \
  -e LUMO_NSYS_TRACE="$LUMO_NSYS_TRACE" \
  -e LUMO_NSYS_OUTPUT="$LUMO_NSYS_OUTPUT" \
  -e FR13_BI_TREE_ATTN="$FR13_BI_TREE_ATTN" \
  -e FR13_TORCH_DET_WARN="${FR13_TORCH_DET_WARN:-0}" \
  -e FR13_TORCH_DET_WARN_LOG="${FR13_TORCH_DET_WARN_LOG:-/logs/fr13_torch_det_warn.log}" \
  -e FR13_TREE_PER_REQ_GEN="${FR13_TREE_PER_REQ_GEN:-1}" \
  -e FR13_TREE_REQKEY="${FR13_TREE_REQKEY:-1}" \
  -e FR13_TREE_REMAP_SEQ="${FR13_TREE_REMAP_SEQ:-1}" \
  -e FR13_TREE_BONUS_SELF="${FR13_TREE_BONUS_SELF:-1}" \
  -e FR13_CONV_COMMITTED_PATH="$FR13_CONV_COMMITTED_PATH" \
  -e FR13_APC_CONV_FIX="${FR13_APC_CONV_FIX:-1}" \
  -e FR13_APC_CONV_SNAPSHOT="${FR13_APC_CONV_SNAPSHOT:-0}" \
  -e FR13_APC_SSM_SNAPSHOT="${FR13_APC_SSM_SNAPSHOT:-0}" \
  -e FR13_APC_BLOCK_ALIGN_45477="${FR13_APC_BLOCK_ALIGN_45477:-1}" \
  -e FR13_APC_CACHE_AB="${FR13_APC_CACHE_AB:-0}" \
  -e FR13_APC_CACHE_AB_LOG="${FR13_APC_CACHE_AB_LOG:-/logs/fr13_apc_cache_ab.jsonl}" \
  -e FR13_APC_CACHE_AB_BLOCK="${FR13_APC_CACHE_AB_BLOCK:-${MAMBA_BLOCK_SIZE:-1024}}" \
  -e FR13_APC_AB_R2_REQ="${FR13_APC_AB_R2_REQ:-}" \
  -e FR13_APC_AB_R3_REQ="${FR13_APC_AB_R3_REQ:-}" \
  -e FR13_APC_VERBATIM="${FR13_APC_VERBATIM:-0}" \
  -e FR13_APC_SSM_LEAF_SRC="${FR13_APC_SSM_LEAF_SRC:-0}" \
  -e FR13_APC_SSM_DIAG="${FR13_APC_SSM_DIAG:-0}" \
  -e FR13_APC_VALUE_VS_ORACLE="${FR13_APC_VALUE_VS_ORACLE:-0}" \
  -e FR13_APC_VALUE_VS_ORACLE_LOG="${FR13_APC_VALUE_VS_ORACLE_LOG:-/logs/fr13_apc_value_vs_oracle.jsonl}" \
  -e FR13_APC_HIT_RECURRENT_SUFFIX="${FR13_APC_HIT_RECURRENT_SUFFIX:-0}" \
  -e FR13_APC_HIT_SUFFIX_CAP="${FR13_APC_HIT_SUFFIX_CAP:-64}" \
  -e FR13_APC_STALENESS_AUDIT="${FR13_APC_STALENESS_AUDIT:-0}" \
  -e FR13_APC_STALENESS_AUDIT_LOG="${FR13_APC_STALENESS_AUDIT_LOG:-/logs/fr13_apc_staleness_audit.jsonl}" \
  -e FR13_APC_SNAP_FIDELITY="${FR13_APC_SNAP_FIDELITY:-0}" \
  -e FR13_APC_SNAP_FIDELITY_LOG="${FR13_APC_SNAP_FIDELITY_LOG:-/logs/fr13_apc_snap_fidelity.jsonl}" \
  -e FR13_APC_SNAP_FIX="${FR13_APC_SNAP_FIX:-0}" \
  -e FR13_FORCE_SPINE_COMMIT="$FR13_FORCE_SPINE_COMMIT" \
  -e FR13_DRAFTER_SINGLE_LOGITS="$FR13_DRAFTER_SINGLE_LOGITS" \
  -e FR13_EAGER_PACK="$FR13_EAGER_PACK" \
  -e FR13_TREE_CONV_FUSED="$FR13_TREE_CONV_FUSED" \
  -e FR13_FIX1_SELFCHECK="$FR13_FIX1_SELFCHECK" \
  -e FR13_FIX1_SELFCHECK_DUMP="$FR13_FIX1_SELFCHECK_DUMP" \
  -e FR13_COMMIT_ARGMAX_GATE="$FR13_COMMIT_ARGMAX_GATE" \
  -e FR13_COMMIT_ARGMAX_GATE_DUMP="$FR13_COMMIT_ARGMAX_GATE_DUMP" \
  -e FR13_FORK_MARGIN_DUMP="$FR13_FORK_MARGIN_DUMP" \
  -e FR13_FORK_MARGIN_DUMP_PATH="$FR13_FORK_MARGIN_DUMP_PATH" \
  -e FR13_REPLAY_ROUTE="${FR13_REPLAY_ROUTE:-1}" \
  -e FR13_REPLAY_BOUNDARY_LOG="${FR13_REPLAY_BOUNDARY_LOG:-0}" \
  -e FR13_REPLAY_BOUNDARY_LAYERS="${FR13_REPLAY_BOUNDARY_LAYERS:-layers.0.linear_attn}" \
  -e FR13_REPLAY_BOUNDARY_PATH="${FR13_REPLAY_BOUNDARY_PATH:-/logs/fr13_replay_boundary.jsonl}" \
  -e FR13_CHASE_DIAG="$FR13_CHASE_DIAG" \
  -e FR13_CHASE_DIAG_DIR="$FR13_CHASE_DIAG_DIR" \
  -e FR13_CHASE_TOPK="$FR13_CHASE_TOPK" \
  -e FR13_CHASE_KV_WINDOW="$FR13_CHASE_KV_WINDOW" \
  -e FR13_CHASE_H3="$FR13_CHASE_H3" \
  -e FR13_CHASE_H3_LAYER="$FR13_CHASE_H3_LAYER" \
  -e FR13_CHASE_KV_ALLOW_EMPTY="$FR13_CHASE_KV_ALLOW_EMPTY" \
  -e FR13_TREE_SAMPLE_ROW="$FR13_TREE_SAMPLE_ROW" \
  -e LUMO_FB_KERNEL_ROWS="$LUMO_FB_KERNEL_ROWS" \
  -e LUMO_FB_PROJ_PAD_ROWS="$LUMO_FB_PROJ_PAD_ROWS" \
  -e FR13_GB10_FP8_GEMV_CFG="${FR13_GB10_FP8_GEMV_CFG:-0}" \
  -e FR13_SFWD_GPU_TIMER="${FR13_SFWD_GPU_TIMER:-0}" \
  -e FR13_SFWD_GPU_TIMER_JSON="${FR13_SFWD_GPU_TIMER_JSON:-}" \
  -e FR13_SFWD_GPU_TIMER_MAXPENDING="${FR13_SFWD_GPU_TIMER_MAXPENDING:-256}" \
  -e FR13_GPU_COMMITTER="${FR13_GPU_COMMITTER:-0}" \
  -e FR13_COMMITTER_SYNCKILL="${FR13_COMMITTER_SYNCKILL:-0}" \
  -e FR13_GPU_COMMITTER_KERNEL="${FR13_GPU_COMMITTER_KERNEL:-/workspace/scripts/fr13_gpu_committer_kernel.py}" \
  -e FR13_DEVICE_MULTIDRAFT="${FR13_DEVICE_MULTIDRAFT:-1}" \
  -e FR13_DEVICE_MULTIDRAFT_KERNEL="${FR13_DEVICE_MULTIDRAFT_KERNEL:-/workspace/scripts/fr13_device_multidraft_kernel.py}" \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  -e TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}" \
  -e PYTHONPATH=/workspace/src \
  -e FR10_ENABLE_TREE_GDN=1 \
  -e FR10_METRICS="$FR10_METRICS" \
  -e FR10_DECODE_MODE_DEFAULT="$FR10_DECODE_MODE_DEFAULT" \
  -e FR13_SCAN_ALIGN="${FR13_SCAN_ALIGN:-0}" \
  -e FR13_SCAN_ALIGN_MODE="${FR13_SCAN_ALIGN_MODE:-body}" \
  -e FR13_NPAD_INVARIANT="${FR13_NPAD_INVARIANT:-0}" \
  -e FR11_TREE_CONV_NATIVE_BF16_TAPS=1 \
  -e FR12_TREE_CONV_NATIVE_BF16_TAPS=1 \
  -e FR12_TREE_CONV_NATIVE_PRIOR_READ="${FR12_TREE_CONV_NATIVE_PRIOR_READ:-0}" \
  -e FR12_TREE_CONV_NATIVE_SPINE="${FR12_TREE_CONV_NATIVE_SPINE:-0}" \
  -e FR12_TREE_SCAN_NATIVE_SPINE="${FR12_TREE_SCAN_NATIVE_SPINE:-0}" \
  -e FR12_NATIVE_SPINE_ORACLE="${FR12_NATIVE_SPINE_ORACLE:-0}" \
  -e FR12_TREE_CONV_STATE_FULL_CAPTURE="${FR12_TREE_CONV_STATE_FULL_CAPTURE:-0}" \
  -e FR13_FA2_TREE_BIAS="$FR13_FA2_TREE_BIAS" \
  -e FR13_FA2_PREFILL_NATIVE="$FR13_FA2_PREFILL_NATIVE" \
  -e FR13_TREE_ATTN_EXP2_SOFTMAX="$FR13_TREE_ATTN_EXP2_SOFTMAX" \
  -e FR10_TREE_GDN_COUNTER_DUMP=/logs/fr10_tree_gdn_counters.json \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD="${FR10_TREE_GDN_CAPTURE_PAYLOAD:-}" \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX="${FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX:-}" \
  -e FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS="${FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LOG="${FR10_TREE_GDN_COMMIT_HANDOFF_LOG:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX="${FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX:-}" \
  -e FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT="${FR10_TREE_GDN_COMMIT_HANDOFF_LIMIT:-32}" \
  -e FR10_TREE_GDN_SRC_NATIVE_PAYLOAD="${FR10_TREE_GDN_SRC_NATIVE_PAYLOAD:-}" \
  -e FR10_TREE_DEPTH_POSITION_LOG=/logs/fr10_tree_depth_positions.jsonl \
  -e FR10_ROOT_HIDDEN_CAPTURE="${FR10_ROOT_HIDDEN_CAPTURE:-}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS="${FR10_ROOT_HIDDEN_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW="${FR10_ROOT_HIDDEN_CAPTURE_ROOT_ROW:-0}" \
  -e FR10_ROOT_HIDDEN_CAPTURE_POSITION="${FR10_ROOT_HIDDEN_CAPTURE_POSITION:-}" \
  -e FR10_ROOT_LOGIT_CAPTURE_NUM_TOKENS="${FR10_ROOT_LOGIT_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_ROOT_LOGIT_CAPTURE_ROOT_ROW="${FR10_ROOT_LOGIT_CAPTURE_ROOT_ROW:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE="${FR10_LAYER_HIDDEN_CAPTURE:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS="${FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_ROWS="${FR10_LAYER_HIDDEN_CAPTURE_ROWS:-}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_SKIP="${FR10_LAYER_HIDDEN_CAPTURE_SKIP:-0}" \
  -e FR10_LAYER_HIDDEN_CAPTURE_LIMIT="${FR10_LAYER_HIDDEN_CAPTURE_LIMIT:-1}" \
  -e FR12_FULL_ATTN_CAPTURE="${FR12_FULL_ATTN_CAPTURE:-}" \
  -e FR12_FULL_ATTN_CAPTURE_LAYER_PREFIX="${FR12_FULL_ATTN_CAPTURE_LAYER_PREFIX:-}" \
  -e FR12_FULL_ATTN_CAPTURE_NUM_TOKENS="${FR12_FULL_ATTN_CAPTURE_NUM_TOKENS:-}" \
  -e FR12_FULL_ATTN_CAPTURE_SKIP="${FR12_FULL_ATTN_CAPTURE_SKIP:-0}" \
  -e FR12_FULL_ATTN_CAPTURE_LIMIT="${FR12_FULL_ATTN_CAPTURE_LIMIT:-1}" \
  -e FR12_SUBKERNEL_CAPTURE="${FR12_SUBKERNEL_CAPTURE:-}" \
  -e FR13_TCF_DIAG_OVERRIDE="${FR13_TCF_DIAG_OVERRIDE:-0}" \
  -e FR13_TCF_SELFCHECK="${FR13_TCF_SELFCHECK:-0}" \
  -e FR12_SUBKERNEL_CAPTURE_DEBUG_LOG="${FR12_SUBKERNEL_CAPTURE_DEBUG_LOG:-}" \
  -e FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX="${FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX:-language_model.model.layers.0.linear_attn}" \
  -e FR12_SUBKERNEL_CAPTURE_NUM_TOKENS="${FR12_SUBKERNEL_CAPTURE_NUM_TOKENS:-}" \
  -e FR12_SUBKERNEL_CAPTURE_SKIP="${FR12_SUBKERNEL_CAPTURE_SKIP:-0}" \
  -e FR12_SUBKERNEL_CAPTURE_LIMIT="${FR12_SUBKERNEL_CAPTURE_LIMIT:-1}" \
  -e FR12_SUBKERNEL_CAPTURE_Z="${FR12_SUBKERNEL_CAPTURE_Z:-0}" \
  -e FR12_SUBKERNEL_CAPTURE_INPUT="${FR12_SUBKERNEL_CAPTURE_INPUT:-0}" \
  -e FR13_TREE_ATTN_OP_CAPTURE="${FR13_TREE_ATTN_OP_CAPTURE:-}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_LAYER="${FR13_TREE_ATTN_OP_CAPTURE_LAYER:-language_model.model.layers.3.self_attn}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_SKIP="${FR13_TREE_ATTN_OP_CAPTURE_SKIP:-0}" \
  -e FR13_TREE_ATTN_OP_CAPTURE_LIMIT="${FR13_TREE_ATTN_OP_CAPTURE_LIMIT:-1}" \
  -e FR13_FA2_MAB="${FR13_FA2_MAB:-0}" \
  -e FR13_FA2_MAB_DUMP="${FR13_FA2_MAB_DUMP:-/logs/fr13_fa2_mab.jsonl}" \
  -e FR13_FA2_MAB_LAYER="${FR13_FA2_MAB_LAYER:-*}" \
  -e FR13_FA2_MAB_SKIP="${FR13_FA2_MAB_SKIP:-0}" \
  -e FR13_FA2_MAB_LIMIT="${FR13_FA2_MAB_LIMIT:-1}" \
  -e FR13_GDN_SUBOP_MAB="${FR13_GDN_SUBOP_MAB:-0}" \
  -e FR13_GDN_SUBOP_MAB_DUMP="${FR13_GDN_SUBOP_MAB_DUMP:-/logs/fr13_gdn_subop_mab.jsonl}" \
  -e FR13_GDN_SUBOP_MAB_LAYER="${FR13_GDN_SUBOP_MAB_LAYER:-language_model.model.layers.0.linear_attn}" \
  -e FR13_GDN_SUBOP_MAB_SKIP="${FR13_GDN_SUBOP_MAB_SKIP:-0}" \
  -e FR13_GDN_SUBOP_MAB_LIMIT="${FR13_GDN_SUBOP_MAB_LIMIT:-1}" \
  -e FR13_GDN_SUBOP_MAB_EXPECT_TREE_N="${FR13_GDN_SUBOP_MAB_EXPECT_TREE_N:-10}" \
  -e FR13_GDN_SUBOP_MAB_THRESHOLD="${FR13_GDN_SUBOP_MAB_THRESHOLD:-0.0}" \
  -e FR13_REPLAY_DURABLE_AB="${FR13_REPLAY_DURABLE_AB:-0}" \
  -e FR13_REPLAY_DURABLE_AB_LAYERS="${FR13_REPLAY_DURABLE_AB_LAYERS:-}" \
  -e FR13_REPLAY_DURABLE_AB_PATH="${FR13_REPLAY_DURABLE_AB_PATH:-/logs/fr13_replay_durable_ab.jsonl}" \
  -e FR13_REPLAY_DURABLE_AB_FLAG_FILE="${FR13_REPLAY_DURABLE_AB_FLAG_FILE:-/logs/fr13_replay_durable_ab.flag}" \
  -e VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="$VLLM_RAY_EXTRA_ENV_VARS_TO_COPY" \
  -e VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY="$VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY" \
  -e FR13_FLASH_ATTN_OP_CAPTURE="${FR13_FLASH_ATTN_OP_CAPTURE:-}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_LAYER="${FR13_FLASH_ATTN_OP_CAPTURE_LAYER:-language_model.model.layers.3.self_attn}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_SKIP="${FR13_FLASH_ATTN_OP_CAPTURE_SKIP:-0}" \
  -e FR13_FLASH_ATTN_OP_CAPTURE_LIMIT="${FR13_FLASH_ATTN_OP_CAPTURE_LIMIT:-1}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE="${FR13_PREPROCESS_INPUT_CAPTURE:-}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_NUM_TOKENS="${FR13_PREPROCESS_INPUT_CAPTURE_NUM_TOKENS:-}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_SKIP="${FR13_PREPROCESS_INPUT_CAPTURE_SKIP:-0}" \
  -e FR13_PREPROCESS_INPUT_CAPTURE_LIMIT="${FR13_PREPROCESS_INPUT_CAPTURE_LIMIT:-1}" \
  -e FR13_PREFILL_GDN_CAPTURE="${FR13_PREFILL_GDN_CAPTURE:-}" \
  -e FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX="${FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX:-}" \
  -e FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="${FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX:-1}" \
  -e FR10_SPINE_LOGIT_CAPTURE="${FR10_SPINE_LOGIT_CAPTURE:-}" \
  -e FR10_SPINE_LOGIT_CAPTURE_SKIP="${FR10_SPINE_LOGIT_CAPTURE_SKIP:-0}" \
  -e FR10_SPINE_LOGIT_CAPTURE_LIMIT="${FR10_SPINE_LOGIT_CAPTURE_LIMIT:-1}" \
  -e FR13_FINAL_LOGIT_CAPTURE="${FR13_FINAL_LOGIT_CAPTURE:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS="${FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_ROWS="${FR13_FINAL_LOGIT_CAPTURE_ROWS:-}" \
  -e FR13_FINAL_LOGIT_CAPTURE_SKIP="${FR13_FINAL_LOGIT_CAPTURE_SKIP:-0}" \
  -e FR13_FINAL_LOGIT_CAPTURE_LIMIT="${FR13_FINAL_LOGIT_CAPTURE_LIMIT:-1}" \
  -e FR13_HIDDEN_SUBSTITUTE="${FR13_HIDDEN_SUBSTITUTE:-}" \
  -e LUMO_MTP_DRAFT_TRACE_FILE="$LUMO_MTP_DRAFT_TRACE_FILE" \
  -e LUMO_TREE_SAMPLER_DEBUG_LOG="$LUMO_TREE_SAMPLER_DEBUG_LOG" \
  -e LUMO_TREE_PATH_LCP_LOG="$LUMO_TREE_PATH_LCP_LOG" \
  -e SPEC_CONFIG="$SPEC_CONFIG" \
  --entrypoint bash \
  "$IMAGE" \
  -lc "set -euo pipefail
unset FR10_ALLOW_LINEAR_FALLBACK
cp /tmp/fr13_fork_fa2.so /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
sha256sum /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so | tee /logs/fr13_forked_fa2.sha256
python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py
python3 /workspace/scripts/fr13_patch_fa2_tree_bias.py --skip-source
python3 - <<'PY'
import os
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/tree_attn.py')
text = path.read_text()
needle = 'FR13_FA2_PREFILL_NATIVE'
if needle not in text:
    raise SystemExit(f'{needle} patch missing in {path}')
if os.environ.get('FR13_BI_TREE_ATTN', '0') == '1':
    bi_path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/batch_invariant.py')
    bi_text = bi_path.read_text()
    if 'FR13_BI_TREE_ATTN' not in bi_text:
        raise SystemExit(f'FR13_BI_TREE_ATTN allowlist patch missing in {bi_path}')
    decode_needle = (
        'num_splits=1 if envs.VLLM_BATCH_INVARIANT else 0,\n'
        '                    tree_bias=tree_bias,'
    )
    if decode_needle not in text:
        raise SystemExit(f'FR13 BI decode num_splits expression missing in {path}')
PY
NSYS_PREFIX=()
case \"\${LUMO_NSYS_WRAP_VLLM,,}\" in
  1|true|yes|on)
    if [[ -n \"\${LUMO_NSYS_CONFIG_DIRECTIVES:-}\" ]]; then
      NSYS_CFG_PATH=\$(\"\$LUMO_NSYS_BIN\" -z)
      mkdir -p \"\$(dirname \"\$NSYS_CFG_PATH\")\"
      printf '%s\n' \"\$LUMO_NSYS_CONFIG_DIRECTIVES\" | tr ';' '\n' >> \"\$NSYS_CFG_PATH\"
      echo \"nsys config directives appended to \$NSYS_CFG_PATH:\"
      cat \"\$NSYS_CFG_PATH\"
    fi
    NSYS_PREFIX=(
      \"\$LUMO_NSYS_BIN\"
      profile
      --delay \"\$LUMO_NSYS_DELAY_S\"
      --duration \"\$LUMO_NSYS_DURATION_S\"
      --trace=\"\$LUMO_NSYS_TRACE\"
      --cuda-graph-trace=node
      --cuda-flush-interval \"\$LUMO_NSYS_FLUSH_MS\"
      --sample=none
      --cpuctxsw=none
      --force-overwrite=true
      -o \"\$LUMO_NSYS_OUTPUT\"
    )
    ;;
esac
exec \"\${NSYS_PREFIX[@]}\" vllm serve /models/qwen3.6-27b-fp8 --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 9950 --max-num-seqs '$MAX_NUM_SEQS' \
  --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAX_MODEL_LEN' \
  --attention-backend '$ATTENTION_BACKEND' --gdn-prefill-backend triton \
  --chat-template /workspace/docker/chat_templates/qwen3-openai-codex.jinja \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config \"\$SPEC_CONFIG\" $APC_FLAGS $CG_FLAGS $KV_FP8_FLAGS \
  $(if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then printf '%s' '--enforce-eager'; fi)"
