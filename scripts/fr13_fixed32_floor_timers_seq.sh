# Fixed-work floor campaign: Tail23 and Hydra27 use the same 31 physical
# drafts plus the implicit root. The arm kind changes only the sampler validity
# mask; every launch must report draft_tokens/drafts=31.
case "${BSIZE:-}" in
  1|4) ;;
  *)
    echo "fixed32 floor sequence requires BSIZE=1 or 4, got: ${BSIZE:-unset}" >&2
    exit 2
    ;;
esac
[[ "${CONC:-}" == "$BSIZE" ]] || {
  echo "fixed32 floor sequence requires CONC=BSIZE, got CONC=${CONC:-unset} BSIZE=$BSIZE" >&2
  exit 2
}
FR13_DRAFT_VOCAB_ROOT=${FR13_DRAFT_VOCAB_ROOT:-0}
case "$FR13_DRAFT_VOCAB_ROOT" in
  0|1) export FR13_DRAFT_VOCAB_ROOT ;;
  *)
    echo "FR13_DRAFT_VOCAB_ROOT must be 0 or 1" >&2
    exit 2
    ;;
esac
FR13_DRAFT_HEAD_FP8=${FR13_DRAFT_HEAD_FP8:-0}
case "$FR13_DRAFT_HEAD_FP8" in
  0|1) export FR13_DRAFT_HEAD_FP8 ;;
  *)
    echo "FR13_DRAFT_HEAD_FP8 must be 0 or 1" >&2
    exit 2
    ;;
esac

export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
_FR13_FIXED32_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_FR13_FIXED32_REPO=${REPO:-$(cd "$_FR13_FIXED32_SCRIPT_DIR/.." && pwd)}
export HF_HOME=${HF_HOME:-"$_FR13_FIXED32_REPO/.cache/huggingface"}
unset _FR13_FIXED32_SCRIPT_DIR _FR13_FIXED32_REPO
export HF_HUB_OFFLINE=1
export DEPLOY_FORCE_TEMP=0.6
export FR10_METRICS=0
export SWE_EMPTY_PATCH_RETRIES=0
# Formal fixed32 campaigns publish curated summaries only. The generic SWE
# runner otherwise force-adds raw per-task traces after each completed task.
export LUMO_SWE_AUTOCOMMIT=0
export BATCH_INVARIANT=0
export FR13_HYDRA23=0
export FR13_TAIL_BRANCHES=0
export FR13_TAIL_BRANCH_DEPTHS=0

# Acceptance-grade asynchronous stage and wall telemetry.
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_SFWD_GPU_TIMER_MAXPENDING=256
export FR13_SFWD_GPU_TIMER_SAMPLES_MAX=200000
# Live sidecar rewrites run on the decode thread. Formal runs persist complete
# samples only through the explicit task/final boundary transaction.
export FR13_SFWD_GPU_TIMER_DUMP_S=0
export FR13_SFWD_SAMPLES_DUMP_S=30
export FR13_SPAN_GPU_TIMER_DUMP_S=0
export FR13_STEP_WALL_CAP_S=1.5
export FR13_TIMER_EXPLICIT_FLUSH=1
export FR13_FIXED32_WORK_CENSUS=1
export FR13_FIXED32_DEVICE_PUBLISH=1
export FR13_FIXED32_ACCEPT_PACK=1
export FR13_FIXED32_REQKEY_DEVICE=1
export FR13_FIXED32_KV_REMAP16=1
export FR13_FIXED32_COMMIT_DEVICE_FILL=1
export FR13_FIXED32_TAW_WALK_CAP=12
export FR13_DEVICE_MULTIDRAFT=1
export FR13_DRAFTER_GRAPH=1
export FR13_DRAFTER_SINGLE_LOGITS=1
export FR13_DM_DEPTHSYNC=1
export FR13_TAW=1
export FR13_PARENT_GATHER=1
export FR13_SUBTREE_PARALLEL=1
export FR13_EAGER_PACK=1
export FR13_COMMIT_BATCH_OUTPUT=1
export FR13_COMMITTER_NATIVE=1
export FR13_COMMITTER_BATCHED=1
export FR13_COMMITTER_GRAPH=1
export FR13_RING_EXPORT=1
export FR13_REPLAY_ROUTE=1
export FR13_ATTN_KV_REMAP=1
export FR13_SLOT_REORDER=1
export FR13_KV_REMAP_SYNCFREE=1
export FR13_FA2_TREE_BIAS=1
export FR13_TREE_CONV_FUSED=1
export FR13_CONV_WB_FUSED=1
export FR13_CONV_WB_BATCHED=1
FR13_FIXED32_CONV_SOURCE_BATCH=${FR13_FIXED32_CONV_SOURCE_BATCH:-0}
case "$FR13_FIXED32_CONV_SOURCE_BATCH" in
  0|1) export FR13_FIXED32_CONV_SOURCE_BATCH ;;
  *)
    echo "FR13_FIXED32_CONV_SOURCE_BATCH must be 0 or 1" >&2
    exit 2
    ;;
esac
export FR13_CONV_PREGATHER=1
export FR13_CONV_COMMITTED_PATH=1
export FR13_APC_COMMIT_TO_RUNNING_ROW=1
export FR13_TREE_RUNROW_INIT=1
export FR13_FLAGS_INKERNEL=1
# Optimistic mandatory-weight-read floor only. Only the exact logical head-read
# ledgers below are admitted; inherited byte/floor declarations are replaced.
#
# FR14 ARM B (2026-08-16): every byte below is re-derived by SUMMING the served
# RadixArk/Qwen3.8-27B-NVFP4 checkpoint's real tensor spans -- see
# scripts/fr13_hardware_floor_ledger.py (--derive-from-checkpoint reproduces
# the arithmetic against the three shards) and
# results/fr14_nvfp4_port_20260816/floor_ledger_radixark.json.
# Nothing here is a constant scaled by a quantisation ratio. The floor still
# does not halve: 119.658015414 (FR13 fp8-3.6) -> 102.479937172 (arm A,
# conservative bytes) -> 92.345089436 (arm B, aggressive bytes) = 0.772x of
# FR13, because the BF16 MTP head got BIGGER and PHASE 1 keeps the five K64
# draft-head reads in BF16. What arm B buys over arm A is almost entirely the
# HEAD: NVFP4 lm_head 0.715 GB vs BF16 2.543 GB. The same aggressive backbone
# with a BF16 head lands at 99.040 ms -- 3.44 ms, i.e. nothing.
# Every ONE_SIDED_U95_CAP_MS derived from these floors is 1.15 x floor and is
# PROVISIONAL: the FR14 objective bar is Mark's open ruling
# (results/fr14_nvfp4_port_20260816/README.md, "Correctness bar -- PROPOSED,
# AWAITING MARK"). No FR13 acceptance number transfers across a lossy requant.
case "${FR13_DRAFT_VOCAB_K:-65536}:$FR13_DRAFT_VOCAB_ROOT" in
  0:0)
    export FR13_MANDATORY_WEIGHT_BYTES=25430574256
    export FR13_WEIGHT_FLOOR_MS=93.15228665201465
    ;;
  65536:0)
    export FR13_MANDATORY_WEIGHT_BYTES=25254282384
    export FR13_WEIGHT_FLOOR_MS=92.506528879
    ;;
  65536:1)
    if [[ "$FR13_DRAFT_HEAD_FP8" == "1" ]]; then
      # FR14 RETIRED ARM. The FR13 sub-arm re-read the five K64 draft heads
      # as FP8 qweight + FP32 scales (1,678,131,200 B instead of
      # 3,355,443,200 B) and claimed a 113.514015414 ms floor for it.
      # Neither FR14 arm leaves an FP8 head to read. Arm A: unsloth shipped an
      # FP8 per-channel head, this vLLM's qwen3_5 built lm_head as an
      # UNQUANTIZED ParallelLMHead and refused to load it, and the lm_head
      # surgery dequantised it to BF16 to make the checkpoint bootable
      # (results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md pass 6). Arm B
      # (live): the head IS quantised -- NVFP4 -- but PHASE 1 of the DVK port
      # dequantises the 65,536 sliced rows to BF16 at boot so the sealed BF16
      # GEMV units and the 128-id block map stay byte-identical, so the five
      # draft-head reads are BF16 all the same.
      # Either way the arm would pin a floor the hardware cannot realise.
      # Refuse rather than measure nothing. The live successor is PHASE 2 --
      # reading those slices AS NVFP4, 188,743,680 B each instead of
      # 671,088,640, worth 8.834 ms (92.345 -> 83.511) -- which needs an FP4
      # GEMV unit and carries its own byte gate.
      echo "FR13_DRAFT_HEAD_FP8 is RETIRED under the FR14 NVFP4 checkpoint: the served head is NVFP4 and its K64 slice is dequantised to BF16 at boot, so the FP8 draft-head floor is unrealisable" >&2
      exit 2
    fi
    export FR13_MANDATORY_WEIGHT_BYTES=25210209416
    export FR13_WEIGHT_FLOOR_MS=92.345089436
    ;;
  *)
    echo "unsupported fixed32 draft-vocab floor configuration: K=${FR13_DRAFT_VOCAB_K:-unset} ROOT=$FR13_DRAFT_VOCAB_ROOT" >&2
    exit 2
    ;;
esac
# FR14: UNCHANGED at 0.54 on purpose. This is the fp8-era MEASURED per-row
# compute cost (feeds floor_ms = max(weight_floor, compute_floor) in
# fr13_measure.py). It is not re-derivable without a GPU, and under NVFP4 the
# same GEMM shapes run on 4-bit tensor-core paths, so 0.54 ms/row is a
# CONSERVATIVE (high) bound here -- it can only make the compute floor bind
# earlier, never later. Re-measure it on the first FR14 B1 profile run.
export FR13_COMPUTE_MS_PER_ROW=0.54
export MAX_MODEL_LEN=131072
export FR13_ENABLE_APC=1
export FR13_APC_CONFIG_ONLY=0
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export APC_MAX_NUM_BATCHED_TOKENS=4096
export APC_BLOCK_SIZE=1024
export LUMO_LONG_PREFILL_THRESHOLD=1024
export CUDAGRAPH_MODE=FULL_AND_PIECEWISE
export FR13_FULL_ATTN_KV_FP8=0
export ENFORCE_EAGER=0
export FR13_SERVE_BATCH_FLAGS=""

# Alternate execution routes and synchronizing diagnostics stay off.
export FR13_MULTIDRAFT_GPU_TIMER=0
export FR13_REPLAY_GPU_TIMER=0
export FR13_COMMIT_FULL_GPU_TIMER=0
export FR13_COMMITTER_SG_TIMER=0
export FR13_REPLAY_ONLY_GPU_TIMER=0
export FR13_GRAPH_TIMER=0
export FR13_KVREMAP_TIMER=0
export FR13_STATEREMAP_TIMER=0
export FR13_DFWD_SPLIT_NEEDLE=0
export FR13_STEP_GRAPH=0
export FR13_COMMIT_OVERLAP=0
export FR13_REPLAY_MULTISTREAM=0
export CUDA_LAUNCH_BLOCKING=0

# Fail closed against inherited probes, profilers, and self-check routes.
export FR13_BRANCH_ACCEPT_DIAG=0
export FR13_FORCE_SPINE_COMMIT=0
export FR13_FIX1_SELFCHECK=0
export FR13_COMMIT_ARGMAX_GATE=0
export FR13_FORK_MARGIN_DUMP=0
export FR13_CHASE_DIAG=0
export FR13_REPLAY_BOUNDARY_LOG=0
export FR13_GDN_SUBOP_MAB=0
export FR13_CONV_SUBOP_MAB=0
export FR13_FA2_MAB=0
export FR13_REPLAY_DURABLE_AB=0
export FR13_TREE_POSREAD_PROBE=0
export FR13_LEAK_PROBE=0
export FR13_SERVE_LOG=0
export FR13_TORCH_DET_WARN=0
export FR13_TCF_DIAG_OVERRIDE=0
export FR13_TCF_SELFCHECK=0
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
export FR13_PARENT_GATHER_SELFCHECK=0
export FR13_TORCHPROF=0
export FR13_TORCH_PROF=""
export FR13_DVK_DRAFTID_DUMP=""
export FR10_TREE_GDN_CAPTURE_PAYLOAD=""
export FR10_TREE_GDN_COMMIT_HANDOFF_LOG=""
export FR10_TREE_GDN_SRC_NATIVE_PAYLOAD=""
export FR10_ROOT_HIDDEN_CAPTURE=""
export FR10_ROOT_LOGIT_CAPTURE=""
export FR10_LAYER_HIDDEN_CAPTURE=""
export FR12_FULL_ATTN_CAPTURE=""
export FR12_SUBKERNEL_CAPTURE=""
export FR13_TREE_ATTN_OP_CAPTURE=""
export FR13_FLASH_ATTN_OP_CAPTURE=""
export FR13_PREPROCESS_INPUT_CAPTURE=""
export FR13_PREFILL_GDN_CAPTURE=""
export FR13_DECODE_GDN_CAPTURE=""
export FR10_SPINE_LOGIT_CAPTURE=""
export FR13_FINAL_LOGIT_CAPTURE=""
export FR13_HIDDEN_SUBSTITUTE=""
export LUMO_MTP_DRAFT_TRACE_FILE=""
export LUMO_TREE_SAMPLER_DEBUG_LOG=""
export LUMO_TREE_PATH_LCP_LOG=""
case "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" in
  0)
    export FR13_FIXED32_NVTX_PROFILE=0
    export LUMO_NSYS_WRAP_VLLM=0
    ;;
  1)
    [[ "${LUMO_NSYS_WRAP_VLLM:-0}" == "1" ]] || {
      echo "fixed32 attribution mode requires LUMO_NSYS_WRAP_VLLM=1" >&2
      exit 2
    }
    export FR13_FIXED32_NVTX_PROFILE=1
    ;;
  *)
    echo "FR13_FIXED32_ATTRIBUTION_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac
export LUMO_FA_ACTIVATION_REPLAY_BATCH4_DIAG=0
export LUMO_FA_REPLAY_COMMIT_BATCH4_RUNNER_DIAG=0
export LUMO_IR_DIAGNOSTIC_UNISOLATED=0
export LUMO_IR_ALLOW_UNVERIFIED_SPINES2_MEASUREMENT=0

if [[ "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" == "1" ]]; then
  # The bounded profiler wraps one server lifecycle. Never schedule a second
  # arm after its report is finalized and the wrapped target is stopped.
  #
  # FR14: the profiled TOPOLOGY is a parameter, because attribution must describe
  # the config it claims to describe. It stays tail6 by default so every banked
  # FR13 profile reproduces byte-for-byte; the promoted FR14 production config is
  # hydra27, and profiling it under a hardcoded tail6 arm would have produced a
  # table labelled with a topology it never ran.
  _FR13_ATTRIBUTION_MODE=${FR13_FIXED32_ATTRIBUTION_MODE:-tail6_fixed32}
  case "$_FR13_ATTRIBUTION_MODE" in
    tail6_fixed32 | hydra27_fixed32) ;;
    *)
      echo "FR13_FIXED32_ATTRIBUTION_MODE must be tail6_fixed32 or hydra27_fixed32," \
           "got: $_FR13_ATTRIBUTION_MODE" >&2
      exit 2
      ;;
  esac
  run_variant "${_FR13_ATTRIBUTION_MODE}_${TAG}" "$_FR13_ATTRIBUTION_MODE" 31 1
  unset _FR13_ATTRIBUTION_MODE
else
  case "${FR13_FLOOR_ORDER:-TH}" in
    TH)
      run_variant tail6_fixed32_${TAG} tail6_fixed32 31 1
      run_variant hydra27_fixed32_${TAG} hydra27_fixed32 31 1
      ;;
    HT)
      run_variant hydra27_fixed32_${TAG} hydra27_fixed32 31 1
      run_variant tail6_fixed32_${TAG} tail6_fixed32 31 1
      ;;
    *)
      echo "FR13_FLOOR_ORDER must be TH or HT, got: ${FR13_FLOOR_ORDER:-}" >&2
      exit 2
      ;;
  esac
fi
