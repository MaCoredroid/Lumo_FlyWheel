# FR13 CONFIDENCE-GATED skip validation (merge16d verdict fix). ONE arm: merged with a min-prob gate on
# the skip (skip only when arctic's match confidence >= FR13_MERGED_SKIP_MIN_PROB, else full MTP =
# never-regress). Compared against merge16d's SAME-CONFIG baseline (accept 3.61, fullstep_tps 18.82).
# The [FR13_MERGED ENGAGED] needle's conf_gated/skip_fired self-calibrate the threshold live (watch the
# first needle: conf_gated ~= match_full => threshold too high; conf_gated ~= 0 => too low/blanket).
# Canonical offload/proxy + ship-cache env (identical to fr13_merged_ab_seq.sh).
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR10_METRICS=0
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1

# The confidence threshold (overridable at launch via SKIP_MIN_PROB=...). 0.5 = node must have appeared
# >=50% of the time after its prefix in-context (a strong repeat) to be skipped.
export FR13_MERGED_SKIP_MIN_PROB=${SKIP_MIN_PROB:-0.5}

export FR13_DRAFT_SOURCE=merged
run_variant merged_gated_t33333_${TAG}  t33333  15  1
