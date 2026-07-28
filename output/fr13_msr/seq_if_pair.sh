# INSTRUMENT-FULL ATTRIBUTION PAIR (queue slot 2i) — the "two kinds of runs"
# discipline: CLEAN arms (no instruments) are the speed-of-record;
# INSTRUMENT-FULL arms (this pair) are the attribution-of-record. Same
# subset, same boot pattern, back-to-back in one driver invocation so the
# pair shares host state. All timers are the deferred-event observer-safe
# family (bv1 precedent); numbers from these arms are ATTRIBUTION ONLY —
# never quoted as clean speed.
#
# Arm A (if_base): passed-lever stack OFF  -> baseline spans
# Arm B (if_levers): all queue-passed levers ON -> span deltas per bucket
# (cfwd = committer, sfwd = verify forward GPU, dfwd = drafter, kvremap /
# stateremap = remap slices; HOST-gap attribution = wall-per-step minus
# GPU spans, reduced by fr13_measure + the nsys window if armed).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
# ARM A MUST BE LEVERS-OFF: explicit zeros guard against inherited env
# (driver-level exports would otherwise contaminate the baseline arm).
export FR13_PARENT_GATHER=0
export FR13_CONV_PREGATHER=0
export FR13_FLAGS_INKERNEL=0
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=0
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_base_sg.json
export FR13_KVREMAP_TIMER=1
export FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_base_kvremap.json
run_variant if_base  tail6  21  1

# Arm B: levers that PASSED their queue gates get flipped ON here at pair
# time (edit before launch — placeholders reflect queue verdicts):
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1      # 2c PASS
export FR13_HC_INTERNAL=1         # 2d PASS
export FR13_CONV_WB_BATCHED=1     # 2h PASS
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_levers_sg.json
export FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_levers_kvremap.json
run_variant if_levers  tail6  21  1
