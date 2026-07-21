# FR13 direction-2: native MTP-5 cache-ON + GPU timers, matched to the tail6 burn-off run.
# Vehicle = nativemtp5_exseed (LAUNCHER=forked -> patcher runs -> GPU timer machinery exists;
# XFLAGS configure clean native MTP-5 behavior: FLASH_ATTN, naive_mtp linear decode, no tree,
# no speculative_token_tree; FR13_ENABLE_APC=1 already baked in -> cache-ON by default).
# This closes the confirmed gap: no cache-ON native + GPU-timer capture exists anywhere in the
# project's history (checked all candidates). Same subset/B4/CONC4/gpu_util as burnoff_bo1 so
# prefill_frac/eff_conc are as close to matched as achievable.
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant native_exseed_${TAG}  nativemtp5_exseed  5  1
