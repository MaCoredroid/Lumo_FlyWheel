# M3: batched committer + KVREMAP/STATEREMAP sub-span timers + SYNCFREE kv
# remap (patch-time baked ON). Measures: (a) kv-remap slice of the cfwd
# remainder, (b) state-remap slice of the sfwd span, (c) syncfree A/B vs M1
# (legacy remap): cfwd 53.5ms early-phase baseline.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr3_sg.json
export FR13_REPLAY_ONLY_GPU_TIMER=1
export FR13_REPLAY_ONLY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr3_replayonly.json
export FR13_KVREMAP_TIMER=1
export FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr3_kvremap.json
export FR13_STATEREMAP_TIMER=1
export FR13_STATEREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr3_stateremap.json
run_variant msr3_syncfree  tail6  21  1
