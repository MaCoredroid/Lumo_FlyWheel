# BATCH-VERIFY arm bv1 (4-task): validates the accumulated fix batch in one
# boot — SSI_PREBUILD (new) + syncfree kv-remap + inputprep guard + reqkey
# repair (all baked). Checks: no degeneration (trace garble screen), accept
# band, cfwd/step vs M1 45.4 & msr4 31.1, zero errors.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/bv2_sg.json
export FR13_REPLAY_ONLY_GPU_TIMER=1
export FR13_REPLAY_ONLY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/bv2_replayonly.json
export FR13_KVREMAP_TIMER=1
export FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/bv2_kvremap.json
export FR13_CONV_PREGATHER=1
run_variant bv2_verifybatch  tail6  21  1
