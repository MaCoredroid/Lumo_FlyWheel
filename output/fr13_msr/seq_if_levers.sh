# 2i ARM B (rerun solo; arm A if_base results stand): bar-stack candidates
# MINUS HC_INTERNAL (incompatible with PARENT_GATHER by launcher guard —
# same-bucket levers; pg chosen for the bar on 16-task evidence; hc deferred
# to a compat-fix gate). Timers = attribution-labeled.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_levers_sg.json
export FR13_KVREMAP_TIMER=1
export FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/if_levers_kvremap.json
run_variant if_levers  tail6  21  1
