# FR13 CONC=1 isolation (user 2026-07-06: "why are 4 running together at B=1?").
# B=1 CONC=4 has 4 concurrent agents SHARING the prefix cache => cross-agent cache interaction, NOT single-agent.
# §66 give-up-extinct was CONC=1 (single-task probes). This runs the cache-on arms at CONC=1 (serial, 1 agent, no
# cross-agent cache sharing) to isolate carrier (B):
#   chain5+cache CONC=1 ~resolves => carrier (B) = CROSS-AGENT concurrent cache-sharing (CONC), NOT single-request.
#   chain5+cache CONC=1 still gives up => single-agent cache-restore issue (independent of concurrency).
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant chain5cache_c1_${TAG}  chain5  5  1
run_variant cat8cache_c1_${TAG}    cat8    9  1
