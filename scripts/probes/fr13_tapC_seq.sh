# FR13 §122 carrier-B localization via the EXISTING Tap C stale-read instrument in the
# LIVE REPLAY_ROUTE restore (layers.0.linear_attn). ENFORCE_EAGER (boundary-log is
# eager-only). Tap C compares consumer src_row vs _last_written[req]; stale_read>0 @CONC=4
# => carrier B IS the replay restore reading cross-req state; ==0 => refutes (look elsewhere).
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_REPLAY_BOUNDARY_LOG=1
export FR13_REPLAY_BOUNDARY_LAYERS=layers.0.linear_attn
run_variant chain5cache_tapC_${TAG}  chain5  5  1
