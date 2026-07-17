# FR13_REPLAY_MULTISTREAM B=4 A/B (the actual gate batch). GPU_UTIL=0.72 is the ESTABLISHED tail6 B=4
# config (fr13_tail6_prewarm_seq.sh:8-12): the n_pad=32 tail6 capture spike dips below the 9000MiB
# oom_guard floor at the 0.78 default => 0.72 leaves headroom. Not config drift — tail6's own setting.
# Both arms identical tail6 + CF2 committer timer; differ ONLY in FR13_REPLAY_MULTISTREAM (kind).
export GPU_UTIL=0.72
run_variant ms_strm_b4 tail6_ms  21 1
run_variant ms_base_b4 tail6_cf2 21 1
