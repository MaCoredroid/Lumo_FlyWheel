# B=1 multistream A/B: less KV => boots under the 9000MiB gpu_oom_guard floor that killed the B=4 attempt.
# B=1 = eff_conc 1 = CLEANEST per-step committer measure (no co-residency confound). CF2 committer timer
# on both; compare committer gpu_s (does the ~66ms replay collapse toward ~5ms?). ms_strm FIRST (fail-fast).
run_variant ms_strm_b1 tail6_ms  21 1
run_variant ms_base_b1 tail6_cf2 21 1
