# Re-run ONLY the fixed ms_strm (FR13_REPLAY_MULTISTREAM=1, now capture-guarded @ 6f55d2a19).
# ms_base baseline is captured by the first A/B run (tail6_cf2_commit.json). Compare via
# scripts/fr13_ms_ab_reduce.py. Verify FIRST: (1) boots past 'Capturing CUDA graphs (decode,FULL)',
# (2) [FR13_REPLAY_MULTISTREAM] ENGAGED fires, (3) no garble, THEN read committer gpu_s.
run_variant ms_strm tail6_ms 21 1
