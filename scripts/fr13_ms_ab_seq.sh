# FR13_REPLAY_MULTISTREAM A/B (see FR13_REPLAY_MULTISTREAM_DESIGN.md + task #43).
# Both arms = identical tail6 tree/flags; differ ONLY in FR13_REPLAY_MULTISTREAM (via kind).
# CF2 whole-committer GPU timer on both (one sync/step => captures stream overlap).
# ms_base = multistream OFF (baseline committer gpu_s). ms_strm = multistream ON (N=4 default).
# Compare committer gpu_s (does the ~66ms replay collapse toward ~5ms bandwidth floor?).
# Lossless: ms_strm output must be COHERENT (non-garble) + accept comparable (cross-boot autotune
# forbids byte-identity; correctness is constructive: scan->replay event + join + independent writes).
# ms_strm FIRST (fail-fast: a concurrency bug in the new path crashes/garbles within ~5min of boot),
# then ms_base baseline. Order does not affect the committer-gpu_s comparison.
run_variant ms_strm tail6_ms  21 1
run_variant ms_base tail6_cf2 21 1
