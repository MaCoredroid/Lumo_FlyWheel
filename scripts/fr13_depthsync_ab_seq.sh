# FR13_DM_DEPTHSYNC phase-3 A/B (committer optimization): legacy per-node walk (~100 .item() syncs/step)
# vs depth-synchronous walk (~2 syncs/level). Both = deployed tail6_mt (FR13_MULTIDRAFT_GPU_TIMER on).
# LOSSLESS GATE: accept ds0 == ds1 (offline byte-gate fr13_dm_depthsync_byte_gate.py = 96/96 byte-identical
#   at same seeds; live confirms same-boot). SPEED: committer_ms(ds1) < committer_ms(ds0); the delta ==
#   the host-sync overhead removed == the phase-2 decomposition-by-intervention. temp 0.6, subset_b4_sixteen.
# Only launch AFTER the dedup A/B (need the legacy committer_ms baseline + no-drift confirmation first).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_DM_DEPTHSYNC=0
run_variant tail6_mt_ds0  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/tail6_mt_ds0_md.json 2>/dev/null || true
export FR13_DM_DEPTHSYNC=1
run_variant tail6_mt_ds1  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/tail6_mt_ds1_md.json 2>/dev/null || true
