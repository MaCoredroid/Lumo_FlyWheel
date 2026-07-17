# FR13_DEDUP_SIBLINGS A/B (no-drift gate) + committer DECOMPOSITION (phase 2), one session.
# kind tail6_mt = deployed tail6 + FR13_MULTIDRAFT_GPU_TIMER (per-step GPU time of the deployed
# rejection committer fr13_device_multidraft_commit). temp 0.6 (deploy), subset_b4_sixteen.
#   dd1 = FR13_DEDUP_SIBLINGS=1 (default, baked) ; dd0 = FR13_DEDUP_SIBLINGS=0 (baseline).
# GATE (no config drift): accept dd1 == dd0 (dedup is provably a NO-OP for tail6 -- native-topk head +
#   spine-only tail are distinct by construction, so the collision-check returns False and the repair
#   never runs; widened topk columns are unused by the packer). A tie => confirmed no-op on the deployed
#   config. PHASE-2 read: multidraft_gpu.json gpu_seconds/n_spans => per-step committer ms (is the ~94ms
#   the KERNEL or DtoH/verify-wait). run_variant is sourced from fr13_b4_campaign_driver.sh.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_DEDUP_SIBLINGS=1
run_variant tail6_mt_dd1  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/tail6_mt_dd1_md.json 2>/dev/null || true
export FR13_DEDUP_SIBLINGS=0
run_variant tail6_mt_dd0  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/tail6_mt_dd0_md.json 2>/dev/null || true
