# FR13_SAMPLED_REPLAY_BATCHED A/B (speed half of "speedy tree pipeline w/ accept>5") + PREWARM (accept>5).
# sbr0 = per-layer replay (baseline, committer ~88ms); sbr1 = batched all-layers replay. Both tail6_mt
# (FR13_MULTIDRAFT baked) + FR13_COMMIT_FULL_GPU_TIMER (whole committer) + FR13_REPLAY_GPU_TIMER (per-layer
# replay; sbr1 should show ~0 spans = per-layer bypassed). GATE: accept sbr0==sbr1 (lossless, semantics-
# preserving kernel) AND accept>5 (prewarm) AND committer_ms(sbr1) << committer_ms(sbr0). temp 0.6.
export GPU_UTIL=0.72
export FR13_PREWARM_TRIE=/home/mark/shared/lumoFlyWheel/output/fr13_prewarm/corpus_harness.jsonl
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_REPLAY_GPU_TIMER=1
export FR13_SAMPLED_REPLAY_BATCHED=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sbr0_cf2.json
export FR13_REPLAY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sbr0_replay.json
run_variant tail6_sbr0  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/sbr0_md.json 2>/dev/null || true
export FR13_SAMPLED_REPLAY_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sbr1_cf2.json
export FR13_REPLAY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sbr1_replay.json
run_variant tail6_sbr1  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/sbr1_md.json 2>/dev/null || true
