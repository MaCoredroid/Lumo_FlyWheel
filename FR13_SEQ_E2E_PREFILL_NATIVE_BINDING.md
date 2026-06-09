# FR13 sequential e2e binding - prefill-native + metrics-off

Commit under test: `a586ac84`.

Run dir: `output/fr13_seq_e2e_prefill_native_20260609T041654Z`.

Config:
- Sequential 9-node tree, `TREE_ATTN`, forked FA2 `.so` sha256 `97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1`.
- `FR13_FA2_PREFILL_NATIVE=1`, `FR13_FA2_TREE_BIAS=1`, `FR13_TREE_ATTN_EXP2_SOFTMAX=1`.
- `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`, `FR10_ENABLE_TREE_GDN=1`.
- B=4, CUDA graph, `MAX_NUM_SEQS=4`, `GPU_UTIL=0.86` after `0.88` failed vLLM's startup free-memory guard.

Capture status:
- FULL decode capture completed: logs show `Profiling CUDA graph memory: PIECEWISE=8 (largest=80), FULL=4 (largest=40)` and `Capturing CUDA graphs (decode, FULL)` completed.

Probe:
- `scripts/fr10_quick_decode_tps_probe.py`, same8 shape: 8 prompts x 4 samples, `batch_size=4`, `max_tokens=64`, temperature `0.6`, top_p `0.95`, mode `tree_mtp`.
- Artifact: `output/fr13_seq_e2e_prefill_native_20260609T041654Z/quick_tree_mtp_b4.json`.
- Tree engagement: `engaged=true`, `gpu_tree_metadata_ok_rows=248/248`, `tree_accept_rows=795`.

Result:
- Tree accept/event: `1.6583442838370566`.
- Saved E5 bar: `3.076171875` accept/event, `17.987313578432634` warm decode TPS (`output/fr10_native_mtp5_same8_20260604T210257Z`).
- Fresh paired native comparison artifact used for bag-TV because saved E5 has no token records:
  `output/fr13_seq_e2e_prefill_native_20260609T041654Z/tree_vs_native_bag_compare.json`.
- Bag-TV vs paired native: `0.42584828811470293`, above the `0.0593` floor.
- First-token TV: `0.03125`; first paired mismatch: prompt `0`, sample `0`, position `8`.
- Tree warm decode TPS: `5.746200172868099`; paired native warm decode TPS: `15.647809832189031`.

Interpretation:
- This is not a lossless binding. Prefill-native improved the older prefill-off e2e (`1.11 -> 1.66` accept/event), but the sequential tree still misses both acceptance and distribution floors.
- The live e2e path log emitted canonical stochastic committer rows (`tree_sample_accept`), not authoritative `tree_path_lcp_max` rows, so this artifact proves the e2e distribution/acceptance miss but does not by itself prove per-depth native argmax equality or the exact argmax-flip layer.
- Per current direction, do not re-chase GDN or literal zero from this result. The next localization target remains the full-attention/tree-verify argmax front, using the authoritative per-depth argmax ladder rather than this stochastic e2e log.
