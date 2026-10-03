# M1-Q C0 path-scope correction: bounded closure

2026-09-28. Accepted as a path-resolution-only source-binding correction. Five independent stdlib/AST controls pass. No candidate, Torch, generator, Docker, GPU, remote or workload code was run; no qualification data was accessed.

The preserved v3.5 manifest is SHA-256 `9b3ea3f7e1d7472692f2336b4b0136f7a960a2f5bf88beae3e86713e306e4da3` (18,037 bytes). Its replacement for C0 path resolution, `experiments/review-response-20260927/FREEZE-M1-Q-C0-PATH-SCOPE-v1.json`, is `85c17249741b3babd38b81719768942d1afd2dac13d1852614c5d4749cf988e4`.

All 89 original members remain. All original hashes and byte counts are unchanged. The only changed member fields are `path` for `tree_conv_fused`, `gdn_kernel` and `topology_module`, each exactly prefixed with `repo:`. Every other field of every original member compares equal. The only additional member, `freeze_v3_5_original`, binds the preserved original manifest and its byte count. Updated top-level schema/status/provenance correctly describes this limited correction; the prior test count is labelled `original_tests_claim`, not a newly executed test result.

The unchanged `m1_stage_collector_v3_1.py` resolver treats an unprefixed `src/` or `scripts/` path as campaign-relative. AST controls verify that all three corrected entries now resolve under the repository root; the old entries resolve beneath the campaign. The retained CPU preflight at `p0/monitor/review-response-20260927/m1q-c0-source-preflight-20260928T220541Z/stdout.json` reports exactly those three missing scoped sources and no other refusal. This note does not claim a newly executed complete runtime preflight.

The C0 entry is byte-for-byte the previously reviewed source except for replacement of the one pinned manifest-hash literal. Its new SHA-256 is `2bb6103e88d84b51c6e022c5c9ad424e151c94c85f8d3e34a70bd94028a37b92`. Both exception-retention repairs therefore remain unchanged. The launcher is byte-identical at `34de4761a587ac7bd1cd3ca0e58a26458e5267269c5239d28cb0bdf3059e68b7`. No scientific algorithm, arithmetic, seed, topology, method order, input generation, repetition count, numerical rule, or gate policy changed in this delta.

Evidence snapshot: `p0/monitor/review-response-20260927/m1q-c0-path-scope-reviewed-20260928T220841Z/`. It retains both manifests, both collector versions, both launcher copies, the resolver source, original failed-preflight evidence, controls and test log. Reproduce with the configured Python and `-B -m unittest -v reviewer_controls` from that snapshot.

The previous bounded source closure carries forward to the new collector hash. Parent still owns the final gate, actual deployed source preflight and all execution. This review neither opens a gate nor establishes a C0 result.
