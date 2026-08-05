# B1 M128 cooperative target and SFWD real gate

Status: **PASS for source-bound byte correctness; non-timing and
non-acceptance evidence**.

Two authenticated one-task SWE-Verified B1 runs used source commit
`a8a904ed6c27a6338d43151038c155ebb76e3656`, fixed physical32,
K64/root1, and `astropy__astropy-12907`. The task resolved in both runs.

- The standalone cooperative target gate completed 320/320 comparisons over
  all five projection shapes with zero mismatching comparisons. Stock was
  served.
- The combined run repeated the target result at 320/320 with zero
  mismatching comparisons, engaged the production Qrow16 attention path on
  all 16 attention layers, and ran the SFWD conv/post-prep candidate in
  shadow-only mode while serving the reference result.
- The source-native SFWD live pass records 336 comparisons: 48 layers times
  seven byte surfaces, with zero mismatches and zero differing bytes.
- The digested SFWD JSONL contains 20,016 clean layer records from 417
  complete attempts. The final 40 complete attempts form the explicitly
  derived 1,920/1,920 record window requested for reporting; every record and
  every one of its seven surfaces is byte-equal. The derived count does not
  replace or rewrite the source-native 336-comparison live-pass field.
- Runtime and external-input manifests were byte-identical at launch and end
  in both runs. The combined SFWD source manifest was also byte-identical at
  launch, installation, and end.

This package qualifies only the historical cooperative target and combined
Qrow16/target/SFWD byte gates at `a8a904ed6`. It is not timing, TPS,
production-enablement, hardware-floor, or acceptance evidence, and it does
not qualify the later exact-shape follow-on source.

The package contains normalized JSON records, source identities, derived
counts, and checksums. It excludes prompts, responses, patches, traces, raw
JSONL streams, Docker logs, environment dumps, process identities, binaries,
tensors, and model outputs.
