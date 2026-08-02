# B4 SFWD pre-task flush rejection

This artifact preserves the rejected exact4 B4 SFWD state-fusion byte-gate
attempt from source commit
`d9763769ceb461f907b7174c21cff325f8e43030`.

- Classification: real SWE-Verified exact4 B4 byte diagnostic
- Result: rejected before task traffic
- Acceptance valid: no
- Timing eligible: no
- Floor acceptance eligible: no
- Speculative events: 0
- Cause: the orchestrator selected the graph-census task bracket for an eager
  SFWD kernel diagnostic. Its first pre-task snapshot reached the runtime flush
  endpoint, which correctly rejected the absent eager graph-capture pregather
  counters.
- Correction: classify
  `FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1` with the existing eager kernel
  diagnostic bracket, which records per-task metrics without invoking the
  graph-census flush protocol.

Environment and process-identity dumps are deliberately excluded because they
can contain credentials. The copied logs and manifests are bound by
`SHA256SUMS`.
