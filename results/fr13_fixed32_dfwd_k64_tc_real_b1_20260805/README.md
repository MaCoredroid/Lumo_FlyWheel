# B1 K64 TensorCore real-task diagnostic

This bundle records the 2026-08-05 real SWE-Verified B1 diagnostic for the
default-off `gemm_m1_tc16x256x64_s2_out` drafter-head kernel at source commit
`ba947b41da22daa0f713d5deec10d6cce8279f2e`.

The candidate served every draft-head call for exact Hydra27 physical32,
K64/root1 B1. Runtime evidence reports `ready`, `engaged`, four captured graph
calls, at least one root call, zero fallbacks, proposal-only output, and
unchanged target authority. The real `astropy__astropy-12907` task resolved.

Measured over 610 decode events, the diagnostic produced:

- 36.3392 ms drafter GPU time per step
- 153.2524 ms verifier/SFWD GPU time per step
- 25.2072 ms committer GPU time per step
- 10.7769 ms other overhead per event
- 225.5757 ms full-step wall time
- 21.9582 full-step wall TPS
- 3.9532 accepted drafts per event

The matched `astropy__astropy-12907` slice from the 2026-08-05 exact4
reference measured 35.0973 ms drafter GPU time and 223.9717 ms full-step wall
time. TensorCore is 1.2419 ms (3.54%) slower on DFWD and 1.6040 ms slower on
the full step. The route is therefore rejected and does not receive an exact4
timing slot.

This is a one-task B1 diagnostic. It is not timing-eligible or valid hardware-
floor acceptance evidence. Launch/end runtime and external manifests are
byte-identical.
