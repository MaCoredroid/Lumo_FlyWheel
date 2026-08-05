# B1 K64 pair8 real-task diagnostic

This bundle records the 2026-08-05 real SWE-Verified B1 diagnostic for the
default-off `gemvx_m1_warp4_pair8_out` drafter-head kernel at source commit
`dcc044225176868b4fdda9d4cbe723afa60e6820`.

The candidate served every draft-head call for the exact Hydra27 physical32,
K64/root1 B1 route. Runtime evidence reports `ready`, `engaged`, four captured
graph calls, zero fallback calls, and unchanged target authority. The real
`astropy__astropy-12907` task resolved.

Measured over 264 decode events, the diagnostic produced:

- 35.1777 ms drafter GPU time per step
- 152.2096 ms verifier/SFWD GPU time per step
- 32.6183 ms committer GPU time per step
- 3.0595 ms other overhead per event
- 223.0651 ms full-step wall time
- 24.3848 full-step wall TPS
- 4.4394 accepted drafts per event

The matched `astropy__astropy-12907` slice from the 2026-08-05 exact4
reference measured 35.0973 ms drafter GPU time per step. Pair8 is therefore
0.0805 ms (0.23%) slower on the phase it changes. Its 0.91 ms lower full-step
wall sample is not attributable to DFWD and occurred with a different task
trajectory and acceptance count. This result does not justify an exact4
timing slot; the TensorCore K64 route is the next DFWD candidate.

This is a one-task B1 diagnostic. It is not timing-eligible or valid hardware-
floor acceptance evidence. Launch/end runtime and external manifests are
byte-identical.
