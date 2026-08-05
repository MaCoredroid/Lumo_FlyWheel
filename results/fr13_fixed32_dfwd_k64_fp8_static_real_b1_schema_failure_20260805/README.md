# B1 K64 static-FP8 real-task schema failure

This bundle preserves the 2026-08-05 real SWE-Verified B1 diagnostic for the
static-I/O block-FP8 K64 draft head at source commit
`fb0647d8026c65d59f872e371da9a814dbefe26d`.

The real `astropy__astropy-12907` task resolved. The FP8 candidate served the
exact Hydra27 physical32 K64/root1 B1 route with one root call, four captured
loop calls, at least one measured replay, direct FP8 proposal logits, zero
BF16 shadow calls, zero fallback calls, and unchanged target authority.

The outer runner nevertheless exited 1 after the task because the runtime
engagement writer omitted its validated `arm` value while the post-run gate
required that key. `fp8_runtime_identity.txt` records the exact arm from the
runtime environment; `fp8_engagement.json` proves it was absent from the
writer payload. Commit `3967fb29cc69335ac930b522f2a5dea20ce11aa0`
adds the field and rebinds the source hashes.

The preliminary diagnostic measured:

- 27.5848 ms drafter GPU time per step
- 153.7337 ms verifier/SFWD GPU time per step
- 19.8502 ms committer GPU time per step
- 16.7761 ms other overhead per event
- 217.9449 ms full-step wall time
- 25.6508 full-step wall TPS
- 4.5905 accepted drafts per event

Against the matched exact4 reference slice, DFWD improved by 7.5124 ms
(21.40%) and the full step improved by 6.0268 ms (2.69%). The corrected
candidate still sits 87.4038 ms above its 130.5411 ms acceptance cap.

These numbers are useful directional evidence only. This bundle is explicitly
not a passed gate, is not timing-eligible, and is not hardware-floor acceptance
evidence. A corrected-source real B1 rerun is required.
