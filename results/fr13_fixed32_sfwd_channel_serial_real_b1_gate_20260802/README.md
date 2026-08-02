# Fixed32 SFWD channel-serial real B1 gate

Status: **REAL_SWE_VERIFIED_B1_BYTE_PASS_SOURCE_ONLY**.

The source-bound `fixed32_sfwd_channel_serial_r32_c64_w2_v1` candidate passed
one authenticated real SWE-Verified K64/root1 B1 task after its convolution
state addressing was repaired for the live channel-major layout. The task
completed with 35,808 candidate comparisons over all 48 layers. Every
`conv_out` and `commit_source_stage` comparison was byte-identical, the
incumbent reference was always served, and no fallback engaged.

The launch and end source manifests are byte-identical and bind commit
`df7f6166257dae0ce40ca9684cd9f280c640906c`. The wrapper failed closed on the
live convolution-state channel/state/row-stride contract, and the successful
run used stride `(2097152, 1, 10240)` for `[bank, channel, state]`.

This was a correctness gate only. It is not timing eligible, hardware-floor
acceptance eligible, or production eligible. No throughput or latency claim is
made from this run.

Prelaunch checks found zero containers and zero GPU compute owners. Periodic
run polls found one container and one GPU compute owner, and terminal checks
found zero containers and zero GPU compute owners. These observations support
correctness-run isolation but are not used to make a timing claim.

This reduced package excludes raw task/model content, requests, responses,
patches, logs, environment values, process/container identifiers, credentials,
and secrets.
