# Fixed32 SFWD channel-serial byte-gate readiness

Status: **READY_FOR_ONE_REAL_SWE_VERIFIED_B1_BYTE_GATE**.

The channel-serial W2 kernel is bound to the existing independent, default-off
SFWD shadow gate. The reference result is always served; the candidate may not
enter timing or production based on this gate. A real one-task B1 run must
first prove exact bytes for both convolution output and commit source stage
across every layer.

The source-bound gate uses the fixed 32-row topology, C64 kernel block, two
warps, K64 drafter vocabulary, and root reduction enabled. The generated
14-file manifest binds the wrapper, kernel, patcher, launcher, runner, gate,
fixed reference GDN source, and task ingress surfaces to source commit
`07e46a9783d1f8347fd9f959309cfeeb95694d2c`.

Offline codegen qualification remains the evidence recorded in
`results/fr13_fixed32_sfwd_channel_serial_codegen_ready_20260802`: B1/B4 and
fresh-cache builds are byte-identical, shared synchronization is eliminated,
and the 64-register/no-spill gate passes.

No GPU, Docker, service, task, timing, or acceptance run was used for this
readiness package. It contains no task or model content, requests, responses,
raw logs, environment values, credentials, process identifiers, or secrets.
