# Fixed32 SFWD late-tap0 B1 byte-gate readiness

Status: **READY_FOR_ONE_REAL_SWE_VERIFIED_B1_BYTE_GATE**.

The promoted late-tap0 SFWD candidate is prepared behind the existing
default-off shadow gate at pushed source commit
`4f1a53eef1bffb083051a292e94c2eddf631d6ea`. The gate always serves the stock
reference result. It cannot enable production, timing qualification, or
hardware-floor acceptance.

The only permitted next run is the canonical one-task real SWE-Verified B1
diagnostic with K64 and root reduction enabled. It must prove exact bytes for
both convolution output and commit source stage across all 48 model layers.
The candidate uses the fixed 32-row topology and B1 C128/W2 geometry.

The host preflight binds 15 committed source and ingress files, including the
unchanged reference GDN source and chat template, and verifies the exact stock
FA2 binary, canonical task subset, K64 block map, pushed source commit, clean
tracked worktree, and repository virtual environment. The source-manifest
SHA-256 is
`3e2593a46a8df6467e693480b4bbd7fef4bbe70f776fbafeff3bedc2dbac7fac`.

Offline SM121a codegen remains recorded in
`results/fr13_fixed32_sfwd_late_tap0_pareto_codegen_20260802`: both B1 C128/W2
and B4 C256/W4 use 44 registers/thread, 91 global-load instructions, 136
global-store instructions, 1,920 static instructions, and 1,936 encoded
instructions, with zero stack, local, shared, spill, barrier, or call
resources. These are compiler proxies, not runtime measurements.

No Docker, GPU, service, task, timing, throughput, acceptance, or production
run was used for this readiness package. It contains no task/model content,
request, response, patch, raw log, environment value, process/container
identifier, binary, PTX, SASS, credential, or secret.
