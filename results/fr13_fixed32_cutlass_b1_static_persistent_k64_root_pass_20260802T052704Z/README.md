# Fixed32 B1 generic static-persistent K64/root byte diagnostic PASS

This reduced artifact packages the authenticated real SWE-Verified B1 run
`b1_static_k64_cap320_20260802T052704Z`. The generic stock-tile
static-persistent candidate passed its bounded K64/root shadow comparison.

## Result

- Source commit: `ac3536a5d27d6a69bea16355d257ed5db9c2f122`
- Candidate SHA-256:
  `88c50e7d1b6060c2bcec68f50985a1db47b43d299b574edfbfc32cac1ce68742`
- Live-result SHA-256:
  `21d4b62663d1a9a3875618f627435b0acee10ae582f33577e7df54a7c1c293c6`
- Issued-sidecar SHA-256:
  `550cb31e0c1d32072976dd59546baba6d1fed4b9b8d0cb7f72039d3e94d1a6ba`
- Comparisons: 320 unique, contiguous invocations `0..319`; all 320 were
  byte-equal, with zero mismatching comparisons and zero differing bytes.
- Fixed rows and shapes: M32 over `(N,K)` values `5120x6144`,
  `5120x17408`, `14336x5120`, `16384x5120`, and `34816x5120`.
- The authenticated task `astropy__astropy-12907` resolved. The orchestrator
  and serving lifecycle both exited 0, the task did not time out, and the real
  task arm reached its ended state.

The diagnostic selector compared candidate bytes in shadow and served stock
bytes. Production remained disabled during the run. Both the live-result and
issued-sidecar validators independently returned 0 against the immutable
candidate and pinned source inputs.

## Lifecycle

The run container was attested and removed, and host memory was recovered.
Because container-removal failure is mapped to a nonzero serving exit, the
recorded serving exit 0 also binds successful removal. Launch/end runtime
manifests and launch/end external manifests are byte-identical.

The archived proxy STOPPED line is only a best-effort record because its helper
does not propagate remote stop failures. An independent post-run probe found no
remote proxy process or listener, but a stale proxy state file remained. No
named run container or tagged local runtime process remained.

## Scope

This is a one-task B1 kernel-correctness diagnostic. It is explicitly not
acceptance evidence, timing or TPS evidence, hardware-floor evidence, a
production-return run, or a production-performance claim. The candidate was
never the served result, and its production default remained off. The issued
sidecar is retained only as a hash-bound credential artifact; it does not turn
this diagnostic into a production run.

The evidence uses three compatible classification layers: the gate's specific
one-task K64/root label, the campaign's `b1_diagnostic` label, and the eager
lifecycle's `eager_kernel_byte_diagnostic` label. Every layer marks the run
ineligible for acceptance.

The package intentionally excludes prompts, model outputs, comparator JSONL,
logs, task patches, environment payloads, secrets, and process/container IDs.
`manifest.json` summarizes the result, `verification.json` records the
independent reductions, and `source_evidence.sha256` binds excluded source-run
evidence by hash. In that list, `arm/` denotes the run's
`hydra27_fixed32_k64_root_b1_static_k64_cap320_20260802T052704Z` directory;
the remaining run files are relative to its parent run root. `SHA256SUMS`
covers every retained file except itself.
