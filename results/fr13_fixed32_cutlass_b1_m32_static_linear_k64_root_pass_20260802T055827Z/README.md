# Fixed32 B1 M32 static-linear K64/root byte diagnostic PASS

This reduced artifact packages the authenticated real SWE-Verified B1 run
`m32_static_k64_20260802T055827Z`. The `m32_static_linear` candidate passed
the bounded K64/root shadow comparison.

## Result

- Source commit: `53b502c6e8ca215e6bf4e509bbd505eba8714bf2`
- Candidate SHA-256:
  `079d82d60426411bf403eb96f4869cb8d3872a4a68d49e9c336a55a90d571f91`
- Live-result SHA-256:
  `583a3be6eb7bd2b1d7b5aebd606f0da69232d5fbb9a2311d1d6038b3e2b88da6`
- Issued-sidecar SHA-256:
  `beb9c1b73d47b15501ce4c5e92d1d2a8b94c8245f5663531ac013bf478ec12ea`
- Observed comparisons: 320 unique, contiguous invocations `0..319`; all
  320 were byte-equal, with zero mismatching comparisons and zero differing
  bytes across 311,951,360 compared bytes.
- Fixed rows and shapes: M32 over `(N,K)` values `5120x6144`,
  `5120x17408`, `14336x5120`, `16384x5120`, and `34816x5120`.
- The authenticated task `astropy__astropy-12907` resolved. The orchestrator,
  evaluator, agent, and serving lifecycle exited 0; the task did not time out;
  and the real-task arm reached its authenticated ended state.

The diagnostic selector compared candidate bytes in shadow and served stock
bytes. Production remained disabled. Formal `validate`, deterministic `issue`,
and `verify` commands independently returned 0 against the immutable candidate
and pinned source inputs; the reissued sidecar was byte-identical to the
retained credential.

## K64 Provenance

The environment records root `1`, K `65536`, the pinned block-map path and
SHA-256, and an empty allow override. The byte-identical full and after-task
runtime logs each contain one K64 gather-shim marker and one root-engaged gather
marker, with no `[FR13_DRAFT_VOCAB] DISABLED` marker.

This runtime-marker audit is independent of the formal reducer: the reducer
validates the environment digest and K64 fields but does not reopen the runtime
logs to check those markers. Also, its reusable comparison-count rule permits
`5..320`; this run's exact `320/320` count is an observed and independently
reduced fact, not an exact-count invariant enforced by that rule. Unrelated
`[FR13_CPG_SERVE]` no-stage fallback messages do not contradict the narrower
absence of the draft-vocabulary disabled fallback.

## Lifecycle

The run container was attested and removed, and host memory was recovered.
Because container-removal failure is mapped to a nonzero serving exit, the
recorded serving exit 0 also binds successful removal. Launch/end runtime
manifests and launch/end external manifests are byte-identical. Authenticated
proxy and engine ledgers finalized 7/7 accepted requests as completed, with no
active, failed, aborted, or campaign-rejected work.

The archived proxy STOPPED line is only a best-effort record because its helper
does not propagate remote stop failures. An independent probe at
`2026-08-02T06:24:24Z` found no named run container, tagged local runtime
process, relevant listener, GPU compute process, remote proxy process, or
remote proxy listener. A remote `sse_capture` directory remains; it may contain
sensitive request or model data, so it was not inspected or included. A stale
archived offload metrics file predates the run and was excluded. The per-request
vLLM metrics file is empty, so completion is supported by the authenticated
finalized ledgers instead. One provenance summary leaves its post-run runtime
attestation digest null; the underlying lifecycle and task result are otherwise
independently hash-bound.

## Scope

This is a one-task B1 kernel-correctness diagnostic. It is explicitly not
acceptance evidence, timing or TPS evidence, hardware-floor evidence, a
production-return run, or a production-performance claim. The candidate was
never the served result, and its production default remained off. The issued
sidecar is retained only as a hash-bound qualification credential; its
`QUALIFIED` status does not turn this diagnostic into production execution.

The evidence uses three compatible classification layers: the gate's specific
one-task K64/root label, the campaign's `b1_diagnostic` label, and the eager
lifecycle's `eager_kernel_byte_diagnostic` label. Every layer marks the run
ineligible for acceptance.

The package intentionally excludes prompts, model outputs, comparator JSONL,
logs, task patches, environment payloads, secrets, the candidate binary, and
process/container IDs. `manifest.json` summarizes the result,
`verification.json` records the independent reductions, and
`source_evidence.sha256` binds excluded source-run evidence by hash. In that
list, `arm/` denotes the run's
`hydra27_fixed32_k64_root_m32_static_k64_20260802T055827Z` directory; remaining
run files are relative to its parent run root. `source@53b502c6` denotes the
pinned source tree, and `immutable_candidate.so` denotes the external binary
that was rehashed but not copied. `SHA256SUMS` covers every retained file except
itself.
