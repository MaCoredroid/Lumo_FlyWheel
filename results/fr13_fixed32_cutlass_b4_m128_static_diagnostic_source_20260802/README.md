# Fixed32 CUTLASS B4 static-M128 diagnostic source route

Status: `source_route_ready_default_off_host_verified`

This reduced artifact records the source-only integration of the pinned
`persistent_b4_m128_static` kernel into the real SWE-Verified exact4 B4
raw-byte diagnostic. The implementation commit is
`432d47a60e0a0a7cef5198a330c72d82fd80bd3f` on
`agent/fixed32-sfwd-m128-static-scheduler-20260802`.

## Bound inputs

- Candidate binary SHA-256: `9c63ed03ad73640293ba544fc5acad9047dcf9e202854d86f83a7ba4ca5a7d39`
- Candidate binary bytes: `113010008`
- Host-build resource credential SHA-256: `7ab2c3223366f4591fc2324a47c805aa0a1e9d4a106743af4256d4089054a2dc`
- Host-build resource credential bytes: `5404`
- Resource audit: 168 registers/thread, 0 stack bytes/thread, 0 local bytes/thread, 1,024 static shared bytes/CTA, 384 threads/CTA
- Static patch source SHA-256: `977c0204d03d022bd3f4b745ad4a0bad8ec36d7bf82ac1c6f82aa42a62094fab`
- Static patched dispatch SHA-256: `446771039af31a2ae386b917540be2a018fdc8d947c001030696ec9a6608a4c4`

The host-only verifier re-read the actual binary and credential and matched all
pinned identities and audited resource fields.

## Diagnostic contract

- Default selector remains incumbent `persistent_b4_m128`.
- Static opt-in uses `CUTLASS_B4_CANDIDATE_SELECTOR=persistent_b4_m128_static`.
- The only installable static selector is `persistent_b4_m128_static_byte_ab`.
- Workload is real SWE-Verified exact4 B4, batch 4, concurrency 4, fixed M=128.
- Selected vocabulary contract is K64 with root included, physical root+31 rows.
- Tail23 and Hydra27 must each produce an independent raw-byte PASS.
- Candidate comparison is bounded by the incumbent 320-call contract.
- Stock output is always served; candidate output is comparison-only.
- The pinned host resource credential is required at verify, install, live-result issue, and sidecar verification boundaries.

Direct static production install, production attestation, and the B4 timing
runner remain unavailable. Passing one or both raw-byte gates does not silently
enable production or timing; a later reviewed authorization change is required.

## Validation

- 81 CUTLASS-focused tests passed.
- 52 broader fixed32 source tests passed; 2 unrelated generated-kernel tests were deliberately deselected because that generated file is absent from this worktree.
- Shell syntax, Python compilation, focused Ruff, and legacy-script parse checks passed.
- No Docker, GPU runtime, synthetic/probe performance, full-step timing, or hardware-floor acceptance run was performed.

This artifact makes no performance or acceptance claim.
