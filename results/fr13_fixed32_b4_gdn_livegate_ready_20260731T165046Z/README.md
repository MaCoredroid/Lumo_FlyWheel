# FR13 fixed32 B4 GDN live-gate readiness

This artifact binds the source and runner needed to execute the missing real
SWE-Verified exact4 B4 byte diagnostic for the batched wide-BV GDN candidate.
It contains no GPU, Docker, real-task, byte-parity, timing, acceptance, or
hardware-floor result.

## Source identity

- Kernel-stack base: `e2a3e6b4ca5faa8d61b0d1018a519c035036ff1b`
- Readiness code: `35758a4a6a2c6a83e1d9232edd08519e53961f40`
- Branch: `agent/fixed32-b4-livegate-ready`
- Canonical exact4 subset SHA-256:
  `0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5`

## What is ready

Authenticated engine ingress now publishes the real-event arm only after a
canonical exact4 request passes admission and before vLLM executes it. The
first ingress publication must win an atomic hard-link create; a marker
injected after boot is rejected. Later requests validate the exact first
marker. Marker and enabled-sidecar reads use no-follow file descriptors,
`fstat`, bounded reads, link-count checks, and mode checks.

A `MAX_NUM_SEQS=4` lifecycle can start or drain at B1. B1 now stays on the
stock per-request BV8 route even while the B2-B4 candidate is selected. Byte
gate state is keyed by `(batch, layer)`, so a B2 or B3 comparison cannot
suppress the required B4 comparison for the same layer allocation.

The runner `scripts/fr13_run_b4_gdn_wide_live_gate.sh` requires the canonical
four-task subset, `B=4`, concurrency 4, eager execution, metrics, a new
runroot, no containers, and a clean tracked tree. It disables timing and all
other live kernel gates. Success requires:

1. a finalized exact4 engine ledger with an acceptance matching the marker;
2. 48 distinct nonzero B4 layer records;
3. raw byte equality for output, K/V/A/B rings, compact and untouched state
   export, flags, and invocation counter;
4. stock reference state restored and served for every comparison;
5. a source-bound v2 B4 PASS with BV8 reference and the selected wide BV;
6. unchanged runtime, external, and runner hashes across the run.

The structural B4 launch count is reduced from eight reference launches per
layer to two candidate launches per layer: 384 to 96 across 48 GDN layers,
or 288 fewer launches per event. This is not a latency estimate. Program work
still scales with batch and BV, and only a real B4 run can establish bytes or
time.

## Safety and remaining blocker

All diagnostic and production selectors remain default-off. The diagnostic
always serves the stock result, including on mismatch. Only B4 may publish the
wide-BV v2 PASS; B2/B3 observations are logged but cannot satisfy it.

This is diagnostic-ready, not production-ready. The existing production
selector binds a PASS to one batch size. A B4 PASS would therefore reject an
actual B2 or B3 production step in the same capacity-4 lifecycle. Production
stays off until a B2-B4 evidence schema and mixed-batch routing rule are
defined and gated.

There is still no B4 timing. The corrected hardware floor is
119.658015414 ms/step and the one-sided U95 acceptance cap is
137.606717726 ms/step (1.15x floor), but this artifact does not move either
comparison.

## Verification

- Targeted and integration suite: `113 passed`
- Full fixed32 source suite with CUDA disabled:
  `609 passed, 8 skipped, 1 deselected`
- Python compilation: PASS
- Ruff relevant checks: PASS
- Runner, launcher, and variant shell syntax: PASS
- Commit whitespace check: PASS
- GPU used: no
- Docker used: no
- Real SWE-Verified task run: no
