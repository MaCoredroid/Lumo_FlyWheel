# FR13 fixed32 unified-attention BM8 real B1 live result

Status: `KERNEL_LIVE_PASS_WITH_POSTVALIDATOR_PERMISSION_DEFECT`.

This artifact packages the real SWE-Verified B1 diagnostic run at
`output/fr13_b1_bm8_live_gate_20260731T175331Z`. The executed source commit
was `d67e405b5f1c37a0f35f329c722af17edfe5fe4a`.

## Kernel result

The launcher-private BM8 candidate for `kernel_unified_attention_2d` used
`BLOCK_M=8, BLOCK_Q=1`; stock used `BLOCK_M=16, BLOCK_Q=2`. On the first four
eligible replay calls inside the real task bracket for
`astropy__astropy-12907`, candidate and stock produced byte-identical 12,288
byte outputs at sequence lengths 22,872 through 22,875. Every call had zero
raw-byte mismatches. The immutable live result has SHA-256
`570caf42e3e75ff0d3717042b0dfc58b23a90041e71103f70a07f6d7563445b5`.

The exact candidate identity has SHA-256
`d12e035acb459640b128a052cba606602d57b3dca898e81bcedbe7d9582d8058`
and binds the executed source commit plus the patcher, emitted unified-attention
source, and Eagle replay-hook hashes. Stock output was served throughout; the
candidate was not production-enabled.

## Real task result

SWE-Verified task `astropy__astropy-12907` resolved with agent exit 0 and eval
harness exit 0. Its exact task marker was live from 18:01:22Z to 18:05:58Z.
The post-task boundary contains 817 contiguous pure-decode forward steps and
817 complete work-census events over `[0, 817)`, with zero pending SFWD, DFWD,
or CFWD work. Final flush generation 3 retained the same complete counts and
zero pending work.

Engine and proxy ingress each accepted and completed 13 requests. There were
no campaign rejections, failed attempts, aborted logical requests, or active
requests at finalize. Every traffic-audit check is true. Runtime and external
manifests are byte-identical at launch and end. The server arm recorded
`serve_rc=0`, completed teardown, and recorded 106 GiB available memory.

## Launcher defect and fix

After the successful task, final flush, and teardown, the top-level B1 runner
repeated the host validator as user `mark`. The executed commit had written the
nonsecret candidate identity as root-owned mode `0400`, so that redundant
post-validator could not read it and the top-level runner returned `rc=2`.
This is why the artifact status is not an unqualified launcher PASS.

The unchanged raw identity and live-result files pass the exact same validator
when it is allowed to read the identity. `sudo_validation_stdout.jsonl`
preserves that PASS output; its SHA-256 is
`e21571ab7a17f35e4c1ad253554e0b506acbe75d975365b7e454fc984b5b5e65`.

Commit `fdb61d773131f703eb1a7390ccdaa5b4998d93a6` fixes future gates by atomically
publishing the nonsecret identity at mode `0444` and requiring that exact mode.
It also adds a negative mode-`0644` drift test. The fix passed compile, shell
syntax, Ruff, and the focused 120-test CPU suite. Per the no-waste decision, it
was not rerun on GPU solely for this post-teardown permission defect, so the
live GPU credential remains bound only to executed commit `d67e405b5`.

## Scope

This is a one-task B1 diagnostic kernel-correctness result. It is not B4,
formal exact4/exact16 acceptance, a production-return run, a timing or TPS
measurement, or hardware-floor acceptance. The source run explicitly records
`run_classification=b1_diagnostic`, `gate_eligible=false`, and
`floor_acceptance_eligible=false`.

`manifest.json` records the result contract, `verification.json` records the
reduced checks and hashes, and `SHA256SUMS` covers every artifact file except
itself.
