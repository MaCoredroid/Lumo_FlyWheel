# Continuous native driver / patch glue review — 2026-09-29

**Final disposition: bounded PASS at repaired driver SHA `6409f27a5273ffdb69c37a25ec8d5ea6e3629367fcb1fdd3d9ad00e2f6e47473`; both findings below are closed.** CPU/source review only; no request was sent to a real server and no launcher, gate, model or scientific count was advanced. Initial sources and failures are preserved under `p0/monitor/review-response-20260927/native-sequences-driver-review/`.

## Reviewed initial source

| File under `experiments/review-response-20260927/tools/` | SHA-256 |
|---|---|
| `q1_reference_driver_sequences_v1.py` | `26c154fad427f3af2df2c5a1e0cc7d3b386ec313962cfa46be6a850749fe61d8` |
| `q1_patch_native_mtp_sequences_v1.py` | `ecde868d14d8bae9b6257405eb20b15eebdf0e325ca91e1d6c6eb93e829834a4` |
| `q1_apply_native_mtp_sequences_v1.py` | `141ceacb0157d1b2ad3f8691629805683665afcadf6255259dba1ceb132e03e9` |
| `q1_spec_off_sequences_config_v1.py` | `a0dff8b46bdf1cdec286e2fafe6acb66526dca359cfe9b63a2220a7c97b9bb7d` |

## Actionable findings

1. **Refuse the wrong native arm before HTTP.** Driver lines 91–100 verify the fixed input population, MTP policy, phase contract, calibration split and request count, but not `job.arm`. Changing just the arm to `native_default_packed` reaches the HTTP loop. The real target raw auditor rejects that arm only after a request; the isolated mock-auditor control completes all 24 requests, which demonstrates the driver preflight omission rather than a real scientific false pass. Require `aligned_nonpacked` before the first request, consistent with the new configuration renderer.
2. **Persist authentication/response-shape failures.** Driver lines 145–146 call seal authentication outside the failure-receipt path. The real authenticator raises on malformed JSON and on hash-consistent malformed raw metadata, leaving zero driver receipts after one request. A non-object HTTP JSON response also escapes at line 156. Catch these failures into the existing stopped-receipt/verdict path, preserving the response/error and issuing no next request. Raw audit exceptions already follow the intended path.

## Checks already passing

The extended independent controls initially pass 31/36. The five failures are manifestations of the two findings above (foreign arm, synthetic seal exception, real malformed seal JSON, real malformed raw metadata, non-object HTTP JSON). All controls are synthetic HTTP/seal/outer-audit boundaries with actual frozen calibration input loading. The initial 33-control output log is retained alongside the extended result.

The driver issues the exact 24 frozen calibration inputs and supports an explicitly supplied repeat expansion without inventing the campaign denominator. Preflight completes source-consumer checks and loads every padded prompt before HTTP; immutable stream, population, contract, final pending extent and escaping prompt paths are refused. Payloads use `ignore_eos=True`, exactly `len(flat_consumed_tokens)+1`, temperature 0, top-p 1, seed 0, no streaming, and distinct observation cache salts. Short/long/boolean completion counts, prompt mismatch, HTTP timeout, missing/invalid seals, salt mismatch and raw-audit failure stop after one request and persist both receipts.

On the exact pinned stock runner, the new patch is byte-for-byte the accepted joint-source patch with exactly one hook import replaced by `q1_reference_hooks_sequences_v2`. Compilation succeeds without importing/executing the runner; tampered or already-patched input is refused. Temporary output/receipt generation binds the exact transformed SHA. The new apply adapter differs from its accepted predecessor only in patch module and receipt schema. The new renderer preserves all base target flags, environment, image, model, mounts and scheduling settings; only its patch path/rendered command and descriptive native-MTP field differ, and it rejects the packed arm.

The version-2 hook is the live-map extension of the accepted version-1 callback contract. The driver reuses version-1 fixed-input/source helpers; this does not discard version-2 runtime witnesses. The separately reviewed target outer auditor is responsible for those witnesses and full cycle/state raw joins. Future source/job/gate binding and source-run eligibility are not provided by this glue review. No accepted core kernel or numerical criterion was reopened.

## Repair closure

The final driver adds explicit aligned-nonpacked/calibration preflight, catches wait/seal-authentication exceptions into the existing stopped-receipt path, validates HTTP response/usage objects and integer prompt count, and extracts summary fields defensively. The other three reviewed implementation files remain byte-identical to the initial hashes above. No input, phase, budget, numerical criterion or engine setting changed.

**42/42 independent CPU controls and 5/5 selected parent tests PASS.** All original failures were replayed and closed. Foreign/missing arms cause zero HTTP calls; real malformed JSON and malformed raw metadata, malformed response/usage, wait failures and malformed failure summaries each stop after one mocked request and preserve the per-request receipt plus DRIVER-VERDICT. Positive controls exercise all 24 actual frozen calibration records and a synthetic R2 expansion with distinct salts. R2 in that control does not set a prospective campaign denominator.

Reproduction from the paper root: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python -B p0/monitor/review-response-20260927/native-sequences-driver-review/independent_controls.py repair1`. The selected parent suite emits an inherited unclosed-read `ResourceWarning`; it passes and this does not affect the scope/receipt results. Initial implementation copies, control versions and failure logs/results remain alongside the repaired snapshot. No unresolved material issue remains within this driver/patch/config scope. This is not acceptance of the separately reviewed target outer auditor, a source/job freeze, launch gate, model result, or full Q1 qualification.
