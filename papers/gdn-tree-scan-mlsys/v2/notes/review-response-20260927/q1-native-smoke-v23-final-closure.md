# Native-smoke launcher F1 closure: PASS

The original wrapper lifecycle finding is closed for the frozen v2.3 package. No remaining F1 fix is requested. The parent may apply its one-run native-smoke gate to these exact reviewed bytes; this note does not modify that gate or authorize any larger experiment.

## Exact final identities

- Freeze `FREEZE-Q1-FULLMODEL-DESIGN-v2.3.json`, frozen2026-09-28T02:26:29.641440Z: raw SHA-256 `a381f936c937d1c29fa4e0ddb94187cd0cac335a980e6e8f643c86930e99691e`; independently verified canonical SHA-256 `32bfac15205d1dd572eb4f4a1ae32e5d1d84b64770d4be86071cb5c932b496cf` (exclude `frozen_utc` and its canonical-hash field, sorted compact JSON).
- Launcher `run_q1_native_smoke_v2_3.sh`: `b1a61fb2a5a142ed882be00b49cfcffb6257c9643c48d9a040435e581f9795be`.
- Cleanup library `q1_native_smoke_cleanup_v2_3.sh`: `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524`.
- Test source: `7062ae4e487199cbe7cbcf0c244578e944465f972c58f6436e8163cda339dd3b`.
- Frozen worker test log: `6a42d594c38a4409a6dcf09a203fcb5001656165b7a4301807d4dfea48ee2faa`.

These new files match the freeze by hash and size. The seven earlier repaired full-model sources remain hash-identical to their accepted review; their semantics and scientific criteria were not reopened. The freeze's expected gate entries bind both the launcher and cleanup library. The library is sourced only after that check succeeds. Preserved final snapshot: `native-smoke-v23-closure-snapshot/` next to this note. Earlier failure reports and snapshots remain intact.

## Closure evidence

Independently ran all six shipped CPU test functions against v2.3. They cover verified stop/remove success, failed stop and a contradictory still-running inspection, failed removal, genuinely absent container, failed Docker enumeration, combined finalization plus actual EXIT trap, driver failure with cleanup failure, unexpected abort, and receipt-write failure. All passed. Stop failure yields8, removal failure9, receipt failure10 when no primary failure exists; an existing primary nonzero status is preserved. Failed query is no longer evidence of absence. The actual trap is a no-op after finalization, including failed cleanup; it cannot retry cleanup and rewrite sealed artifacts.

Also executed five independent CPU controls using the actual launcher `write_receipt` function, actual cleanup library and actual EXIT trap, with only Docker/NVIDIA shell mocks. Final statuses: success0; stop failure8; driver2 plus stop failure2; Docker query failure8; boot6 plus stop failure6. In all five, `engine.finalized` equals the final marker; `FINALIZED.txt` is in the file index; every indexed file hash and byte count remains valid after process exit (respectively10,8,8,4,8 files checked). The review-only reproduction is `check_native_smoke_final_receipt.py` beside this note; run it with the final snapshot directory as its single argument.

`FINALIZED.txt` is written before the atomic receipt. No output artifact is written afterward by normal finalization or the EXIT trap. Cleanup state and diagnostic logs are retained even when stopping fails; the stopped timestamp requires a verified terminal container state. The parent-requested finalization-order correction is therefore verified using the real receipt function, beyond the shipped receipt mock.

## Scope

This closes only launcher readiness for one aligned/nonpacked native process A, shortest calibration root-only case, two repeats, pinned current image and patched FA2. It does not establish successful engine boot, full-model numerical qualification, candidate behavior, held-out results, timing or workload claims. No GPU/model/Docker execution, remote mutation, implementation edit or gate change was performed in this review.
