# Memory v2.2 / readiness consumer: bounded source closure

**Original F5 is closed for the reviewed failure cases. The narrow no-reclaim M1 idle-readiness path is acceptable for the exact parent-described authority state, conditional on the gate and launch-time state checks.** This is byte-specific source closure; no standalone v2.2 freeze or operational/scientific authorization is implied. General recovery admission has the scanner limitations below and is not approved by this note.

Reviewed snapshot: `p0/monitor/review-response-20260927/m1-v31-reviewed-20260928T0514Z/repo/`, with sources under `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/`.

| Source | SHA-256 |
|---|---|
| `memory_recovery_v2.py` | `4f336819a9a71e029adfed6979deb865204ac9ad845bc6ae7560f0d1e4f76f55` |
| `q1_readiness_consumer_v1.py` | `2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9` |
| `run_m1_init_stage_v1.sh` (readiness caller only inspected) | `0964766f91855e9ad1b1df0952c27da38e1ce897b59e5fd87848548f9ac66373` |

## F5 closure

The actual producer now writes an exclusive, durable authority-wide `finalization_in_progress` hold (**1510–1544**, invoked at **1602–1606**) before releasing the lock. It durably seals the final receipt (**1631–1638**) before clearing that hold (**1639–1641**). A failed terminal write routes through `_seal_incomplete` and retains the in-progress hold; a visible success receipt is explicitly insufficient while that hold remains. `verify_terminal_receipt` checks the real nested authorization/condition fields, authority state, and absent in-progress hold (**1647–1711**).

Independent standard-library controls called the actual `_finalize`, hold placement/removal, `_boot_hold`, `verify_terminal_receipt` and `verify_authority_idle` methods. All filesystem work was confined to disposable fixture directories. The operation constructor and operation entry point were bypassed; the receipt projection contained the actual fields consumed by the verifier. Commands were never invoked.

| Control | Finalizer result | Remaining holds | Receipt admission | Authority idle |
|---|---:|---:|---|---|
| Clean finalization | 0 | 0 | true | true |
| Terminal write failure before replace | 12 | 2 | false | false |
| Terminal write failure after visible replace, modeling directory-fsync failure | 12 | 2 | false | false |
| Interrupt after lock release, before terminal seal | interrupted | 1 | false | false |

These close the exact prior F5 counterexamples. No broad F1–F4 re-review or operational test was performed.

## Actual producer/consumer and no-reclaim M1 scope

The consumer delegates receipt and idle verification to the hash-bound producer (**consumer 31–50**) instead of maintaining the former incorrect flattened schema. M1 requires the gate's recovery authority and runner hash (**launcher 55–57**), obtains that authority from the gate (**75**), and calls this consumer (**124**). The consumer checks the producer bytes before dispatch. These caller facts matter: the consumer alone does not independently approve a caller-supplied authority or gate.

Independently executed the actual consumer `decide()` with its real, bounded subprocess dispatch to the producer's **read-only `--verify-authority-idle` CLI**. Temporary fixtures yielded:

- `LATEST_DIR.txt` plus a completed legacy-v1-style run directory, no v2 reservation/lock/hold: admitted, verifier rc 0.
- Real `.campaign-operation.lock`: refused, verifier rc 13.
- `holds/any-name`: refused, verifier rc 13.

Additional direct verifier controls refused an arbitrary hold subdirectory and a parseable top-level v2 consumed reservation whose run had no terminal receipt. Thus the earlier `.lock-v2`/`HOLD-*.json` and flattened-receipt errors are resolved for these actual states.

The parent reports the intended live authority is exactly:

`/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/runs/memory-recovery`

with only `LATEST_DIR.txt` and the completed `memory-recovery-20260928T031357Z` legacy directory, no v2 authorizations/actions/reservations, no campaign lock and no holds. **That remote observation is parent evidence, not independently inspected by this review.** For that exact state, binding this root and the checked producer/consumer bytes in the M1 gate, retaining `no_reclaim_boot_approved`, and rerunning the idle check at launch is an acceptable narrow readiness path. It invokes no memory operation. The future-recovery limitations below do not block this already-verified idle-only case. This note does not approve the rest of the M1 launcher or its scientific scope, which the parent reviews separately.

## General recovery-consumer limitations: fix before relying on arbitrary v2 recovery histories

`authority_state` (**1714–1756**) only examines parseable top-level `*.json` files with the consumed-marker schema. It skips nested paths, non-JSON extensions and malformed JSON; it also treats an existing `holds` path that is not a directory as empty (**1720**). Producer authentication, however, permits consumed-marker paths anywhere inside the authority and imposes no filename extension (**797–804**).

Independent disposable-fixture controls produced `idle=true` (and consumer admission under the no-reclaim disposition) for each of:

1. `nested/AUTH-x.consumed.json`, valid consumed schema, no terminal receipt;
2. `AUTH-x.consumed`, valid consumed schema, no terminal receipt;
3. truncated `AUTH-x.consumed.json`;
4. an ordinary file at the `holds` path.

These are limits of the **new scanner**, not findings against the accepted pre-write/cleanup logic or evidence that the parent-described M1 authority contains such states. A real retained lock or hold independently blocks admission, as demonstrated above.

Minimal future fix: align the producer's allowed reservation namespace with what the verifier enumerates (either a fixed producer-enforced reservation layout or complete enumeration of the declared marker paths); classify malformed declared reservations and malformed holds paths as unknown/refused. Keep the exact legacy-only M1 case distinct. No additional host probes, GPU tests or reclaim are needed to test this.

## Verification limits

Source hashes were checked before and after the controls and remained the values above. Only standard-library source imports, temporary fixture I/O, direct safe methods and the explicitly read-only idle-verifier subprocess were used. No SSH, Docker/container command, CUDA query, model call, sysctl/reclaim, real process cleanup, operation-mode runner invocation, source mutation or gate edit occurred. No author suite was claimed as independently rerun. Final source-freeze equality and all authorization remain parent-owned.
