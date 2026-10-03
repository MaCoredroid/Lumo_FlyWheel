# Global clean-cache v2.8: bounded source review

**Disposition: hold receipt-consumer acceptance for one concrete binding fix.** The global producer action path has no additional blocker identified in this review. Fresh user approval remains pending; this review authorizes no live write, query, boot, or workload attempt.

Reviewed `tools/memory_recovery_v2_8.py` SHA-256 `c510f617fd3f4344247901cc71f2e67e5a831b4e1e4ec7ddc0f524d808616359` against preserved v2.7 `db30c0db380adedcf32a461f36dc7b8242fa2e08c6cfe24390535387f195adfa`. New test file SHA-256 `76b8ca70e5287eed438107630d55d054e9b055f7674fee04ce85bb0341b26d69`; proposal v1 SHA-256 `0a08c95f89f64459cc739f2eb9b351868f7cc43c4c69a3fb2537070273f87655`.

## R1 — Bind extracted authority fields to the preserved authorization

At lines 3601–3608, `global_receipt_problems` compares the source, operational configuration and global approval block to `receipt.authorization.fields`. However, `profile_binding_problems` lines 3639–3653 checks the hash-bound `authorization.copy.json` only for the operation profile, and the durable marker only for authorization hash and profile. Thus the receipt's extracted gate/config/approval fields can disagree with the actual consumed authorization and still admit.

Two independent CPU reproductions used `TestGlobal.run_case()` from the supplied test module, the real producer with its injected command transport, and the actual `verify_terminal_receipt`:

1. Clean fixture: producer exit 0, consumer admit true. Change only `r['authorization']['fields']['scientific_gate_sha256']` to `'b'*64`, save the receipt, then pass `'b'*64` as the expected gate. **Consumer still admits**, although `authorization.copy.json` retains gate `ffbc2dfc402782325da71132100e74ff511d1585dd80e4ea196ed4bcace3fef2`.
2. Starting from the clean fixture again, replace `global_clean_cache_readiness` in the receipt's extracted fields and `operational-config.json` with `{'proposal_sha256': 'b'*64, 'user_approval': 'changed receipt approval, absent from consumed authority'}`. Recompute the configuration's canonical SHA in the receipt fields/operational-config summary and the raw file's manifest hash/size. **Consumer still admits**. The untouched, hash-bound authorization copy retains the original `'a'*64` proposal and CPU-fixture approval; the consumed marker retains the original configuration SHA.

The mutation fixtures were temporary and removed afterward; no campaign source, authority or gate was modified. This is a consumer provenance defect, not evidence that the normal producer performed an unauthorized action.

**Minimal correction:** within the new global-profile consumer, hash-verify and parse `authorization.copy.json`, require the authority-required fields and normalized global block to match the extracted fields, and compare the consumed marker's source/config identifiers to those authoritative values. Derive the expected-gate/config/source/approval checks from that verified copy. Retain old profiles' behavior. Add the two focused negative controls above, with the clean control still admitting.

## Checked boundary and remaining handoff

`python3 -B -m unittest -v tools/tests/test_memory_global_v28.py` passed all **3 test methods**, including six refusal subcases and the legacy bit2 control. These are CPU fixtures, not recovery or experiment executions. Static review confirms the distinct explicit profile, profile-specific exact bit3 argv, durable one-use reservation before the write, maximum one write/query, fresh under-lock idle/client and Dirty/Writeback checks, both `drop_slab +1` and `drop_pagecache +1` at immediate and 10-second checkpoints, failure blocking the query, fixed 88,326,002,442-byte query threshold and retained exact-CID cleanup evidence.

Before any parent-issued authority, retain the explicit operational values in its source-bound proposal/config: Dirty ≤65,536 KiB, Writeback ≤16,384 KiB, both counter deltas +1 at both checkpoints, one bit3 write and at most one unchanged fixed-image capacity query. The authority must cite the actual approved proposal hash and the fresh user response; nonempty text in `user_approval` is an authority declaration, not independent proof of user consent. All earlier consumed authorities stay consumed. No additional scientific criterion, performance claim, or experiment is proposed here.

## Repair closure — R1 closed

Repaired runner SHA-256 `b85c13a77c9a6060d2f9b6bd221997851159f716efff8aa0c53751edfed22616`; test file SHA-256 `d042aee85e6352f27c5eb34c0ea760648369cbe5a88662b66fe666727a2fe49c`.

The global consumer now hash-verifies the preserved authorization, compares all 12 required values (11 common authority fields plus the global block), requires the original `approved` value to be exactly true, and checks the consumed marker's hash and source/config/profile/run/authorization binding. This is confined to the new global consumer.

Independently repeated **both original reproductions**, each starting from clean fixture state. The gate-only substitution and the independent approval/proposal substitution with a consistently rehashed operational config both now return `admit: false`, with `global receipt authority fields differ from original copy`. The clean fixture still exits 0 and admits. The supplied 3 test methods also pass, including six producer refusal subcases and the legacy bit2 control. The second independent attack was reset to the original valid gate before mutation, so its refusal is not inherited from the first negative control.

**Final bounded disposition: R1 closed; no remaining source/CPU blocker identified for this requested delta.** Original failing-source evidence above remains preserved. Fresh user approval and an exact source/config/proposal-bound new authority remain prerequisites owned by the parent; no live recovery, query or boot occurred during this review.
