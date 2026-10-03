# Memory v2.4: reserved-name directory closure

**PASS — the single reserved-name-directory finding is closed for these frozen bytes.** No remaining issue was found in the changed region. Earlier F1–F5 dispositions remain unchanged; the accepted v2.2 M1 path and preserved v2.3 source are unaffected. This review grants no recovery, launch or scientific authorization.

Snapshot: `p0/monitor/review-response-20260927/memory-v24-reviewed-20260928T0546Z`, relative to paper v2. Independently verified **10/10** freeze members' hashes and sizes, zero mismatches.

| Item | SHA-256 |
|---|---|
| `FREEZE-MEMORY-RECOVERY-RUNNER-v2.4.json` | `d03409f90c7525eb26b5b8ec4dca5c6a222872871002dc27238166b50b8a3b1c` |
| `tools/memory_recovery_v2_4.py` | `8117bd6ae6f719b291caedd85a7a13368fb580f19e847990b626e70fc0d9c558` |
| `tools/tests/test_memory_recovery_v2_4.py` | `692c843a1ff50b68c759e128b188e73065db363b065d46e958d888129996f2c4` |
| `tools/identity/memory_recovery_v2_3_to_v2_4.diff` | `b5c9b11591990b6de6b49920c2316fd004628546d8ea3cef0a19f7e2949aac83` |
| Author CPU log, attempt 1 | `9294c7a021db2eb2f97c8ec94bf4e83667db2819ec4c0e05bc481e5730cbdbca` |

The tool paths above are under the snapshot's `repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/` directory.

The reviewed diff classifies marker-like nonregular entries **before** generic directory descent at runner **1796–1799**, and checks nested directory names at **1807–1809**. This addresses both previously admitted directory shapes. Ordinary run/legacy directories remain allowed. Other differences are version/schema compatibility and identity/documentation updates; the previously reviewed F5 ordering is unchanged.

Independent standard-library controls imported the pinned module under a non-main name and called only its idle/terminal verification functions against disposable local fixtures. **All 13 control cases passed:**

- Empty authority and legacy-only directory layout: idle admitted.
- Empty `AUTH-x.consumed.json/` at the authority root or nested below an ordinary directory: idle refused, with the offending path classified malformed.
- For each terminal receipt schema v2.2, v2.3 and v2.4: a valid canonical-reservation fixture admitted; adding a stray reserved-name directory at either level made **both** terminal admission and idle verification refuse.

These were verifier fixtures, not real recovery receipts or an executed recovery. The supplied 143-test author log was hash-checked but was not rerun; no broad suite was necessary. The preserved v2.2 and v2.3 files also matched their prior hashes in the freeze inventory.

No remote commands, operation-mode entry point, containers, GPU/model calls, process cleanup, memory operation, source edits or gate edits were used. The runner hash remained unchanged after the controls. Final authorization remains parent-owned.
