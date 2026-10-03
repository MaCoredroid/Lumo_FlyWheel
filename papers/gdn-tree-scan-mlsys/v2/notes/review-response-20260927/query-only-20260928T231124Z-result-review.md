# Query-only capacity result — independent review

**Verified terminal result: INSUFFICIENT_CAPACITY; no model boot is admitted.** The measurement and cleanup are valid. CUDA free bytes are insufficient under the unchanged requirement, so an alternate host-MemFree guard is not a remedy for this result. No qualification, model-boot or workload counter advances.

Run: `q1-capacity-query-20260928T231012Z`, directory `experiments/review-response-20260927/runs/memory-recovery/memory-recovery-v2-q1-capacity-query-20260928T231012Z-20260928T231124Z/`. This review only read saved files and made one bounded read-only SSH evidence/inventory/verifier check. It performed no new capacity query, GPU work, container mutation, reclaim, source edit or gate change.

## Capacity and scope

The actual saved query stdout—not just the receipt summary—contains:

| Quantity | Exact bytes | GiB |
|---|---:|---:|
| CUDA free | 51,571,183,616 | 48.0294075 |
| CUDA total | 126,175,309,824 | 117.5099144 |
| Unchanged requirement | 88,326,002,442 | 82.26 |
| Shortfall | 36,754,818,826 | 34.2305925 |

The same raw JSON identifies NVIDIA GB10, Torch 2.11.0+cu130 and CUDA 13.0. The exact owned container image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. The query completed normally (rc 0, no timeout, 1.326 seconds); the runner correctly returned **exit 10** for inadequate capacity. The full operational interval was 23:11:24.474032Z–23:11:26.870835Z. These are operational durations, not performance measurements.

The gate explicitly authorizes query only and sets `candidate_launch_authorized=false` and `host_memfree_launch_guard_changed=false`. Source, image, fixed device/total, profile and exact configuration agree across gate, authority, launch argv, canonical operational config and receipt. Both emitted configuration and actual launch use `--reservation-gib 82.26`; the earlier default-budget mistake is absent.

## Raw evidence and lifecycle

- Independently hashed all **75 receipt-indexed members** and all **76 files** in the remote run directory (the latter adds the terminal receipt). Every local/remote SHA-256 and byte length matches. All **31 command records** match their saved stdout/stderr hashes and byte counts.
- Exactly one `docker create` and one `docker start -a` occur, followed by exact-CID ownership checks/removal/absence proof. CID is `f2f695ca0fe71b6ea26b95ab2569a844e9917c72f414873ddca0601ce00b8db1`; nonce `eb16092b234507611a3baebe`. Create stdout, cidfile, start target, both ownership inspections and removal target all agree. Raw inspections bind the immutable image and nonce, with states created then exited; `docker rm -f` returns zero, and the final exact-CID inspect reports `No such container`.
- The query command bytes match the frozen runner's `QUERY_PY` exactly. No model or vLLM command was invoked. Recorded action counts are **one query, zero advice calls, zero advisory workers, zero slab writes**. Raw `vmstat` confirms both drop counters unchanged. The receipt has no targeted-advice record or cache treatment.
- The durable consumed marker was written at **23:11:25.216714Z**, before create began. Its exact SHA, run, authorization hash, source hash, config hash, profile and `reserved-before-query` status agree. The authority remains **CONSUMED despite insufficient capacity**. It cannot be reused for another query or cache operation.
- Finalization is complete, cleanup proven and the source-bound postconditions pass. Host desktop GPU identities are unchanged; the final live check finds the campaign lock absent, holds empty and all authority reservations terminal. The receipt's `cleared_after_seal` field is null by its finalization ordering; the independent filesystem/producer-idle check establishes actual hold absence rather than inferring it from that field.

At the independent read-only check, **23:13:34Z**, `docker ps` and the NVIDIA compute-process listing both returned success with empty output. This establishes no running container or listed GPU compute job at that check, not a claim that unrelated CPU work or desktop graphics ceased.

## Other recorded conditions and limits

Both privileged read-only /proc passes inspected all **623 listed PIDs** with no unknown/vanished entries, mappers or writers. Their only three target-reader records are the readiness runner's own read-only held descriptors (PID 1668734, starttime 482455425). Pinned target identity remained continuous. All three served model shards report **0 resident bytes** at precheck, final recheck and post-query mincore boundaries; no advice was performed. This is residency evidence at those observations only and does not identify who owns the remaining RAM or guarantee future coldness.

The final pre-query Dirty/Writeback values were 808/0 KiB, within unchanged 65,536/16,384-KiB bounds. Host `MemFree` was 50,788,344 KiB before and 50,731,632 KiB after; neither is substituted for actual CUDA free bytes. The CUDA measurement resolves the previous capacity unknown. It does not demonstrate a driver leak, assign ownership of unavailable RAM or authorize a recovery action.

## Connected disposition

I ran the exact frozen runner's **read-only** `--verify-receipt` against the saved remote receipt, correct run/gate hash and expected query-only profile. It returns **13, admit=false**, identifying the non-SUCCESS capacity outcome and unmet reservation. It does not flag malformed command evidence or uncertain cleanup. Its separate `--verify-authority-idle` returns **0, idle=true**. Idle authority means cleanup/terminal bookkeeping is settled; it does not restore this consumed authority or admit a model boot.

Counts for this operation: **1 operational capacity measurement; 0 new model boots; 0 numerical-qualification results; 0 workload attempts.** Preserve the insufficient receipt and stop this automatic admission path. The observed shortfall is now an actual CUDA measurement, so this audit requests no alternate-guard change or repeated query. CPU implementation work is separate.

## Bindings and reproducibility

| Artifact | SHA-256 |
|---|---|
| Terminal receipt | `aa8926daeb1c4ebc8177f4710627510e26601f75ee463942d5da1fb9065bf856` |
| Runner v2.7 | `db30c0db380adedcf32a461f36dc7b8242fa2e08c6cfe24390535387f195adfa` |
| Gate | `e2dbb6d66808982a3026f85cd3e268c6f05df81b1502481ce016bf6e9408f0fa` |
| Authorization | `d63433a97dd11c05dd11a5324a5904d8d787e8cd0db454ab3fe666d0cb48eb04` |
| Canonical effective config | `093abee7af501b5cd82f85ec8a579ded182f5505d36b5f40c36f9c02e69fab2f` |
| Consumed marker | `431bbc0923e1fda01c333affdcbdbefb544ca0469e43e0b86d80ff14293af521` |
| Parent launch argv record | `63dbefc06798d8a5b5b18b2f8a69575f43440411ae901ce0aeac0cdd70dc8787` |
| Independent hash verification | `67a9c7dabdf46c479339332ccdc1510151eef9f840fad4c2942c9637ed8c2d5c` |
| Read-only remote evidence | `0555a9d6cb6b0245a2a6e47794feadd54da5c20893f5d24fad95d51e6fe20d22` |

Reproducer scripts `audit_local.py` and `audit_remote_readonly.py`, full hashes/byte counts, current-inventory output and exact consumed-marker bytes are retained in `p0/monitor/review-response-20260927/query-only-20260928T231124Z-independent-review/`. The canonical config hash deliberately differs from the pretty-printed config file's byte hash; the independent audit checks both the content equality and the canonical hash bound by the authorization.
