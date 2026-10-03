# Large non-payload campaign inventory: bounded completion

The separately authorized read-only pass found **zero non-payload regular files >=16 MiB** in `/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927`, recursively excluding the entire active `runs/m1-native-c1` mirror and symlinks. Thus no qualifying campaign-local archive or transport tar adds observed residency to the completed payload inventory.

The pass completed at **2026-09-28T23:38:52.985197Z**, elapsed **0.656 seconds**, with no cap truncation or reported stat errors. Limits were10,000 qualifying files/30 seconds. The excluded suffixes were `.pt`, `.bin`, `.npy`, `.npz`, `.safetensors`, already handled for the selected run roots by the prior pass. Root process inventories completed624/624 PIDs before and after, with no unknown or vanished entries. There are no target inodes, so zero matching readers/maps is vacuous; it does not mean the excluded native mirror or other host jobs finished.

No artifact content was read. There are no target-file identities to link against archived hashes or to propose for cache treatment. The exact root, excluded mirror path, suffix list, minimum size, process scans and memory counters are saved in the raw receipt. Its empty target set is preserved rather than converted into an invented target list.

## Combined interpretation

The previous complete selected-run payload inventory remains **20,416 unique inodes /22,464,141,520 resident bytes (20.921 GiB)** outside the excluded mirror. It is13.309 GiB below the earlier measured CUDA shortfall, even under a hypothetical one-for-one release assumption. This new pass adds zero qualifying non-payload files. **No capacity-sufficient evidence-file target set has been demonstrated.**

This is still a scoped accounting result: non-payload files below16 MiB, payloads outside the previously selected completed-run roots, symlink targets, the excluded native mirror, and unrelated/model/host directories were not covered by this new pass. In particular, these findings are not a bound on all campaign cache or global `Buffers`/`Cached`, and cannot identify another memory owner or authorize reclaiming global buffers. Historical remote hash checks and independently completed local archives remain distinguished in the preceding completion note.

No cache operation, CUDA query, model/container launch, process mutation, gate change, numerical criterion change, or experiment-counter increment occurred. Existing one-use authorities remain consumed. No further scan is implied by this note.

## Sealed evidence

- `LARGE-NONPAYLOAD-RESIDENCY.json`: `555e75b6b0af1dcb0d395fb1bbfe10959216bea7eddd7fa072bd3c15bd596f90`.
- Exact read-only source: `65f87526c552731380d6fa5d10bae61f6b05d72813e7aa0cbf6340dcbb8dff5c`.
- Prior payload completion note: `owned-campaign-payload-residency-completion.md`, SHA `ad146b4830c172ceb659783100921438fe6f78212a56d2dfa68c488dc2dba4f1`.
- Prior payload binding ledger: `c54247d4e0ff793ad0d8e03faf710d790e2c46bb185355d66b2ae14e0b84dbf9`.

All raw artifacts and seals are under `p0/monitor/review-response-20260927/owned-m1-residency-diagnosis/`; earlier capped receipts and review notes remain unchanged.
