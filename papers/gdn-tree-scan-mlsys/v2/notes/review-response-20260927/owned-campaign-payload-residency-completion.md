# Completion of the bounded campaign payload-residency inventory

The separately authorized larger pass completed **all 20,416 payload inodes in the selected campaign run roots**, excluding the active `m1-native-c1` mirror entirely. It took **1.593 seconds**, below both the 100,000-file and 60-second limits. No truncation, stat/mincore errors, or inode identity changes occurred. Observations ended **2026-09-28T23:34:48.224017Z**. This closes the previous 5,000-file coverage gap for the named payload suffixes and roots; it does not cover arbitrary non-payload archives or unrelated directories.

**Total observed resident payload: 22,464,141,520 bytes = 20.921 GiB.** This is **13.309 GiB below** the earlier 34.231-GiB CUDA shortfall even under a hypothetical one-for-one release assumption. Therefore this pass identifies **no sufficient evidence-file target set** for the unchanged 82.26-GiB reservation. The comparison is not a new CUDA measurement or a reclaimability guarantee.

| Root under campaign `runs/` | Payload inodes | Resident bytes | GiB | Same-size local payload paths |
|---|---:|---:|---:|---:|
| `m1-c-baseline` | 16 | 35,635,004 | 0.033 | 0 |
| `m1-init-stage` | 205 | 82,100,224 | 0.076 | 205 |
| `m1-q-c0` | 440 | 4,092,561,875 | 3.811 | 16 |
| `m1-q-native` | 48 | 3,959,025,648 | 3.687 | 0 |
| `m1-q-reference` | 110 | 10,048,536,529 | 9.358 | 8 |
| `q1.2a` | 200 | 101,580,800 | 0.095 | 0 |
| `q1.2b` | 6,220 | 1,602,797,568 | 1.493 | 0 |
| `q1.2b-heldout` | 6,200 | 1,599,709,184 | 1.490 | 2 |
| `q1-native-smoke` | 6,977 | 942,194,688 | 0.877 | 6,977 |

The selected heldout-reduction, init, candidate-stage1, and candidate-console roots contained no matching payload files. The filter was exactly `.pt`, `.bin`, `.npy`, `.npz`, `.safetensors`. Resident totals clip partial final pages; the raw receipt also retains whole-page counts and full file sizes. All sampled inodes have `st_nlink=1` and are on **ext4 `/dev/nvme0n1p2`, device259:2 (st_dev66306)**. Both root `/proc` inventories covered615/615 PIDs with no unknown, vanished, or matching reader/writer/mapping. These are point-in-time checks of the sampled inode set; the excluded native mirror was not rescanned or treated as finished.

## Archive and completed-run binding

The complete inode ledger and historical/local-stat crosswalk are `PAYLOAD-COMPLETION-RESIDENCY.json` and `PAYLOAD-COMPLETION-BINDINGS.json`. Prior receipts establish completed, reviewed M1/Q1 component runs; a preserved failed attempt remains a terminal artifact, not a successful qualification. Calibration q1.2b's6216-object store contains5832 record-referenced objects checked by the completed CPU reducer and384 extra native states whose names/size/census were reconciled. Heldout's6198-object store has5814 independently rehashed referenced objects and384 extra states not independently rehashed in that review; its later CPU repair/reduction is separately preserved. Do not silently upgrade every content-addressed filename into independently verified bytes.

**Historical remote hash verification and an independent archive copy are different facts.** Current local stat checks find all6977 native-smoke objects and205 init-stage payloads at matching sizes. Most M1-Q/reference/native and q1.2a/b payloads are not mirrored at the same local paths: only16 C0 and8 reference payloads, zero heldout-native or q1.2a/b payloads, and2 heldout fixtures are present. Some other archive location could exist, but this review does not establish it. The JSON retains exact path/inode/size/hash-receipt provenance rather than calling these absent local payloads archived copies.

The strongest completed independent local archive in this pass is the recovered native smoke: its prior independent audit authenticated all6977 objects /1,195,911,168 bytes and complete finite observations; all same local paths/sizes remain present. Its current remote residency is **942,194,688 bytes (0.877 GiB)**. Even this is historical authentication plus current stat consistency, not a fresh content hash. The M1 historical hash indexes still bind exact path/size identities where available; no full tensor was reread remotely or locally for this diagnosis.

## Prospective boundary

The sampled no-reader payload population has a measured residency upper bound of20.921GiB at this observation, before any restriction to proven independent archives. That is insufficient to justify a capacity-covering proposal. No cache advice, eviction, slab operation, CUDA query, container, model boot, process termination, gate change, or experiment-counter increment occurred. Completion of the excluded mirror would remove its reader restriction only after a fresh check; it would not itself evict cache. No unchanged guard or scientific threshold has been relaxed.

The parent has separately requested one final bounded **non-payload file >=16MiB** pass within the owned campaign directory to test archive/transport extensions omitted by this suffix filter. That is a separate read-only scope, not an authorization to reclaim this set. Per-page cleanliness and actual CUDA release remain unproven; aggregate Dirty/Writeback and host cache totals cannot supply those guarantees.

## Hashes

- Raw complete inventory: `926e7ca246855cb7ec532adbd37f0e539bd4ad1287a7dfa2c5c733bda31e3168`.
- Exact source: `5a0e2dfe2ba0697a29c0d20e50c27f3b7dc641e2cd88fce6816f6f29f08a00f1`.
- Historical/local-stat bindings: `c54247d4e0ff793ad0d8e03faf710d790e2c46bb185355d66b2ae14e0b84dbf9`.
- Original capped diagnosis is preserved unchanged at `owned-m1-residency-diagnosis.md` (SHA c3672737c3d99a48f183f829559db4e02e23837464558e842a93874c33b04ef4).
