# Continuous source-input inventory v1: bounded review

2026-09-29. **PASS for this exact deterministic CPU input inventory.** No implementation edits, GPU/model/runtime operation, held-out output inspection, gate change or experiment count advance.

Reviewed source `experiments/review-response-20260927/tools/q1_continuous_source_inventory_v1.py` SHA `7c1fca833b050b74bc34ff971b0ee09da59b5b60752fba273ae9283391f7c0a0`; its unchanged schedule dependency SHA `c7c1e53bc2cb2be78f9ef143f6eacd804c2c50765b23f9798fca142c1755afb1`. The regenerated canonical JSON exactly matches `p0/monitor/review-response-20260927/CONTINUOUS-SOURCE-INPUT-INVENTORY-v1.json`, SHA `4bc6c582377e1741c832780a7cd0b47c8e6b937f49ec34e10135a4a2ef4d8bd1`.

## Verified identity and scope

- Lines 25–30 authenticate the exact fixed sequence fixture (`e9683d2f8097ecfe195331d9977f9e57e1ab8482f1d583e04c5268ad92777318`) and accepted three-source manifest (`49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8`). Neither is selected by an observed output.
- Lines 35–49 resolve each original prefix under the declared prefix-plan root, verify its bytes, append exactly the already-frozen little-endian uint32 padding, and require the resulting extent and root/split. Lines 50–57 group by split plus complete prefix/root digest and admit reuse only on simultaneous prefix hash, extent and root equality. Evaluation cannot reuse a calibration source outcome.
- Independent grouping used complete byte strings, rather than the implementation's digest key. It reproduces all **24 groups / 48 record mappings / 108 existing cycle descriptions**. The three reused unpadded calibration sources cover 15 records, with `(extent, root)` equal to `(13487, 11352)`, `(28941, 363)`, `(60083, 7643)`. All accepted record/file identity fields propagate unchanged. There are **9 new calibration inputs and 12 new evaluation inputs**. The 18 padded F4 inputs retain their existing padding; no new token or task population is generated.
- The three accepted sources are inventory references only. Line 66 explicitly preserves mandatory raw authentication at consumption. This tool does not authenticate/reopen the source tensors or establish their reuse under an eventual live job by itself.
- Instrumented `Path.read_bytes` for the real build permits only the frozen fixture, accepted manifest and six frozen prefix token input files; the build uses exactly that set. No held-out inference/output file is read. Held-out *inputs* are inspected only to plan the already-fixed source identities.
- The output retains `source_only=true`, `launch_authorized=false`, `qualification_denominator=null`, `process_repeat_expansion=null`, and zero workload attempts. Source materialization counts are neither process/repeat denominators nor completed experiments. Evaluation execution remains separately gated.

## Powered CPU checks

**23 controls passed** in `p0/monitor/review-response-20260927/continuous-source-inventory-independent/controls.py`.

The real builder reproduces exact output bytes and independent full-byte grouping. In reviewer-owned copies, it refuses changed fixture whitespace, padding, root, extent or path; changed accepted-source root or record path; corrupted/truncated/missing prefix bytes; and a same-content prefix symlink escaping the declared root. Hash-pinned fixture changes are refused at the outer integrity boundary, not treated as a permissible new population.

The actual CLI refuses an existing output, output equal to the fixture, and output equal to the accepted manifest via exclusive creation, preserving original bytes. The canonical implementation and inputs are unchanged after testing. Exact snapshots, six prefix byte identities, read-set/count evidence and test results are sealed locally.

No remaining concrete blocker found in this source-only planner. Eventual executable admission must bind this source, its schedule dependency and exact output inventory; authenticate the referenced accepted raw source at use; and admit newly materialized sources under their separately reviewed jobs. These are existing boundaries, not new experiment requirements.
