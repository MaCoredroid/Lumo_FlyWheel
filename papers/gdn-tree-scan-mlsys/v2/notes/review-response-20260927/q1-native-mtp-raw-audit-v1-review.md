# Native MTP raw auditor v1: bounded CPU review

Disposition: hold this prospective auditor until the three grouped defects below are repaired. Reviewed auditor SHA-256: `decf0d2d865307913fe3a1617f1549775b6819a371e70d8b69e3d8dbf112fa72`. The implementation was not edited. No model, Torch native path, GPU, remote operation, qualification, or launch was performed.

The retained CPU package is `p0/monitor/review-response-20260927/native-mtp-raw-audit-v1-review-20260929/`. It includes byte-preserved sources, `controls.py`, full synthetic documents and content-addressed raw objects, `RESULTS.json`, and a manifest. Reproduce with bundled Python `/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` and `PYTHONDONTWRITEBYTECODE=1`. NumPy 2.3.5 was used locally.

The 31 controls comprise two valid fixtures, eight correctly rejected negatives, and 21 malformed documents incorrectly accepted. The script deliberately asserts the current vulnerabilities for reproducibility; a zero exit status means those reproductions matched, not that the auditor passed. Both valid fixtures use the actual vocabulary 248320, hidden width 5120, BF16 KV `[2,8,64,4,256]`, full blocks plus tails, four history records, and exact first/follow raw-score SHA pairs. One fixture crosses a block between the first and follow MTP rows (positions 63 to 64). These tensors are synthetic, not model outputs.

## F1: Snapshot labels are accepted without authenticating complete finite logical KV

Auditor lines 42–48 call `q1_reference_hooks_v2.authenticate_objects`. That helper (lines 86–148) authenticates only the objects explicitly referenced, deriving lengths from unpinned metadata. It does not validate complete coverage or numerical finiteness. It accepts an empty full-block list and absent tail without reading KV bytes.

Twelve controls pass incorrectly: all snapshot payload references removed; forged logical digest; correctly rehashed BF16 NaN; dtype changed to FP16; head geometry changed to 8×128 with the same product; shape block dimension changed to 1024 while the kernel field remains 64; wrong logical index; physical block outside the allocation; aliased or out-of-range physical block; storage pointer changed; and previously materialized prefix bytes overwritten with new finite content and a recomputed digest.

Minimal correction: use the accepted `q1_native_corpus_v1.Objects.get` finite/hash/length behavior and the per-attention validation in `snapshot` (lines 158–181), adapted to the singleton MTP layer rather than calling the whole 48/16-target validator. Pin the deployed native kernel geometry `[2,N,64,4,256]`, BF16, positive stable storage identity, and group consistent with the bound allocation. Derive exact full-block/tail counts from extent; require consecutive logical indices, valid tail count, unique positive in-range physical blocks equal to the allocation's corresponding columns, and exact K/V lengths. Recompute the ordered K-then-V SHA digest. Bind allocation block size to the snapshot. Compare the previously materialized logical prefix across the three checkpoints; new suffix rows may extend it, not replace it. The source `_attention_kv` at hooks-v2 lines 375–422 defines this serialization and digest order.

Do not report `raw_bytes` using fixed BF16 arithmetic while accepting arbitrary shape/dtype metadata (current lines 46–48). Count bytes through the same authenticated geometry and object store.

## F2: Record identity is not joined to its actual allocation and scratch row

Auditor lines 19–20 compare record generations, but ignore each allocation's generation. Lines 32–35 check only first-pass start/count and a limited block-prefix relation. They never validate allocation column count, block validity, independently recomputed slots, or the follow allocation. The recorded `scratch_slot` is ignored entirely.

Five controls pass incorrectly: allocation generation changed, allocated column count set to zero, wrong first-pass slot, empty follow allocation, and a nonsensical scratch position/block/offset.

Minimal correction: a shared allocation check should require exact integer generation/group/start/count/block-size/column fields, generation equal to the record, sufficient distinct positive in-range blocks, and `slots == [blocks[p//64]*64+p%64 for p in range(start,start+count)]`. Preserve the prior materialized block prefix, as `History.pre` does. The follow must be the before-Z checkpoint, whose preceding first pass has count one; its allocation must match the following real first pass's allocation and preserve the earlier allocated columns as required by `History._follow`. Reconstruct its scratch block/offset from `zstep`, join it to the follow allocation, and require the same row in the same-input record. The next real first pass consumes that position. These are existing source contracts from `lease`, `History.pre`, and `_follow`, not new numerical thresholds.

## F3: Same-input SHA equality is checked while contradictory row/API/decision metadata passes

Auditor lines 53–54 check three booleans, the number of speculative tokens, and the two score hashes. Four controls still pass with a false schema/API, aliased first/follow rows, a false recorded native greedy ID, or an incompatible native score dtype.

Minimal correction: require schema `lumo.q1.native-mtp-same-input.v1` and API `proposer.propose`; join first slot to the preceding one-token allocation and follow slot to the independently reconstructed scratch row; require distinct in-range rows. Require exactly two native score entries and compare each SHA, greedy ID, and dtype with its corresponding raw-audited history record. Pin or validate the history native dtype against the supported bound run policy. These fields are supplied by `q1_native_mtp_same_input_v1.verify`; the raw full-vocabulary rows already provide the greedy comparison.

This remains a source-bound consistency audit. The existing same-input record does not archive both pre/post cache-row byte images or all temporary cloned inputs, so CPU review cannot independently replay the native API or prove restoration beyond the pinned runtime's checks. Keep that boundary explicit rather than converting the booleans into an independent experiment claim.

## Existing controls and admission boundary

The auditor correctly rejects the tested wrong request, record-generation change, raw declared length, nonfinite logits, nonfinite hidden tensor, wrong same-input score SHA pair, target-history gap, and missing follow. Full-score smallest-ID greedy/tie/margin and exact-k checks remained active in the positive fixtures.

Parent reports the prospective driver authenticates outer seals, expected job/run/process/repeat, and per-observation policy. This review therefore treats outer admission as a required caller precondition, not a fourth standalone defect. Document that contract; the function `audit(doc, objects_root)` alone does not authenticate the job, fixture, or record seal. Reuse the driver's verified expected geometry/owner binding when available. This review did not independently approve that new driver.

No source or gate changed. Qualification and experiment denominators remain unchanged.
