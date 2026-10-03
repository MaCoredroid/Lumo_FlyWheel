# Target-KV witness v1 independent source review

Disposition: **repair the identity/layer-coverage check before accepting this raw auditor as a sixteen-layer publication witness**. The independent logical map and simultaneous-copy byte oracle passed the bounded review. This is source/CPU review only; it grants no launch authority or full-Q1 qualification.

Reviewed source `tools/q1_target_kv_witness_v1.py`: SHA256 `195dadc17ddb0204786fccf3aaa5cbd68217bda2c0d3cfc3ce68ee66807133e4` (8,433 bytes). Supplied tests: `349ce54802e79d1f8e762d2666f3500b8585465af9aa5fe1654485f9b8494ea3` (6,176 bytes). Both are preserved under `p0/monitor/review-response-20260927/target-kv-witness-independent-review/snapshot/`. Scope excludes the evolving hooks-v3 bridge.

## Required closure: complete identity and actual distinct layer views

At source lines 114–120 the auditor compares the two identity dictionaries, but only validates shape and dtype. Equality of incomplete records is insufficient to establish the recorded storage was retained. The following three independently reproduced corruptions still return `target_kv_bytes_match=true`:

- Remove every identity field except `shape` and `dtype` on both sides of all layers.
- Set both sides' `ptr` and `storage_bytes` to zero and every stride to zero.
- Copy one layer's entire valid pre/post record under every one of the sixteen expected names. Names remain distinct, while the same object/view is counted sixteen times.

The third case also has a direct collection path: `capture()` lines 85–88 requires sixteen unique names but accepts a repeated tensor object under those names. For a root-only or other no-op remap, the duplicate-cache case can preserve all bytes and appear to cover sixteen layers. No numerical divergence needs to be fabricated for this omission.

Minimal closure: require the full signature schema, exact integer types and valid nonzero object/address/storage identities, valid stride/offset/storage bounds, and consistent pointer displacement for bf16. Reject repeated or overlapping target-layer views in capture and audit, while permitting proven disjoint views backed by the same allocation. The separately implemented bridge should provide the actual ordered target-layer name-to-tensor binding; labels alone cannot establish which layer was observed. Add positive disjoint-shared-storage coverage and negatives for missing identity, impossible stride/span, and the repeated-view case. No change to numerical criteria is needed.

## Checks that passed

The pure planner correctly uses the request block table to derive every physical tree slot, rather than trusting the remap's destination result. Its physical parent table equals `scripts/fr13_fixed32_topology.py`'s current `PHYSICAL_PARENT`; the shifted validity mask equals `(HYDRA27_VALID_MASK << 1) | 1`, rejecting physical rows 18, 23, 25, and 27. The implicit root and eleven-draft maximum are consistent. Root bytes are untouched; accepted drafts alone populate logical continuation positions. Neutral tails, physical-block aliases, slot-map corruption, and inactive paths are refused.

The production reference at `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:10039` gathers all source and destination-prior rows before writing each layer. The witness's expected bytes also come exclusively from the pre-copy object, so overlapping accepted source/destination positions are handled simultaneously. Every captured destination and complementary row is compared, including allocated neighbor blocks. No claim is made for uncaptured cache bytes or another request's state.

Independent standard-library controls passed 420 combinations: all 28 frozen paths × three block sizes (16, 32, 64) × five offsets (0, block-end minus 1/3/11, and next-block plus 7). Expected copies in that control are derived from the production operation's accepted-path and destination-permutation indices, independently of the planner's copy list. Eight corruption controls refused: negative slot, inactive path, block alias, missing layer, changed pointer, row geometry, resealed destination byte, and resealed untouched byte. These synthetic raw-byte controls do not execute Torch or a device.

The parent-supplied connected test log reports four methods passing in 0.703 seconds, including 280 cases using AST-extracted actual production KV16 functions. Its printed kernel SHA256 is `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`, matching the current source. I inspected that extraction and log; I did not independently rerun the Torch suite because the available local Python has no Torch. The original mask-numbering failure log is preserved rather than discarded.

## Evidence and limits

Reviewer controls and results: `p0/monitor/review-response-20260927/target-kv-witness-independent-review/{check_stdlib.py,STDLIB-CONTROLS.json}`. A reviewer script initially used the wrong repository parent depth and failed before reading fixtures; correcting that reviewer-only path allowed the controls above to run. No implementation bytes changed.

Additional source/evidence hashes:

- Topology: `c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc`.
- Frozen token fixtures: `607634e19235a58d2b6d73f8260dac3c49f842f9f1cba9a1789f329c56012223`.
- Parent connected success log: `4a7437960ed182cc224bc6a893a7e868d464d135e8eb29f1c898d7526ccb7ad3`.
- Preserved initial mask-numbering failure log: `408f23efb0dc3c1d0b97c67dd4589b6a884d1a2d8fbc64f273737650003ac480`.

The intended result remains a byte-publication witness over tree-suffix blocks and one allocated neighbor on each side, with `whole_cache_checked=false` and `candidate_qualified=false`. This review neither verifies runtime hook placement nor establishes MTP, full-cache, numerical, or full-model qualification.
