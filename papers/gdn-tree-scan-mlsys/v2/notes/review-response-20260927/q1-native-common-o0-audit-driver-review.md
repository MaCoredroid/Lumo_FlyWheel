# Common-O0 auditor and driver: initial bounded review

Disposition: hold the new common-O0 audit/reducer for the four scoped corrections below. The actual driver-to-auditor fail-stop connection passed its powered CPU test. No implementation, gate, launcher, container, GPU, or model execution was changed or run by this reviewer.

Reviewed auditor: `8bf80355385cbef91f231cbf52b0deba8d479b5e747abd43474217a1a32b0e75`. Reviewed driver: `cdc4e13ff85d6647d5dc4dc0cfa4a8918e8087d8e4a65a5b10b53c29a1153eec`. Accepted design note: `029ff514df7076f7dbeac3a15f2dc80928120faca2deb9dee8dc8481be46221e`. The real frozen `SOURCES.json` bytes match the supplied SHA `e4f7e3bb7944f197a62b02bad6fa85ca3b4f1792db9be3471b5e02855dc7e343`; this review did not independently reread its 6.7 GB source population. The parallel importer/freezer review remains separate.

Evidence is under `p0/monitor/review-response-20260927/native-common-o0-audit-review-20260929/`. `controls.py` uses real `NC.observation`, `NC.snapshot`, `NC.Objects`, driver `run`, and seal authentication with locally created content-addressed tensors. The fixtures have all 48 GDN layers (native 3×10240 BF16 convolution and 48×128×128 FP32 SSM), all 16 native64 BF16 KV layers, and the full 248320-score O2 row. Identical synthetic payloads are deduplicated. Neither target raw validation nor the new auditor is stubbed. Only HTTP is replaced with a local producer of declared synthetic seals. NumPy 2.3.5 and bundled local Python were used.

There are 19 raw-audit controls: two valid positives, five correctly refused negatives, and twelve malformed records accepted. A separate five-case reducer set and a connected three-request driver scenario are retained. Zero script exit means those expected reproductions matched; it is not auditor approval.

## F1: Natural bootstrap storage is not joined to the import/readback owner

Auditor lines 27–28 authenticate bootstrap tensors and their internal digest, but do not compare its storage identities to imported O0 or the receipt. Four correctly resealed controls pass despite changing only the bootstrap GDN bank pointer, selected native row, attention pointer, or attention physical block mapping. This contradicts the claimed same native allocation before and after import while all raw digests remain valid.

Minimal correction: compare bootstrap and imported O0 per-layer group/row/storage identities and native logical-to-physical KV maps at the same prefix extent. Join those identities to the receipt's native identity snapshot. Bootstrap values/digests must remain free to differ: the positive `different_natural_bytes_allowed` control intentionally changes finite bootstrap state and must still pass. This checks owner preservation, not equal natural-prefill numerics.

## F2: The independent native receipt maps and geometry are mostly unused

Auditor lines 35–45 compare request/prefix and selected pointer/stride/offset fields. They ignore `ids.tables`, layer groups, GDN bank shape/dtype, and KV dtype. Six controls pass after changing the attention table's first block, a selected GDN table row, a GDN layer group, convolution bank shape, convolution dtype, or KV dtype.

Minimal correction: validate the receipt's four native group tables and join its used entries to the readback. GDN row must equal its group's selected column at `prefix_len // 1024`, with native 1024 units and group `layer_index % 4`; attention full/tail physical blocks must equal the corresponding group-3 table prefix in native64 units. Check counts, bounds, distinct attention blocks, and group identities. Compare bank shape suffix/dtype with the native row geometry and require the selected row to fit the bank; compare KV shape/dtype with its readback. Pointer/stride/offset checks already present remain required. These fields come directly from `live_destinations`; do not substitute source-process physical addresses or add candidate fixed32 assumptions.

The complementary guard algorithm and its runtime identity callback are being reviewed independently. This finding concerns using the serialized independent map at the new raw-audit boundary, not reimplementing H4.

## F3: Fixed-source prefix/fixture identity is not tied to the audited destination controls

Lines 17–25 check source population, case selection, exact bound-row equality and the manifest SHA label, but never join the bound row's prefix identity to `prefix` or the manifest fixture SHA to the job fixture SHA. Two controls pass with a contradictory bound prefix SHA or a contradictory manifest fixture SHA. The exact mapping/manifest remains caller-authenticated; these controls demonstrate missing internal consistency checks rather than forging that outer authentication.

Minimal correction: require `bound.prefix_len == prefix.prefix_len`, `bound.prefix_sha256 == prefix.token_ids_sha256`, and `manifest.fixtures_sha256 == job.fixtures.sha256`. Keep the existing bound source record equality and imported logical digest check. The importer already applies related source-control checks; the raw audit should not silently accept a sealed contradiction.

## F4: The fixed-denominator reducer can count invalid or wrong-arm rows as success

`repeat_summary` lines 49–69 uses only case/process/repeat, O0 digest and winner. Its clean 84×A/B×R2 fixture returns true; missing or duplicate observation controls correctly make it false. But an explicitly `valid=false` row with an error still produces `all_target_decisions_pass=true`. Labelling every input row `arm=native_default_packed` likewise leaves the result true.

Minimal correction: return an explicit successful audit marker and case/arm/process/repeat/observation identity from `audit`; require usable aligned rows before a case can satisfy the aggregate prerequisite. An invalid, missing, foreign-arm or unaudited row must prevent that case's pass while retaining all 84 cases in the denominator. A future outer collector may instead reject those inputs before calling this function, but that required connection must be executable and source-bound; it is not part of the current function. Preserve diagnostic winners from valid observations without using invalid rows to satisfy the prerequisite. Do not count archived source rows as new hydrated repeats.

## Driver connection: bounded positive result

The connected test passes the actual 84-case/R2 population preflight, supplies real synthetic raw state/seals for two requests, and changes the third seal's common-source record hash. The actual new raw auditor refuses the third binding, the driver writes its failure receipt/verdict, and exactly three mocked requests occur; there is no fourth request. The two preceding observations authenticate successfully. This verifies the new insertion at driver lines 122–127 and stop at 135–139, not an HTTP/server run or completion of 168 real requests.

Existing checks correctly rejected a nonfinite bootstrap payload, wrong exact source row, wrong post-import digest, wrong import pointer, and missing import. The connected raw check also re-reads the sealed path, so the temporary `_raw_objects` field added by `authenticate_seal` does not invalidate the canonical seal.

Outer image/source/launch/boot identity remains an explicit caller obligation. New corpus/config/job/launcher sources are not approved by this note. Target-only scope, unchanged scientific criteria, and no MTP/full-Q1 qualification remain intact.
