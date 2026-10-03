# Supplemental joint raw/state/patch review

Reviewed 2026-09-29T16:34:13.406443+00:00. **BLOCKED for the supplemental raw auditor at the initial snapshot below.** Three concrete raw-admission gaps were reproduced. The physical256 state successor and pure patch composition pass this bounded review. No GPU, engine/model, container, production-source edit, patch installation, or gate action occurred.

## Initial source identities

| File | SHA256 |
| --- | --- |
| `tools/q1_candidate_joint_raw_v1.py` | `3bf9fcf41205333460af788a1259f869f9dbc30ef8d2c22729620562c66f2c69` |
| `tools/q1_candidate_state_audit_v2.py` | `cf2ae8f378371e0626ad10346c1e679b412782a1463f87bff1f601516bde645c` |
| `tools/q1_patch_candidate_joint_v1.py` | `a954eab93927a28a57e590a644197cc40d20c96dc9d60350699f8f018ce8df28` |
| `tests/test_q1_candidate_joint_raw_v1.py` | `878dd1d59af8508329a687bba5e2064bae093ec0e3e60bdd8ee0d1cf038aaa34` |
| Preserved source inventory | `87b572624df8f74bc002a9c3d752104c9274395cf44761fb2f26a05dcc9fa119` |

Sources and exact reproduction scripts are preserved under `p0/monitor/review-response-20260927/candidate-joint-raw-independent/initial/`. Line numbers below refer to that immutable raw-auditor copy, not any subsequent parent repair.

## Concrete corrections

**R1 — required graph input/provenance fields are ignored (raw69–80).** Starting from the complete two-event CPU fixture, deleting `live_graph_signatures`, replacing them with an empty dict, deleting graph `model_input_kind`, and changing that kind to `inputs_embeds` while retaining no `input_embeddings` raw tensor all pass. Every result still says `actual_inputs_graph_and_two_seals=True` and `candidate_selector_observed=True`. The known MM-capable route needs the captured embedding payload to remain mandatory, and the newly recorded graph signatures must be consumed by the raw verifier if used as provenance evidence.

Minimal fix: validate the actual graph-witness record schema and required tensor set against its input mode; check physical input dimensions/position axes/dtypes and finite embeddings when present; validate the required live-buffer signature fields against the accepted witness's declared geometry. Do not compare CPU export pointers to live CUDA pointers: these are different objects. Preserve a valid input-ID and padded embedding positive so the repair does not accidentally refuse the real route.

**R2 — graph selector validation is weaker than root validation (raw80,85–88).** Root lines65–67 require int64 selector tensors, in-vocabulary values, and distinct ordered top3. Graph validation checks only the two shapes. Replacing the first-follow graph raw selector with `[248320]` and `[[248320,248321,248322]]`, duplicate `[[17,17,17]]`, or float32 tensors `[[17,18,19]]`, while updating the matching projection, all passes. These are content-addressed, correctly hashed replacement tensors, so this is not a corrupt-file test.

Minimal fix: enforce int64, vocabulary range, and ordered-top3 distinctness on graph spine/top3 for both phases, then compare the projection exactly. Keep verification of actual returned selectors; do not replace them with recomputed top-k.

**R3 — request-map labels are not joined to the saved KV map (raw68,77–82).** Changing both root and graph `blocks` from `[1,2]` to `[2,1]` and changing captured active slot IDs accordingly passes even though every `mtp_prefix`/`mtp_first_follow_prefix` physical page list remains the original mapping. Each KV snapshot passes `State.attention` separately, but the graph's claimed actual write map is a different map.

Minimal fix: compare the covered prefix of the full root/graph request map with each corresponding raw KV `physical_blocks` list, at the snapshot's recorded extent; require the root and first-follow physical cache identity/geometry to agree. Full request maps may include spare pages whereas snapshots include only materialized pages, so exact whole-list equality would be incorrect.

## Executed controls

The raw controls ran through read-only SSH using host Python3.12/Torch2.4.1, with `CUDA_VISIBLE_DEVICES=''`, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `PYTHONDONTWRITEBYTECODE=1`. Exact raw/state/test source hashes were checked before and after. Script input was sent on stdin; no remote script or production source was written. The existing fixture creates only temporary CPU tensor objects and raw files, which it cleans up. Twelve controls comprise one valid positive, eight invalid acceptances above, and three correctly rejected owner/epoch/deferred-commit negatives. This is synthetic CPU evidence, not observed live candidate output.

- Reproducer `remote_controls.py`: `958795f98008d477c8f19d731c63aa1180a88dd40e406b1dfc3fe3c732aa07e6`.
- Result `AUDIT.json`: `9cdb4c4ac1475bbc194995357056cbb5cb5a398537066e481c37d723dcca9232`.

The separate local CPU/source control script passes **12/12** checks. An actual reblocking export spanning three physical256 pages (513 tokens, native64 export chunks, page order `[2,1,3]`) passes; wrong tail page/offset, aliased page, missing chunk, undersized storage, and prior1024 geometry refuse. The v2 state diff preserves GDN and digest semantics while extracting reusable attention validation and changing the physical-page constants. No material new state-auditor defect was found in this changed scope.

The pure patcher reproduces all three prepared source outputs byte-for-byte and compiles them. Eagle differs from the accepted padded graph-prepared Eagle only in the intended hook-module import; the preserved source triple hashes equal the preparation receipt. Fresh-hash missing/duplicate anchors, stale expected hashes, and already-patched sources all refuse. No material new patch-composition defect was found. This does not attest an installed runtime patch or final helper/source freeze.

- Reproducer `state_patch_controls.py`: `12a9e3bfff03ac6ed0d96169843583665cb8b611b2ac45bc931ee43946366603`.
- Result `STATE-PATCH-AUDIT.json`: `7db44992b6833adf9f0b626dcd69a145da8bd6ebdadf9ff3e99da98a6ce31210`.

The new raw auditor is explicitly supplemental: accepted v9 target/publication witnesses, joint source import/union-complement, native traces, complete run provenance, and final caller integration remain separate obligations. These findings concern its own claimed graph/input/selector validation and do not reopen those previously reviewed components.

## First repair recheck — 2026-09-29T16:35:52.485248+00:00

Raw auditor `23a6c7bba780abf4c5c52bbee31efef98c02665dc670ea9186f1d4291cc2fb9c`; hooks `fcd4f3ee3012fb45c7eaf8f4ee6477b212e482fca472a8ee1387755c7baf1526`. All original twelve controls now behave correctly. The additional valid padded embedding/three-position-axis raw fixture also passes. R1/R2 are closed; the collector's added `actual_input_signatures`/`actual_embedding_signature` only copies accepted witness bindings into the receipt and does not change model or graph behavior.

**One R3 remainder:** only shifting first-follow KV `ptr` and `storage_ptr` together by4096, preserving all raw bytes, hashes, page lists, strides and internally valid bounds, still passes. The raw root/follow cache allocation identity is not joined. Require the source-recorded allocation tuple (`group`, `shape`, `dtype`, `stride`, `storage_offset`, `ptr`, `storage_ptr`, `storage_bytes`, `kernel_block_size`, `export_chunk_size`) to agree across root and follow; extent and materialized page coverage may grow. This is the cache-identity part of the original R3 request, not a new numerical criterion.

Fourteen controls: thirteen expected outcomes, one residual invalid acceptance. Reproducer `repair1/remote_controls.py` SHA `9f38a6aaa3126bca7145d30308477224bdf9fa928f243c6fdfebe0a8e07a7156`; result `repair1/AUDIT.json` SHA `8c54b66bbb89b927d310883179beee72fcb620571cce4ddb42792e8182d38613`. State/patch verdict remains unchanged.

## Final bounded closure — 2026-09-29T17:54:02.573164+00:00

**PASS for this supplemental source/CPU review. R1, R2 and R3 are closed; no remaining material issue found within the reviewed delta.** Current raw auditor SHA `1f2b2acec875918aa604cb368824f2c6b25d306e5d6c2999c0512a689566124e` adds the exact cache-allocation tuple comparison at111–112, joining every root/follow snapshot to natural and imported MTP O0 while permitting materialized page coverage to grow. The previously accepted displaced-pointer mutant now refuses with `actual MTP cache allocation changed between import/root/follow`.

All **14/14 independent raw controls** now produce their expected outcomes: both valid positives (input IDs and padded CPU embeddings/three axes), the eight original malformed-record cases, the allocation-identity residual, and the three owner/epoch/deferred-seal negatives. Exact remote source hashes were checked before and after this CPU-only replay. No raw candidate GPU output exists or was created by these controls. The unchanged separate state/patch checks remain **12/12 PASS**.

Final identities:

| Component | SHA256 |
| --- | --- |
| `q1_candidate_joint_raw_v1.py` | `1f2b2acec875918aa604cb368824f2c6b25d306e5d6c2999c0512a689566124e` |
| `q1_candidate_hooks_joint_v1.py` | `fcd4f3ee3012fb45c7eaf8f4ee6477b212e482fca472a8ee1387755c7baf1526` |
| `q1_candidate_state_audit_v2.py` | `cf2ae8f378371e0626ad10346c1e679b412782a1463f87bff1f601516bde645c` |
| `q1_patch_candidate_joint_v1.py` | `a954eab93927a28a57e590a644197cc40d20c96dc9d60350699f8f018ce8df28` |
| `repair2/remote_controls.py` | `09ff648df1c65447300ca13c6dec2f8e7926699b42619acd4ac8395314994dd2` |
| `repair2/AUDIT.json` | `8cd56e4ed26e6fe9fd69cbe65a151ea2d2f01c1907c31253475f33cbd46fabc5` |
| `repair2/SOURCE.json` | `6787b592f3a1d61f1f0f7bcaeffd4e60b0ce17abb45657d856fbccec19596f14` |

The hook delta only retains already-bound model-input and embedding signatures. Previously accepted callback fail-stop, graph copy/epoch semantics and numerical settings are unchanged. Final connected source/patch/job freeze, outer target/publication/import/run checks, and actual runtime qualification remain caller-owned and unapproved by this note. Initial failures and both repair snapshots remain preserved.
