# Native MTP history v1 implementation: bounded review

**Disposition: repairs required on the recorded snapshot before source closure.** No launcher, gate, GPU/model call, or implementation was modified by this review. The implementation is prospective; this note does not describe new experimental evidence. The parent is already repairing the findings below; later closure must bind the repaired bytes separately.

Snapshot identities:

| File | SHA256 |
|---|---|
| history v1 (after the first top-k repair) | `95856dc824cdd7207d85d8cc64c3694388a65caa5c3fd5c61298cfb72b1cba7d` |
| owner v2 | `cde48a325666a6abfa77a45ac614d793e6080d8f3a5b23cbcd1bb559f0912ec9` |
| program v2 | `d3c60ba6cebb3cdb4fe07e1185cbd6e291e4ca7be5998177624a3a6b50df729c` |
| hooks v2.3 | `061e5204b0cf8d0306c54041de20ddf6960fe2590a4f91e4ffe3954aba1b0bb9` |
| worker patcher v2 | `cdcdf33835f0ba3ed181ceade051ed471509484aaa3208bcb7375598a96837d8` |

Sources and reproducer are preserved in `p0/monitor/review-response-20260927/native-mtp-history-v1-implementation-review/`.

## F1: previously materialized prefix mapping is not checked on ordinary steps

`History.pre` (60–76) validates the new lease, contiguous token count, IDs and positions. It compares the prior block mapping only through `_follow` (110–112), which runs at the declared checkpoint. Ordinary prompt-chunk and chain transitions can therefore change an earlier materialized MTP block while passing every new-lease check. They then read another physical history on the next first pass.

The independent CPU reproducer executes the exact snapshot's `pre` and `lease` with structural metadata: the first 32 materialized rows are in physical block 1, then the next contiguous 32-row step changes both matching CPU/GPU tables to block 2. The unchanged-map control and the remapped-prefix case both accept. No MTP model is needed to reproduce the missing admission check.

Minimal repair: retain the last successfully materialized allocation; before each new MTP write require unchanged generation/group/block geometry and unchanged physical columns covering the already materialized prefix. Reject relocation rather than inventing a copy path. Newly allocated columns beyond the materialized extent may be appended. Add a noncheckpoint prompt-chunk control and a noncheckpoint chain-step control, both refusing before any model call.

## F2: exact top-k operation and k must match the declared categorical comparison

The initial `scores` implementation used stable descending `argsort` for `top32_ids`, thereby inventing smallest-ID tie ranking. The parent has changed the snapshot's line 87 to actual `torch.topk(logits[0], k=32, sorted=True)` on the native device/dtype; smallest-ID greedy remains separately computed. This fixes that operator mismatch for a **top-32 diagnostic**.

However, the first three entries of `topk(k=32)` do not establish `topk(k=3)` equality under ties. Current candidate Eagle explicitly describes the fused `argmax + torch.topk(logits, 3)` contract at line 3159. Collect each exact declared comparison k on native device/dtype (at minimum k=3 for that route), or label top32 diagnostic-only. Preserve full raw logits, actual operator/k/dtype/device, native ordered IDs, and smallest-ID argmax separately. A tied-score control must distinguish the native top-k result from stable argsort, and should not infer one k's tie ordering from another.

## F3: builder failure after capture leaves the owner ready

The worker wraps the real builder with `try/finally: end_real_metadata(scheduler)`. If the builder raises **after** `capture_common` has populated a record, `end_real_metadata` sees a matching scheduler/nonempty common and succeeds. Independent execution of the exact owner method leaves `failed=false`, `ready=true`, and captured common retained. The original builder exception still propagates; this is a state-machine integrity defect, not a demonstrated end-to-end numerical false PASS.

Minimal repair: an explicit exception path/abort invalidates the owner, common metadata and armed scheduler before rethrowing; normal finalization validates exactly one complete capture. Exercise errors both before and after capture. Preserve the original exception context where possible; no metadata may be consumed after either failure.

## Connections that are consistent on this snapshot

The partial-prefill path bypasses the old early return and selects the next actual prompt token. Normal steps explicitly call `complete_first`, retaining KV but releasing pending context. The checkpoint is the declared before-Z boundary; the next real pre-forward authenticates allocation, refreshes only the retained block table, uses retained first-pass MTP state, and restores the exact scratch K/V row even when follow-up raises. The terminal target-Z MTP first pass uses the raw O2 greedy token while separately recording the server's terminal token. History completion precedes a valid terminal seal. The native shifted-input mathematics is not reimplemented by the history wrapper.

The all-hidden-row capture is inserted before the selected-logits gather and excludes padded rows by the actual scheduled count. Owner capture is armed only around the real metadata builder; dummy calls do not populate it. Case validity additionally requires completed, nonfailed MTP history. None of these source observations replaces the still-needed repaired connected tests or runtime evidence.

## Cache-salt ingress progress and scope

The four new completion-source files match their exact-image manifest. Completion protocol line 145 declares `cache_salt`; 438–443 rejects nonstring/empty values. Live completions delegate to `render_completion`, then `preprocess_cmpl`, which passes `prompt_extras['cache_salt']` to `renderer.render_cmpl_async` (render serving 509–517). The separately pinned engine input processor consumes `decoder_inputs['cache_salt']`, and the native block-hash code salts the first block and chains later hashes.

The retained files still do not include the actual `BaseRenderer.render_cmpl_async` and its extras-to-EngineInput merge. The separate render-server `GenerateRequest(cache_salt=...)` branch is not proof of that ordinary completions path. Bind that narrow missing source edge plus the actual runtime salt receipt before relying on unique per-observation salts; initial zero-offset history refusal remains mandatory regardless.

Current history records persist full MTP logits and a target-hidden fingerprint, but do not persist full MTP KV boundary tensors. `scratch_restored` is a checked producer assertion without saved before/after scratch bytes in this snapshot. These limits must remain explicit: this collector is not yet a raw MTP-state comparator. Saving the tiny scratch before/after hashes or objects would support independent restoration auditing, without requiring a larger experiment or weakening criteria.

The supplied Torch history tests were read; they use a synthetic `Program` rather than the extracted native program and real owner/hook bridge. Parent-reported passing tests are not relabeled as reviewer-executed. The independent controls here are CPU standard-library only, cover the two reproduced gaps above, and preserve the finite positive control.
