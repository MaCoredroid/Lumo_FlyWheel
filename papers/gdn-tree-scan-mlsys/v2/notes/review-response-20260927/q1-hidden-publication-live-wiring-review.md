# Hidden-publication live wiring: bounded source map

The accepted helper can be wired without replacing any production copy, forward or gather. Required additions are one Runner-to-Eagle binding plus observed boundaries inside the existing first-pass path. This is an implementation map, not a live-qualification or launch approval. Existing helper criteria, request/path cohort, model and serving flags remain unchanged.

## Exact anchors

All following exact-line anchors occur once in the pinned generated sources; counts/hashes are retained in `p0/monitor/review-response-20260927/hidden-wiring-source-map-20260929/SOURCE-ANCHORS.json`.

| Source boundary | Safe anchor and required actual operands |
|---|---|
| Runner producer → Eagle | Runner8065, `draft_token_ids = self.drafter.propose(` followed by `target_token_ids=...`. Bind actual runner/drafter identity, request row, metadata, `target_hidden_states`, and parent `hidden_states` before this call. The supported padded/non-aux slice is8056, `hidden_states[:total_num_tokens]`; require actual total32, false `disable_padded_drafter_batch`, false auxiliary outputs, actual B1 speculative metadata. Compare view storage/offset/shape/stride and bytes with that actual parent slice, not a reconstructed expected tensor. Other `propose` methods at7856+ are different branches and must not receive this boundary. |
| Post-TSR selected index | Eagle797, `num_tokens, token_indices_to_sample, common_attn_metadata = (`. Read the actual index after the production rewrite764–795, query-start `[0,32]`, actual accepted paths/lens and local `_fr13_tsr_nrows==1`; join the selected index to the frozen accepted leaf using the existing helper. Do not derive the purported observed index from expected nodes. |
| Target-hidden buffer copy | Bracket exact Eagle6775, `self.hidden_states[:num_tokens] = target_hidden_states`. Alternatively bracket its unique call797–807 while proving the returned no-extra-slot route. Capture actual32-row source before/after and full logical destination buffer before/after; retain untouched logical tail. Require returned token count32, same metadata/request and unchanged selected index. |
| Actual model input | Immediately before exact first Eagle832, `ret_hidden_states = self.model(**model_kwargs)`. Require `method=='mtp'`, `pass_hidden_states_to_model is True`, and actual `model_kwargs['hidden_states']` from6906 as a same-storage prefix of the copied buffer. Bind observed `num_input_tokens`, mode and kwargs; first32 bytes must equal the copied source, and padded rows must match the already captured buffer. Padding beyond32 is permitted and observed. A successful copy into an unused buffer is insufficient. |
| Actual first model return | After normalization832–838, before unique marker840 `FR13_FIXED32_MTP_KV_POSTFORWARD`. Retain actual return identity and normalized `last_hidden_states` / `hidden_states`; this precedes MTP KV remapping. In current methodmtp, `model_returns_tuple()` is false (6925), so both sources intentionally alias the same returned tensor. Do not require independent allocations or equate forward output with target input. |
| Sampling gather | Bracket exact Eagle1011, `sample_hidden_states = last_hidden_states[token_indices_to_sample]`. Capture the actual normalized first-return source and actual produced one-row result. Recheck current indices/request/event against the retained entry. |
| Continuation gather | Bracket exact Eagle1031, `hidden_states = hidden_states[token_indices_to_sample]`. Retain the original source reference inside the observer before this rebinding; after the unchanged statement capture the original source and actual new result. Release temporary live references after capture. Do not accidentally treat the newly rebound selected row as the source-after tensor. |

## Request/event and lifecycle binding

Use the current candidate observation/case/control/run/process/PID and actual `runner.input_batch.req_ids==(c.req_id,)`, row0. Only arm for the one active accepted tree event: O0 imported, forced TAW call1, O1 absent, completed same-case target-KV/conv/replay witnesses, and matching accepted paths/lens. Prefill, warmup, capture and the later terminal/O2 event are outside this boundary and must not supply replacement records.

Runner7397 calls the real proposal-begin before7413. Generated GDN5935–6036 creates `_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT` from the actual pending event with `mode`, `batch_size`, `request_ids`, `measured`, `forward_step_index`. At entry retain the same live proposal object identity and pending event identity, require measured B1/current request, target-KV complete and drafter-KV not yet complete, and bind the MTP payload's request/forward-step/event-index fields. Keep one observer invocation token and an ordered boundary state machine; reject duplicate, reordered, foreign or stale callbacks instead of accepting an observation label alone. Snapshot only fields expected stable; the production dictionaries legitimately acquire completion evidence.

Two source-specific traps:

- Eagle641–646 consumes `_LUMO_FA_TREE_COMMIT_NROWS` by setting the global to0. After TSR, use the actual local `_fr13_tsr_nrows` plus retained owner/evidence, not a requirement that the global stay1.
- Eagle1004 clears `self._fr13_fixed32_mtp_kv_payload` before1011/1031. Capture its identity/fields before forward; at gathers bind to the retained invocation and actual pending/proposal transition. Do not substitute a new payload or require the cleared field still exist.

**Do not use `proposal['mtp_forward_calls']+1` as evidence for first forward832.** The counter's eager hook is at Eagle5265/GDN6330+ for later post-root forwards; the first32-row pass is not that counter's unit. Its occurrence comes from the unique actual832 pre/post callbacks and captured return. Runner7475's existing seal then requires completed event+1/pendingNone through hooks v6:561–582; join all three helper records and producer/consumer receipts to that same event before case acceptance.

## Refusals, retention and source closure

Refuse an active unsupported route: extra input slots; wrong batch/query span; non-MTP or tuple-return branch; missing hidden kwargs; auxiliary/compacted source branch; missing/foreign proposal/pending/payload; stale raw/identity; absent selected index. Eagle1014–1025 early-returns for one speculative token or parallel drafting before continuation: it must remain incomplete/refused, never synthesized. Do not hook later forwards5263/6730 or gather a later overwritten drafter buffer as first-pass evidence.

Retain partial records and stage/error on every failure; mark the active case invalid/fail-stop as existing hooks do. Require exactly one complete copy, sampling gather and continuation gather plus dispatch/return provenance. Offline audit must authenticate every raw object and join source→buffer→actual kwargs and actual return→both gathers; a self-consistent foreign tensor can pass the standalone helper and must fail this bridge.

Patcher v6 currently patches only Runner and rejection sampler. A versioned successor must add Eagle as a third pre-hash/anchor/marker/compile/receipt participant, preserving original operations. Pin helper/new collector/job/patcher and generated Eagle pre/post hashes through launch dependencies and the offline auditor. Validate all expected files/anchors before modification; retain failed patch receipts. No serving flag or source kernel change is required.

Focused CPU controls for the implementation: actual pinned copy/gather statements with a fake first forward; same-object alias positive; padded kwargs positive; foreign Runner source, unused-buffer kwargs, swapped return tensor, stale proposal/step, changed index, duplicate callback, wrong gather source-after, early exit and partial forward failure negatives. These are integration controls, not new numerical criteria or experiment requests.

## Source hashes

- `eagle.patched.py`: `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62`
- `gpu_model_runner.patched.py`: `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79`
- `gdn_linear_attn.patched.py`: `23df7748f02a742753e586487e3e905e0ffb23815a9c5aaf87f5cf2c4a2ff91a`
- `q1_candidate_hooks_v6.py`: `9bae999518bdfb36803052ee8c2a9943d6aedd41a9af43f5bc4e7bcf59be77af`
- `q1_patch_candidate_v6.py`: `c8dc13419b770d6c0bae50000e792efb3eb5d86bc0cf27b026bb22dcd7819f21`
- `q1_hidden_publication_witness_v1.py`: `92ac01369e8e337716183172ec8f64ec266b6601bca21384abef834361f2cd18`
