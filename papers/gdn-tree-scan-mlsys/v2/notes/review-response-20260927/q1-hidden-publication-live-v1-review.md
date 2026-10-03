# Hidden-publication live owner v1 — bounded review

Disposition: PASS for corrected source/CPU owner binding at SHA256 `e634c349434f29cbc5d5045fdd2098144a56871a4a66721a1ff85dfd0bab3eae`. No remaining concrete blocker found in this bounded owner-only pass. It is not wired into the real collector and provides no runtime, numerical or launch qualification. Parent made all implementation repairs; this review used no remote, GPU, container or model execution.

## Source-grounded findings and closure

1. **Actual payload types.** Initial owner attempted `list(payload['batch_rows']) == [0]`, which is incompatible with the real scalar count. Generated Runner6833,7110 construct integer `batch_rows`;7111 retains tuple `spec_batch_indices`, validated at6861–6874. The corrected owner requires integer1 and exact tuple `(0,)` with an integer element. A retained intermediate version still admitted list `[0]` by tuple conversion; the final version refuses it.
2. **Fresh registry anchors.** The initial owner compared proposal/pending/payload only to each other. The corrected owner also checks the live `_FR13_FIXED32_CURRENT_FORWARD_STEP` and the current census-list length, matching Runner6903–6940 and7084. Stale mutually consistent saved fields cannot supply an independent owner by themselves.
3. **Restored pure-B1 route.** The owner now requires payload `slot_restore_complete is True`, `batch_indices is None`, and `permutation_query_start_loc` identical to the retained actual metadata tensor. Runner7815–7847 restores before the producer call8065; Eagle899–907 also requires the restored query identity. The entry requirement is therefore placed after a real source transition, not before it.
4. **Pending completion stage.** Before first-pass completion, payload remains the actual drafter payload, `drafter_kv_complete`, `kv_complete` and pending `event_index` must all be None. At sampling/continuation callbacks, payload must have been cleared, both completion flags True and pending event index equal the retained event. Eagle998–1004 performs these updates. The intermediate owner admitted a non-null old pending event index before KV; that reproduced owner-stage gap is closed. Proposal/pending object identity and complete-event count must remain unchanged until the existing outer seal.

These were owner-stage checks. Existing production validation remains independent; no complete engine false pass was claimed or executed.

## Request and metadata compatibility

The strict request tuple is correct for this stage. Retained `gpu_input_batch.py:108,289–292` stores an active `_req_ids` list, with None only transiently during updates; `condense():786` trims it to active count. `num_reqs:1056` derives from the active map, and Runner1422 condenses before use. It is not a permanently capacity-padded request list. A trailing None at this active B1 boundary is rightly refused.

Constructor requires the actual speculative route, non-auxiliary hidden source, padded drafter batching,32 actual tokens, one request, batch1 and query `[0,32]`, plus completed same-case target-KV/convolution/replay witnesses. It preserves the actual parent/target values for the accepted byte bridge. The owner does not mistake the post-root MTP forward counter for the first32-row call, require the already-consumed global TSR row count, or require the cleared payload remain present at gathers.

## Retained CPU evidence

`p0/monitor/review-response-20260927/hidden-live-binding-independent-20260929/` contains scripts, source snapshots, source bindings and results. Eighteen independent controls pass: two source-shaped positive transitions and16 expected refusals. Refusals cover payload type/row errors, stale current-forward/census indices, unrestored slots, equal-valued foreign query tensor, unsupported compact-index tensor, early pending event index, trailing inactive request entry, bad metadata extent, missing predecessor witness, premature KV completion, early complete-event count, uncleared post-KV payload, absent post-KV completion and foreign proposal identity.

The owner-calling Observer stub intentionally isolates Binding and does not pretend to test Torch, byte capture or live callback origin. Accepted bridge behavior was reviewed separately. The intermediate `cce75164…` source/control results are preserved under `before-final-owner-closure/`, including the two admitted malformed owner states; final `e634c349…` refuses both. Initial scalar-type and independent-anchor defects were source findings repaired before that intermediate fixture run.

## Remaining connection, unchanged scope

The future collector must supply actual Runner/Eagle hook arguments and current case, retain a partial case record before constructing Binding, provide the bridge's same-record sink, and mark every constructor/callback exception unusable. Bind the first-forward and gather callbacks to this same Binding; retain observed ownership in the sealed case/offline audit. Successful owner checks alone do not establish that the actual first forward, copy or gathers ran. The existing source map and accepted byte bridge specify those separate joins. No additional cohort, experiment, criterion or serving change is proposed.

## Exact source bindings

- `q1_hidden_publication_live_v1.py`: `e634c349434f29cbc5d5045fdd2098144a56871a4a66721a1ff85dfd0bab3eae`
- `q1_hidden_publication_bridge_v1.py`: `00d30bc67e34027b09eae5a0386017ffc7f8766d0a7399218abb5920a15622b8`
- `gpu_model_runner.patched.py`: `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79`
- `eagle.patched.py`: `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62`
- `gdn_linear_attn.patched.py`: `23df7748f02a742753e586487e3e905e0ffb23815a9c5aaf87f5cf2c4a2ff91a`
- `gpu_input_batch.py`: `2d6d6d4293ce70b1bc7a7fe47217727148959706c8ad6245e4ad060cd6a9a2bf`
