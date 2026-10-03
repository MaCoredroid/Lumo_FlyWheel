# Actual stale-forward consumer and natural driver: bounded review

The proposed negative is technically source-supported for **rejection of actual old forward-event metadata at the fixed32 commit consumer after real same-internal-ID replacement**. It does not establish refusal of arbitrary old product tensors paired with fresh metadata, and is not an executed L5 result or a lifecycle gate.

## Real source path and minimal connection

1. `identity/lifecycle-source-composition-20260930/runner.py:3122–3144` assigns the current pure-decode forward number from the process counter, calls `_fr13_fixed32_observed_begin` with that number and actual request IDs, then increments the counter. The producer implementation in `scripts/fr10_phase4_patch_vllm_tree_gdn.py:2114–2125,3623–3655,3768–3778` places the actual forward/request/mode/batch identity into the observed dictionary. It includes set-valued work fields, so retained evidence needs an explicit deterministic typed encoding rather than arbitrary stringification of sets.
2. The actual consumer call is `rejection_sampler.py:2567–2572`, after real TAW products and the existing diagnostic forced-product callback. `_fr13_fixed32_device_commit_route` validates current product/caller geometry first, then at lines 1977–2005 rejects observed/current forward mismatch. The first output writes are lines 2008–2009. No cache/publication work precedes that freshness guard in this function.
3. Retain the old observed dictionary from the real source request's corresponding consumer boundary, with its original forward index and request IDs, source observation/job/process identity and actual request object/incarnation. Observe the old request retire and a genuinely new request object with the same internal ID in the same worker; require a strictly newer live forward. Do not fabricate or rewrite the old forward field, reset counters, or infer replacement merely from two identical client keys.
4. At the replacement's actual consumer boundary, retain the exact current observed object and valid current products/destination arguments. Substitute only the retained old observed metadata, call the unchanged loaded consumer, require its precise freshness rejection, and restore the exact current object in `finally`. Assert current pure B1/spec counts and all earlier argument checks so the generic “pure/mixed row geometry drift” error cannot be credited to a different defect. Authenticate loaded consumer/producer bytes. A direct call to a rewritten stand-in would not suffice.
5. Compare protected bytes/counters immediately around the rejected call, after all legitimate current forward/TAW work has already happened. Include output-token and accepted-row destinations, persistent slot/spec path and length buffers, recurrent/convolution/target/MTP stores and relevant pending/current events, census/replay/completion counters. Cache-storage capture alone omits output/publication buffers. Restore metadata, then allow exactly one ordinary current commit/deferred seal; verify the expected one-event advance independently. If the negative returns, has a different exception, changes protected state, or restoration fails, stop the diagnostic and preserve the failure rather than continuing normal serving.

No inherent source flaw blocks this narrow negative. Its supported claim is a forward-generation freshness guard operating even when request IDs are reused. It does not prove a separate request-key comparison in this early guard, arbitrary stale-product detection, physical slot reuse, every lifecycle path, or APC state equality. The executable L5 plan must say which of those it actually tests; no existing result is relabelled by this source analysis.

## New driver findings

`tools/q1_natural_lifecycle_driver_v1.py` correctly uses the captured completion request-id/body route, keeps fixed token/sampling/budget inputs, authenticates natural raw records, stops on an error and does not infer a hit/reuse/refusal. Its reset endpoint query matches the exact API (`entrypoints/serve/cache/api_router.py:21–44`); HTTP 200 alone is deliberately insufficient. Requiring matching actual scheduler before/returned events and `success=True` is the right distinction. The endpoint is installed only with `VLLM_SERVER_DEV_MODE` (router lines 69–72), which the eventual launch must retain.

Two concrete receipt corrections remain:

- Driver lines 110–116 count `len(receipts)` as `completed_requests`, including an entry whose reset fails at line 85 before any completion HTTP call. Distinguish attempted plan entries, completion calls sent, successfully completed HTTP requests and authenticated raw observations. A reset-only failure must not increment a request-completion counter.
- `reset_prefix` raises before returning on failed HTTP/scheduler reset, so the outer `reset` variable remains None and the request receipt loses the partial reset status/body/event evidence (lines 43–61 and 81–107). Preserve a structured reset-attempt receipt as it progresses, including available response/error and matching observed events, even when it refuses. Existing raw event files remain evidence, but a generic exception string is not the claimed failed-reset receipt.

No tests, synthetic fixtures, new inference, GPU/container/cache actions or implementation changes were performed. This note does not grant a plan, population or launch authority.

## Inspected source identities

- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/lifecycle-source-composition-20260930/runner.py`: `06f2c52b5e0b1b9df3d9b62691f7ec95bf762bd16a2ef8e1021bbcebb17b8277`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/lifecycle-source-composition-20260930/rejection_sampler.py`: `89e249621d8be3a1ce3e3b8fcd1848446c71914adf77d599592cc272de947ae5`
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`: `c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_natural_lifecycle_driver_v1.py`: `3cdb1336d0a53352b307d4eafeb1390a730da0603c977168da8e8416e97b2039`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/identity/lifecycle-http-source-20260930/entrypoints/serve/cache/api_router.py`: `aba8ba22ca0885bce2031773c3cdeca9b53ce19402e0ffb22671a51d16768691`
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_lifecycle_scheduler_events_v1.py`: `7ef8190d5b4d18a4be50e677c38ec3bc2334593b60491ca964efa319e8d74df3`
