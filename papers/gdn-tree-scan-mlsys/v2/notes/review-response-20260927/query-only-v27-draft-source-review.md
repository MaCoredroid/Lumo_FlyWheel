# Query-only v2.7 draft: bounded source and connected-receipt review

**Disposition: one bounded verifier repair required before source acceptance.** The producer's explicit query-only path and one-use behavior passed the focused injected controls below. The new query-only verifier still admits incomplete/contradictory saved query evidence. This is an evidence-validation defect, not a finding that the actual producer executed an extra operation or reported a false real capacity result. No v2.7 freeze or operation is approved by this review.

Snapshot at 2026-09-28 20:00:36 UTC: runner `426f016c5458264cca6ff17f55270af1feca267c4d9188a470845bc475d41efa` (268391 bytes); tests `c712da087afabfef159c90ac283f411d195ed7bea5e86e8ba19aec6b98337c6b`; wrapper integration tests `a20eaa079ce8b302e2a4ef81d11c9b92703f596f73c6aac9726e0123275e63d0`. This is an unfinished source snapshot, not a worker freeze. The supplied diff is `b544df40c5565f464d39491912f0b9d16e9bd575c8112d151b5323cbcb486d1c`; a reviewer-derived exact diff against accepted v2.6.1 `548c6c26…` is preserved separately. Consumer v1 remains `2230ba1d…`.

## F1: query-only verifier permits a missing start or inconsistent CID evidence

At `query_only_receipt_problems`, lines 3404–3410, `len(starts) > 1` allows zero starts. The subsequent checks trust the `query` summary's successful return code/result/cleanup assertions, without linking the actual start command, ownership evidence and removal to the same CID. `profile_binding_problems` checks the configuration and authorization files, but does not validate the recorded query stdout file. No later consumer layer repairs these omissions: consumer v1 delegates directly to this verifier.

I generated a positive receipt through the actual v2.7 `main → Recovery` path with injected host/command dependencies and a local authority directory. Its unchanged receipt passed the full verifier. Each of the following isolated edits still passed that same full verifier:

- Remove the sole `query-docker-start` command.
- Change only its command argv CID to another 64-character CID.
- Remove `query.ownership` entirely.
- Remove the saved raw stdout file of the one query, while retaining its index/command metadata.

The unchanged consumer also admitted the missing-start case when its subprocess adapter was replaced with a direct call to the exact same verifier. That seam avoids process execution locally; it does not bypass the consumer's runner hash, receipt selection, run/gate parameters or admission decision.

Minimal repair: require exactly one recorded create **and** start; require their expected successful argv/return/timeout identity, link the created/owned/start/removal/absence CID and immutable image/nonce, and reject missing or contradictory ownership evidence. For the query result consumed here, authenticate the saved stdout against its existing command/file SHA and size, parse it with the existing strict result validator and compare it to the recorded result. This need not become a general adversarial filesystem verifier: focus on the query's own evidence chain and ordinary missing/corrupt/mismatched artifacts. Keep this validation query-only so accepted old profiles are not silently redefined.

Focused closure controls: retain the valid producer-generated connected receipt; reject each of the four cases above through the full verifier and unchanged consumer; retain below-capacity/failed-cleanup refusals. A source-only helper test is insufficient because the full consumer path is the admission boundary. No live query is needed to close this issue.

## Checks that passed in the submitted draft

The `_flow` branch is explicit (1603–1658): query-only returns through `_query_only_readiness` before a slab write, and unknown profiles raise instead of falling through. `_advise_one` is guarded, and the existing command write guard restricts the slab operation. Authentication/config emission bind the profile, source, gate, immutable image/GPU contract, exact 82.26 GiB reservation and exact existing dirty/writeback bounds (1661–1793, 4129–4136). No arbitrary query-only reservation was admitted. The new `_reserve` records `reserved-before-query` before creation (1915–1943); the query stage rejects a second attempt (2888 onward). The new after-query identity/residency checks are diagnostic for residency and retain the pinned target continuity requirement.

Independent local producer controls passed for all seven declared scenarios: SUCCESS, insufficient capacity, invalid query JSON, uncertain cleanup, failed creation, identity drift across the query, and a query dependency exception. Every scenario recorded zero advice calls/workers and zero slab writes, at most one query creation, and a consumed authority after the marker. Failures did not admit. A separate positive connected receipt passed; an altered effective config was rejected. Unapproved-authority and wrong-source refusals remained explicitly UNCONSUMED, with no query. These observations support the draft's repaired pre-profile wording; no extra wording change is requested.

The local harness replaced only the unchanged absolute pinned-input file loader and used the author's fake host/command dependencies. It guarded real advisory calls and slab commands. Consumer invocation used an in-memory verifier adapter. These are transparent seams: they do not revalidate the already accepted pinned absolute input loader, Linux shell wrapper or live host, and they executed no Docker, CUDA, process inspection or cache operation. The parent's separate report of 82 focused v2.7 pytest controls passing is consistent with this review; it does not cover the four newly demonstrated missing-evidence cases.

## Scope and artifacts

`p0/monitor/review-response-20260927/query-only-v27-draft-reviewed-20260928T200036Z/review_control.py` reproduces the bounded cases in temporary directories. `REVIEW-CONTROLS.json` retains positive and refusal results plus the four false admissions and consumer verdict; `REVIEW-CONTROLS.log` preserves producer output. The frozen input snapshot and reviewer-derived diff identify the reviewed bytes. No worker/parent implementation, gate, authority, threshold or remote source was edited. No GPU/container/cache query or operation was launched. The prior v2.6.1 worker/source closures remain closed.
