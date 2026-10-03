# Natural lifecycle driver correction and worker byte observer: bounded source review

Disposition: no concrete blocker found in the requested source delta. This is preparation only: no executable lifecycle request plan, runtime observation, F5 verdict, or launch approval was examined or established. No tests, GPU operations, runtime imports or implementation edits were performed.

## Reviewed bytes

- `q1_natural_lifecycle_driver_v1.py`: `3cdb1336d0a53352b307d4eafeb1390a730da0603c977168da8e8416e97b2039`.
- `q1_lifecycle_worker_bytes_v1.py`: `837fcb3017cffa013c3b9d2108569ab3b0b8b179fb8377d6dd1a003730eb6d3c`.
- `q1_lifecycle_device_store_v1.py`: `c23449ed4a77e4268488b2132633799d04e8065a76a8a16ec2e5464df3b9d36e`.
- `q1_natural_lifecycle_binding_v1.py`: `3003e737c3e4f77bbc8c3f698a844b485834cf5fcc13d239b8ce5dc8b31d5b4f`.
- Reused storage witness: `854b122a4d17d3e1d8099c8feafb6da10eac98c3a7bb21c2a146758608d0edf2`.
- Reused host worker observer: `493445f362fa74083efdedcc961c47b43472496070f802539d6652e71ca91ad9`.

## Driver closure

Lines 85–127 now distinguish entries attempted/completed, API dispatch attempts, returned API responses and authenticated raw observations. A reset that fails before completion dispatch no longer increments the API count. Lines 37–65 update the caller-owned reset dictionary before checking the actual scheduler outcome, retaining the HTTP result and fresh reset events when that outcome fails. This closes the two findings in `q1-natural-lifecycle-l5-consumer-driver-source-review-20260930.md` for the requested scheduler-failure scope. `requests_sent` describes entering the HTTP dispatch, not independent server acceptance. A transport/HTTPError can still leave status/body null; the explicit error remains and is not success evidence.

## Boundary and store review

Worker bytes lines 85–91 retain the old actual request object and its original validated control, require it still occupies the retiring request ID, then capture before the production removal. Lines 99–110 keep the new-entry context only during the actual worker update, clear it even on failure, and capture the resulting actual request before its first forward. Lines 113–127 bracket the real zero operation and preserve its parent update call ID and exact block list; an exception does not manufacture an after-zero record.

The preserved actual production runner removes finished states before zeroing newly allocated blocks, creates new states afterwards, and clears/moves fixed32 slot buffers before `_update_states` returns (`identity/lifecycle-source-composition-20260930/runner.py:1123–1152, 1220–1233, 1429–1550`). Its optional deferred callable only corrects async token counts; it is not a delayed allocation or zero operation. The caller invokes the update before model execution (line 5584), so the observer's stated boundary order is compatible with this source.

The byte wrapper binds the natural job, host-event job, actual request identity, source hashes and host call ID; uses the accepted independent 113-view registry; checks a whole-snapshot disk reserve; and retains read-only synchronized full backing-storage capture. Source closure now includes the new device observer and explicit per-entry phase lists. The content store validates encoded hash/length and raw hash/length, rejects linked existing objects, bounds raw decompression to one 16 MiB chunk plus the detection byte, and rejects trailing or incomplete compressed streams. No cache write was found in these helpers.

## Required connection limits

1. The future source composition must **replace** the existing host-only decorators with `worker_bytes` decorators, not stack both: `worker_bytes.update_states/zero_blocks` already nest the host observers. Stacking would duplicate host events and invalidate the predicted outer call-ID join.
2. This helper captures only explicitly planned phases. Zero snapshots are tied to a new-request update; an absent zero call produces no zero witness. An independent complete-plan/terminal reader must reject missing required phases and distinguish observed absence from proof of successful reset/reuse. The current driver correctly makes no lifecycle qualification claim.
3. The complete named cache backing storages are not all mutable commit products: output rows, accepted paths/lens and other control buffers needed by the separate stale-consumer diagnostic still need their own unchanged-byte/counter evidence. No claim beyond the helper's named cache scope is granted here.
