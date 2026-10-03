# Lifecycle backing-storage witness: bounded source review

Reviewed `experiments/review-response-20260927/tools/q1_lifecycle_cache_storage_witness_v1.py`, SHA-256 `920e45f9fc81c05f4b1471d33dbb7ef856a6637861f2b2d4ab9288da1f2b4829`. Source-only review; no imports, tests, GPU/container/runtime actions or implementation edits.

## Concrete correction

The graph-capture guard at line 56 is not bound to the observed device. `registry_views` accepts any single CUDA device (lines 41–45), while `torch.cuda.is_current_stream_capturing()` queries the current device/stream. The later copies and synchronizations use the cache device. If these differ, the helper has checked the wrong device before synchronizing the observed one. Put guard/capture/synchronizations inside `torch.cuda.device(device)`, or explicitly require the current device to equal this device before checking capture. The currently intended single-device route may already satisfy that condition externally; this helper does not enforce it.

## Otherwise supported within the declared scope

- The independent registry gives exactly 48 target GDN names, 16 target attention names and the separate MTP singleton. The resulting dictionary contains 96 recurrent/convolution views plus 17 KV views. The existing production identity snapshot checks the complete GDN layer-to-bank mapping, including convolution transpose and offsets, before and after the read.
- An empty uint8 tensor is rebound to the existing untyped storage at offset zero with unit stride. This mutates the temporary view metadata, not observed cache bytes. Reading the entire storage includes unused regions and padding, regardless of the logical view dtype or stride. Deduplication uses storage base pointer plus byte size on the already-single device; every named view retains its original offset/shape/stride/dtype signature.
- CPU transfers are chunked at 16 MiB. Each content-store response is checked against exact chunk bytes/hash; offsets and a whole-storage streaming hash are retained. The existing ContentStore.put returns the required `bytes`/`sha256` fields. Post-read registry/signature checks refuse changed cache views or GDN bank bindings.
- These records are raw observations. Signature collection does not establish expected model dtype/geometry, live MTP producer/group ownership, a particular request/slot owner, valid initialized values, or which writes are allowed. Only run_id is currently checked in event_binding. The docstring and record correctly leave joined ownership/allocation/complement decisions to the future independent reducer; they must remain prerequisites for any L3–L5 verdict.

## Future connection limits

Before/after synchronization waits for queued work; it does not lock out another thread that submits writes during the multi-chunk read. The actual callback connection must provide an exclusive/quiescent boundary, and must not describe this alone as an atomic storage snapshot. Graph-capture checks must apply to the actual device as above.

The 16 MiB limit bounds a transfer buffer, not total disk output. Every unique backing allocation is read and can be emitted once per boundary; reserve against the summed storage bytes before admitting a full witness rather than assuming content-address deduplication will save space. This is a future operation admission condition, not evidence that any storage capture has run.

No outer collector, zero/retirement callback, job, lifecycle population or scientific verdict is supplied by this helper.

## Observed-device correction closure

Current helper SHA-256 `854b122a4d17d3e1d8099c8feafb6da10eac98c3a7bb21c2a146758608d0edf2` differs from the preserved reviewed `920e45f9fc81c05f4b1471d33dbb7ef856a6637861f2b2d4ab9288da1f2b4829` only by requiring `device.index == torch.cuda.current_device()` immediately before `is_current_stream_capturing()`. This closes the single concrete source finding above. The helper's raw-observation limits and future callback/admission conditions are unchanged. Closure was read-only diff/hash review; no tests or device execution.
