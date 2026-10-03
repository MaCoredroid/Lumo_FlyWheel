# Runtime probe producer v1.2 — bounded independent source review

2026-09-28. **PASS for this source wiring and raw-retention delta.** No material blocker found in the reviewed scope. This does not approve a gate, live request, workload attempt or serving route. No GPU, engine import, live `/proc`, Docker, socket, HTTP or workload operation was performed; all execution used temporary files and injected host/HTTP/ownership boundaries.

Final reviewed producer `runtime_probe_producer_v1_2.py` SHA `4ea777422afd973ccb22ed7dcefbf654dd516c15266b56892223b812b6f092bf`; supplied tests SHA `ca276d1ef0f63e58132d5140b422be1bba5f78e22a12018c9ffcc813cee740e4`. Paths are relative to paper v2. Helpers reside in `experiments/review-response-20260927/workload-plan/tools/runtime-collectors/`.

## Source findings and closure

The previous join's trusted allocation/completion boundary is now implemented inside this producer rather than discharged by caller-supplied bundle JSON or completion scalars:

- Lines 437–450 check the prospective observation policy/common sampling and request scope, then call the actual accepted `Allocated.collect` before the one pre-agent request. The full inspector refreshes the exact container through `reload()`. Allocation source, target layer coverage, precision, process epoch and owned-CID checks therefore execute before the request. The public function takes directories/policies, not a preconstructed `allocated` object.
- The existing `_collect` retains the HTTP response and pre/post engine metrics; `_build` checks successful status, exact response ID, a single completed choice, finish state, counter attribution, measurement markers and proxy exclusion. The new `_augment_worker_observations` checks those actual results again before proceeding. The arguments passed to `RequestEvidence.collect` are the received response ID and computed exact-one completed counter delta, not fields declared by the caller.
- Allocation is collected again after the probe; record bodies, file bindings and policy digest must equal the earlier accepted allocation. The actual request join then enforces the mapped ID, allocated worker epoch, installed sampler/seed/cap, and applicable fixed32 host-dispatch evidence. Earlier source reviews of those helpers remain applicable; their bytes are unchanged here.
- The additive `_retain_bundle_files` helper (lines 481–489) preserves the exact original allocation/request JSON bytes, checks regular nonsymlink members, basename safety, SHA and size, and puts them in the source index. Allocation raw bytes are retained before the request. Accepted request raw bytes are retained after its real join. The pre/post allocation bundles and request bundle are also retained separately as derived records. On later failure, existing write-once refusal/source receipts preserve collected evidence and prevent another probe through the claimed output directory.
- Effective storage dtypes are derived from the accepted allocation records. Compute dtype is still the instantiated model compute policy, not a numerical proof of every operation. Named sampling comes from the accepted worker-request bundle. `Projection.check_sampling` labels the result as a named 32-token probe and explicitly sets `workload_limits_observed_by_probe=False`. Timeout, compaction and other controller limits are not fabricated as observations of this request.
- Lumo's dispatch projection uses the accepted request graph records, identifies the same target process/CID/scope and explicitly records `HOST_FUNCTION_DISPATCH_NO_KERNEL_TIMER` with device completion/timing inference false. Removing the old unavailable tree-counter placeholder does not establish kernel timing, numerical qualification, task coverage or scientific success.

The delta leaves the previously reviewed v1.1 HTTP/accounting/cancellation logic and original v1.1 file unchanged (SHA `5fc0273c24a4d1e66a4785312a741b8bc19c8546bc6b655b7ef53f6e11d335d4`). No threshold, engine setting or scientific computation was modified by this review.

## Independent CPU evidence

Snapshots: `p0/monitor/review-response-20260927/runtime-producer-v12-independent-v1/` preserves initial v1.2 SHA `d78d220b…`; sibling `runtime-producer-v12-independent-v1-repair1/` preserves final retention-enabled bytes and all controls. `SNAPSHOT.json` binds ten source/dependency/fixture files. Canonical bytes were compared with the final snapshot at closure.

**9 supplied tests pass.** These tests mock the complete allocation/request collectors. To test the connection beyond those mocks, the independent harness executes the **real** allocation collector, metadata projection, request-evidence collector, producer, source recorder and verifier over synthetic input files. Only HTTP, Docker inspect and `/proc` ownership observations are injected; no live process is consulted.

**11 independent connected controls pass:** a positive Lumo-shaped evidence chain; stale initial allocation refusing before any probe; changed post-probe allocation; stale worker event epoch; wrong graph descriptor; missing dispatch; extra completed request; missing counters; wrong HTTP response ID; measured-phase overlap; and proxy transit. The positive's observed call order is exactly:

```
Allocated.collect → one injected HTTP completion → Allocated.collect → RequestEvidence.collect
```

It also checks exact retention of one original allocation file and three original request-event files, source-index verification, allocated precision, no observed workload-limit fields, and one host dispatch without device-completion inference. Failed post-request cases preserve refusal receipts; the initial allocation failure sends no request. Synthetic `ALL_OBSERVED` means the producer assembled its required observed fields, not that a real or scientifically qualified serving route ran.

**6 additional raw-retention controls pass:** exact-byte positive; wrong hash, wrong size, missing file, symlink and traversal refusals. Commands run only inside the final snapshot:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_runtime_probe_producer_v1_2
PYTHONDONTWRITEBYTECODE=1 python3 independent_connected_controls.py
PYTHONDONTWRITEBYTECODE=1 python3 independent_retention_controls.py
```

| Independent artifact | SHA256 |
|---|---|
| Connected controls | `9f125e16bece3d54426eb722330a42aa77cbec676d0624581dc793fb73ba1d23` |
| Connected results | `d8fdbb9d19b36d15a4b25c5d1ccdcdcac16412153099071c3e10902bee6e17a0` |
| Retention controls | `a04786aa8f24c2b3703b2be05403a979c3223067da74ba80ce3446850a0e0b78` |
| Retention results | `eb950254c510e0342d791b3eb8619abdcb23a9e9498fde520f081676c972ec34` |
| Final snapshot inventory | `1eb41753e7a646c8bc08b8fe3484fcc9406e7689a79d3ba0bfcdbcc33c49ccba` |

## Remaining caller boundary

The function closes the earlier arbitrary-allocation/completion-assertion gap **when invoked as this reviewed producer with its reviewed dependencies**. Parent-frozen policies, expected model/source/precision/layer maps, exact directory mounts and helper/import identities remain authority inputs. This helper does not grant those policies authority merely because their status says frozen. The actual owned executor must install the source-bound worker hooks before allocation, invoke this producer in the declared pre-agent order, retain all referenced receipts, and verify that the later measured baseline excludes the probe.

Final source freeze, startup/consumer wiring, expected-route projection, controller enforcement, model/weight identity gates and scientific qualification remain separate. `verify()` checks retained hashes; neither `VERIFIED` nor `ALL_OBSERVED` is an experiment-admission verdict. No gate is opened by this review.
