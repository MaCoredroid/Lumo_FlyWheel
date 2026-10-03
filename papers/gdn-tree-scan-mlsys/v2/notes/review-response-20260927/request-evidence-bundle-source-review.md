# Request evidence host join — bounded independent source review

2026-09-28. **Bounded source closure PASS after F1 and its repair regression were corrected; source/CPU preparation only.** No GPU, live process/container inspection, HTTP, model or workload ran. The source and controls are preserved in `p0/monitor/review-response-20260927/request-evidence-bundle-independent-v1/`.

`R` below means `experiments/review-response-20260927/workload-plan/tools/runtime-collectors/` relative to paper v2. Initial `request_evidence_bundle_v1.py` SHA `dec269c046b40b0441bd0b1c4b3c2041c1467426e3d8a7693abcc1558826f941`; supplied test SHA `0b95daef0fd2e0a2ac94f0352c7b33010c426d945d6b37534e4f6daf4bc75095`.

## F1 — replay event identity/descriptor and event-ordinal completeness

At lines 87–101, the graph loop validates its request/mapping link, dispatch-count semantics, canonical manifest bytes/signature and independently supplied manifest requirements. However, it does not validate the event's `graph_id` or `descriptor`. The producer explicitly emits and checks these fields; their absence or contradiction in the consumed record currently passes. Lines 53–54 validate each non-map filename but do not reject a reused event ordinal for the same PID across record kinds.

Independent synthetic positive graph evidence passes. Each following isolated mutation also passes unexpectedly, with all other record and policy bytes unchanged:

- Delete `graph_id`, or change it to boolean `True`.
- Delete `descriptor`, or replace it with `{'num_tokens': 999}` while the signed manifest describes 32 tokens.
- Rename `pid<P>.event2.graph_dispatch.json` to `pid<P>.event1.graph_dispatch.json` when this worker's sampling event already has ordinal 1.

Minimal repair: require a positive strict-integer graph ID; validate the exact producer-emitted descriptor fields against their signed manifest values; reject duplicate `(PID, event ordinal)` pairs across non-map events. Do not infer device completion or add a new scientific criterion. No contiguity requirement is needed: the special API mapping filename has no ordinal and workers can be separate processes.

The old reproduction is retained in `independent_controls.py` and `independent-results.json`. It uses toy records/owned-process callbacks and does not forge or change any real run or authorization.

## Other checked boundaries

The nine supplied controls pass. Independent positives/negatives confirm the named ID, randomized vLLM internal ID and mapping-file hash join, exact source-binding equality, strict completed-dispatch integer, worker process epoch and required graph presence. A wrong event producer, wrong source binding, wrong mapping digest, missing canonical payload, boolean dispatch count, absent graph, bad signature, foreign internal ID and different worker epoch all refuse. The actual ownership helper separately validates exact CID, private namespace and live ancestry; this review leaves that accepted implementation unchanged.

All sampling/graph events must come from the accepted allocation target's PID and epoch. API mapping may come from a distinct owned process. Request records retain filenames, original byte hashes and bodies. The two inspect calls and repeated ownership observations bracket this join, while inventory/byte checks detect changes during it. These are bounded observation guarantees, not continued liveness after return. No GPU query, tensor access or synchronization appears in the host join.

The signed manifest is captured-work evidence. Its presence alone cannot prove dispatch: the join additionally requires a named worker `graph_dispatch` record from the source-bound observer. That observer records only an already-captured FULL graph's successful host dispatch and buffers it until its enclosing worker call succeeds. Neither helper asserts device completion or numerical correctness.

## Trusted inputs and pending caller integration

`allocated` is an **already accepted allocation bundle** supplied by the caller. Lines 32–35 check its claimed producer, CID, attempt/boot and worker-policy digest, then select its target record. They do not replay the allocation/projection checks, and a producer-hash string is not authentication by itself. The supplied tests intentionally use a minimal allocation-shaped object, which this join accepts. This is an explicit input trust boundary, not a second validator for arbitrary caller-provided allocation JSON. The final owned caller must pass the actual validated `A.collect` result, or a retained artifact whose bytes and accepted validation have been independently bound, with the same frozen worker policy and CID. It must retain that bundle and its projection/source/ownership evidence.

Likewise, `completion_id` and `completed_request_count` are arguments, not raw HTTP/counter receipts. The function proves equality and strict count=1; it does not itself inspect HTTP success, finish state or counter population. The caller must derive them from the matching successful pre-agent request and completed-request counter bracket and retain those source receipts. The returned allocation digest binds the object used but does not independently confer acceptance.

The independently frozen request/worker policies, expected sampler/seed and capture-route requirements are trusted authority inputs, never to be constructed from the observed manifest. Final wiring must bind the exact helper dependency bytes, actual post-patch loaded sources, private startup import policy, source/installation receipts and request/worker ownership. The API and worker data directories require an explicit host/container mapping. No hook, host join or final freeze has yet been demonstrated on a live boot by this review.

Commands executed inside the snapshot:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_request_evidence_bundle_v1
PYTHONDONTWRITEBYTECODE=1 python3 independent_controls.py
```

Initial disposition was source-blocked only by F1, with all live caller/integration gates separately pending. The final closure is recorded below.

## First repair — valid producer-shaped graph still refuses

Repair SHA `db741173117e63e6be994a5a1f201a462739eae8d06021f083c688f2226eedda` adds the positive strict graph ID and per-PID ordinal guards, but its whole-dictionary descriptor equality (lines 98–100) is incompatible with the accepted producer. `request_worker_observer_v1.py` lines 188–190 emits exactly five descriptor fields (`num_tokens`, `num_reqs`, `uniform`, `has_lora`, `num_active_loras`); the capture manifest descriptor also has `runtime_mode`. The independent valid-graph control now refuses before any negative controls can run. All nine supplied tests still pass because there is no positive Lumo-graph fixture in that suite.

Minimal correction: require exactly the five emitted fields and compare them to the corresponding signed manifest projection. Preserve the separately required `descriptor.runtime_mode == FULL` manifest condition. The first repair and failed positive receipt are preserved under `request-evidence-bundle-independent-v1-repair1/`; no implementation or live records were modified by this review.

## Final repair closure — 2026-09-28

**F1 is closed with no remaining material issue in this bounded delta.** Final `request_evidence_bundle_v1.py` SHA `ee26a0e28bf0912b410ea5fede7e1525c4adcf4628357e8ce9bff1cbea7bbd62` rejects duplicate `(PID, ordinal)` at lines 54–59 and requires a strict positive integer `graph_id` at line 98. Lines 100–104 require exactly the five fields emitted by the accepted observer and compare them canonically to the corresponding signed-manifest projection. The separately frozen FULL runtime-mode requirement remains intact. The whole-dictionary mismatch from the first repair is removed. No numerical, ownership, completion or policy authority criterion changed.

The final snapshot is `p0/monitor/review-response-20260927/request-evidence-bundle-independent-v1-repair2/`. All seven snapshotted source/dependency files equal their canonical copies at closure. The accepted observer, allocation bundle, ownership and projection helpers are unchanged. The original failing snapshot and the first repair's valid-positive failure remain preserved.

The exact preserved F1 counterexamples now refuse: omitted/bool graph ID, omitted/wrong descriptor, and duplicate event ordinal. The producer-shaped valid Lumo graph now passes. **11 supplied tests and all 16 independent controls pass** against final bytes; the latter include two accepted-input controls and fourteen refusals. The existing nine supplied tests alone would not have exposed the earlier false refusal; the expanded supplied suite now includes a positive Lumo graph and relevant regressions.

| Final artifact | SHA256 |
|---|---|
| Host join | `ee26a0e28bf0912b410ea5fede7e1525c4adcf4628357e8ce9bff1cbea7bbd62` |
| Supplied tests | `90ac843e585978df3ae20c5ea328c549e7faa633375920a886f6cac6a3bf72d8` |
| Independent controls | `1976530ba9f250f72746bc94d5417d8e950145ff83eae1a4653ca24b66cc5911` |
| Independent results | `5b6f69a18295645e168c8a75a7712b93ee5e19c4f5b40d83d8dd7ade3b2c637b` |
| Snapshot inventory | `8bc5e26832e5c33b3e64cc862a1c9787397b96a4c3de83d162e0d9a8da8aefce` |

This disposition accepts the source join only under its explicit trusted-caller contract: a genuinely accepted allocation bundle and frozen policies, actual successful HTTP/counter receipts, exact inspected owned CID/worker epochs, and frozen helper/import provenance must be supplied and retained. The join does not independently turn producer-hash assertions or completion scalars into that evidence. Caller/bootstrap wiring, final freeze and actual runtime qualification remain pending; no launch authority is conferred.
