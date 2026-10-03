# Workload observation caller v3.7: bounded source review

**Verdict: PASS for the reviewed source/integration delta; no remaining material caller defect found.** These are draft bytes, not a frozen campaign or runtime qualification. This review grants no execution authority, does not open WP, and does not substitute for the separate review of `probe_boundary_v1.py` / `runtime_evidence_reader_v1.py` or final parent source/configuration binding.

## Scope and preserved identities

Reviewed the new caller and successor chain against their prior versions. No production files were edited; no Docker, SSH, HTTP, GPU, workload, evaluator, real cleanup, or live process operations were executed. Only standard-library synthetic controls and read-only source inspection were used. The exact 23-file source snapshot is `p0/monitor/review-response-20260927/workload-caller-v37-independent-v1/`, with source root and per-file size/hash in `SNAPSHOT.json`.

| Source under workload-plan/tools | Reviewed SHA-256 |
|---|---|
| `attempt-runtime/sole_executor_v3_7.py` | `88f9519022c51511dd7546f5752f0c272651136f8e63623cd781ff9fcdfa3595` |
| `attempt-runtime/runtime_v3_3.py` | `21b22376b1e6914cfdb84975ce74ec326d795f8a9b33b20a2ee6ac87121a3b39` |
| `attempt-runtime/dependencies_v3_3.py` | `88afc280c4fc90cc3323ed6be47f00a49c5b89c2a23ea1ae5033a1853c749890` |
| `attempt-runtime/pin_observation_successors_v1.py` | `e883b0942f051b0ac070c5b8519916cc8979de02aee2de7e3217b67e36779fe1` |
| `runtime-collectors/collectors_v3.py` | `2c70218d2947b5b3ee212d6eabacfbf2ed187d37f1c86c5f6dbc69b908c7e3ed` |
| `runtime-collectors/integration_v3.py` | `3ff02436b33406cb1af3035cfece3daa9ed311619ca35829cf886318acf5b11f` |
| `runtime-collectors/contracts_v2.py` | `b6c623d1afc641f0ae738d0fea876b9e4f2324dc59d087cb54b41e0a05e44393` |
| `identity-adapter/e3_preflight_v4.py` | `bb8bfceb095ea0742f38825745f3631205d249f1db274c8b89bc873b1ec84b7b` |
| `attempt-closure/closure_v3_3.py` | `de8a98fa8f4069ee319ba52b98ad19b529bba4464913cb7262c256a8f02fa9b4` |

## Source findings

1. **Exactly one caller-owned pre-agent observation, with the correct measurement boundary.** `sole_executor_v3_7.py:499–507` validates its frozen specification, refuses a supplied instrumentation sidecar, and writes the attempt's `STARTED_NO_RETRY` ledger before entering the owned lifecycle. Lines 618–622 prepare the observer before boot. Lines 669–698 call the producer once after healthy boot/proxy setup, retain its result, capture metrics-pre, validate the probe-exclusion boundary, then create the agent and hand the producer's exact observed-runtime path to `Runtime.execute`. The producer's actual request goes directly to the engine; its `proxy_origin` parameter is an exclusion check, not the probe endpoint (`runtime_probe_producer_v1_2.py:425–428,648–651`). Thus the added request does not enter the proxy workload capture; its counters are included in metrics-pre rather than the measured delta.

2. **Refusal consumes the attempt and enters existing owned cleanup.** Preparation/probe/baseline exceptions are caught inside the owned block (`sole_executor_v3_7.py:729–771`). A post-boot refusal before agent start invokes `setup_failed`, exact-CID release and the previously accepted agent-cleanup/closure chain (lines 437–468). The result ledger is retained before ownership ends; the entry check at 478–480 rejects reuse of either existing STARTED or RESULT. No new retry loop was added. This control exercises cleanup calls through fakes; it does not requalify the unchanged cleanup implementation or claim real cleanup occurred.

3. **The exclusion receipt remains a required evidence dependency.** `probe_boundary_v1.py:191–201` stores original post-probe and baseline bytes plus a hash-bound exclusion record in the packet/evidence store. `Runtime.checkpoint` retains the full packet (`runtime_v3_3.py:54–64`), and the admission gate retains its packet before validation (`integration_v3.py:59–75`). `contracts_v2.py:85–115` re-reads the retained bytes, checks exact producer/attempt/CID/runtime/post-metrics bindings, enforces completion ≤ baseline ≤ agent gate, and requires equal completion/prompt/generation counter families. It is called by actual runtime collection, the admission/evaluator gate, and admitted-runtime terminal closure (`collectors_v3.py:299–303`; `integration_v3.py:65–69`; `closure_v3_3.py:482–485`). Unchanged setup-failure closure remains allowed before a probe/runtime was admitted; it does not invent a successful observation.

4. **Observed quantities and controller declarations are separated.** The collector consumes `effective_precision` from the reviewed allocated-record projection and `effective_probe` from the one observed request. It labels workload limits `DECLARED_ENFORCED_CONTROLLER_SETTINGS` instead of claiming the 32-token probe measured compaction/timeouts/workload concurrency (`collectors_v3.py:280–290`). The successor adapter checks these distinct contracts and the independent observation-policy hash (`e3_preflight_v4.py:400–418`). The actual route's observed dispatch is separate from the already-required qualification receipt. Lumo host dispatch does not establish GPU completion, numerical fidelity, task success, or new route qualification.

5. **Coordinator source binding is in the correct namespace.** `Collector.instrumentation` requires the producer hash to be approved for each of six runtime-instrumentation roles (lines 172–196). The coordinator reads/stores its actual producer source and matches the runtime record (lines 239–243), while container source manifests still authenticate container-loaded bytes separately. `contracts_v2.py:60–62` checks the producer in `observed_coordinator_sources` and authenticates its retained bytes. This removes the false requirement that the coordinator's producer file exist inside the engine container, without replacing the container source-map checks. Approved source metadata alone is not accepted as runtime observation: the evidence-reader/bundle checks remain in the actual collector path (lines 207–228).

6. **Successor pins are coherent but intentionally not a final caller freeze.** All 18 listed members in the three new successor manifests matched their recorded bytes and hashes. The manifests explicitly say `DRAFT_SOURCE_REVIEW_ONLY_NOT_EXECUTION_AUTHORITY`. `dependencies_v3_3.py` checks those exact manifest hashes and member bytes before importing the successor collector, integration and closure; inherited seals remain checked. `pin_observation_successors_v1.py` is an authoring helper, not an authorization mechanism, and was not executed. The future parent-owned freeze must also bind the caller/runtime, source approvals, independent worker/request/observation policies, exact configuration and authority; this report does not promote the draft manifests to approval.

## Independent CPU controls

Command (from repository root):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-caller-v37-independent-v1/controls.py
```

**18/18 controls passed.** Six controls execute the exact selected caller functions from the preserved AST with inert fake boundary objects: successful ordering; preparation failure; producer failure; baseline scrape failure; baseline validation failure; external instrumentation refusal. Consumed-failure cases also attempt the same ordinal again and verify refusal without additional work. Eleven controls execute the actual retained-baseline validator and actual metric parser: vLLM and SGLang positives; missing receipt, changed counter, missing token family, altered raw bytes, baseline before completion, baseline after agent gate, foreign runtime, foreign producer, and unapproved boundary source. The eighteenth control authenticates all successor manifest members. No live adapter, scientific module, container, HTTP service or engine was imported or invoked by this test.

The synthetic caller test replaces the runtime, boot, proxy, agent, producer and cleanup boundaries. It proves ordering, exception propagation and refusal mechanics; it is not an end-to-end real-runtime test. The separate helper review and final frozen integration still need to establish the real source/mount, policy, event, allocation and process joins. No new scientific experiment is requested by this review.

## Review artifacts

- `p0/monitor/review-response-20260927/workload-caller-v37-independent-v1/SNAPSHOT.json` — SHA-256 `c336f8bf5702923985d4af3abb0dd0a875834c0b9d42ae5110641e14ef6e42c7`.
- `p0/monitor/review-response-20260927/workload-caller-v37-independent-v1/controls.py` — SHA-256 `26add5ec2441b733f081501d2e37c97a2891fc1312d00b342e5c3f413a36c1f3`.
- `p0/monitor/review-response-20260927/workload-caller-v37-independent-v1/control-results.json` — SHA-256 `2c74d7cee7cbbc56e034973a4269471bd845e8a8b9323d2df9ee90c4878699f9`.

Final source recheck: 23 / 23 snapshot files unchanged.

Source-only verdict applies to the identities above, not to future edits or an unreviewed configuration. No runtime gate was opened.
