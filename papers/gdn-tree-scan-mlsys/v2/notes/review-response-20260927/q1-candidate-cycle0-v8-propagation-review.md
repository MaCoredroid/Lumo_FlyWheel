# Cycle0 v8 source propagation review

**Disposition: bounded source-readiness PASS.** This review found no new integration blocker in the v8 propagation relative to the accepted v7 chain. This is not a launch gate, a result acceptance, a numerical qualification, or evidence of a v8 model execution. Hidden tensor-connection semantics are separately reviewed by `review_response_p0`; this review only checks their executable source propagation and raw record/event/case join. Unchanged H4, convolution and replay internals were not reopened.

## Reviewed integration

- `q1_candidate_job_v8.py:21–38` changes the collector revision to 8, pins hooks/patcher v8 and all four hidden modules, retains the same case builder, and keeps the request seed null. `q1_candidate_cycle0_gate_v6.py` selects this exact builder and revision. The wrapper has 78 distinct dependency keys, including the hidden bridge, live binding, witness and event binder, active raw auditor/authenticator/driver, and inherited required imports.
- `q1_candidate_raw_audit_v7.py:46–52` authenticates each new hidden module against the job. Its `hidden_publication()` joins the bridge owner to the run, observation, request, process, PID and B1 identity, joins the completion counter to the independently checked publication record, uses the exact case nodes, and validates the sealed production census with the real event binder. The call occurs inside `audit_record()` after the existing authenticated KV/convolution/replay checks and before raw logits are reduced. Missing evidence raises and is preserved as a failed authentication by `q1_candidate_cycle0_seal_auth_v2.py`; it cannot become an authenticated driver receipt.
- The authenticator still selects the exact case-bound native A/r0 record under `runs/q1-native-corpus`, authenticates the bound receipt and job, and uses separate candidate/native object stores. The driver successor only changes its authenticator import. Its real loop preserves all 168 expected requests, null request seed, per-case prompt/tokens, raw responses and driver receipts on a partial failure. A raw disagreement remains a diagnostic and does not acquire a qualification label.
- `run_q1_candidate_cycle0_v6.sh:156,165` records the complete probe hash map and derives the third Eagle prehash from that pinned probe. The generated launcher carries it through required-value and SHA syntax checks, container forwarding, the recorded effective diagnostic environment, and `--expect-eagle-sha256`. `q1_patch_candidate_v8.py:114–135` refuses absent/mismatched prehashes for any of the three files before mutation, compiles all outputs before its writes, and records all three pre/post hashes. The final provenance JSON authenticates the complete patch receipt by hash. It does not duplicate Eagle in a separate `expect_eagle_sha256` top-level field, but the complete hash-bound patch receipt and launch binding retain that value; this is not a lost source binding.
- The generator reproduces its saved launcher and diff exactly. Relative to v3.7, the generated shell has only the five patcher selector substitutions and five Eagle propagation additions (three variable loops, one environment record and one patcher CLI argument). Model, numerical criteria, request scope and operational behavior are unchanged by this delta.

## Independent CPU controls

The reproducible script and logs are in `p0/monitor/review-response-20260927/cycle0-v8-propagation-review/`.

1. Four successor gate test methods passed with the accepted predecessor test's native-builder stub, changing only the selected gate/revision.
2. The actual authenticator routed the final fixture case to the correct native corpus record, job and raw store. Eight wrong-stage/case/native-run/receipt/expectation/seal controls refused before the raw auditor (the expensive raw audit was stubbed for this routing test).
3. The actual driver loop completed 168 synthetic authenticated responses. Injecting failure on request 2 retained both responses/receipts, reported one authenticated observation, and preserved `expected_requests=168`. A valid raw-disagreement diagnostic did not fail completeness or become qualification.
4. The actual hidden raw join and real census validator accepted a consistent control and rejected 16 malformed identity, counter, missing-event, census, request-digest and bridge-owner controls. Only the hidden tensor audit was stubbed; this does not substitute for the separately owned tensor/production-connection review.
5. On temporary copies of all three retained generated sources, missing and wrong Eagle prehashes refused without writes. The positive patch compiled 21 anchors, and a duplicate patch refused. No model/Torch/GPU import or execution occurred.
6. Exact generated-shell/diff reproduction, probe-to-retained-Eagle equality, unique wrapper keys and required active import pins passed. All snapshotted source hashes still matched after the controls.

The initial added pin-control assertion expected a literal generator filename inside the dependency array, overlooking its correctly pinned `$DIAG_GENERATOR` variable. That reviewer-control error is preserved as `controls.attempt2.txt`; the corrected assertion checks the variable binding and exact generator rendering. It was not an implementation failure. Parent-reported seven connected hidden tests, gate/RNG tests and target Bash syntax checks were not claimed as independently rerun here.

## Scope of a later parent decision

These source bytes support prospective collection of the unchanged 84 calibration cases × R2 in one explicitly selected A or B process, with the same native per-case references and operational gates. Separate hidden-connection acceptance, final freeze equality, native result acceptance and fresh run-bound readiness/launch authority remain parent responsibilities. No GPU job, container, cache operation, gate, implementation file or qualification counter was changed by this review.

## Source snapshot

Review recorded: `2026-09-29T10:10:50.566189+00:00`. Full immutable copy/hash ledger: `p0/monitor/review-response-20260927/cycle0-v8-propagation-review/SOURCES.json`.

| Source | SHA-256 |
| --- | --- |
| `q1_candidate_hooks_v8.py` | `d6977ec69324c68fd82492da18472430c268808ea062b5608255cd03390638d3` |
| `q1_candidate_job_v8.py` | `f80d2544765552aaf5331dd3afbe468023b9c279cafa235b0001f549e8f83420` |
| `q1_patch_candidate_v8.py` | `73ea046c8fa32eef3af919c3ce22890c26822df1267cb5335fed88ac0e0349d2` |
| `q1_candidate_raw_audit_v7.py` | `d65225715f06d0d80855e1af46e41192df5559a7f44d967266848d0962bc01e4` |
| `q1_candidate_driver_v2_3.py` | `059c223453c4434cba10128abd6bf4cc2a1fa6cdd3e6d3689c5f7f6d244c686e` |
| `q1_candidate_cycle0_seal_auth_v2.py` | `05dcb82296764c3fea4b9e3f1b7cc39fb572d7d20c7dd32ca131e2285271749e` |
| `q1_candidate_cycle0_gate_v6.py` | `0f21f3df5e9e304fbf9ce901972915b70421cbec1d94d840397b6f03b135b049` |
| `run_q1_candidate_cycle0_v6.sh` | `b1a3f2dac1f45240fc0c71b934e1c2dc49233a117d33c5bab4c87bdf1b9ec0c7` |
| `q1_make_diag_launcher_v3_8.py` | `55250ae275abbfd69bd73b68b2add9b04072a4bded7e7283b7efbf9f27f59232` |
| `generated/fr14_leg3_launch_nomiddleware.q1diag.v3_8.sh` | `194539c6382730d3b967445aefc1ce39045b4eb3bb0a8463749da1638fbed717` |
| `q1_hidden_publication_bridge_v1.py` | `00d30bc67e34027b09eae5a0386017ffc7f8766d0a7399218abb5920a15622b8` |
| `q1_hidden_publication_live_v1.py` | `e634c349434f29cbc5d5045fdd2098144a56871a4a66721a1ff85dfd0bab3eae` |
| `q1_hidden_publication_witness_v1.py` | `92ac01369e8e337716183172ec8f64ec266b6601bca21384abef834361f2cd18` |
| `q1_hidden_event_binding_v1.py` | `9ea8e10f4eb290840a908f4b78f38c5f29c0144d0e40dcab5b3750650c1550e4` |
| `identity/launcher-diagnostic.v3_8-from-v3_2.diff` | `c1adf4574fb977806310c023468d1853d995e10eb524fbcf1fc5b014cccbe4f6` |
| `identity/generated_source/probe-20260928T035847Z/MANIFEST.json` | `32262907a36919bdc7f539bc5542ae73a8cb5066558e49bd5a93c23250923d3a` |
