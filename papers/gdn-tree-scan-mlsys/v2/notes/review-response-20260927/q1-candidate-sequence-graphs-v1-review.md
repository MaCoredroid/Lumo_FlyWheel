# Continuous graph/head callbacks — independent source/CPU review

**Disposition: PASS for the repaired prospective graph/head mixin only**, 2026-09-29. Three powered source-integrity/ownership findings are closed. No unresolved material finding remains in this bounded callback delta. This is not a model/numerical result, CUDA replay qualification, launch admission, or full-Q1 acceptance.

## Source binding

Paths are relative to the v2 directory. Primary sources:

| Source | Reviewed SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/q1_candidate_sequence_graphs_v1.py` | `04e7d11e8da4cf85797be226a06f5cd9451586fe3e2951c1daa4c4742a837258` |
| `experiments/review-response-20260927/tools/tests/test_q1_candidate_sequence_graphs_v1.py` | `ed16731ab183fb53621f7ccdc0c64d1cee84640d44a59d5053d2208cbdcf93d2` |
| `experiments/review-response-20260927/tools/q1_candidate_sequence_publications_v1.py` | `bbdb3d258983f2c11e96cbea09a17116fd6518837baca36b8f56bbd42a22dc79` |
| `experiments/review-response-20260927/tools/q1_candidate_hooks_joint_v1.py` | `fcd4f3ee3012fb45c7eaf8f4ee6477b212e482fca472a8ee1387755c7baf1526` |
| `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py` | `4fd8b2fef029e0b702406662db0afcb3b575a4376760331c2e8a2c2faf79e181` |

`FINAL-SOURCES.json` additionally binds the accepted Sequence/publication/hidden dependencies and reused CPU fixtures. The publication mixin stays at its accepted `bbdb3d25…` bytes; Basev9, Joint, and GraphWitness are unchanged. The initial graph source `14ee7e735906e64a7e0a414f6370234dcb5c1d88bfccc08a027daa690daa839d`, initial probes/results, reviewed test bytes, repaired source and exact diff are preserved under `p0/monitor/review-response-20260927/sequence-graph-callbacks-review/`.

## Findings and closure

1. **Foreign graph producer accepted.** On initial source, after the actual hidden/root-selector callbacks, preparing/capturing/beginning/exporting a witness under a distinct drafter object and restoring the real drafter before sealing still succeeded without a process latch. `PROBES.initial.json` preserves this connected counterexample. Repaired `on_joint_graph_begin`/`export` (`q1_candidate_sequence_graphs_v1.py:164–177`) require callback drafter identity to equal both the current hidden binding's drafter and the Sequence runner's drafter. The same powered control now refuses and latches.
2. **Retained phase fields could change after export.** Initial final seal accepted a changed captured root spine or input position. The initial probes are retained. `_sequence_graph_frame` (`:53–63`) now compares exact retained keys and independent canonical digests for forward, phase and graph records, in addition to the original target-root digest and live object identities. Digests are recorded as each actual callback completes. Modified spine, position, graph epoch/IDs, forward token, missing/foreign phase, replaced joint storage and changed target reference now refuse before successful seal.
3. **Raw object corruption escaped final sealing.** Flipping a byte of the content-addressed target-root head or graph head after capture, without changing its SHA/path, initially still sealed. `PROBES.raw.initial.json` preserves both failures. `_authenticate_graph_captures` (`:200–221`) now re-reads the retained HW tensor captures and checks supported dtype, positive shape, shape-derived byte count, available byte-count reference fields, actual length and SHA-256. Final seal invokes it on both joint and forward evidence (`:237–238`). Missing or corrupt target/graph objects now refuse and latch. Metadata digests and byte authentication are separate checks.

## Independent controls

`CONTROLS.repaired.json` and `.log`: **35/35 local CPU controls pass, 35.731 seconds**. This includes four parent connected controls and 31 added controls; these are software tests, not experimental observations.

The positive path traverses three interior cycles and the terminal cycle using the actual callbacks, preserved production hidden copy/gather statements, and the accepted GraphWitness. It retains one witness across frames, preserves the existing global census from 2 to 6, and observes cumulative epochs `[2,2,2,2]` through `[5,5,5,5]`. Interior records contain root and first follow-up phases; terminal records contain only the root after the pending token. A separate control changes the terminal target-head greedy ID to 89 while the fixture still supplies 7: the actual forward callback refuses, demonstrating that terminal input checks use the newly captured head rather than the fixture's constant.

Other controls reject wrong target-head indices, BF16 dtype, finite-root condition, runner or computed extent; duplicate head and seal; missing/repeated replay epochs; wrong first-follow token or hidden; wrong logical position, request slot or active padded slot; and changed selector output. Foreign-frame and incomplete-graph failures also remain covered. A returned prior-cycle receipt can be modified without rebinding the next live frame, confirming the deep-copy boundary.

Failures latch `ProcessUnusable`, preserve the marker, and prevent further guarded work. The five initial accepted-invalid probes remain preserved; repaired controls require refusal. No implementation file was edited by this reviewer.

## Scope and remaining connection obligations

The parent fixture deliberately stubs MTP KV snapshots and supplies synthetic model/selector values. Later replay is simulated by CPU copies into the actual GraphWitness diagnostic buffers; this is **not actual CUDA graph execution**. The accepted live MTP/publication byte callbacks were separately reviewed in `q1-candidate-sequence-publications-v1-review.md` and were not requalified here. This component records a full root vocabulary row and actual selector outputs; it does not prove categorical agreement with native MTP or invent numerical tolerances.

A future outer collector must initialize and retain the graph maps once, install these callbacks around the real producer/capture/replay sites (including initial graph capture), supply the authenticated per-frame case and live inputs, and connect pre-forward/import/state/O1/O2/serialization and final Sequence sealing. The mixin does not perform those actions or grant their admission. No new experiment is needed to close this source/CPU delta.

Reproduction from repository root:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python -B papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/sequence-graph-callbacks-review/independent_controls.py repaired
```

The fixture explicitly sets `_cpu_test=True`; the production CUDA head guard remains. Local Python 3.9 uses the previously disclosed fixture-scoped `zip(strict=True)` and synthetic census `bit_count` equivalents. Production sources and global builtins are unchanged. No SSH, GPU, container, model, launch, gate, or running native-B state was touched.
