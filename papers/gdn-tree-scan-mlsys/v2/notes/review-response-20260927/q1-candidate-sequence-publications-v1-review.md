# Continuous publication callbacks — independent source/CPU review

Disposition: **PASS for the repaired callback mixin only**, 2026-09-29. One powered failure was found and repaired; no unresolved material finding remains within this bounded delta. This is source preparation, not a runtime or numerical result, qualification, gate, or launch authorization.

## Exact source

All paths below are relative to the v2 paper directory; the JSON seal lists byte hashes and evidence files.

| Source | Reviewed SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/q1_candidate_sequence_publications_v1.py` | `bbdb3d258983f2c11e96cbea09a17116fd6518837baca36b8f56bbd42a22dc79` |
| `experiments/review-response-20260927/tools/tests/test_q1_candidate_sequence_publications_v1.py` | `c76defdc69b8f4a05359f94b0effa699be117bd223de33c932f41d69130c4a96` |
| `experiments/review-response-20260927/tools/q1_candidate_hooks_v9.py` | `905e35898a504357b8a96f2315395e0bec37d5b2509d2ae0a0709b827271317c` |
| `experiments/review-response-20260927/tools/q1_hidden_publication_sequences_v1.py` | `5bb05ae87fd9c778537998f96e668d5d8d3585cbbdfe1373237b526610bf61bb` |
| `experiments/review-response-20260927/tools/q1_candidate_sequence_forcing_v1.py` | `93db3462d89a643a590482c55796eadc9041023896cdde217e31a1c18e108f22` |

The only implementation delta during review is the `@failstop` wrapper at `q1_candidate_sequence_publications_v1.py:98–102`. Initial source `ff0889fd9a63c21bd448d7dc87688df2a6136a9cba16226e7ff9da333675e20a`, all initial snapshots, and `source-repair.diff` remain under `p0/monitor/review-response-20260927/sequence-publication-callbacks-review/`. The remaining four primary source/test hashes are unchanged. `CONTROL-INPUTS.json` additionally binds the byte readers, reused fixtures, and preserved production statements used in CPU controls.

## Closed finding

**F1 — argument evaluation escaped the failure latch.** On the initial source, after a valid active frame, `on_hidden_forward_before(SimpleNamespace(), {}, 32)` raised `AttributeError` while reading `drafter.method` before inherited `_hidden_dispatch` entered its guarded region (`q1_candidate_hooks_v9.py:637–640`). Memory latch remained unset and `PROCESS_UNUSABLE` did not exist. This is reproduced in `PROBE.initial.json` and `probe_initial.py`; it is not a model failure. The repaired wrapper encloses those inherited argument reads. Both a foreign/malformed drafter and a raising `model_returns_tuple()` now raise `ProcessUnusable`, retain the failure latch, and prevent further active work. The inherited copy/gather and MTP entrypoints already place their operand collection in guarded regions; common activation itself is also guarded. No accepted Basev9 source was edited.

## Connected evidence and scope

`CONTROLS.final.repaired.json` and `.log`: **22/22 local CPU controls pass** (29.585 seconds). This includes the four original connected tests and 18 added controls, not 22 experimental observations. The first 16-control run and original powered failure remain preserved.

- `PublicationCallbacks:33–56` binds the actual current Sequence/frame, request/observation/run/process, path/root extent, event census baseline, and target-forward index. Parent-frame drift refuses and latches. Disabled collection and the deliberate interval after `Sequence.seal()` are inactive. A callback after the actual deferred event but before outer Sequence sealing refuses.
- `:70–90` invokes the unchanged actual target/conv/replay AFTER collectors and registers the **same completed record object only after** their audits succeed. The independent connected control executes all three interior cycles and the terminal cycle with real target-remap callbacks, real conv capture callbacks, and unchanged production replay-staging statements; neural recurrence is mocked. Corrupt convolution/replay bytes refuse before registration; duplicate AFTER callbacks refuse.
- `:110–123` binds the new hidden owner to the actual runner and case. CPU controls surround the preserved production hidden copy and sampling/continuation gathers with the actual callbacks. The synthetic neural output is explicitly `hidden + 2`; it is not evidence of MTP numerical fidelity.
- Inherited `q1_candidate_hooks_v9.py:663–725` is exercised through **actual** MTP KV before/after callbacks and real CPU cache captures, not only an injected completed receipt. The control supplies the exact metadata/query/payload/registry identities, a sufficient physical 256-row block map, and disjoint target/MTP storage. A synthetic simultaneous MTP KV copy is checked byte-for-byte. Foreign drafter, changed device block table, wrong slot-buffer identity, rebound payload, modified untouched cache row, and duplicate MTP callbacks refuse and latch.
- `PublicationCallbacks:125–148` requires completed hidden/census evidence plus a completed MTP record with no live in-flight tuple, then reruns the MTP byte audit against the actual hidden owner, accepted nodes, and prefix. Missing MTP evidence, a stale forward owner, and a **newly hash-sealed corrupt raw MTP after-image** refuse at this final boundary; no successful publication receipt is created. Duplicate seal refuses. The successful receipt is copied and marked `qualification=False`; it neither collects O1 nor advances Sequence by itself.

The canonical Sequence registration and hidden-owner final checks remain the previously accepted implementations (`q1_candidate_sequence_forcing_v1.py:77–114`; `q1_hidden_publication_sequences_v1.py:71–148`). Their unchanged oracles were reused, not independently requalified here.

## Reproduction and limits

From the repository root:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python -B papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/sequence-publication-callbacks-review/independent_controls.py repaired
```

Local Python 3.9 required two disclosed fixture-scoped compatibility equivalents: `zip(strict=True)` for unchanged topology loading and `int.bit_count()` for the synthetic census value. No production source or global Python builtins were changed. Conv/MTP byte writes and neural outputs are synthetic; target remap, replay staging, hidden copies/gathers, all relevant raw captures/readers, callback ownership, and failure latching execute on CPU. No SSH, GPU, container, model, active collector, or campaign state was touched.

The future outer collector still must provide the per-frame case fields, actual pre-forward/import/O0/O1/O2/state and graph integration, callback installation, and outer seal order. This mixin alone cannot establish those obligations, numerical equivalence, MTP categorical qualification, or a launch gate. No additional broad test campaign is required to close the reviewed callback delta.
