# Candidate graph observer: bounded repair closure

Date: 2026-09-29. Verdict: **PASS for the repaired CPU tensor/source seam; no remaining blocker in F1/F2.** This is not CUDA replay qualification, a connected-caller review, or launch approval. The original findings and original source snapshot remain unchanged in `candidate-graph-observer-source-review.md` and `p0/monitor/review-response-20260927/candidate-graph-observer-independent/`.

## Reviewed identities

Paths below are relative to paper `v2/`.

| File | SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py` | `323938955da3c528a0f45710fd22328e1bf5dd50ec0632eb0c1d86c97c8b273c` |
| `experiments/review-response-20260927/tools/tests/test_q1_candidate_graph_observer_v1.py` | `bde639a1c0525f0a686d7aa85061d10c5a3e39d38fd05d6ccc0efe8f0517f827` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-independent/repair1/SOURCE.json` | `b9ae825d9d1e91a8cad1ab087ae9b15fdc85ddb4751f582c87736520d2ba28e2` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-independent/repair1/supplied-tests.txt` | `d3f038c4b5bda2b6ae4d89100b551f5a8143e7b36d15d90c4344b23930ed1cff` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-independent/repair1/ADVERSARIAL.json` | `d5144b2fbe5a47fd4402230ffc2010b6db03ae8d24faf99db5f532dbdb899c8a` |

The helper and test were copied before execution. Canonical bytes were checked again at report completion and still match these identities.

## Closure evidence

**F1 — actual embedding and position surfaces: closed.** Lines 38–46 allocate diagnostic position storage with the original graph position shape and optionally allocate an embedding home from the supplied actual prototype. Lines 59–73 distinguish the actual `input_ids=None`/`inputs_embeds` route, require explicit preembedding token IDs, and copy the actual model kwargs. Export preserves the original position shape and checks every axis (106–107), rather than accepting only the first axis. This covers the previously identified source-permitted embedding and three-axis route without changing production flags or replacing embedding computation.

**F2 — stale later-loop snapshots: closed.** The epoch is now a four-element tensor (42); each captured after-hook increments its own entry (87). `begin_replay` records all four prior entries (94), and export requires every entry to advance by exactly one (101). The prior scalar/loop-0 counterexample is rejected. Diagnostic storage signatures (48–53), deep-copied owner values (94, 115), and capture failure poisoning (22–29) also pass the supplied controls.

Local CPU execution used `/usr/bin/python3`, Torch 2.8.0, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, and one Torch thread. All **18 supplied controls passed** against the frozen copy. All **10 independent controls passed**: partial epoch advancement of 0, 1, 2, or 3 of four loops refused; all four single advances accepted; one double advance refused; actual embedding plus three position axes preserved; missing prepared token IDs refused; nonfinite embeddings refused; and corruption confined to a nonfirst position axis refused. The independent epoch mutations simulate captured-operation effects on CPU; they are not executions of a CUDA graph.

## Boundaries retained

The future caller must still create these buffers before the first actual graph capture (including bootstrap), supply the actual embedding prototype/preembedding IDs/model kwargs and returned head/selector tensors, and bind the real graph, request, event, owner, and cache registry. It must place the hooks around all four actual calls and bracket the genuine replay with `begin_replay`/`export` outside capture. The helper cannot establish live provenance for arbitrary caller-supplied tensors. The separately required categorical comparison, source freeze, connected seam tests, and actual runtime qualification remain pending. No GPU operation, engine/model import, flag change, source edit, or gate action was performed in this review.
