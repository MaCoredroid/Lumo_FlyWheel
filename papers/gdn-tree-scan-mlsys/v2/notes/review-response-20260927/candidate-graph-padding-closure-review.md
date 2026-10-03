# Candidate graph input padding: bounded closure

2026-09-29. **PASS for the source/CPU repair.** The physical-input compatibility gap in `candidate-graph-patcher-source-review.md` is closed without assuming that logical B1 implies physical N=1. No remaining critical defect was found in this changed seam. This does not qualify CUDA replay or the unconnected callbacks, and does not authorize a launch.

## Source identities

Paths are relative to paper `v2/`.

| File | SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py` | `4fd8b2fef029e0b702406662db0afcb3b575a4376760331c2e8a2c2faf79e181` |
| `experiments/review-response-20260927/tools/q1_candidate_graph_patcher_v1.py` | `f3b23b57d25f61a664211aa125ef0ef7a72f2af6d0012c2818fbe86f56fb273d` |
| `experiments/review-response-20260927/tools/tests/test_q1_candidate_graph_observer_v1.py` | `5f62f6cc8f3897efd1dad46ad0f89024c28a25f521930c2cfafbb3a8a78aef35` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-cpu/PATCH-PREPARATION-PADDED.json` | `1659056a9017a85ad1628bd982bba0465060133a4ba62bfa2a1663e83105a1e9` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-cpu/eagle.graph-witness.padded.prepared.py` | `9e968c2124affa420d54f4ade3e84eef1979af77989b494cffb9c692603e01fb` |

Canonical helper, patcher and test hashes still matched the frozen review copies at completion. Prior snapshots and findings are preserved.

## Why the repair closes the seam

Observer lines 43–64 now derive physical N from the actual model-input hidden prototype, require corresponding complete ID/position/slot prototypes, and retain their storage signatures. Full padded hidden, positions and embeddings are copied; logical graph outputs and sequence lengths remain one-row surfaces. The embedding route correctly retains the one active preembedding token ID because the real embedding call consumes that one token, while preserving the full N-row embedding tensor actually passed to the model. The ID route retains the complete N-row input IDs. Capture lines 91–95 reject substituted storage rather than accepting a shape-compatible replacement.

Patcher line 21 takes prototypes from the same full `self` buffers used by pinned Eagle's model kwargs, before graph capture. Line 26 observes `_slot_mapping_buffer[:input_batch_size]` inside the forward context. In pinned Eagle (SHA `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62`), `_get_slot_mapping` lines 538–555 returns that same view when called without its optional copy argument, as the loop forward context does at 5261. The new hook therefore captures the actual padded slot storage, rather than the separately truncated logical `common_attn_metadata.slot_mapping`. No new model computation or slot mutation is inserted.

Export lines 132–138 apply logical position/axis/map checks only to active row zero, require every padded slot to be -1, retain the other position values instead of pretending they are logical positions, and check finiteness across the full captured first input hidden/embedding tensors. The result explicitly distinguishes `physical_input_rows=N` and `logical_active_rows=1` (142). The previous complete-loop epoch, distinct sampling/continuation, owner, and failure-poison checks remain intact.

## Independent checks

All **24 supplied CPU controls** passed against frozen copies. **13 independent controls** passed: ID and embedding routes at physical N=2, 32 and 64; cloned position storage; cloned embedding storage; truncated hidden input; a padded cache-write slot; corruption of an active nonfirst position axis; a nonfinite padded embedding row; and incomplete per-loop progress. Positives use overallocated 80-column homes with three-axis position views of stride `(80,1)`, exercising actual view/stride preservation. Only the two pure `_get_positions` and `_get_slot_mapping` methods were AST-extracted from pinned Eagle; no Eagle module, engine or model was imported. Tests independently confirm the context's slot view and the observer's slot view have identical storage signatures.

The nine-anchor preparation was reproduced byte-for-byte from unchanged v9. Compilation succeeds, and stripping only the added observer statements and preembedding alias gives an AST identical to the v9 scientific source. Prepared output SHA is the value above; no settings or scientific call changes were made.

Reproducer and evidence are under `p0/monitor/review-response-20260927/candidate-graph-patcher-independent/padding-repair/`:

- `SOURCE.json`: `f98d811842364ddb063d96f145fab09d5964ac4cd31da64d3e00ca9fc12c5dfc`
- `review_controls.py`: `03e16dab3f28fb67f14a74027a2013462034e4f7c3ac60873a0207cfef52dbf1`
- `supplied-tests.txt`: `beafc68b4a4b3171de12e5dc4fa6ba81effdd9e8c707618a2d76f8421b441a75`
- `AUDIT.json`: `2e8219cea4f52e355dcb4f2e317e27c387793163762658d3319f095ee6e494a9`

Command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -W ignore::ResourceWarning <review-directory>/review_controls.py`; Torch 2.8.0, one CPU thread.

The tested N values are synthetic tensor extents, not measured dispatcher outcomes. The actual caller must pass the new complete prototypes, connect hooks before the first real capture, bind graph/owner/request/registry and exact source identity, and preserve the full actual kwargs. Runtime embedding provenance, categorical reduction, source freeze, and actual CUDA execution remain separately pending. No GPU, remote, source-setting, gate, or production action was taken.
