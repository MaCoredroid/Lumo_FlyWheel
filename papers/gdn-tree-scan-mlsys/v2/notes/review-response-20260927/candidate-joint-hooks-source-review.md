# Joint candidate hook source review

Reviewed 2026-09-29T16:21:56.448856+00:00. **Bounded source/CPU PASS at the hashes below.** The initial two blocking defects are reproduced and closed. This is preparation review, not candidate CUDA execution, numerical qualification, a final source freeze, or launch admission.

## Preserved initial findings and closure

1. **F1 — root selector read a nonexistent observer key.** Initial hooks SHA `5c780e101ee50f682f9787391d06d7e41fe9920f1a66a2d0474b33d706e13e62`, lines169–170, used `sampling['selected']` and `continuation['selected']`. The real bridge emits `selected_after` (`q1_hidden_publication_bridge_v1.py:133–137`). An independent real-CPU bridge positive reaches exactly `KeyError('selected')` in the initial hook. Current hooks lines190–191 read `selected_after` and check the actual advanced-index gather objects with `HB.same_entry`; equal-byte clones are rejected. **Closed.**

2. **F2 — new callback failures could escape without poisoning the process.** Initial second-event `on_hidden_runner`, `on_sealed`, and third-TAW failures propagated while `unusable` remained empty. The runner hook precedes the `drafter.propose` containment context (`q1_patch_candidate_v9.py:73,115`), and the deferred seal is outside it, so the inherited context did not protect these boundaries. The independent controls reproduce all three unlatched failures. Current `failstop` at lines26–34 decorates the new external callbacks and mutation-bearing bind/hydrate methods; the same failures now raise `ProcessUnusable`, latch in memory, create the owned marker, and make `_guard` refuse reuse. It reuses accepted v9's in-memory-first, best-effort evidence path. **Closed.**

Initial and repaired sources remain under `p0/monitor/review-response-20260927/candidate-joint-hooks-independent/{initial,repair1,recording-delta}/`; neither attempted source was replaced. The late recording-only delta `e6cee814…` → current `7f3c5c93…` copies the four pre-replay epochs before export clears `pending`, then retains actual graph IDs, segment pass counts, executed segment count, and bound graph-buffer signatures. No numerical, model-input, or replay control-flow change was introduced. The independent positive checks `[0,0,0,0]` → `[1,1,1,1]`, one executed four-pass segment, and retained signature/ID fields. **Closed.**

## Source seams checked

- The common source mapper and authenticated source seal are connected at hooks50–91 before target/MTP import. Existing accepted mapper, transaction, raw-source auditor, and v9 behavior are reused; this review does not requalify them.
- Actual selected root input is checked at hooks147–162: shifted preembedding IDs, the actual input/embedding buffer identity, and every position axis must consume Z in the first event or actual candidate O2 in the second. Root sampling/continuation tensors remain distinct actual objects. The prepared Eagle calls the root observer immediately after the real fused selector (`eagle.graph-witness.padded.prepared.py:4790`), supplying the actual head and returned spine/top3, not a surrogate argmax.
- Prepared Eagle5158 passes the full physical input prototypes. Its capture call5279 passes the same full slot-buffer view used by the model context and the real preembedding token variable. Capture-after5342 passes separate sampling and continuation results. Existing and first replay branches are bracketed at5038/5074 and5504/5546. The accepted graph witness enforces all four epochs and preserves physical padding; this review makes no CUDA-replay claim.
- `_mtp_snapshot` uses the unique actual MTP cache group and physical256 map (hooks66–79). The pinned runner chooses speculative `common_attn_metadata` specifically from `drafter.kv_cache_gid` (`gpu_model_runner.py:3710–3730`), consistent with checking that map at the root and graph export. The first-follow snapshot is taken after the four loops; accepted graph evidence bounds later writes to later positions. Raw reader validation remains separately required.
- Second terminal forcing uses actual finite O2 and does not seal at the TAW callback (hooks252–265). The real deferred seal requires all three categorical phases, both graph receipts, and accepted `AfterZ.audit_seal()` (269–277). Qualification flags remain false.

## CPU evidence and exact limits

The reviewer executed **13/13 controls** locally, with Torch2.8.0 on CPU and no engine/model import. These compile exact hook ASTs and exact v9 lifecycle methods, then use the real HB, AfterZ, and graph-witness modules. Controls preserve the original schema/latch counterexamples; exercise the repaired root → physically padded graph → second seal; and refuse wrong O2, wrong positions, rebound input storage, and cloned sampling/continuation tensors. The second-seal positive uses explicit earlier-phase placeholders and a synthetic KV provider/census; it is not proof of the whole first-event production path or live KV publication. Parent's independently retained `attempt4.log` reports nine connected CPU controls, including the first actual selected-leaf Z/follow path; this is read as author evidence, not represented as independently rerun here.

Direct import of the whole current module on local Python3.9 encounters unchanged production `zip(strict=True)` during topology import. That is a reviewer-platform limitation, not a candidate defect; the exact-AST controls avoid changing production code. Parent's full connected tests ran on Python3.12. No SSH, GPU, engine/model, container, gate, production write, or remote synchronization was performed for this review.

Command, from repository root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -W ignore::ResourceWarning papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/candidate-joint-hooks-independent/recording-delta/review_controls.py
```

Remaining boundaries are the explicitly unfinished connected source/patch/job freeze, independent raw reader, and actual runtime/graph qualification. No new blocker was found in the bounded repaired source examined here; this disposition does not waive those boundaries.

## Exact identities

| Item | Paper-relative path | SHA256 |
| --- | --- | --- |
| current hooks | `experiments/review-response-20260927/tools/q1_candidate_hooks_joint_v1.py` | `7f3c5c93211b43d1aa08e10bbe015413c738510da591ad2964b81ea6206e3a86` |
| parent connected tests | `experiments/review-response-20260927/tools/tests/test_q1_candidate_hooks_joint_v1.py` | `52fdd713ec152df40164928ac3af4ee5da6dbd57f03a8289d98bcd61927bfac8` |
| accepted base v9 | `experiments/review-response-20260927/tools/q1_candidate_hooks_v9.py` | `905e35898a504357b8a96f2315395e0bec37d5b2509d2ae0a0709b827271317c` |
| AfterZ binding | `experiments/review-response-20260927/tools/q1_candidate_after_z_binding_v1.py` | `a6850423e1318f88706f1b5a52fc343fa89ebc1d21fb2b5727c2dc457eda9b53` |
| graph observer | `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py` | `4fd8b2fef029e0b702406662db0afcb3b575a4376760331c2e8a2c2faf79e181` |
| graph patcher | `experiments/review-response-20260927/tools/q1_candidate_graph_patcher_v1.py` | `f3b23b57d25f61a664211aa125ef0ef7a72f2af6d0012c2818fbe86f56fb273d` |
| prepared padded Eagle | `p0/monitor/review-response-20260927/candidate-graph-observer-cpu/eagle.graph-witness.padded.prepared.py` | `9e968c2124affa420d54f4ade3e84eef1979af77989b494cffb9c692603e01fb` |
| independent reproducer | `p0/monitor/review-response-20260927/candidate-joint-hooks-independent/recording-delta/review_controls.py` | `9f14172f6651726fd03e50576d6872cf53a7cef3b6da5ece10679cea552d73ca` |
| independent result | `p0/monitor/review-response-20260927/candidate-joint-hooks-independent/recording-delta/AUDIT.json` | `4277bc9fe5b31738d9ecaf52a00b483a4964edd53f5081396ac851b5ff66c562` |
| parent nine-control log | `p0/monitor/review-response-20260927/candidate-joint-hooks-cpu/attempt4.log` | `2b0fde2f1f958eac36e345deda366f67783ca1028bff957b82d9651a7e01287e` |
