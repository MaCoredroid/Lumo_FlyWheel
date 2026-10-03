# M1-Q C0 reducer — independent prospective source review

Date: 2026-09-28. **Bounded source/control-flow closure PASS** for `tools/m1/m1_q_c0_reduce_v1.py` SHA256 `4e2aa7dab16e385ddee610d85b69ef9e23f0ce0d0cdf51fc3538578117f736de`. The one material finding below is repaired. This verdict is preparation only: it neither qualifies a candidate nor opens a GPU, reduction, timing, full-model, or workload gate. No candidate output was available or examined. No M1 held-out inputs were generated; no Torch, author kernel, GPU, container, remote process, or actual reduction was executed.

Paths below are relative to `papers/gdn-tree-scan-mlsys/v2`; `c/` denotes `experiments/review-response-20260927/`.

## Finding and exact repair

**F1, CLOSED: complete numerical cycles could override a failed collection finalization.** The initial reducer SHA256 `b7f97c2a95736f59ef5cd742692a8180e5deb6149b433103395970c471f52a09` checked owned-container absence, authenticated the collection receipt, and classified each complete cycle, but did not consult either terminal status or `stage_failures`. This is a reachable producer path: `m1_q_c0_collect_v1.py:197–212` can complete all eight cycles and then fail mandatory memory-observation finalization, recording `FAILED_PRESERVED`; the owned launcher also preserves failure while successfully removing its exact container. The original reducer could nevertheless issue overall PASS.

An independent full-`main()` synthetic control reproduced that combination: both launch/collection statuses `FAILED_PRESERVED`, nonempty `stage_failures=[{"stage":"memory_finalization"}]`, all eight synthetic numerical cycles passing, and erroneous overall PASS with 571,392 repeated observations. The original bytes and reproducer are preserved, not overwritten.

The repaired reducer checks both terminal statuses, an empty collection failure list, mandatory memory-observation presence/content binding, and tensor-index finalization (`:119–136`). It records `collection_admissible` and reasons. Otherwise passing numerical surfaces/cycles become UNCOVERED when collection is not admissible (`:185–201`); actual numerical FAIL remains FAIL. Thus diagnostics remain inspectable without promoting failed collection evidence to qualification. Separate controls for failed launcher, failed collector, nonempty failure list, and missing mandatory memory document all now produce UNCOVERED. Nonfinite/negative sealed bounds are also rejected as malformed (`:52–54`), without changing the prospective numerical rule.

## Checked numerical, population, and raw-schema boundaries

- The original CPU baseline/source and function bindings are checked before scoring (`:70–85`). The authorization binds the reference pack, eligibility seal and immutable launch receipt; the actual C0 gate must contain this exact reducer SHA before C0 (`:87–118`). This review assumes the stated parent-owned authorization trust boundary, not an attacker-created replacement approval.
- The ordered pack domain is exactly seed 20260929, layers 0–47 (`:108`). The fixed method/repeat domain is four named methods, two repeats each; missing cycles remain missing, duplicates/extras/reordering/path mismatch refuse (`:20–30`). No averaging, favorable repeat, or subset estimate is used (`:194–201`).
- `m1_stage_collector_v3.py:190–235` emits physical-order first outputs and indexed state captures. The reducer correctly loads BF16 output `[32,48,128]`, then selects frozen `B.ACTIVE` (28 nodes), and FP32 captures `[48,48,128,128]` at indices 0/1/2/3 (`:152–165`). Capture 0 must equal each pristine S0; publications 1–3 are scored against the corresponding uninterrupted-history C2 references. The collector's frozen runtime receipt checks already validate publication/final-flush execution before setting a cycle `complete`; this numerical scorer reuses that source-bound producer contract rather than inventing a second execution protocol.
- Per-repeat metric arrays are `[48,28,48]` output cells (64,512) and `[48,3,48]` state cells (6,912): **71,424 unique cells per method**, 285,696 across four methods. Two repetitions yield 571,392 expected observations, not additional unique cells. Recorded nonfinite stops produce FAIL and zero invented evaluated cells; missing cycles produce UNCOVERED (`:142–146`).
- Raw candidate references enforce geometry, dtype, content-addressed filename, index geometry/dtype, byte counts and actual SHA256 (`:33–46`). The candidate metadata, tensor index, pack operands/references, C2 tensor hash and sealed bound arrays are authenticated (`:110–118,134–136,149–179`). The score uses the unchanged baseline `cell_metrics` FP64 per-cell RMS/max reduction, not the collector's generic in-image diagnostic C2 results. Both inequalities must hold independently, with finite candidate/metric requirements (`:49–61`).
- The reference-seal producer requires finite repeated references and structural witness power, creates an all-eligible mask only for a resolved comparator, and hash-binds saved bound arrays. The scorer does not fit constants or produce a new reference. Runtime/scientific qualification remains contingent on the exact frozen source/pack/seal/gate and actual retained output evidence.

## Independent CPU checks and preserved identities

The reviewer harness uses NumPy-backed small mask operations and shape-only tensor sentinels. Its full-main tests mock numerical backends/runtime-version guards to reach the actual JSON/hash/domain/status/repeat control flow; those tests are **not** a real Torch runtime or numerical equivalence test. A separate tiny FP32 raw-byte test exercises the real `load_tensor` authentication path. No scientific fixture generator is imported.

Original snapshot: `p0/monitor/review-response-20260927/m1-q-c0-reducer-independent-snapshot-20260928/` — 24 checks, including the preserved F1 reproduction.

Repair snapshot: `p0/monitor/review-response-20260927/m1-q-c0-reducer-independent-snapshot-20260928-repair1/` — **35 checks passed**, covering equality at both limits, RMS-only/max-only violation, nonfinite metrics/candidate flags, ineligible references, malformed bounds/shapes, exact/missing/duplicate/extra/reordered cycle sets, incorrect paths, raw tensor shape/dtype/path/size/content alterations, full nominal population, missing repeat, recorded nonfinite stop, the four F1 variations, and a numerical failure confined to the first, second, or final repeat. Every single-repeat failure remains FAIL.

Reproduce from either snapshot with:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 independent_controls.py
```

| Reviewed dependency | SHA256 |
|---|---|
| `c/tools/m1/m1_q_c0_collect_v1.py` | `2bb6103e88d84b51c6e022c5c9ad424e151c94c85f8d3e34a70bd94028a37b92` |
| `c/tools/m1/run_m1_q_c0_owned_v1.py` | `34de4761a587ac7bd1cd3ca0e58a26458e5267269c5239d28cb0bdf3059e68b7` |
| `c/tools/m1/m1_q_reference_seal_v1.py` | `1d2009774c385c8a559ae7d975806125071ad7b626f8f15d07175ba4950a4692` |
| `c/tools/m1/m1_c_baseline_v1_1.py` | `00d1c8e5a593f2f1377e43aef2002f446a6ffb7c04ef0689b38cdbe163fd721b` |
| `c/tools/m1/m1_stage_collector_v3_5.py` | `0dc680fa0a9ec81496926b177144ff7211e38fbeb63c37a14b6805384db37f3c` |
| `c/tools/m1/m1_native_reference_pack_v1.py` | `098de87f5b8c1e9ac61be2b02746a34897bc7d7f8da6e64703c8b3ef2a40e257` |
| Original independent controls | `6ec944c5839f62ad7c55788f1a9cc4632e3eacbfc35b47fd4f457ec69b3d35e5` |
| Original controls result | `1fc7aa01f7d6f07160412090d0a8d1371bfddfb4c3165e5e70a2a5c939860339` |
| Repaired independent controls | `c03da58560c920b18e17f6ac0ce121027a94c57df7ade1f19ae4700d0dc99095` |
| Repaired controls result | `8d68b667aa0b2f94bb434d2760d5f724089520cdc94666fe9c432ffd4f30a5ef` |

Each snapshot also contains a full source-path/hash/size `SNAPSHOT.json`. The exact repaired source must be frozen in the pre-C0 gate; final source equality and any launch authority remain parent decisions. No remaining material defect found within this bounded scorer/collector-schema review.
