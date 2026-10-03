# M1 v3.1 bounded runtime-closure review

**Disposition: HOLD for one remaining failure-lifecycle correction.** The delivered-v3 path resolver, protected executor construction, empty-index guard, exact reset premise, and raw-evidence completeness repairs close the prior findings within this untimed diagnostic scope. A late mandatory-evidence serialization failure can still leave a method marked `complete` and the stage exiting zero. This is a preparation defect, not a failed GPU experiment or a new numerical acceptance criterion.

## Reviewed identity and boundary

Snapshot: `p0/monitor/review-response-20260927/m1-v31-reviewed-20260928T0514Z`, payloads under its `repo/` tree.

| Artifact | SHA256 |
| --- | --- |
| `FREEZE-M1-ADAPTERS-v3.1.json` | `17bd8f2acb191392ac0dda5b613b996c18f8d4e3ecd68ab25641016d166ffffe` |
| `tools/m1/m1_stage_collector_v3_1.py` | `52d6a2e52fb4794e679c6aa66e55fceef50c218f41cc8f6c4ce5cedc4f7e1f63` |
| `tools/m1/m1_cycle_driver_v3_1.py` | `aee4b81155c75b1bac52b9cadc65bf5c5c826ace5beff79c0e458d06f7796854` |
| `tools/tests/test_m1_v3_1.py` | `91a4222687dfe4a991c881bcbbb218999c507c1b052cb1da249468293f1a23a9` |
| `tools/test_log.m1_v3_1.attempt1.txt` | `1611b5f06cecf99fc24b45915f35027b30ca7ba5f157ebbbb6480881901ff28d` |

Independently hashed all 36 snapshot payloads against `PARENT-SNAPSHOT.json`: 36 matched, zero mismatches. The log reports 87 passing CPU tests; that full suite was not independently rerun. This review executed only selected actual function ASTs with stdlib/in-memory dependencies. No Torch import, CUDA, GPU, SSH, Docker, model, collector launch, production-source edit, or gate change occurred. The parent separately reviews the host launcher.

## Remaining blocker F1: late evidence failures preserve a successful disposition

Source: `tools/m1/m1_stage_collector_v3_1.py:194–211`, with stage disposition at `:260–268`.

The collector promotes `cycle_complete` to `complete` at lines 194–197, before saving and hashing `receipt.json` at 207–208 and saving `status.json` at 210. `safe()` records later exceptions in `secondary_failures` but never demotes the successful status. Consequently `run_stage` considers the method successful at line 261. Its own finalization exception handler at 266–267 similarly appends a failure after choosing the exit code without changing that code.

Independent reproduction used the **actual `_exc` and `run_method` ASTs** from the pinned collector. Dependencies supplied a completed runtime cycle and an accepted completeness/receipt result; file writes used `io.StringIO`, with a targeted `OSError` for the named final file. No real files or tensors were created. These stubs isolate lifecycle control flow; they do not represent executed kernels or establish numerical correctness.

| Control | Returned method status | Retained failure | Stage code under current status mapping |
| --- | --- | --- | --- |
| Successful cycle and serialization | `complete` | none | 0 |
| Executor factory raises `RuntimeError` | `failed`, `executor_construction` | original traceback, partial failure and empty call log | 3 |
| Opening `receipt.json` raises `OSError` | **`complete`** | `secondary_failures[0].stage == "receipt"`; no receipt | **0** |
| Opening `status.json` raises `OSError` | **`complete`** | `secondary_failures[0].stage == "status_json"`; no status file | **0** |

The positive control populated the fields `_receipt()` reads, so both late-write failures are reached after successful receipt validation/reduction, rather than through an unrelated missing-key error.

**Minimal correction:** determine final success after all mandatory evidence steps, or explicitly demote a successful status when any such step fails. Preserve an earlier primary execution error; retain serialization/finalization errors alongside it. A stage-level finalization failure must also force a nonzero exit. Add focused controls for failed receipt writing/hashing, status writing, and stage finalization after an otherwise complete cycle. No extra experiment or numerical threshold is needed.

## Closed deltas

**H1, actual freeze path resolution:** collector lines 53–99 use `entry.path`, normalize the absolute path, distinguish explicit repository paths and the `papers/gdn-tree-scan-mlsys/` prefix, and reject disagreeing hashes that resolve to one path. The legacy path-key fallback is limited to v2. Executing the actual resolver AST against the delivered `FREEZE-M1-ADAPTERS-v3.json` resolved and hash-verified **all 21 entries in the snapshot**, including the repository-relative stage design. A missing v3 `entry.path` is refused. The launcher passes the original v3 freeze, not the v3.1 delivery manifest; this closure is specifically for that actual interface.

For transparency, a first audit pass resolved against the canonical local repository and found six v3-only files absent there; the complete mirrored snapshot contains them and all 21 actual-v3 hashes match. That local mirror placement is not a resolver failure or a remote-runtime attestation.

**H2, original constructor/empty-index cases:** construction now occurs inside the protected block at 167–180. The actual-AST constructor-negative control returned `failed/executor_construction`, retained the original exception and traceback, and wrote partial evidence, empty call log, endpoint evidence and status through the in-memory writer. `inner=None` and `cap=None` are handled. `TensorStore` writes a valid empty index at 125–128; stage finalization guards index existence at 264. Original constructor failure is closed; F1 above is the remaining late-stage hole.

**Reset premise and raw completeness:** the index-zero callback checks shape, float32 dtype, and exact S0 equality for every layer before the measured verification sequence (`reset_s0_premise`, lines 134–141; callback line 164; inherited cycle start capture after reset/build). A failed premise raises while the capturing wrapper already retains the raw initial state. The inherited Lumo `published_state()` returns a CPU clone, and author-state captures also copy to CPU, so this exact comparison uses the CPU shared operands rather than mismatched devices.

The new completeness checker requires 48-by-28 first-verification coverage, per-head metric arrays, 48 raw output references with matching store metadata, four finite endpoint captures, digest equality, correct state shape/dtype/size, all exact reset flags, and publication-step diagnostics. Its runtime validator now inspects summaries and endpoint captures rather than relying solely on a boolean declaration. No error tolerance is introduced.

Independent controls on the actual completeness AST: the complete 48-layer metadata fixture was accepted; seven variants were refused (one false reset flag, missing output reference, unavailable diagnostic cell, missing final-flush capture, mismatched state digest, nonfinite state, and state absent from the index). Four actual reset-function controls with tensor-shape/equality stubs accepted exact equality and refused a changed layer, missing tensor, and wrong shape. These are CPU contract controls, not raw tensor authentication from a real run. The real run's stored bytes and receipts still require result auditing.

**Earlier adapter closures remain preserved:** the v3 adapters, executor, cycle driver and capturing collector match the previously reviewed v3 files byte-for-byte. The accepted canonical indexed-device and persistent `out=` buffer changes therefore remain intact. No normalization, tree geometry, proper-ancestor replay, publication, or final-flush semantics were reopened.

## Scope after correction

After F1 is corrected and resealed, this review has no further identified blocker in these closure deltas for the fixed L48/B1, seed 20260928, root-only/n14/n15, four-method serial, `dot_bf16=True`, untimed initialization/diagnostic stage. That statement is neither launch authority nor GPU/runtime validation, timing readiness, numerical qualification, or an M1 result. Host-launcher approval and actual-run evidence remain separate.
