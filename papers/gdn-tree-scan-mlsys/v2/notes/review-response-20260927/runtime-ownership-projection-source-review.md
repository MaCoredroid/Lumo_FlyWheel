# Runtime worker ownership and projection — bounded source review

2026-09-28. Scope: parent-written draft `runtime_worker_ownership_v1.py` and `runtime_projection_v1_1.py`, and compatibility with the two allocation-metadata producers. **Bounded source closure PASS after F1–F3 repairs; preparation only, no runtime admission.** No container, GPU, HTTP request, workload, live `/proc` scan, or engine import was executed. Tests use temporary synthetic `/proc` trees, fake tensor descriptors, and mocked process identity. The final projection SHA is `24fa08459abf28811793f19d0b89f5cfe84f2015431b7bec383267e76791adb5`; all pending integration boundaries below remain unchanged.

All paths below are relative to `papers/gdn-tree-scan-mlsys/v2`. Helper filenames refer to `experiments/review-response-20260927/workload-plan/tools/runtime-collectors/`.

## Findings and current disposition

**F1 — CLOSED: allocation metadata lacked the worker's process epoch.** Both original producers emitted only `os.getpid()`. The consumer matched this namespace PID to a current ownership envelope but never bound allocation metadata to the observed `start_ticks`. The observer's end recheck could detect reuse during its own read, not reuse between the metadata write and that observation. Independent control: retain the allocation record, replace ownership `worker.start_ticks` with 99999, and the original consumer accepted it.

The new `worker_metadata_v1_1.py:31–38` reads `/proc/self/stat` start ticks and `/proc/self/ns/pid`; both new producers emit this `process_identity`. The consumer now exactly compares it with the observed worker epoch and namespace (`runtime_projection_v1_1.py:103–104`). Independent stale-epoch refusal and both actual producer-function-to-consumer fixtures pass with mocked identity. The self-stat parser correctly handles a comm containing spaces and parentheses, and refuses zero start ticks and malformed namespace names. Original v1 producers are byte-preserved.

**F2 — CLOSED: “isolated namespace” initially meant only “not the observer's host namespace.”** A synthetic owned init with namespace PID 2 and inspection `HostConfig.PidMode=container:foreign-container` passed ancestry checks. This admits a shared container namespace, contrary to the stated private-namespace premise. The repaired observer requires default/private PID mode and owned init's namespace PID 1 (`runtime_worker_ownership_v1.py:46,51`). Both the shared-mode and private-mode/non-init variants independently refuse. Existing end-of-observation identity checks remain intact (`:72–75`).

**F3 — CLOSED; original reproduction preserved at projection SHA `d2ab4720446a2acbd5a3dec61c50ba6ceab154c55313bfcfced2d54626c1e6d6`: unscoped dispatch could be accepted.** `check_route` defaults `attempt_id`, `boot_id`, and `request_id` to `None`, then compared those values with `dispatch.get(...)`. A dispatch omitting all three fields passed when those optional caller arguments were also omitted. The independent reproduction returned `dispatch_observed=true` without naming a boot, attempt, or pre-agent request.

The final source adds exactly one precondition at line 161: each expected ID must be a string with nonempty `strip()` before exact record comparison. Removing that one guard produces byte- and AST-identical repair1 source; ownership and both producer bytes are unchanged. The original omission now refuses. Fourteen independent scope controls passed: the original all-omitted case, a valid fully named dispatch, and `None`, empty, whitespace-only, or integer expected values for each of the three IDs. The supplied combined suite now passes 21 tests (including nine new scope subcases). No dispatcher was executed; caller integration remains pending.

## Producer compatibility and accepted scope

The vLLM producer returns named attention, convolution and recurrent logical cache views, target-role metadata, compute-policy/quantization fields, exact source bindings and producer identity. The SGLang producer returns the same target projection, with packed Mamba layer descriptors and separately described K/V components. The consumer's per-layer name/dtype/positive-shape checks accept both actual record-construction functions on synthetic instantiated-object fixtures. It rejects draft-role records, missing/duplicate target layers, `auto` as an observed storage dtype, wrong producer/engine/CID/namespace, and malformed ancestry.

These checks establish schema and provenance preparation, not real allocation observations, source installation, tensor arithmetic, or physical memory usage. Packed/logical views are not counted as unique allocation bytes. Compute dtype is correctly labeled an instantiated model policy rather than a numerical theorem. The route checker requires explicit Lumo host-function dispatch evidence and labels it without claiming kernel timing; a qualification receipt cannot masquerade as engine-observed route state.

## Explicitly pending caller/freeze integration

- Fresh exact-owned-client inspection must bracket the `/proc` observation, comparing CID, image, running state, init host PID and container start identity. The helper accepts an inspection object and cannot establish when that external read occurred. Rechecking process start ticks within the helper is a bounded-interval observation, not permanent liveness or freshness of caller-supplied inspection.
- Archive/authenticate producer records and the ownership envelope under the exact attempt/boot and freeze all helper/producer/source pins. Self-reported SHA strings alone are not an authenticated transport. The caller must refuse absent, stale, partial, unexpected or multiple worker records and bind the selected live target worker.
- Derive and freeze the complete target-layer map independently from pinned model/source information. `layer_map_source_sha256` is a required policy binding, not a file the projection helper independently opens.
- Observe actual TP/PP through the serving configuration/worker evidence. SGLang metadata contains and compares rank/size values; the current vLLM branch checks only that the **expected** policy equals TP1/PP1. Do not present that declaration as a vLLM worker observation.
- Wire and qualify the actual route-dispatch producer for the named pre-agent request; derive its PID/CID from the accepted worker ownership rather than an independent unchecked caller dictionary. Preserve the distinction between probe sampling/cap observations and declared controller timeout/compaction limits.

These are known draft integration boundaries, not evidence that an actual attempt was admitted. No WP/WC gate or attempt outcome is approved by this note.

## Evidence and identities

Original snapshot/reproducers: `p0/monitor/review-response-20260927/runtime-ownership-projection-independent-v1/`. Initial 17 supplied tests pass; four independent controls reproduce F1/F2 and confirm the two producer schemas.

Repaired snapshot: `p0/monitor/review-response-20260927/runtime-ownership-projection-independent-v1-repair1/`. **20 supplied tests and eight independent controls pass** for F1/F2 closure and producer compatibility. `unscoped-dispatch-reproduction.json` separately preserves F3. `SNAPSHOT.json` binds all source/test paths, sizes and hashes. Commands, executed in the corresponding snapshot:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_runtime_worker_ownership_v1 test_runtime_projection_v1_1
PYTHONDONTWRITEBYTECODE=1 python3 independent_controls.py
```

Final F3 snapshot: `p0/monitor/review-response-20260927/runtime-ownership-projection-independent-v1-repair2/`. Its `SNAPSHOT.json` preserves final bytes; `dispatch-closure-controls.json` records the exact one-line delta and 14 independent results. The 21-test supplied suite was rerun against this copied snapshot. All three source findings are closed, with no remaining material source defect found in this bounded review; the helpers remain draft and unwired.

| Source | Reviewed SHA256 |
|---|---|
| Repaired `runtime_worker_ownership_v1.py` | `23eeb7494d23ad228bb98c46491afa235de624e27036248bc3e89d3e27636c93` |
| Final `runtime_projection_v1_1.py` (F1/F3 closed) | `24fa08459abf28811793f19d0b89f5cfe84f2015431b7bec383267e76791adb5` |
| Intermediate projection (F3 reproduction preserved) | `d2ab4720446a2acbd5a3dec61c50ba6ceab154c55313bfcfced2d54626c1e6d6` |
| `worker_metadata_v1_1.py` | `e02fea266846223296b9d64bfa5e857524d98bb21c7d05a9c0c78c1c0907ec4b` |
| `sglang_worker_metadata_v1_1.py` | `08175a1a7a70857925012992295233294b046168c922122f97a8f9be26ef5988` |
| Original ownership helper | `f8c23345c500facabba98fa518878b7aa64924899b8cf28d7835f4a337f98068` |
| Original projection helper | `6a6ebec635806838f54cefc3c58ddda149f598e96498ee160ad1c321794a99fa` |
| Original `worker_metadata_v1.py`, unchanged | `97587d1be791c6621afd834a9014d57a3f2f14167e976fbde195268146492dbb` |
| Original `sglang_worker_metadata_v1.py`, unchanged | `cfb20c7952eb65ce7a35c13ecaae3cd249d56316f3e082374f874cf0c8903a5c` |
