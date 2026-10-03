# M1 v3 runtime closure review

**Disposition: HOLD for two narrow collector fixes.** R1 device canonicalization and R3 persistent gather/remap destinations are closed at source level. R2 now has independent pin comparisons, but the delivered freeze format cannot pass its expectation builder. Executor-construction failures also escape the promised method failure accounting. This review covers only the proposed first untimed stage: L=48, B=1, seed 20260928, root-only/n14/n15, four serial methods, TreeWY `dot_bf16=True`.

No SSH, container, GPU, Torch/CUDA import, actual collector run, implementation edit or gate change was performed. Original normalization/tree/leaf/publication closures were not reopened. Parent owns the raw-C2/state population and host-launcher review.

## Reviewed identity and checks

Source snapshot: `p0/monitor/review-response-20260927/m1-v3-reviewed-20260928T0455Z/`.

`FREEZE-M1-ADAPTERS-v3.json` SHA-256: **`abd45b3c72e8eed43a35fd608bef9288cee38d8e1d9e5f3399832746f4259599`**, frozen `2026-09-28T04:54:40.953855+00:00`. All **21** payload files independently match their hashes and sizes.

| Changed file | SHA-256 |
|---|---|
| `tools/m1/m1_adapters_v3.py` | `c5c3b3b328f8f67d04ddb4d0a421da478a2d9e2c38958ef5496919f63372aeda` |
| `tools/m1/m1_executor_image_v3.py` | `24593eca86443ac65210fc4703ba9e8acab4e08e8c52142756df74c77166d6e1` |
| `tools/m1/m1_cycle_driver_v3.py` | `4ad560d96844ad4fd6abe7bfc484f01ac0fbb16925c217c27d46c64801a29d75` |
| `tools/m1/m1_stage_collector_v3.py` | `512bdab2a0b0e7283d1f09b7b9d85793a147b242a7853179bd3e5b4e937c4940` |

Independent safe checks extracted the actual functions through AST: four canonical-device controls passed using injected device objects; the actual expectation builder was exercised against the **actual delivered v3 freeze**; and the actual `run_method` was exercised with a failing constructor/no-op filesystem interface. AST checks also confirm both gather/remap call sites use `out=dst` and no subsequent `copy_`. The author log reports 78 CPU tests; that suite was not independently rerun locally.

## H1 — Actual v3 freeze keys are not file paths

`m1_stage_collector_v3.py:100–104` iterates `for rel, e in f3['files'].items()` and adds `exp['files'][rel]`. The delivered freeze instead has keys such as `m1_adapters_v3`, `runner_v2_2`, `stage_design`; the actual relative path is in **`e['path']`**. Consequently `observe_identities:123–124` and `_resolve:80` look for `<campaign>/m1_adapters_v3`, etc. The genuine package is refused before any method executes.

**Reproduction:** the source-extracted `expected_identities` read the delivered freeze with the pinned v2/author manifests. All **21** v3 logical keys were inserted as expected paths; all 21 differed from their entry's `path`, and none names a payload file. No source substitution or collector/GPU execution was used. `tools/tests/test_m1_v3.py:72–76` builds a different fixture schema whose keys are already file paths, explaining why its positive case misses this defect.

**Minimal repair:** normalize v3 entries from `e['path']`, then merge/check conflicts against earlier frozen paths. Resolve campaign-relative and repo-relative members explicitly: the `stage_design` entry begins `papers/gdn-tree-scan-mlsys/...` and needs repo resolution (or a declared `repo:` prefix), not a campaign join. Keep missing/mismatched paths fail-closed. Add a positive construction/preflight test using the delivered freeze schema and exact source-bound files, followed by one altered-file refusal. Do not use only the synthetic path-key fixture.

The independent runner comparison in `m1_executor_image_v3.py:61–63`, exact image/runtime comparisons in collector `:143–166`, gate freeze/design/runner bindings, and receipt checks at `m1_cycle_driver_v3.py:276–283` replace the former tautology/format-only checks correctly in principle. H1 prevents certifying their executable positive path.

## H2 — Constructor failures escape method accounting

`m1_stage_collector_v3.run_method` calls `executor_factory(method)` at **`:289`**, before entering its `try` at `:291`. A constructor failure therefore bypasses both failure handlers, `finally`, `status.json`, and the return to the stage loop. `run_stage:350–352` then aborts without retaining that method's terminal status or proceeding according to the declared serial-stage failure policy.

**Reproduction:** executing the actual AST-extracted `run_method` with an injected factory that raises `RuntimeError('INJECTED executor construction failure')` propagated that exact exception and invoked **zero** JSON writes. `makedirs` was a no-op; no files or CUDA objects were created. This tests the real placement of the factory call, not a synthetic successful executor.

**Minimal repair:** put constructor/wrapper creation inside the protected method lifecycle, initialize `inner`/capture state safely, and emit a failed method status plus partial evidence even when no executor exists. Continue or stop exactly as the frozen stage policy specifies. Preserve a valid empty tensor-store index, or handle its absence at `run_stage:355`, when all methods fail before the first tensor capture; otherwise finalization itself raises. Verify one constructor failure and the all-constructors-fail case without a live runtime.

## Closures supported by source

- **R1 closed:** `m1_adapters_v3.canonical_device:40–50` resolves unindexed CUDA through the current index; `ImageExecutorV3:44` stores it before staging/allocation and `_dispatch:80–81` compares canonical devices. Safe controls accepted cuda/cuda:0 at injected index 0, distinguished cuda:1, accepted matching named tensor devices, and refused unresolved unindexed CUDA without availability.
- **R3 closed for the reviewed defect:** `m1_adapters_v3:68–87` writes both permutations with `torch.index_select(..., out=dst)` and asserts destination pointer stability. The explicit intermediate-result allocation and second copy found in v2 are gone. Rebound plan builders use these new callable objects through `G3:93–113`; unchanged math/call arguments remain in the frozen v2 code objects. Allocation/work quantities remain labeled computed, not measured (`cycle_driver_v3:96–100`). This is not a claim that backend internals allocate no workspace or that allocator residency/traffic has been measured.
- **Executable chain after preflight:** the stage imports the real image executor only after successful preflight (`collector:324,328–330,348–349`), stages common operands, calls the v3 cycle through `CapturingExecutor`, and delegates real callable resolution to the pinned executor. Warmup, tree builds, persistent state restoration, accepted publication and final TreeWY flush retain the previously reviewed structure. The capturing wrapper forwards unhandled executor attributes and intercepts the correct returned output/state interfaces. Parent assesses its diagnostic populations and raw persistence separately.
- **In-call exceptions:** the v3 executor appends resolved-call evidence and exception information before re-raising (`executor_image_v3:73–107`), and cycle execution wraps failures with partial evidence (`cycle_driver_v3:142–208`). H2 concerns the earlier constructor boundary, not these repaired call failures.

After H1/H2 and the parent's separate stage findings are repaired, a new sealed source review can determine readiness for the specified untimed stage. This note supplies no launch authority and no M1 numerical or performance result.
