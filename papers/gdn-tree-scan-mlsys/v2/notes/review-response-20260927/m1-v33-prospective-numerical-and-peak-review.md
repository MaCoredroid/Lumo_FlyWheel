# M1 v3.3: prospective numerical contract and peak observations

Disposition: **hold the numerical freeze and use of the new peak collector pending the four bounded corrections below.** The separation of baseline-only reference preparation, unopened qualification inputs, reproducibility, and timing is appropriate. This is a local source/design review, not a qualification result or launch approval. The parent retains every gate. No GPU, remote operation, author-kernel execution, or campaign-source change was performed.

## Reviewed identity and scope

Snapshot: `p0/monitor/review-response-20260927/m1-v33-reviewed-20260928T154354Z/repo/`, relative to this paper directory. Independently checked all **69/69** snapshot members against their recorded hashes and sizes; all **56** retained v3.2 entries are unchanged. Relevant SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `FREEZE-M1-ADAPTERS-v3.3.json` | `7eb5a8c73c5e7965ec07e13ed5bd52fdd02e2c22d68643c10e7de15734d91588` |
| `tools/m1/m1_stage_collector_v3_3.py` | `90fd825020aa314df097977a5a69cba912f7963125e10ba02f66dccec49fbfd9` |
| `m1/M1-NUMERICAL-CONTRACT.v4.json` | `8c3130826e67653a931a6268d15be6390dfdb4f9cf2a1b80edad0a8d57c795c2` |
| `identity/m1_stage_collector_v3_2_to_v3_3.diff` | `31598aa62a3725d8ed8cc53dc8a2ef43b1eec4c91a5895058f10e88d971e26da` |
| v3.3 test source | `c548a913db166540e197526af0e10a05192119400d83f58199e62fe4a7a7ba67` |
| supplied test log | `60c86dc353046787d1d7ac6576487ae3aa004308caba9c1b95eed4b7167686d2` |

The supplied log reports 74 CPU tests across retained versions and the new tests. I inspected the changed source/tests and ran the focused stdlib lifetime control below; I did not rerun the accepted dependency suites. Earlier source closures and the initialization raw-tensor review remain closed. References below to contract/collector lines use this snapshot's campaign payloads.

## Required corrections

### F1. Specify the actual replay arithmetic and the scope of the proposed emulator

Contract lines 69–80 label Weaver accepted replay as TF32 merge dots and apply a TF32-rounded sequential comparator. The named replay callable actually executes FP32 stash loads and `tl.sum(b_h * b_k[:, None], 0)` at `experiments/author-code-20260926/weaver/chunk_tree_verify.py:898–906`. The cited merge kernels are another route. This affects both author-default and aligned replay comparators (also contract lines 94–103). Preserve the declared distinction between author BF16-rounded beta/normalized-key storage and aligned FP32 storage; correct replay's reduction policy.

Freeze explicit preparation, scale placement, rounding points, accumulation/reduction, state storage, and output storage for each C2p/C1-emu surface. A sequential emulator can be an independently implemented, declared comparison operator without reproducing compact matrix execution. Finiteness, deterministic repeats, and detection of injected faults do **not** establish that it emulates the author's Tensor Core arithmetic. Acceptance must therefore be described as agreement relative to that named comparator on these inputs, not established native nonregression or exact hardware emulation. The existing decision to keep same-author alternatives diagnostic is sound. No full-model oracle or extra author-kernel experiment is requested.

### F2. Freeze the real observation domain and continuation history

Contract lines 19 and 46–59 omit part of the tested domain. The 28 first-verification nodes span depths **1 through 12**, with counts `{1:1,2:3,3:5,4:5,5:4,6:4,7:1,8:1,9:1,10:1,11:1,12:1}` in the pinned topology. Root-only/n14/n15 publish paths contain **1, 5, 5** nodes; their consecutive continuation histories contain **1, 6, 11** updates. An extension mentioning only depths 2–5 does not cover the existing captures. Q1's frozen domain also identifies particular fixtures; a new M1 seed does not enter it merely by matching depth.

The expected-observation manifest should include input, layer, cycle step/history, surface, physical/logical node, path depth or cumulative accepted depth, and value head. State references for publications two and three must continue the earlier accepted history from the same pristine S0, not independently restart each accepted path from S0. Freeze the prospective new input/domain explicitly and adopt the native paired rule on that domain before qualification outputs. This is a coverage clarification for existing observations, not a request to enlarge the matrix. Keep missing/extra/duplicate and nonfinite handling as declared.

### F3. Make baseline-only negative controls applicable and interpretable

Contract lines 148–157 require every witness to exceed 10 times the budget at every cell, and require nonzero C1-emu/C2p error everywhere. These conditions are not suitable as written:

- No-op/stale publication witnesses concern publication states, not every output surface. Beta/normalization/decay changes require recomputing the recurrence from changed primitives; they cannot be defined simply as changes to the final emulator output.
- Ordinary-random inputs need not expose an epsilon or beta-storage swap at every cell. Rounding can erase such a perturbation. Failure of that witness is not proof that the underlying comparator is invalid.
- Exact agreement on a cell is valid. Requiring positive numerical disagreement does not establish reference independence and needlessly rejects correct exact cells.

Freeze a small witness-to-surface applicability map before running M1-C. Reuse structural witnesses such as no-op/stale state or wrong head/path where they are observably different. Treat subtle policy swaps as sensitivity diagnostics unless a prospectively specified discriminating fixture makes the claimed detection requirement meaningful. Enforce policy identity through source/operand bindings as well. Remove the nonzero-error validity requirement, report zero error honestly, and keep comparator independence as a code/source property. Do not tune tolerances or omit weak cells after candidate results. These are CPU reference-definition corrections; no broader GPU sweep is needed.

The contract already refuses U2/timing for unresolved policies and fixes constants before calibration; it contains no explicit default-to-pass. Retain those boundaries. Passing finite/deterministic/power checks alone must not automatically confer a stronger numerical claim than the declared comparator supports.

### F4. Undo observer wrappers before the executor's normal lifetime ends

Collector lines 150–159 install closures onto each executor instance. Each closure retains a bound method, which retains that executor: `executor -> wrapper -> bound method -> executor`. There is no restoration before the inter-method observation. This can retain staged inputs/results until cyclic garbage collection, changing the live allocation population and later method peaks.

Reproduction used the **actual `PeakObserver` AST**, a dummy executor with its three methods, no Torch import, no sampler thread, and weak references. With automatic cyclic GC temporarily disabled inside the isolated Python process: an unwrapped executor vanished when its last external reference was deleted; an instrumented executor remained alive; `gc.collect()` released it. This is a source-level lifetime counterexample, not a measured GPU leak.

Use weak-bound dispatch or restore instance attributes in `finally` before the inter-method point. Preserve whether the method originally existed in the instance dictionary: delete a temporary override for an original class descriptor rather than restoring a bound method onto the instance and introducing another self-cycle. Do not clear production/global caches to solve this instrumentation issue.

## Accepted peak design and remaining execution boundary

The new observer keeps host/system and CUDA counters separate on UMA, retains the **12 GiB** entry floor, folds nested CUDA peaks, and labels observations as stage-specific rather than universal bounds (collector lines 103–141, 163–167; contract line 173). This is consistent with the accepted live-allocation inventory. No arbitrary multiplier or addition of overlapping host/device counters is introduced. The scientific complete-cycle driver is unchanged; observation synchronizations and the host sampler belong only to untimed phases and must remain absent from future measured intervals.

Keep the limits explicit: sampled RSS/MemAvailable can miss transients; VmHWM is process-lifetime evidence; observer errors or missing counters cannot establish a peak bound. Minor phase-label corrections are also warranted: the shared-operands phase ends before input hashing, and evidence persistence ends before status writing. These are labels, not reasons to add another experiment.

The v3.3 collector does not yet implement M1-C references, evaluate this numerical contract, or provide the future measurement launcher. That unfinished integration is correctly separate from source/design acceptance. Declare v4's precedence over superseded policy/memory paragraphs in retained proposals so a future launcher cannot select an older rule.

## Next admissible stage

First close these four items with CPU source/specification changes and a frozen observation/witness mapping. A separately authorized **baseline-only M1-C** may then characterize the independently defined references/comparators on its already proposed disjoint seeds, without author outputs or the qualification seed. Parent review must resolve the policy-specific contract before U2 or timing. Apply the peak lifetime fix to the already planned untimed observations; no additional memory-clearing action, dedicated GPU memory trial, enlarged workload, or relaxation of the entry floor is needed. Actual execution and timing remain subject to the parent's separate launcher/source gate.
