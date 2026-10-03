# M1 native C1 and GPU qualification: bounded next-step review

**Disposition:** the next M1 GPU work can be independent of the blocked full-model candidate boot. M1 is a tensor/kernel component experiment, without model weights, an agent server, or a model-memory reservation. Retain its accepted **12 GiB actual CUDA-free check inside the one owned job**, plus separate host/UMA observations. Do not apply the full-model 82.257 GiB MemFree floor to this different experiment, lower either threshold, or infer present readiness from an earlier observation. This is implementation guidance, not launch authority.

## What is complete

The accepted v3.5 package closes the observer/lifetime, manifest and numerical-contract issues. The later M1-C v1.1 run completes the finite CPU comparison baseline on both calibration seeds: 48 layers, six author-policy surfaces, 428,544 C1/C2 cells and 732,672 detected eligible structural witness evaluations. Its additional 142,848 Lumo cells are **C2 only**. The authenticated receipt explicitly leaves native Lumo C1 unimplemented and substitutes no CPU recurrence. The CPU result is neither an execution of author GPU kernels nor an adapter qualification. See `m1c-baseline-result-review.md` and its preserved result snapshot.

The existing real executor and complete-cycle driver can be reused. `m1_executor_image_v3.py:33–76` binds the immutable image/runtime and current fixed32 backend. `m1_cycle_driver_v3.py:115–240` stages/reset inputs, verifies and publishes each of three paths on the same continuing state. Its TreeWY branch includes the final deferred commit/flush and inverse output mapping; retain that accounting. There is no need to rebuild these adapters for the next step.

## Minimal remaining implementation

1. **Add an actual native-C1 collector for the two Lumo surfaces, baseline first.** Use the pinned `vllm` `fused_sigmoid_gating_delta_rule_update`, with the exact argument/dtype pattern in `q1_component_runner_v2_2.py:454–465`. The pinned native module SHA is `000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f` (`:67`); record actual module/function/dispatch identity as in `:769–790`. Use independent FP32 scratch state and raw BF16 q/k/v/a/b, FP32 A_log/dt_bias/S0, native in-kernel normalization, and BF16 stored output. No CPU rewrite or candidate-produced intermediate is C1. The baseline-only collector does not need to construct the full Lumo Backend merely to call this native operator.

   For first-verification outputs, collect the native result at every one of the 28 active nodes after a native sequential root-to-node history starting at pristine S0. For publication states, run **one uninterrupted** native history and capture at cumulative updates 1, 6, 11. Also retain the intermediate history states and the designated alternate final sibling needed for W1–W4. Reset to pristine S0 for the second independent repeat; require bitwise repeat equality. A wrapper around the existing `native_chain` can reuse its call signature, but restarting each accepted publication path from S0 would be wrong. The frozen C2 implementation is `m1_c_baseline_v1_1.py:361–438`; metric, bounds and witness utilities are `:442–502`. Its author-only `evaluate_layer_surface`/declaration table cannot accept Lumo unchanged: give the native collector its explicit source-bound declaration and observation adapter, while preserving all arithmetic functions/constants.

2. **Finish the reference-only Lumo resolution before candidate results.** Generate the existing two calibration populations (20260930/20260931), compare native C1 with independent C2, check all finite/repeat/coverage/witness criteria, and seal per-cell results. Parent resolution of the two Lumo surfaces and the existing six CPU surfaces must precede M1-Q. Archived M1-C tensor values were hash-bound, not saved in full: regenerate operands/C2 using the frozen implementation and recorded CPU environment and require the archived hashes when claiming reuse. Torch 2.4.1 CPU/four threads is the accepted baseline environment. Do not silently substitute image-Torch reference bytes if they differ; either preserve/reuse authenticated regenerated CPU references or resolve a separately versioned reference-environment repair before evaluating methods. Archive the raw operands and reference/native values for the new evidence.

3. **Add a manifest-driven numerical reducer and a stage-specific entry/design.** Run the existing four real adapters serially on the prospective qualification seed and compare their raw first-verification outputs and three durable states against their own policy C1/C2. Reuse the collector's raw tensor store, exact reset-S0 premise, exception/evidence finalization, and v3.5 memory observer. Enforce observed extents/cell identifiers, not scheduled counts; match every raw input/source identity; fail or mark uncovered as prescribed below. Repeats start from pristine S0 with stash/pending/ring reset, while retaining the accepted cache lifecycle. Seal the reference-only eligibility before C0 dispatch and do not tune from C0 outcomes.

   This is real missing functionality: `m1_stage_collector_v3_5.py:296–306` explicitly declares no numerical pass/fail and no qualification; `:327–348` is still the initialization-design loop. Existing `m1_stage_entry_v1_1.py:29–30` imports **v3.2**, and `run_m1_init_stage_v1_1.sh:47–51,139` pins initialization seed 20260928/old entry. A successor must explicitly bind the new stage/entry, v3.5 observer, v4.2 contract, expected manifest and scope. Do not merely relabel a successful initialization receipt or override its seed externally.

## Already frozen population and decision rule

- Batch 1; 48 independent operand instances/layers; ordinary-random fixtures; 48 value heads; 32 physical slots, 28 active nodes with depths 1–12. Inactive rows are 18/23/25/27.
- First-verification surface: 48 × 28 × 48 = 64,512 cells. Durable publication surface: 48 × 3 × 48 = 6,912 cells. **71,424 cells per policy/input**, not a pooled summary.
- Paths are root-only `[0]`, n14 `[0,1,4,9,14]`, n15 `[0,1,4,9,15]`. The continuing reference history is `[0,0,1,4,9,14,0,1,4,9,15]`; root recurs and contributes an update on each cycle. Reset S0 is an exact premise. Outputs of verifications 2/3 are outside the currently declared observation cells.
- Calibration seeds are 20260930/20260931. Seed 20260928 is initialization reproduction only. **20260929 remains the prospectively declared M1-Q input**; its generation must occur only under the new stage's frozen scope, not during exploratory CPU tests.
- At every cell, both RMS and maximum absolute error must satisfy `error(C0,C2) <= 1.1*error(C1,C2) + 2^-24*metric(C2) + 2^-149`. No extra BF16 ULP allowance, cross-head pooling, observed-max replacement or calibration retuning.
- Nonfinite reference/comparator means UNCOVERED; nonfinite method means FAIL; missing cells UNCOVERED; extra/duplicate cells MALFORMED. Required baseline witnesses are no-op/stale publication, head permutation and applicable wrong-sibling substitutions. Every applicable, observably different witness must violate the same rule; exact-identical excluded cells are counted and cannot establish power. No universal 10x margin is required. TreeWY/Weaver use their named CPU comparators and surface-specific preparation, not Lumo-native claims.

## Smallest defensible launch sequence and resource independence

First freeze/review a **native-only calibration collector + owned launcher/entry** and run the two existing calibration seeds, without author/candidate outputs or timing. Its data then closes the two native baseline surfaces. Next freeze/review the untimed M1-Q reducer/entry and prospectively adopted input domain, run the four existing GPU adapters with exact repeated evidence, and inspect the numerical verdict. Timing can follow only for resolved/qualified methods under the already retained complete-cycle boundary and 56-cycle design; it is not required for the immediate native-C1 step.

Each future launch needs a fresh one-use M1 gate/output namespace, pinned image/source/fixture identity, exclusive GPU ownership, bounded timeout, exact owned-CID cleanup and terminal evidence. The accepted init launcher already expresses `no_reclaim_boot_approved` and disallows separate CUDA queries/reclaim (`:55–63,120–125`); bind the current accepted readiness-consumer dependency without reusing a consumed full-model authority. The in-process entry's free check (`m1_stage_entry_v1_1.py:18–26`) is the appropriate resource seam. Native-C1 uses a small scratch bank; full adapters use the already audited tensor banks/caches, not model weights. The earlier init completed all four methods with 12.048 GiB CUDA free, which supports feasibility but is not a current-memory promise or a peak bound. Keep the 12 GiB floor and v3.5 separate host/CUDA peak observations; abort and preserve evidence on resource failure. No reclaim, new memory trial or model boot is necessary for this plan.

Focused CPU controls before launch should exercise native call shape/dtype/history/reset construction with injected operators; omission/duplication/nonfinite/repeat-drift and wrong-sibling reducer refusal; raw/hash/shape binding; reference-before-method execution order; and the new entry/gate source selection. These close the new seams without repeating the accepted adapter campaign.

## Source identities checked in this review

| Artifact | SHA-256 |
| --- | --- |
| Accepted M1 v3.5 freeze | `9b3ea3f7e1d7472692f2336b4b0136f7a960a2f5bf88beae3e86713e306e4da3` |
| Contract v4.2 | `b98c3af2fd1d7cccf492cc10040a5847b82aefdc940c7570cb777b805ee4ee80` |
| Observation manifest v1 | `c5dd3936ca2a94caea87b0f3e68794f892e01c5759204d149af10b7daa10c3f3` |
| Collector v3.5 | `0dc680fa0a9ec81496926b177144ff7211e38fbeb63c37a14b6805384db37f3c` |
| Existing entry v1.1 | `3374cbb55279705ecdf58746566c077abdfe399e2686f45f7776aa786f02b355` |
| Executor v3 | `24593eca86443ac65210fc4703ba9e8acab4e08e8c52142756df74c77166d6e1` |
| Cycle driver v3 | `4ad560d96844ad4fd6abe7bfc484f01ac0fbb16925c217c27d46c64801a29d75` |
| Adapters v3 | `c5c3b3b328f8f67d04ddb4d0a421da478a2d9e2c38958ef5496919f63372aeda` |
| CPU baseline executable v1.1 | `00d1c8e5a593f2f1377e43aef2002f446a6ffb7c04ef0689b38cdbe163fd721b` |
| Authenticated CPU calibration receipt | `0c6d5918fe2c982c7ae58dabf7e4bd3cb3251ef46a61dac21fd249b5de23cb36` |

Contract/manifest/old launcher were read from the immutable accepted v3.5 review snapshot; current local executable hashes above agree with their preserved source identities. Prior CPU result audit was reused, not numerically rerun. No Torch/CUDA import, remote operation, implementation edit, gate change or experiment was performed for this review. Full-model qualification and all experiment counters remain unchanged.
