# M1 v3.5 bounded closure review

**Disposition: sufficient for baseline-only CPU M1-C implementation preparation. F2, F3 and F4 are closed within their stated scope. F1's factual arithmetic error is closed; before calibration, record the explicit C1 preparation/product/storage declaration below and bind its implementation.** This is a small remaining specification closure, not grounds for another broad design review. No numerical constant, input population, workload or timing scope needs to change. Calibration has not run, the executable is not yet present, and no policy is qualified or launch gate opened by this note.

## Independent snapshot and checks

Read-only SSH copied the freeze and its **89 directly listed payloads**, plus the review request, into `p0/monitor/review-response-20260927/m1-v35-reviewed-20260928T165708Z/repo/`. Every payload hash and size matches. The freeze was identical before and after transfer. All entries retained from v3.3 and v3.4 match those earlier manifests. The supplied log reports **125 passed**; I did not rerun the unchanged dependency suites.

| Artifact | SHA-256 |
| --- | --- |
| `FREEZE-M1-ADAPTERS-v3.5.json` | `9b3ea3f7e1d7472692f2336b4b0136f7a960a2f5bf88beae3e86713e306e4da3` |
| `M1-V3.5-REVIEW-REQUEST.md` | `3cdefa96a19352af676c1c4fb0e22583e2a77f66ad663a3a2b3355d47f7736b6` |
| `m1/M1-NUMERICAL-CONTRACT.v4.2.json` | `b98c3af2fd1d7cccf492cc10040a5847b82aefdc940c7570cb777b805ee4ee80` |
| `tools/m1/m1_stage_collector_v3_5.py` | `0dc680fa0a9ec81496926b177144ff7211e38fbeb63c37a14b6805384db37f3c` |
| `tools/m1/m1_observation_manifest_v1.py` | `72d62e39024f29b9848cae4276e9fb88e5d90e8af27e74b2c6a762f584e8652f` |
| `m1/M1-OBSERVATION-MANIFEST.v1.json` | `c5dd3936ca2a94caea87b0f3e68794f892e01c5759204d149af10b7daa10c3f3` |
| supplied v3.5 test log | `9eecc0dc42c85e98f65d3e699de7ecb9f5bfd8b51082e24bdc40b5d818dc8ea2` |

`INDEPENDENT-SNAPSHOT.json` records the transfer and 89 checks. `CLOSURE-CPU-CONTROLS.json` records **16 independently passing local stdlib/source controls**: retained entries, unchanged constants, exact manifest regeneration, unique cells for all four policies, cumulative histories, observer call behavior/lifetime/restoration, and a scalar storage illustration. Only inspected stdlib topology/manifest code and the actual observer class extracted by AST were executed. No Torch import, tensor-input generation, author kernel, GPU, container, remote job or gate mutation occurred.

## Closure of the four findings

**F1 — corrected arithmetic; one declaration remains.** Contract v4.2 correctly identifies Weaver replay as FP32 elementwise reduction, preserves author BF16 versus aligned FP32 stash policies, and explicitly disclaims Tensor Core emulation/native nonregression for the named author-policy comparators. The retained Weaver source at `chunk_tree_verify.py:898–906` supports this correction. However, C1 entries at contract lines 119–123, 142–145, 166–170, 188–192, 212–217 and 235–239 contain names/claims rather than complete arithmetic fields. In particular, output-store rounding remains implicit. `test_m1_contract_v4_2.py:46–56` requires the six arithmetic fields for the method and C2p, but checks only name/claim for C1.

This affects the rule, not just prose: for a scalar ideal output 1.001 and its BF16 stored value 1.0, an FP32 C1 yields a budget about `1.11e-7`, while a BF16-stored C1 yields about `1.10006e-3`, under the same frozen constants. The observed error is about `1e-3`. This is a stdlib rounding illustration, not a measured kernel result or proposed tolerance.

**F2 — closed.** Exact regeneration matches the sealed manifest. Each policy/input has **71,424 unique cells**: 48 layers × 28 first-verification nodes × 48 value heads, plus 48 layers × 3 publications × 48 heads. Depths cover 1–12; uninterrupted publication histories contain 1, 6 and 11 updates. Role, seed, layer, cycle step, surface, node/depth, cumulative history and head are bound. The contract explicitly states that the new inputs are outside Q1.2b's frozen fixture manifest. Reset S0 remains an exact premise. The reducer should follow the contract's missing=`UNCOVERED`, extra/duplicate=`MALFORMED`; the generator's introductory comment loosely calls all three malformed, but no reducer implementing that comment exists in this package.

**F3 — closed as a prospective policy.** Structural witnesses have surface-specific applicability; primitive-policy swaps are recomputed sensitivity diagnostics; the universal 10× requirement and the requirement for nonzero C1/C2 error are removed. Exact agreement is valid. Constants remain exactly `1.1`, `2^-24`, `2^-149`. No weak/failing candidate cells may be dropped and no tolerance may change after candidate outputs. Implement the declared counts of identical/excluded versus eligible/detected cells explicitly; a witness with no eligible observations supplies no power evidence. No powered-negative result is asserted yet.

**F4 — closed.** Collector lines 174–199 use `weakref.WeakMethod`, skip existing instance overrides, and delete only the temporary overrides. Lines 234–244 put cleanup in `finally`; the inter-method point is after the method returns. Repeating the original lifetime control with cyclic GC disabled now releases the executor at its last external reference. Calls still return their original results; cleanup is idempotent, restores class descriptors, preserves pre-existing instance attributes and works on an injected exception path. Phase descriptions now match the boundaries. The 12 GiB floor, separate UMA counters, persistent production caches and absence of timing instrumentation changes remain intact.

## Exact C1 declaration to adopt before calibration

For each C1 surface, record `preparation`, `scale_placement`, `rounding_points`, `accumulation_reduction`, `state_storage`, `output_storage`, plus function/source hashes. The following is a concrete declaration for implementation preparation; its software quantizers and reduction order must be fixed in source before M1-C. These author-policy C1s are named sequential comparison operators, not reconstructions of compact-kernel execution.

| C1 surface | Independent preparation and arithmetic |
| --- | --- |
| Lumo output/state | The **actual pinned native** `fused_sigmoid_gating_delta_rule_update`, sequential on the manifest paths/history. FP32 rsqrt normalization with additive `1e-6`, FP32 gates/beta/state, query scaled before readout, BF16 stored outputs. Do not label a CPU rewrite as this native C1. |
| Weaver author verification | Convert pristine BF16 inputs to FP32; normalize q/k with FP32 `sqrt(sum(x*x)+1e-6)` division; round normalized q/k to BF16 and promote back to FP32. Gates and beta are FP32. Scale normalized query in FP32 before the readout projection. For the named sequential comparator, round both operands of the `k·state` and `q·state` projections to the declared TF32-width quantizer, accumulate in FP32, keep the sequential recurrence/update FP32, and store output BF16. |
| Weaver author replay state | The same independently prepared BF16-stored normalized key; beta computed in FP32 then BF16-rounded/promoted, g in FP32, v from pristine BF16. FP32 elementwise products/reduction and FP32 state updates; no TF32 quantizer and no output store. |
| Weaver aligned verification | Independent FP32 `x*rsqrt(sum(x*x)+1e-6)` preparation, FP32 normalized q/k/g/beta storage; scaled query and the named TF32-width projection arithmetic as above. BF16 stored output; FP32 state. |
| Weaver aligned replay state | Aligned FP32 key/g/beta stashes prepared independently, FP32 elementwise recurrence/reduction; no TF32 or BF16 stash rounding. FP32 state; no output store. |
| TreeWY output/state comparator | Independent FP32 `x/max(sqrt(sum(x*x)),1e-12)` preparation and FP32 gates/beta. In the **named sequential comparator**, use BF16-rounded normalized keys/scaled queries and BF16-rounded state operands for the reduced `k·state`/`q·state` products, with FP32 accumulation. Declare the sequential decay/residual/outer-product update as FP32 on its prepared operands, with FP32 persistent state; round the readout once to BF16. This defines a comparison operator rather than reproducing WY intermediates, rounded `Co/vt`, or Tensor Core ordering. |

For all CPU comparison operators, specify the reduction implementation and rounding mode explicitly (for example a named software round-to-nearest-even quantizer for BF16/TF32-width operands), and do not let library defaults silently choose TF32 or intermediate dtypes. Store BF16 outputs once, then promote those stored values for error computation; keep state surfaces FP32. The C2p ideal-policy definitions remain separate as frozen in v4.2; do not replace their float64 preparation with C1 intermediates or author buffers.

BF16 output storage is source-grounded: pinned native `identity/native_source/fla_ops__fused_sigmoid_gating.py:153–154,225` stores to q's dtype, BF16 for these fixtures; Weaver `gdn_tree_triton.py:381–385,444` stores to `O` allocated with v's BF16 dtype; the TreeWY adapter `m1_adapters_v2.py:396` requires returned `qs.dtype`, BF16 here. FP32 persistent states remain separate. Query scaling precedes product precision conversion in Weaver `gdn_tree_triton.py:201,356` and TreeWY `tree_wy_triton.py:236–243`.

Minimal implementation checks are therefore: all C1 fields present, actual prepared-operand dtypes/digests match them, quantizer/reduction sources are bound, output tensors are BF16 and state tensors FP32, manifest population matches exactly, and witness applicability/eligible/detected counts are retained. No numerical threshold changes are needed. If a different sequential outer-product convention is desired, declare it now before calibration; do not infer it from the word “BF16” or select it after results.

## Next stage

The parent can authorize **CPU implementation preparation now**, using this declaration to close the remaining documentation gap directly. Before a baseline calibration execution, seal the implemented comparators/references and their exact arithmetic declaration. M1-C remains on seeds **20260930 and 20260931**, without method/author outputs or qualification seed **20260929**. Its CPU author-policy comparisons cannot establish the actual GPU native C1 for Lumo; that named operator remains a separately executed native reference. Contract resolution, GPU qualification, launcher integration and timing remain later parent-owned gates. No additional experiment matrix, memory-clearing operation or repeat review of accepted source is requested.
