# Native continuous map witness — independent review

**Current disposition: PASS for the single F1 repair, at hooks SHA `204b07938c5dc7bf69a35f2e1e417fe7a5be24715ed33f79353f3293c634f456`.** Mapper SHA remains `204e5b4f9b8f0004fafe580e88147ba9bfc610127b06bc00a5039f612782ca65` unchanged. The original finding and source/results below remain preserved as history.

## Successor closure

The exact diff changes only the materialized-prefix count in `bind_snapshot`: group3 retains attention `ceil(extent/64)`; groups0–2 retain GDN `floor(extent/1024)+1`, including the previously selected column at the exact boundary. No mapper, numerical, import, ownership, callback order, or phase logic changed.

The existing controls were replayed without additional cases: **17/17 descriptor controls and 18/18 actual CPU-mapper controls now meet expectation.** Both previously accepted remap counterexamples refuse. Valid63/64,255/256,1023/1024/1025 boundaries and future attention-page extension still pass. Actual cache tensors remain unchanged by mapper calls. No broad unchanged callback suite or raw auditor was run; the newly written raw auditor is explicitly outside this review and still awaits its separate review.

Successor artifacts are `source.repaired.py`, `bind_controls_repaired.py`, `bind-repaired-results.json`, `mapper_controls_repaired.py`, `mapper-repaired-results.json`, and final manifest/seal in the same review directory. Original failure sources, controls, results, `REVIEW.initial.md`, and initial manifest/seal are unchanged. No runtime, GPU, remote, model, container, implementation, or gate operation was performed. This closure does not qualify or authorize a serving route.

## Original snapshot review (superseded finding, retained)

Initial disposition: **one narrow boundary finding remains** at hooks SHA `86b67fa00b387cbfc31f4d618b5b97253421c9d009d92c961702a2c30b6e149e`; mapper SHA `204e5b4f9b8f0004fafe580e88147ba9bfc610127b06bc00a5039f612782ca65` has no new substantive finding. Original sources and controls are preserved under `p0/monitor/review-response-20260927/native-continuous-map-review/`.

## F1 — previously selected GDN column omitted at exact 1024 boundary

`q1_reference_hooks_sequences_v2.py:46–47` uses `ceil(previous_extent / block_size)` for every cache group. For attention this covers materialized KV pages. For the GDN groups, the actual selected state column is instead `extent // 1024`, as independently enforced by `q1_native_continuous_maps_v1.py:86–90` against current request, runner Mamba index, CPU/GPU table and metadata SSI.

At previous extent 1024 the selected GDN column is 1, but the stable-map check compares only column 0. At current extent 1025, column 1 can change to a different bank row, consistently across request/CPU/GPU/SSI and the captured state, and the new `bind_snapshot` accepts it. The bank allocation, layer and owner can all remain unchanged. This contradicts the intended preservation of the previously materialized recurrent state mapping.

The independent reproduction first uses the exact `bind_snapshot` AST and complete 48/16 descriptor sets. A second reproducer uses the actual mapper, the accepted native CPU fixture, and an owned MTP singleton; it records extent 1024, changes group0 column1 in the actual table/request/SSI, then binds extent1025. Both accept the bad remap. Minimal repair: for GDN groups preserve `previous_extent // 1024 + 1` entries (inclusive of the previously selected state column); keep the attention `ceil(previous_extent / 64)` rule. No numerical policy or serving setting needs to change.

## Bounded checks and scope

- `bind_controls.py`: 16/17 expected outcomes, with only F1 accepted incorrectly. Positives cover 63→64→65, 255→256→257, 1023→1024→1025 and future attention-page extension. Negatives reject ordinary materialized attention/GDN remaps, changed request/prompt/MTP generation, changed convolution alias storage/layer identity, duplicate and regressed boundaries.
- `mapper_controls.py`: 17/18 expected outcomes, with only the connected F1 reproducer accepted incorrectly. Actual CPU Torch checks use 48 GDN and 16 target attention layers plus MTP; extents 63,64,65,255,256,257,1023,1024,1025; wrong live SSI or logical column; missing layer metadata/SSM state; stale computed/sequence extent; and altered original prompt length. Cache tensor pointers and mutation versions remain unchanged across mapper calls. A local Python3.9 annotation-only fixture compatibility issue is preserved in `mapper.initial-python39.stderr`; deferred annotation compilation fixes the test loading without changing any callable fixture or implementation bytes. CPU Torch is 2.8.0, threads2, CUDA visibility empty.

The mapper's executable differences from the accepted joint O0 mapper are limited to separating immutable original-prompt length/hash from current computed extent and removing the mutation/import function. Its mapping formulas, exact cache groups, native64 attention geometry, GDN metadata predicates, owner identity, and selected row joins are otherwise unchanged.

The v2 callback adds read-only descriptors after the inherited snapshot, at initial O0 and every O1, and includes them in the existing canonical case seal. It adds no model call, numerical kernel, state copy/import, token change, or interior hydration. The complete hook still inherits the already accepted **one initial** joint import; saying the whole hook performs no hydration would be inaccurate. Existing outer callbacks retain fail-stop behavior. No numerical criterion changes were found.

This is unconnected preparation. Raw-auditor verification of the extension marker/maps, source-bound job/patch/launcher selection and final launch/result admission remain pending. No remote runtime, GPU/model/container call, scientific execution, production source edit or gate change was performed. The unchanged 24-control base callback suite was not rerun.
