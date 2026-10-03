# M1 adapter v1 independent source review

**Disposition: three concrete wiring closures are required before runtime freeze.** The source/topology preparation is useful and the named arithmetic variants are substantially faithful, but the returned plans are not yet executable as advertised. These are preparation findings, not failed experiments. The package explicitly defers its real executor; lack of GPU results is not a finding. No GPU, author computational-module import, remote operation, source edit or gate change was performed.

## Reviewed identity and reproduction

Campaign-relative paths below refer to `experiments/review-response-20260927/`; author paths refer to its sibling `author-code-20260926/`.

- Freeze file SHA-256: `6cc862ea5d7b86768ed3ce8fc4e3b17e1e4296ff5f16a2aa88cad17029e15b6f` (frozen `2026-09-28T03:34:42.184926+00:00`). Independently verified all **21** freeze payload hashes/sizes and all **22** author/source-license hashes/sizes.
- `tools/m1/m1_adapters.py`: `375a177b1da203a15b8deecc21f971a87a328d037e2f52913c41bd40f1444066`.
- `tools/m1/m1_topology_mapping.py`: `8ec415ecfb71f79e195efb3ee66bc68c52471c0e1c0778021eaae8912e177794`.
- `tools/m1/m1_loaders.py`: `f9f2d979ff9e88acf3ea796475a01ab32daaa6241f7cb2fd0faa3507fff9a5ba`.
- Accepted component runner `tools/q1_component_runner_v2_2.py`: `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29`.

Independent checks used Python standard-library hashing/AST and imported only the pure topology module: all 28 frozen accepted-path records reproduce exactly, both permutations invert, TreeWY's 18 positional argument names and four keyword names match the pinned signature, and Weaver's 12 positional/five optional argument layout matches. AST checks confirm the two direct leaf-tensor factories omit `device`. Neither Torch nor Triton was imported during these checks.

The source-bound author log reports **50 CPU tests passed**, SHA-256 `2bb2b81f2e06e6fa894b5ef0063563e40f3d0bb8f76db1d76fbafb1507fd6d59`. I attempted the 19 adapter/topology/loader tests locally with `python3 -B -m pytest -q -p no:cacheprovider ...`; the bundled interpreter lacks pytest (and Torch), so this is **not an independently reproduced test count**. No packages were installed and no remote tests were run.

## Required closures

### F1 — Bind Weaver's returned normalized query to its verifier input

`m1_adapters.py:143–146,165–168` creates a normalization call without `out=`, records its result only as descriptive text, then independently allocates `q_norm = torch.empty_like(q4)` and passes that unrelated tensor to verification. The pinned helper `m1-author-dependencies/upstream/python/sglang/srt/layers/attention/fla/l2norm.py:175–192` allocates/returns its own result when `out` is absent. Executing the listed author call cannot fill the placeholder. The test at `test_m1_adapters.py:32–42` checks names/strides and the key output alias, but never the query result alias.

**Minimal repair:** make result references explicit and have the executor pass the actual returned normalized query into the verifier. Alternatively, use a preallocated output buffer with an explicit `out=` call and label/account that allocation choice. A CPU injected-call test should return a distinguishable tensor and assert the verifier receives that exact result or explicitly filled storage. It must fail when the query normalization result is discarded. Keep the aligned variant's `writes_into` semantics equally explicit; numerical helper values must not be mistaken for completed stash writes.

### F2 — Build and bind both Weaver tree metadata representations

`m1_adapters.py:91–104` allocates zero ancestor tensors, fills only `tree.parent`, and returns. `weaver_prepare_calls` never builds the closure; its final positional argument at `:166` is the string `"TreeStructure(bufs['tree.*'])"`. `weaver_replay_kwargs` returns the still-zero `ancestor_masks` at `:182–191`. The CPU reference at `:194–200` is not connected to either execution path. The cycle driver has no metadata builder either.

This has two distinct consequences: the literal tree string cannot satisfy `tree.anc_u8` in `weaver/gdn_tree_triton.py:437`; merely wrapping the existing zero arrays would still give incorrect path sums. Replay uses `ancestor_masks[leaf] | (1 << leaf)` (`weaver/chunk_tree_verify.py:858–864`), so an unchanged zero bitset would replay **only the leaf** for a nonroot accepted path.

**Minimal repair:** add explicit initialization/build entries for the real author `TreeStructure` and its inclusive ancestor closure, plus the separate proper-ancestor replay bitset. Source-supported builders are `gdn_tree_fused.build_tree_structure_into_fast` (`:114–127`, after `alloc_tree_structure_buffers`) and `chunk_tree_verify.build_tree_ancestor_masks` (`:104–141`). Bind the actual object, not a description. For this fixed topology, declare whether initialization is reused or rebuilt and charge it according to the agreed boundary. CPU injection must prove both builds happen before their consumers and compare every populated row with the already verified mapping. Root-only alone cannot detect the replay defect; include `[0,1,4,9,14]` and require all five nodes.

### F3 — Accepted-leaf tensors must share the kernel's device

`m1_adapters.py:190` creates `last_correct_steps` with `torch.full(..., dtype=torch.int64)` and `:276` creates TreeWY `leaf_full` the same way. Neither inherits `cache_indices.device`/`slots.device`; under the normal default-device setting both remain CPU tensors even when the buffers were allocated with `device="cuda"`. Both author kernels dereference these arguments with `tl.load` (`weaver/chunk_tree_verify.py:858`; `treewy/tree_wy_triton.py:143`). The current assertions do not reject the mixed-device plan. Lumo metadata at `:333` is also CPU-only and requires an explicit copy into the accepted runner's persistent device metadata, as its real `Backend.publish` does (`q1_component_runner_v2_2.py:405–418`).

**Minimal repair:** allocate or fill persistent device leaf buffers using the owning state/index device, assert all directly passed tensor arguments share the execution device, and retain explicit host-to-device copies for host metadata. Prefer the already allocated TreeWY `bufs['leaf_full']` instead of creating an untracked leaf allocation per call. A CPU/meta-device or factory-injection test can verify propagation without launching CUDA. Keep allocations/transfers inside their declared timing/accounting boundaries.

## Source conclusions that do not need repair

- **Topology:** physical level order and declared DFS preorder are invertible; root is retained; all four inactive slots remain physical work but never enter any of the 28 accepted paths. Both accepted-leaf identifiers are node IDs, not path lengths. TreeWY's `kt=anc_i` matches the pinned accepted-path convention; its positive `gc` sentinel prevents a spurious first commit. The extra final fused call can publish the last accepted state while discarding its new verification output; it must remain charged. Parent reviews actual cycle dispatch/accounting separately.
- **Lumo:** the fixed environment names, root-inclusive replay through drafts-only acceptance metadata, in-place running-row endpoint and neutral fixed16 tail agree with the pinned Q1.2b runner. This is a source binding, not a new executed component result.
- **Weaver precision:** the default preparation preserves author strided normalization with bf16 storage and bf16-rounded replay beta. Its verifier independently recomputes fp32 beta from raw `b` (`gdn_tree_triton.py:146`); therefore the rounded seam is specifically the **replay stash**, not all verification arithmetic. The aligned local helper explicitly changes normalization/storage and replay beta. This labels an aligned arithmetic policy, not demonstrated bitwise equivalence to the native kernel. The explicit scale, `precision="tf32"`, `bf16_mode="none"`, and proper-ancestor depth 11 agree with the selected route.
- **TreeWY precision:** its floor-normalization and default bf16 dot operands are preserved, with fp32 in-kernel sigmoid. The freeze explicitly corrects the stale mapping text that described a rounded TreeWY beta. `treewy_aligned_local` refuses execution instead of silently substituting arithmetic.
- **Loader:** the 13 planned bindings cover the selected imports in dependency order, use the exact absolute FLA names, preserve real external dependencies, and reproduce only the declared `torch_release` definition. No hidden computational replacement was found in this static graph. Actual importability/capability receipts remain a future in-image check, as stated in the package.

The parent independently reviews dropped publication arguments, cycle dispatch, accounting and synthetic-versus-measured receipts. Those findings are not duplicated here. Closing F1–F3 and the parent's findings is preparation for a new review; this note does not authorize a launch or assert numerical qualification.

Unreviewed scope: numerical execution/resource use, unavailable in-image executor behavior, unused chunk-verifier/conv kernels, and the parent-owned cycle/receipt/accounting review. No exhaustive arithmetic proof or timing readiness is implied.
