# Clarifying nontrivial implementation changes and method differences

Revision dated 26 September 2026 (local time). The manuscript now distinguishes the execution/commit choice from the supporting transformations that implement it. No experiments or new performance claims were introduced.

## Architectural difference

The closing comparison in Section IV names the overlap explicitly. SpecLA shares resident path/chain execution, with accepted factors applied in a later fused update. LumoTree uses immediate replay of recorded operands through native per-layer updates. Weaver shares selected-path replay but uses triangular verification. Bole reconstructs accepted states from factors in a batched commit. TreeWY's inspected author implementation can combine prior-state reconstruction with next verification. The LumoTree choice is path recurrence plus immediate native replay and coordinated hybrid-state publication, with accepted-update work included in the cost.

This is a comparison of specific execution policies, not evidence that state coordination, replay, resident tiles, or native cache slots were invented here. Bole also shares a selected path across layers; that property alone is not a distinguishing claim.

## Section V changes

The title is now “Optimizing the LumoTree Verification Cycle.” Four subsections explain the operations changed and the constraints on those changes:

- Device acceptance products feed native replay, convolution, KV and drafter state. Pending and padded tokens must not advance the materialized prefix.
- Spine-first packing coordinates KV writes, mask columns and accepted-path remapping while preserving logical query order. Isolated layout benefit remains an ablation question.
- Static ancestry gathers and fused convolution replace per-node preparation while retaining tap order, cast boundaries, padding and per-layer operand identity.
- Shared full-vocabulary scores and fused argmax/top-k preserve both reference tie conventions and duplicate behavior. Fusion and graph replay are identified as established supporting techniques.

Table II now compares starting work or constraints, the LumoTree change, and what must remain consistent. Numerical and workload evidence is unchanged; no isolated optimization gain is inferred from the combined workload result.

## Evidence checked

- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`: `_fr13_fixed32_device_commit_route`, device product publication and fused draft-selection setup.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`: `_fr13_fixed32_committer_replay` and per-layer native graph scope.
- `src/lumo_flywheel_serving/fr13_tree_conv_fused.py`: static ancestry indices and explicit ordered-tap/cast design.
- Existing current-production and optimization-supersession notes bind active code to the reported workload; disabled candidates remain excluded.
- Primary pages rechecked: https://arxiv.org/html/2607.16673v1 (SpecLA accepted-factor and delayed-state policies), https://arxiv.org/html/2608.01651v1 (Bole finite-Neumann and batched factor commit), https://arxiv.org/html/2608.20961v1 (TreeWY reconstruction). The previous pinned TreeWY code review remains the source for fused prior-commit/next-verify details.

Build and visual receipt: `p0/monitor/2026-09-26-design-distinction-build.json`. Abstract, references, algorithms and empirical result source were preserved. The current private bundle and hashes are recorded in `artifacts/FINAL-DELIVERY.json`.
