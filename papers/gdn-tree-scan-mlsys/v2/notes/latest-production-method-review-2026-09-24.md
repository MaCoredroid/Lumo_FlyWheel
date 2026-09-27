# Latest production-method review — 2026-09-24

**PASS for the checked manuscript and component-evidence scope. No remaining material source inconsistency found.** This is a read-only review of the Hydra27 rewrite and a CPU replay of its evidence reducer. No model execution, new numerical experiment, manuscript edit, or alteration of attempted experiments occurred.

The implementation/history inventory is in `latest-optimization-supersession-2026-09-24.md` (SHA256 `3d9aa321a3894d72c9990060086955d4e1511e57ee96fb1ce2870e0887c89ed9`). That note distinguishes the latest completed workload route from later unserved candidates and the superseded Cat10 campaigns. This review additionally read the complete current main/abstract and the three included TikZ figures.

## Source and manuscript agreement

- **Active route:** the method follows the completed NVFP4 Hydra27/fixed32 workload: 27 logical drafts, 31 physical draft slots plus root, depth eleven, four post-root MTP iterations, Arctic suffix candidates, patched FA2 GQA-pair split-K4, graph execution and prefix caching. It does not import Cat10's geometry, flat-slot layout, eager/cache-off configuration, or qualification outcomes. The source/receipt chain is documented in the inventory; newer Hydra31 work lacks a completed successor task record.
- **GDN schedule:** `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:17261–17361` implements the active two-level path schedule: one native-state-rooted path supplies cut states to independent terminal paths. Programs also span value tiles and heads. The manuscript correctly distinguishes this schedule from the disabled single-launch candidate and avoids a no-temporary-state-traffic claim.
- **Numerics:** `_gdn_node_step` at kernel lines 10409–10491 and the actual fixed32 native committer graph at 14551–14693 are different arithmetic implementations. The text now states FP32 carried/bank state, raw gate operands, normalization and possible cast differences without asserting common-helper or bit-identical execution. Generic replay-helper comments are not used as proof for the deployed native graph.
- **Acceptance and publication:** fixed32 TAW returns device tensors, while the graph contains 48 native recurrent updates. The manuscript correctly states that convolution and attention-KV publication separately consume the same selected path. Patcher `_fr13_fixed32_device_commit_route` at 19980ff calls convolution publication at 20215, then recurrent replay at 20222; the native graph is not a fused all-layer kernel or a single graph containing all three storage surfaces.
- **Layout and convolution:** main line120 correctly limits the spine-first permutation to KV locations and ancestry-mask key columns, retaining logical query-row/position order. The fused-convolution source preserves path windows, ordered tap accumulation and casts; stacked rings and static preparation do not make branch scratch durable request state. Figures represent the logical selected-prefix boundary and do not reintroduce the old flat-map or all-state-materialization narrative.
- **Evidence claims:** candidate-selection parity is confined to the tested inputs/graph behavior. Attention bounds concern synthetic tensors at banked-operand scales and are not described as exact banked-tensor replay or token preservation. Workload outcomes are not promoted into a full-model conditional-distribution proof. Current workload rates retain their separate, previously reviewed retrospective patch-producing selection scope.

Two wording findings were corrected during this review and rechecked in the final source: (1) “captured device acceptance” became “device acceptance and captured recurrent replay”; (2) the replay paragraph stopped attributing convolution/KV publication to the recurrent graph. The active `FR13_STEP_GRAPH=0` and the separate call order support these corrections. No additional experiment is required to substantiate the remaining, explicitly scoped mechanism descriptions.

## CPU evidence replay

Ran `python3 papers/gdn-tree-scan-mlsys/v2/results/current-production/reduce.py` and compared stdout with `audit.json`: byte-identical. Repeated after the final manifest expansion in a temporary, isolated repository-shaped tree containing only the 20 pinned inputs, `inputs.json`, `reduce.py`, and `audit.json`: byte-identical again. Appending one newline to a pinned input made the isolated reducer fail on that file's hash, as required. Temporary test files were removed.

The reducer correctly binds the fused-selection binary to the recorded deployment environment and the split-K credential/binary/patch identity to the positive sixteen-layer engagement receipt. It reproduces 1,368 selection cases / 6,840 configurations / zero byte mismatches, 24 graph replays / zero mismatches, sixteen determinism cases across two processes, 93.30657958984375% within two ULP, maximum output difference 0.00390625, maximum LSE difference 4 ULP, and nine passing declared bounds. It emits no performance claim. The manifest now includes the actual selection probe, synthetic split-K probe and topology source as well as result/route evidence. These checks validate the archived component receipts and their deployment binding; they do not rerun GPU kernels or imply an unrecorded full-model equality gate.

## Checked identities

SHA256 values, paper-relative paths:

| File | SHA256 |
|---|---|
| `main.tex` | `4793cefe3803d70e817031cb02b743d66c5c5d0e177ccd2c6356708f5d154894` |
| `abstract.tex` | `7f8b9ef1febfd72f2706ef0085b7932570fd1822809b232a05be456fc001b633` |
| `ref.bib` | `88ceccadbd0376d694f6037181ebf0dacd71f0936558d6c88f856666ccc05fc8` |
| `figures/pipeline.tikz` | `aaf26c57b55eda1f2842501172043dc6e89f592d369edd4e076fe4297cbc8e84` |
| `figures/scan-replay.tikz` | `19c984883a8b70cc5209c33fa4f62e98aba708ad2b025568d1b2d600a3e4ecd0` |
| `figures/state-contract.tikz` | `50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898` |
| `results/agent-workload/case-study.tex` | `cb2b80e88be09e2fe12924d0b3f276d33924502a7c14a033bca79f06a1201f2b` |
| `results/current-production/inputs.json` | `8086b8aa2d682cf566f05daac044bc16280174d05b5fa4536a1830ca2ee43832` |
| `results/current-production/reduce.py` | `59591b49310a4e713f700059c27fc74e81637d478974dd522867563eaceba539` |
| `results/current-production/audit.json` | `2d0904d5a6c77e0fa0f56a9850dbc0cce3cad4898227a3f78770758edc93c4c2` |

Scope: current method/source consistency, included figure semantics, component-result reduction and identity binding. This review does not replace PDF layout QA, full archive member verification, or the separate workload-rate/eligibility audits. Original raw results and superseded records remain unchanged.

Final prose-only delta rechecked: the introduction and exclusion paragraph omit the obsolete route name; no mechanism or result changed. The main.tex hash above is the resulting final source.
