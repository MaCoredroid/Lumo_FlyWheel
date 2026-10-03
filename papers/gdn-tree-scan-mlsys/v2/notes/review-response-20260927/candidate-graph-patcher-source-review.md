# Candidate graph patch preparation: bounded source review

2026-09-29. **Nine-anchor source preparation passes; one physical-shape compatibility boundary remains unresolved before connection/qualification.** No production installation, engine import, GPU operation, or gate change was performed. The earlier observer closure applies to the reviewed one-physical-row tensor seam and is not evidence that the runtime dispatcher always selects that extent.

## Exact reviewed bytes

All paths are paper-v2-relative unless noted.

| File | SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/q1_candidate_graph_patcher_v1.py` | `6161a71f2b79855f956e1eed6ae956f34a7809cec0788cf8f5b6957f845f79b5` |
| `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py` | `323938955da3c528a0f45710fd22328e1bf5dd50ec0632eb0c1d86c97c8b273c` |
| `experiments/review-response-20260927/tools/q1_patch_candidate_v9.py` | `93894ce2758c8283f60dd90f64b364fd1efd6f310ab63ff3f5e25814928ab202` |
| Pinned `workload-plan/inspections/cpu-lumotree-source-20260929T012000Z/result/sources/v1/spec_decode/eagle.py` under the campaign | `8b22dd4fb15044813c8da28afa43b90f178709f7ed3111b3925b28abea717d62` |
| `p0/monitor/review-response-20260927/candidate-graph-observer-cpu/PATCH-PREPARATION.json` | `f1e394a69f3a7ee887f9de75ca8766e0f47fdfae93a66a0d6baa06c72b87ea97` |

Independent immutable source copies and receipts are in `p0/monitor/review-response-20260927/candidate-graph-patcher-independent/`. `SOURCE.json` hashes to `76f531ecaa4ea304ca4a6ed1e8594d75e87968236a73f955cb28011446de84fc`; `AUDIT.json` hashes to `23516b00b2afabbf61097ad86a9f1ee9b0050e2a3227cbc3fb1c6080d7aa119c`.

## Physical rows: unsupported extent, not a measured dispatcher conclusion

Pinned Eagle 4881–4883 calls `_determine_batch_execution_and_padding(batch_size)`. Its implementation at 7943–7947 takes `input_batch_size` from `batch_desc.num_tokens`, with an additional DP coordination branch at 7953–7978. The logical `batch_size` remains distinct. In the graph allocation, `_dg.hidden` is explicitly sliced to logical rows (5104), and `_dg.pos` copies logical positions (5125). The actual loop model receives `inputs_embeds[:input_batch_size]` (5242), `_get_positions(input_batch_size)` (5249), and `hidden_states[:input_batch_size]` (5253). The nonembedding path similarly passes padded input IDs (5244). These are not necessarily the one-row graph-output surfaces.

The patcher correctly forwards the actual embedding prototype and model kwargs. However, observer lines 34–46 require a one-row embedding prototype and allocate input hidden/position homes from the logical graph buffers; lines 67–69 require the corresponding exact one-row shapes. Thus any resolved physical extent greater than one refuses during prepare/capture. Three independent CPU seam controls demonstrate this: embedding prototypes with physical rows 2 and 32 refuse at preparation; a two-row actual ID/position/hidden model kwargs bundle refuses at capture. Those values are hypothetical extents, not assertions about a GPU run. Receipt `PHYSICAL-SHAPE-CONTROLS.json` SHA-256: `c2e3d9cb9eae73a4ae1f6f9f82ee44c474c510d50369e05b772e0fa085adbfb5`.

Existing candidate run `experiments/review-response-20260927/runs/q1-candidate-stage1/q1-candidate-stage1-20260929T092706Z/engine.log` (SHA-256 `9277743c9e11f146fd1809102a3dc3504472cb0b24b535538e306a3e85dd8537`) records initial capture sizes including 1 and DP=1 at line 54, but later reports only two PIECEWISE graph keys with largest=64 at line 126 and two captured PIECEWISE graphs. The startup list precedes runner resolution. Pinned runner 9777–9801 resolves graph sizes and then initializes the drafter dispatcher. Neither logical `captured bs=1` nor the initial list proves the physical dispatcher result. The exact dispatcher and compilation-resolution source were not present in the local inspected source set used for this review; I do not infer a definite padded value from the count of graph keys.

**Minimum resolution:** source-bind the actual resolved dispatch/shape under the unchanged configuration, or extend the diagnostic input prototypes to retain complete actual padded kwargs while keeping logical active B1, logical write rows, and one-row sampled/continuation outputs distinct. Do not slice away padded model inputs to make this observer pass, change graph settings, or declare the served route unsupported merely from logical B1. This is a connected-shape prerequisite, not a new experiment axis or a numerical-policy change.

## Placement and tuple roles

The independent reconstruction applied unchanged v9 first and reproduced all nine exact anchor positions and prepared bytes. Prehash is `ff6c97fe5f3db9ce14361d3960d2c6716516ccb0fbdb41f37a29e7445a7330a0`; posthash is `eb901886d6f1b031cf91f3016a0261dae203ee877ffe4e2183ac3813650268b1`. Compilation succeeds. Removing only the inserted observer expression statements and the new preembedding-token alias yields an AST exactly equal to the v9 input. All 20 narrow refusal controls passed: each of nine anchors missing or duplicated, missing v9 marker, and double patching.

- Preparation precedes graph creation/synchronization/capture (original Eagle 5144–5157). The preembedding alias is taken from the actual token variable before `model.embed_input_ids` and before it is replaced by `None` (5237–5242).
- The before-hook is inside the existing forward context immediately before the actual model call (5263). The after-hook follows the actual fused selector and receives its existing static output buffers (5317–5324), not a recomputed selection.
- Tuple handling is correct: the model returns sampling hidden as `last_hidden_states` and continuation hidden as `hidden_states` (5269–5275). The head uses `last_hidden_states[:batch_size]` (5290–5292); the hook passes this surface and the separate sliced continuation value. `model_returns_tuple` excludes method `mtp` (6924–6925), which is the resolved method in the saved startup receipt. The generic tuple branch is nevertheless kept correctly separated.
- Existing replay begin/export surrounds the actual segment replay loop (5026–5061); initial begin occurs after capture ends and segment-count checks, immediately before replay (5459–5489), and export follows replay/storage installation (5526). There is no inference that capture itself executed a model call. Per-loop epoch evidence still depends on connecting the accepted helper inside the captured operations.

Actual `on_joint_*` callbacks remain unwired in v9, as the preparation receipt explicitly states. The successor must bind exact source hashes, graph/segment identity, request/event ownership, actual embedding provenance, registry, and categorical reduction, and preserve refusal behavior. This pure patcher's anchor checks and emitted hashes are preparation evidence, not a standalone runtime source-authority gate. No other material placement or sampling/continuation-role defect was found in this bounded review.
