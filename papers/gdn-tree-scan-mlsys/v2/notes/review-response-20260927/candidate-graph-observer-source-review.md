# Candidate graph observer: initial bounded source review

2026-09-29. **Two prospective connection blockers; unconnected preparation only.** No production route, gate or experiment was changed. These findings do not block an unrelated native source inventory that does not import this helper.

Reviewed helper `experiments/review-response-20260927/tools/q1_candidate_graph_observer_v1.py`, SHA `e756265aa4c6fedddfe4d59308f7b589017b2bddea00e5b7037a1d2399981893`; tests SHA `e8fbd8a818e10fe8f2d13faed03dad2b8734f37e4c57a94f905569ac4e3be4a6`. Initial bytes, `SOURCE.json`, test output and independent `ADVERSARIAL.json` are preserved under `p0/monitor/review-response-20260927/candidate-graph-observer-independent/`.

## F1. Scalar-only positions and ID-only kwargs do not cover the pinned model path

Constructor line 38 insists on `_dg['pos'].shape == (1,)`; capture lines 56–59 reject embeddings and require every position operand to be one-dimensional. Pinned `workload-plan/inspections/cpu-lumotree-source-20260929T012000Z/result/sources/v1/spec_decode/eagle.py:301–315,5194–5250` expressly preserves the draft model's M-RoPE axes, and can supply `input_ids=None` with actual `inputs_embeds`. The pinned `model_executor/models/qwen3_5_mtp.py:401–426` implements that multimodal embedding call, so the proposer load fallback at `eagle.py:7555–7566` cannot be assumed to disable the route. Text-only task content does not prove one-dimensional positions or ID-only kwargs.

Independent real CPU controls show the helper refuses a valid three-axis position shape at construction and poisons itself on supported embedding kwargs. This is a source-compatibility gap, not a claim that a new candidate run has observed either form.

Minimal repair: preserve the actual bound `_dg['pos']` shape/axes; allocate corresponding per-loop observation homes before capture; validate every relevant axis. Capture the real embedding operand when the actual kwargs use embeddings, along with the source-bound prepared IDs used to create it. Pass those IDs from the actual preparation seam, not from an embedding inversion or a recomputed surrogate. Keep actual model kwargs unchanged; do not force the route into the observer's former assumptions.

## F2. One loop-zero epoch does not establish four fresh write observations

Only `capture_after(index==0)` increments `epoch` (72–77); `export` accepts scalar `old+1` (91). The Python `before/after == [0,1,2,3]` lists prove all callbacks were captured once, not that every captured segment executed during the current replay. If the loop-zero segment executes while later rows remain from a prior replay at the same logical positions, the exported assertion `later_writes_outside_first_prefix=True` can pass with stale evidence.

Independent CPU reproduction: create and export one valid populated witness, begin a new owner, advance only the scalar epoch once, leave the four stored positions/slots/lengths unchanged, and export again. It accepts and emits `later_writes_outside_first_prefix=True`. This simulates the observable effect of a partial replay; no CUDA replay was executed. Parent independently identified the same weakness.

Minimal repair: four device epochs, increment each within its own existing loop's captured operation stream, and require every epoch to advance exactly once between begin/export. Reject missing/partial/double loop execution even when the stale values happen to match the new expected coordinates. Retain external binding to the actual graph/signature, current request/proposal owner and replay lifecycle; epochs do not replace those obligations.

## Retained sound behavior and limits

All 12 supplied tests passed independently with `/usr/bin/python3`, Torch 2.8.0, CPU tensors, no GPU access and one CPU thread. The helper binds exact static spine/wide storage, refuses fresh surrogate selector tensors, preserves separate sampling and continuation hidden, copies the original full BF16 row without recomputing a head/selector, checks sequential logical writes against a unique physical block map, and fail-stops on capture/begin/export errors. These useful checks should remain unchanged.

The eventual caller still must install diagnostic copies before the first graph capture (including bootstrap), pass actual model/selector operands, begin outside capture before the genuine replay, and export outside capture after it under the exact graph/owner/registry binding. No helper result alone proves that arbitrary supplied kwargs came from the live model. Raw stored scores still require the separately frozen categorical reducer; this helper does not independently certify selector correctness against the logits. No CUDA behavior, collection readiness, numerical equivalence or qualification is established by these CPU controls.

Repair review should remain limited to the two findings and their connected tensor controls. No production flag change or broader experimental axis is needed.
