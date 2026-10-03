# Continuous MTP graph input position — semantic repair review

**PASS for the one-line source repair `9a0d6249c62566a64ab35f39684b2c32bf002ab6dc093f457997bd92a8ece89e`**, with test source `9d2fc81066a5b6f413fc85762988140626dd11e159081d8cb1b38a84d0de9522`. Date: 2026-09-29. No GPU, model, launch, runtime qualification, or scientific observation was added.

The parent-found off-by-one is real. The prior `04e7d11e…` callback/ownership review and seal remain immutable historical records. **Their fixture mirrored the incorrect position, so that PASS must not be reused as evidence of correct MTP position semantics.** This note supersedes only that semantic boundary; its prior ownership/raw-integrity findings and repairs remain recorded separately.

## Source-grounded position contract

Let `E` be the current frame's target publication extent (a token count, not the last materialized position). The first MTP forward consumes the shifted pending token together with the selected last-accepted target hidden at **position `E−1`**. Its materialized MTP prefix has extent `E`; the first follow-up's input position is `E`.

| Frame | Independent frozen/source definition of first MTP position |
|---|---|
| Interior cycle | `cycle.materialized_positions[-1] = cycle.z_position−1 = E−1` |
| Terminal root-only forward | final preceding cycle's `z_position = terminal root_position = E−1` |

Evidence (paths relative to the campaign unless otherwise stated):

- Pinned native `workload-plan/inspections/cpu-vllm-reference-source-20260929T004505Z/result/sources/v1/spec_decode/eagle.py`, SHA `2c569fd2b60e81b897360071fe433b259575c74b4c7a2a7832560abe2c7cc4b3`: `set_inputs_first_pass` at 646–678 shifts **IDs**, then copies **unchanged target positions** and hidden. The selected position/hidden is read at 488–491. `q1_native_mtp_program_v1.py:20–43` preserves the pinned native first pass and selected-position/follow-up operations; it does not add a root-position increment.
- Served generated `identity/generated_source/probe-20260928T035847Z/logs/generated/eagle.patched.py:6745–6777` has the same shift-ID/keep-position operations; 1027–1031 selects the position at the accepted hidden index. This independently establishes the candidate's intended position instead of deriving it from the new test.
- `tools/q1_native_mtp_history_v2.py:61–80` records actual consecutive target positions; `q1_native_mtp_history_sequences_v1.py:76–113` sends those exact target positions/hidden plus the pending token to the native first pass and records the position input. A single target checkpoint ending at extent `E` therefore supplies position `E−1`.
- Accepted `tools/q1_candidate_hooks_joint_v1.py:148–162` already uses `z_step−1` before the pending token and `z_step` after its root-only consumption. The continuous schedule maps those respectively to current interior and terminal `E−1` (`q1_continuous_schedule_v1.py:32–45,65–75`). The unchanged phase contract requires selected last-accepted hidden plus shifted pending token, and terminal actual-O2 shifted input; it does not authorize a position shift.

The repaired graph source changes only line 110 to `position = frame.item.publication_extent - 1`. The existing root snapshot `input_position+1`, graph first-follow position, and later positions consequently align with `E`, `E`, and subsequent positions. No numerical threshold, phase population, fixture token, or deployed setting changes.

## Independent CPU check

**8/8 controls pass** in 15.282 seconds (`CONTROLS.json`, `CONTROLS.log`). They include the four existing callback checks plus focused semantic controls. The source-statement check extracts the unchanged `set_inputs_first_pass` and `_set_positions` methods from both pinned native and served source, removes only Python type annotations, and executes their actual tensor-input operations on CPU. Across all **78 calibration frame descriptions** (54 cycle descriptions plus 24 terminal descriptions), the selected candidate token, position and hidden equal the corresponding one-row native inputs and independently match the frozen materialized-position fields. These counts describe input validation, not runtime observations or an expanded experiment.

A connected three-cycle-plus-terminal callback path checks first-input position and root-prefix/follow extent against the frozen fields. Powered interior and terminal controls substitute the old `E` input position and are refused with a process failure latch. Existing graph epochs and event counts remain unchanged. Model results, KV snapshots and graph replay remain the disclosed synthetic CPU fixture; no CUDA replay or numerical equivalence is claimed.

The initial semantic test run stopped because the reviewer's minimal AST fixture omitted `vllm_config.model_config.uses_mrope`; source and failed result are retained in `attempt1/`. Adding that existing source-required fixture field allowed the unchanged extracted operations to run. No production source was edited by the reviewer. Local Python 3.9 fixture-scoped compatibility shims are unchanged from the previous review.

`SOURCES.json` and `REVIEW-SEAL.json` bind the complete evidence and exact source hashes. The previous `04e7` review files were not overwritten. Future outer-runtime/job/raw-audit admission remains separate.
