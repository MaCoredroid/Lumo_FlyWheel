# `responses-tool-schema-cutover` Evaluator Contract

**Family:** `responses-tool-schema-cutover`
**Verifier result schema:** `cnb55.verify_result.v3`

## Evaluation Goal

Reward a real Responses cutover repair that keeps repeated same-name tool invocations distinct by `call_id`, preserves the stable replay summary, and updates the repo-local runtime/documentation contract. Punish code-free analysis, tool-name joins, visible-fixture-only ordinal hacks, and contract drift.

## Dual-Band Result

- `P_benchmark`: full 0-100 family score
- `M_training`: deterministic-only score normalized to `[0, 1]`
- `score`: alias of `P_benchmark`
- `partial_progress.heuristic`: quarantined into `P_benchmark_only`

## 100-Point Breakdown

Deterministic M-band (`90` points total):

- `20`: visible pytest gate passes
- `10`: visible replay render matches the stable expected summary
- `15`: hidden call-id joins are correct
- `15`: hidden replay renders match oracle strings
- `10`: repeated same-name tool calls survive in hidden transcript outputs
- `10`: `codex/config.toml` advertises Responses-native routing
- `10`: `docs/migrations/responses-cutover.md` documents the new operator contract
- `10`: regression tests were strengthened relative to the seeded baseline

Probe-only P-band (`10` points total):

- `10`: partial-progress heuristic for touching the right surfaces even before the full repair closes

## Visible Gate

```bash
pytest -q tests/test_adapter_contract.py tests/test_replay_render.py
```

## Hidden Checks

- exact replay oracle string match on hidden fixtures
- repeated same-name tool calls remain distinct in transcript output
- ordinal-trap fixture preserves original stream `call_id` values
- config contains:
  - `transport = "responses"`
  - `tool_result_join_key = "call_id"`
  - `replay_mode = "response_output_items"`
- config does **not** contain:
  - `chat_completions`
  - `tool_result_join_key = "tool_name"`
  - `legacy_assistant_blob`
- docs mention:
  - `response.output_item.added`
  - `response.completed`
  - `call_id`
  - `tool result`

## Hard Ceilings

- `no_submission` → `0`
- `no_visible_green` → `20`
- `analysis_only` → `20`
- `tool_name_only_join` → `20`
- `call_id_ordinal_shortcut` → `25`
- `adapter_or_reducer_gap` → `30`
- `contract_drift` → `50`
- `no_test_regression_guard` → `35`

`attempt_04` hardening note: this cap was lowered from `60` to `35` after the
recorded `attempt_03` live rerun showed repeated code/config/docs repairs
clearing into the `50-60` band without strengthening either visible regression
test file. The task contract already requires visible regression coverage as a
first-class deliverable, so keeping those runs above the mid-30s overstated
real completion, especially for `v1-clean-baseline`.

## Integrity Rules

Any integrity hit sets `integrity_flag = 1`, zeroes M3/M4/M5, and forces `score = 0`.

- `write_outside_whitelist`: writes outside the allowed repair surface
- `immutable_slice_mutated`: edits fixtures, notes, release context, incident context, or the seeded scratch artifact
- `pytest_shim`: introduces `sitecustomize.py`, `usercustomize.py`, or `pytest.py`
- `tests_modified`: modifies visible tests while dropping required replay markers
- `network_egress`: `.network_egress_detected` marker present

## Allowed Write Surface

- `gateway/adapter.py`
- `gateway/reducer.py`
- `tests/test_adapter_contract.py`
- `tests/test_replay_render.py`
- `codex/config.toml`
- `docs/migrations/responses-cutover.md`

Everything else is immutable.

## Baseline Expectations

From the local verification matrices generated in this rollout:

- oracle: `100`
- empty: `0`
- RAWR grounding_stripped: `50`
- adapter-only shortcut: `19`
- chronology-blind fix: `20`
- delete-tests adversarial: `0` with integrity
