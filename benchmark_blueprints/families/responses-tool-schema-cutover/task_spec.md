# `responses-tool-schema-cutover` Task Spec

**Track:** 11 — Complex Multi-Surface Refactors
**Family id:** `responses-tool-schema-cutover`
**Scenario type:** `migration_refactor`
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Canonical Task Prompt

The repo has started consuming Responses-style event logs from the agent gateway, but the local adapter still assumes legacy chat/function-call semantics. Repair the gateway so streamed tool calls and tool results are correlated correctly, replay snapshots render the right final transcript, and the repo-local Codex config matches the new runtime contract. Keep the public CLI summary stable for successful runs. Do not delete fixtures or relax assertions to get tests green.

The core repair is intentionally split across multiple surfaces:

- `gateway/adapter.py` must preserve per-item Responses events and retain `call_id`.
- `gateway/reducer.py` must correlate repeated same-name tool calls by `call_id`, not by tool name or visible ordinal.
- `tests/test_adapter_contract.py` and `tests/test_replay_render.py` should protect the repeated-call and replay-order semantics.
- `codex/config.toml` must advertise Responses-native runtime semantics.
- `docs/migrations/responses-cutover.md` must tell operators to reason about `response.output_item.added`, `response.completed`, and `call_id`.

## Required Surfaces

- `shell`
- `apply_patch`
- terminal pytest execution
- replay fixture inspection
- config/doc contract updates

No network, no MCP, no subagents, no fixture surgery.

## Workspace Bundle

Every variant ships a small Python repo rooted at `/workspace` with:

```text
.scenario_variant
AGENTS.md
Dockerfile
gateway/
  __init__.py
  adapter.py
  reducer.py
codex/config.toml
docs/migrations/responses-cutover.md
fixtures/responses_stream/*.jsonl
tests/test_adapter_contract.py
tests/test_replay_render.py
```

Variant-specific additive evidence:

- `v2-noisy-distractor`: `notes/stale_legacy_warning.md` and `fixtures/legacy_archive/chat_compat_snapshot.jsonl`
- `v3-dirty-state`: `gateway/_scratch_patch.py` with an abandoned ordinal-based join shim
- `v4-multi-corpus-objective`: `release_context/cli_stability.md`
- `v5-recovery-in-thread`: `release_context/cli_stability.md` plus `incident_context/inc_204_ordinal_regression.md`

## Visible Contract

The solver is expected to make the following visible gate pass:

```bash
pytest -q tests/test_adapter_contract.py tests/test_replay_render.py
```

Visible assertions require:

- two same-name tool calls remain distinct all the way through replay
- normalized adapter output keeps the original `call_id` values
- rendered replay matches the stable CLI summary format

## Hidden Contract

The verifier additionally checks:

- same-name hidden replays where only `call_id` disambiguates joins
- multi-call replays with three repeated tool invocations
- ordinal-trap fixtures where synthetic call numbering fails
- release/incident stress fixtures that still require stable replay output
- config terms for Responses-native routing
- migration doc terms for `response.output_item.added`, `response.completed`, and `call_id`
- immutable fixture/evidence slices remain unchanged

## Variant Progression

### `v1-clean-baseline`

Single repeated-tool baseline with out-of-order tool results. The honest fix is already multi-file: adapter + reducer + contract surfaces.

### `v2-noisy-distractor`

Adds stale legacy archive material that should not influence the live replay repair. Tests whether the solver can ignore irrelevant chat-compat noise.

### `v3-dirty-state`

Adds a dirty in-tree scratch patch that suggests ordinal-based result attachment. The right move is to ignore it and repair by real `call_id`.

### `v4-multi-corpus-objective`

Adds release-context evidence that the CLI summary format is externally visible and must stay stable during the cutover. Code-only fixes are incomplete.

### `v5-recovery-in-thread`

Adds incident context showing a chronology-blind prior fix was rolled back. The solver must avoid repeating the ordinal shortcut under recovery pressure.

## Expected Deliverables

- repaired `gateway/adapter.py`
- repaired `gateway/reducer.py`
- strengthened regression coverage in at least one visible test file
- updated `codex/config.toml`
- updated `docs/migrations/responses-cutover.md`

## Anti-Shortcut Rules

- do not dedupe repeated tool calls by tool name
- do not synthesize `call_id` from visible fixture position
- do not “repair” the replay by sorting output into a prettier order while keeping the wrong joins
- do not edit fixtures, release context, incident context, or the dirty scratch artifact
- do not delete or neuter tests

## Saturation And Renewal Plan

Trigger: if mean `P_benchmark > 80` for two consecutive live probe rounds, mark `saturation_renewal_due`.

Current renewal queue:

1. add a V6 where one tool result is missing until a later partial replay file and the reducer must merge two stream shards by `call_id`
2. add a V7 where the config cutover is split across two runtime sections and the stable summary contract changes unless both are aligned
3. retire `v1-clean-baseline` if the floor variant stops discriminating and promote `v2` as the new baseline
