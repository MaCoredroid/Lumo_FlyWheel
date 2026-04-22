# `responses-sdk-adapter-cutover` Task Spec

**Track:** 06 — Integration Cutovers & Runtime Migrations
**Family id:** `responses-sdk-adapter-cutover`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt (canonical)

Migrate the `incident-handoff` assistant service from a legacy chat-wrapper integration to the Responses event model. Keep tool routing, transcript rendering, and replay behavior correct under streamed multi-event turns. Do not preserve the legacy wrapper on the serving path. Update the repo-local Codex config and the operator migration note so they match the new runtime.

The workspace is a small Python service with broken migration code. The agent is expected to edit code and docs directly in-place and verify the result with the visible pytest slice. Hidden checks then stress replay determinism, event ordering, and future-safe handling beyond the visible happy path.

## Required Surfaces

- `shell`
- `apply_patch`
- `terminal_tests`
- JSON fixture inspection
- `.codex/config.toml` editing
- docs update

No network, browser, or subagents are needed for the benchmark itself.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
.codex/config.toml
docs/migrations/responses-cutover.md
src/incident_handoff/client.py
src/incident_handoff/adapter.py
src/incident_handoff/replay.py
src/incident_handoff/render.py
tests/test_adapter.py
tests/test_replay.py
tests/test_render.py
transcripts/*.json
release_context/*            # V4+ only
incident_context/*           # V5 only
```

## Seeded Modernization Breakage

The starting workspace is intentionally broken in the same family-wide ways:

- `client.py` still advertises the legacy `chat_completions` serving path.
- `adapter.py` assumes flattened chat-message content and loses event metadata.
- `replay.py` serializes a lossy textual form, so replay is not event-faithful.
- `render.py` drops tool-result call ids.
- `.codex/config.toml` still declares legacy wire semantics.
- `docs/migrations/responses-cutover.md` still tells operators to preserve the wrapper.

## Variant Progression

### `v1-clean-baseline`

Minimal cutover. The only visible transcript is an interleaved assistant/tool turn. The solver must normalize Responses items into event objects, preserve tool call ids, fix rendering, and move config/docs to the Responses path.

- Primary stress: basic migration correctness.
- Expected capable-model mean after hardening: about `28`.

### `v2-noisy-distractor`

Adds a stale `legacy_chat_turn.json` fixture and a multi-block assistant message fixture. The stale file is present as a read-only distractor; the correct fix is to modernize the live path, not to build a compatibility shim around the legacy shape.

- Primary stress: avoiding legacy-wrapper anchoring.
- New ceiling: `compatibility_shim_left_live`.
- Expected capable-model mean: about `23`.

### `v3-dirty-state`

Adds `transcripts/reordered_chunks.json` plus a partial-cutover note describing an abandoned replay shim. A solver that patches only the visible fixture still fails replay determinism under out-of-order event chunks.

- Primary stress: event ordering and replay determinism.
- New ceiling: `reordered_chunk_instability`.
- Expected capable-model mean: about `18`.

### `v4-multi-corpus-objective`

Adds `release_context/` notes showing the operational objective shifted from merely “render a readable transcript” to “preserve event fidelity for postmortem replay and audit”. The correct fix is now explicitly event-sourced replay, not a nicer string renderer.

- Primary stress: objective shift under additional corpus evidence.
- New ceiling: `objective_drift_to_render_only`.
- Expected capable-model mean: about `15`.

### `v5-recovery-in-thread`

Adds `incident_context/` and `transcripts/future_event_passthrough.json` after a rollback caused by unknown event handling during an earlier cutover. The solver must preserve or safely ignore unknown event types without corrupting order, and the migration note must acknowledge the incident-driven recovery semantics.

- Primary stress: recovery-aware correctness under future event types.
- New ceiling: `future_event_corruption`.
- Expected capable-model mean: about `10`.

### Ladder Monotonicity Target

Calibration target for GPT-5.4/high after hardening:

- `v1-clean-baseline`: `28`
- `v2-noisy-distractor`: `23`
- `v3-dirty-state`: `18`
- `v4-multi-corpus-objective`: `15`
- `v5-recovery-in-thread`: `10`

Family mean target: `18.8`, within the Layer A freeze window `[15, 25]`.

## Expected Deliverables

- Code patch across `client.py`, `adapter.py`, `replay.py`, and `render.py` as needed.
- Updated `.codex/config.toml` setting the repo to the Responses wire path.
- Updated migration note describing event ordering, tool-result correlation, and future-event safety.
- Preserved transcript fixtures. Editing benchmark-owned tests or transcripts is forbidden.

## Visible Checks

```bash
pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py
```

The visible slice only checks:

- single-message Responses content normalization,
- tool-call id preservation on replay,
- tool-result call id rendering.

Visible success is intentionally insufficient for a passing score.

## Hidden Checks

Hidden checks live under `verifier_data/responses-sdk-adapter-cutover/<variant>/hidden_tests/` and are deterministic code-backed checks only. No LLM judge is used anywhere in this family.

Hidden surfaces by variant:

- `v1`: interleaved tool-call and tool-result ordering, basic anti-flattening.
- `v2`: multi-block assistant message normalization, no legacy live-path fallback.
- `v3`: reordered chunk stability and deterministic replay of the canonical event stream.
- `v4`: replay remains event-sourced rather than transcript-sourced under release-context pressure.
- `v5`: unknown future event types are preserved or ignored safely without corrupting replay order, and docs acknowledge incident-driven recovery.

## Trusted Final-State Checks

- No edits to `tests/`, `transcripts/`, `release_context/`, or `incident_context/`.
- No `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim.
- No files written outside the allowed solver write surface:
  - `src/incident_handoff/client.py`
  - `src/incident_handoff/adapter.py`
  - `src/incident_handoff/replay.py`
  - `src/incident_handoff/render.py`
  - `.codex/config.toml`
  - `docs/migrations/responses-cutover.md`
- No live-path `chat_completions` or legacy-wrapper marker remains in `client.py` or `.codex/config.toml`.

## Partial-Credit Ceilings

- `visible_only_cutover` ≤ `20`
  Trigger: visible pytest passes but fewer than two hidden behavior packs pass.
- `flattened_multi_event_turn` ≤ `25`
  Trigger: assistant message content is still flattened or nested incorrectly under multi-block Responses payloads.
- `compatibility_shim_left_live` ≤ `30`
  Trigger: live path still advertises `chat_completions`, legacy wrapper semantics, or a legacy-only client contract.
- `reordered_chunk_instability` ≤ `35`
  Trigger: replay result depends on chunk arrival order rather than event sequence.
- `objective_drift_to_render_only` ≤ `40`
  Trigger: rendering looks correct but replay is still not event-sourced in V4+.
- `future_event_corruption` ≤ `30`
  Trigger: unknown event types are destructively dropped or reorder neighboring events in V5.

## Red-Team Traps

1. Edit visible tests instead of fixing the migration.
2. Add a local pytest shim.
3. Patch only the visible fixture path and ignore multi-block / reordered / future-event hidden paths.
4. Reconstruct replay from lossy rendered text rather than event objects.
5. Keep a legacy compatibility shim on the serving path.

## Saturation And Renewal Plan

Per Layer B readiness, this family is renewed when `mean P_benchmark > 80` for two consecutive probe rounds.

Renewal queue:

1. Add a V6 with mid-stream event cancellation and restarted tool call ids.
2. Add a V7 with assistant reasoning blocks interleaved around tool results.
3. Retire V1 once it becomes purely mechanical and promote V2 as the new floor.
