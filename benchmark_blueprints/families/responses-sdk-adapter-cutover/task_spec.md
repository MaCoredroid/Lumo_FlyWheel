# `responses-sdk-adapter-cutover` Task Spec

## Task Prompt
Migrate the `incident-handoff` assistant service from a legacy chat-wrapper integration to the Responses event model. Keep tool routing, transcript rendering, and replay behavior correct under streamed multi-event turns. Do not preserve the legacy wrapper on the serving path. Update the repo-local Codex config and the operator migration note so they match the new runtime.

## Workspace Bundle
- `src/incident_handoff/client.py`: current SDK wrapper and request entrypoint.
- `src/incident_handoff/adapter.py`: converts provider output into internal events.
- `src/incident_handoff/replay.py`: transcript serializer and replay reducer.
- `src/incident_handoff/render.py`: operator-facing transcript renderer.
- `transcripts/legacy_chat_turn.json`: stale pre-Responses fixture.
- `transcripts/interleaved_tool_turn.json`: streamed tool-call and tool-result fixture.
- `transcripts/reordered_chunks.json`: assistant chunks arrive out of display order if naively sorted.
- `transcripts/future_event_passthrough.json`: includes an unknown event type that must not corrupt replay.
- `.codex/config.toml`: still points at legacy wire semantics.
- `docs/migrations/responses-cutover.md`: operator note with outdated examples.
- `tests/test_adapter.py`, `tests/test_replay.py`, `tests/test_render.py`: visible test slice.

## Seeded Modernization Breakage
- The SDK dependency was bumped and now returns Responses-style event items instead of legacy message objects.
- `adapter.py` still expects flattened `message.content` and loses tool metadata on multi-event turns.
- `replay.py` rebuilds history from rendered transcript lines instead of raw events, so replay order becomes unstable.
- `.codex/config.toml` still advertises legacy chat wiring.
- The doc examples still describe the deprecated wrapper and wrong transcript shape.

## Required Surfaces
- `shell`
- `apply_patch`
- `terminal_tests`
- JSON fixture inspection
- `.codex/config.toml` editing
- docs update

## Expected Deliverables
- Code patch across the adapter, replay, and render surfaces.
- Updated `.codex/config.toml` setting the repo to the Responses wire path.
- Updated migration note describing the event model and replay expectations.
- Preserved or regenerated transcript fixtures only where semantics actually changed.

## Grader Contract
- Phase 2 visible checks:
  - `pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py`
- Hidden checks:
  - Multi-tool turn preserves event order and role boundaries.
  - Tool routing is driven from event objects, not transcript text.
  - Replay is deterministic across `serialize -> render -> replay -> render`.
  - Unknown future-safe event types are preserved or ignored safely, not dropped destructively.
  - `.codex/config.toml` declares the Responses runtime path.
- Trusted final-state checks:
  - No legacy wrapper import or compatibility shim remains on the serving path.
  - `replay.py` is event-sourced rather than transcript-sourced.
  - Docs mention event ordering and tool-result correlation, not only command syntax.
- Scoring intent:
  - Docs/config changes alone cannot exceed `0.10`.
  - A happy-path adapter shim without replay correctness cannot exceed `0.25`.
  - Scores above `0.50` require hidden replay-order and anti-flattening checks to pass.

## Red-Team Traps
- Flatten all assistant chunks into final text and lose intermediate events.
- Route tools by parsing rendered transcript strings.
- Rebuild replay from transcript lines instead of raw event objects.
- Update fixtures only, without fixing ordering semantics.
- Keep a legacy compatibility shim and translate only the visible happy path.
- Overwrite nearby doc edits in dirty-workspace variants.

## Quality Gate
- GPT-5.4/high adversarial probe judgment before hardening: `under_30_for_naive_solver = no`.
- Probe found an easy partial-credit path via a thin compatibility shim plus docs/config churn.
- Hardening applied in this spec:
  - Added interleaved, reordered, and future-event fixtures.
  - Elevated replay determinism and anti-compatibility-shim checks into the grader contract.
  - Capped docs/config-only credit and kept replay correctness as the score separator.
  - Explicitly banned transcript-sourced replay and text-driven tool routing.
- Final quality judgment after hardening: `under_30_for_naive_solver = yes`.
- Rationale: a naive solver can still patch the visible adapter path, but without true event-sourced replay and temporal semantics it should stay below the 30-point band.
