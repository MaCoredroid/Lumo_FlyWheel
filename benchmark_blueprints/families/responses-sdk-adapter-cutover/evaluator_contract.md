# `responses-sdk-adapter-cutover` Evaluator Contract

## Evaluation Goal
Score whether a solver actually migrates a small assistant service to Responses-style event semantics, instead of patching only happy-path text output.

## Visible Checks
- `pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py`

## Trusted Checks
- No legacy wrapper import remains on the serving path.
- Replay logic operates on event objects, not rendered transcript lines.
- Config declares the Responses wire path.
- Docs describe event ordering and tool-result correlation.

## Hidden Checks
- Interleaved tool-call and tool-result ordering remains stable.
- Unknown future-safe event types do not corrupt replay.
- Transcript rendering preserves role boundaries for multi-event assistant turns.

## 100-Point Breakdown
- `10`: visible tests pass without editing benchmark-owned tests.
- `5`: adapter handles the visible message-block shape.
- `5`: renderer keeps visible tool-result correlation.
- `20`: hidden replay pack proves event-sourced deterministic replay on withheld fixtures.
- `20`: hidden interleaved-order pack proves multi-event turn stability.
- `20`: hidden future-event pack proves safe preservation without transcript flattening.
- `10`: trusted anti-legacy-shim checks pass.
- `5`: `codex/config.toml` is aligned with the runtime contract.
- `5`: migration doc is accurate and specific.

## Caps
- If replay is transcript-sourced, total score is capped at `25`.
- If a legacy compatibility shim remains on the live serving path, total score is capped at `35`.
- If the solver edits benchmark-owned visible tests, total score is capped at `20`.
- If only the visible fixture pack is proven and no withheld replay/interleaving pack is exercised, total score is capped at `20`.
- If only docs/config are updated and code semantics are unchanged, total score is capped at `10`.
- If visible tests are bypassed or fixtures are edited without semantic fixes, total score is capped at `0`.

## Passing Interpretation
- Target benchmark-design outcome: a naive GPT-5.4/high solver should land around `20/100`.
- Scores above `30` indicate the family is still too shallow and must be hardened before freeze.
