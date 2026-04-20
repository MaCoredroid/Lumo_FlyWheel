# MCP Routing / Parameter Fidelity

## Task Identity
- `task_id`: `mcp-routing-parameter-fidelity/router-contract-repair`
- `family_id`: `mcp-routing-parameter-fidelity`
- `scenario_type`: `stateful_tool_execution`

## Task Prompt
Repair a local MCP router so requests are dispatched to the correct tool with the correct normalized parameters. The visible failure is a wrong-tool fallback, but hidden cases also probe parameter coercion, ambiguity handling, and provenance logging. Keep the routing table readable and preserve unrelated local tool registrations.

## Workspace Bundle
- `gateway/router.py`
- `gateway/tool_registry.py`
- `fixtures/mcp_calls/`
- `tests/test_router.py`, `tests/test_registry.py`
- `docs/mcp_routing.md`

## Seeded Tooling Drift
- One fix path makes visible dispatch succeed by hardcoding the happy path.
- Parameter normalization differs across two otherwise similar tools.
- Provenance logs still reflect the stale routing decision after successful dispatch.

## Required Surfaces
- `shell`
- `apply_patch`
- MCP routing reasoning
- terminal tests

## Expected Deliverables
- Router or registry patch.
- Updated routing note.
- Preserved provenance logging.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_router.py tests/test_registry.py`
- Hidden checks:
  - Correct tool selection on withheld calls.
  - Parameter normalization fidelity.
  - Provenance logs reflect the actual route taken.

## Quality Gate
- Target naive score: `20/100`.
