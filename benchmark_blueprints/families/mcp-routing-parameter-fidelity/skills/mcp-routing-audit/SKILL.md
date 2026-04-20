# MCP Routing Audit

Use this skill when a task requires repairing MCP tool routing and parameter fidelity.

## Workflow
1. Inspect route-selection logic and parameter normalization together.
2. Verify that fallback behavior is not masking wrong-tool dispatch.
3. Preserve provenance logging as a first-class output.
4. Avoid hardcoding the visible fixture path.

## Anti-Patterns
- Fixing only one visible route.
- Normalizing parameters differently from the intended tool contract.
- Leaving provenance logs stale.

## Done Signal
- Correct tools are selected with correct normalized parameters and accurate provenance.
