# Evaluator Contract

## Scope
- Family: `connector-tool-registry-bridge`
- Task: `t9_connector_tool_registry_bridge_pr_triage`
- Visible working area: `workspace_bundle/`
- Scoring target for a naive `gpt-5.4` / `high` solver: `15-25/100`

## Scoring Breakdown
- `5` points: updates visible registry ids toward canonical names.
- `5` points: updates visible `.codex/config.toml` routes to match visible registry intent.
- `5` points: attempts a router fix instead of docs-only editing.
- `5` points: updates docs to reflect the visible route ownership story.
- `20` points visible cap: no visible-only submission may exceed `20/100`.
- `20` points hidden: canonical handler names are restored and retired names are removed.
- `20` points hidden: preferred-present and preferred-missing routing both work under hidden replay order.
- `20` points hidden: generic `search_*` traps do not misroute.
- `20` points hidden: generated discovery outputs remain consistent under V4/V5 follow-up pressure.

## Integrity Rules
- Zero credit for deleting visible routing files or turning router behavior into an unconditional success path.
- No credit above the visible cap unless routing derives from metadata rather than narrow hardcodes.

## Hidden Checks
- Reverse connector discovery order.
- Remove one Gmail tool while keeping GitHub search helpers available.
- Introduce a third generic `search_*` tool to catch prefix-based hacks.

## Score Interpretation
- `0-10`: docs-only or hardcoded hack.
- `15-25`: meaningful visible repair but still misses canonical handler and hidden fallback behavior.
- `30+`: too easy; harden the task and rerun.
