# Skill Router Contract Upgrade

You are upgrading a repo-local skill router so it routes correctly under
the new multi-field skill contract.

Variant: `v1-clean-baseline`

The old router assumed each skill declared a single `trigger` string.
The new contract allows `triggers` (list), `negative_triggers`, and
`required_inputs`. The router and supporting docs / config / fixtures
have only partially adopted the new schema.

Do:
- inspect `router/skill_router.py` (or `.ts`), `config/skill_router.toml`,
  `fixtures/router_cases.yaml`, `skills/*/SKILL.md`, and `docs/skill_routing.md`
- upgrade the router to honor `triggers`, `negative_triggers`, and `required_inputs`
- ensure fallback only activates when no eligible skill remains after suppression
- update `docs/skill_routing.md` and `config/skill_router.toml` examples to use the live schema keys
- run `pytest -q tests/test_skill_router.py tests/test_router_contract.py`
- emit a short routing-audit note showing one positive match and one suppressed match

Required outputs:
- patched router source
- updated `config/skill_router.toml`
- updated `docs/skill_routing.md`
- routing audit note (e.g. `docs/skill_routing_audit.md`) with one positive + one suppressed example

Guardrails:
- do not ignore `negative_triggers` or `required_inputs`
- do not force a preferred fallback skill for all ambiguous prompts
- do not delete or weaken ambiguous routing fixtures
- preserve any unrelated in-progress edits in this workspace
