
# Sandbox Policy CI Drift

Repair the repo so local CI, the workflow preview generator, and the
checked-in workflow all use the same Codex sandbox and approval-policy
contract.

Required outcomes:

- Canonical active tokens are `workspace_write` and `on_request`.
- The deprecated input spelling `workspace-write` must still parse from
  config files for compatibility.
- Emitted preview JSON and the workflow file must stay canonical-only.
- Update the release note so operators understand the compatibility rule.

Editable surfaces:

- `codex/policy.py`
- `codex/config.py`
- `scripts/render_workflow_preview.py`
- `scripts/run_ci.py`
- `.github/workflows/ci.yml`
- `codex/config.toml`
- `docs/releases/policy-rename.md`

Immutable surfaces:

- `tests/*`
- `tests/fixtures/*`
- `docs/archive/*`
- `repo_evidence/*`
- `release_context/*`
- `incident_context/*`
- `Makefile`
- `Dockerfile`
- `AGENTS.md`

Final validation command:

- `make ci`

Variant details live in `.scenario_variant`.
