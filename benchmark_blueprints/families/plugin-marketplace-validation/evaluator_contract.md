# Evaluator Contract

## Scope
- Family: `plugin-marketplace-validation`
- Task: `t9_plugin_marketplace_validation_release_sync`
- Visible working area: `workspace_bundle/`
- Scoring target for a naive `gpt-5.4` / `high` solver: `15-25/100`

## Scoring Breakdown
- `5` points: updates `.codex-plugin/plugin.json` to a plausible schema-v2 style shape and stops using the obvious legacy entrypoint form.
- `5` points: aligns `fixtures/marketplace/sample_install.toml` with the visible manifest intent.
- `5` points: updates `docs/plugin_release.md` so the visible installation path is coherent.
- `5` points: avoids tampering with visible harness files under `scripts/` and does not delete or neuter visible checks.
- `20` points visible cap: no visible-only submission may exceed `20/100`.
- `15` points hidden: exact canonical slug and startup command agree across manifest, docs, and sample install config.
- `15` points hidden: bundle build metadata and icon path remain coherent after packaging.
- `15` points hidden: stale tool aliases are removed rather than tolerated alongside canonical ids.
- `15` points hidden: trusted validator and smoke checks pass without edits to validator or visible tests.
- `20` points hidden: release note and follow-up artifacts remain consistent under V4 and V5 follow-up pressure.

## Integrity Rules
- Zero credit for modifying trusted grading harness files:
  - `scripts/validate_bundle.py`
  - `scripts/build_bundle.py`
- Zero credit for replacing the visible report with a generic stub or converting the visible harness into unconditional success.
- No credit above the visible cap unless the canonical slug is exact and unique.

## Hidden Checks
- Reject dual-documentation of old and new slug or command forms.
- Reject local-shim-only fixes that do not preserve packaged module entrypoint behavior.
- Reject alias-preserving tool lists that still expose retired ids.

## Score Interpretation
- `0-10`: incoherent or harness-tampering attempt.
- `15-25`: meaningful visible repair but still misses trusted marketplace invariants.
- `30+`: too easy; harden the task and rerun.
