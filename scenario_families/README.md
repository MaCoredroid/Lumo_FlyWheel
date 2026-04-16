# Codex-Long Initial Authored Pack

This repo now carries an initial real authored scenario pack for the Codex-Long
framework. The pack is intentionally below the signed-off freeze threshold from
LLD-13, so it does not include `split_assignment.yaml` or
`benchmark_manifest.lock`.

Included families:

- `report-cli-markdown-evolution` (`feature_evolution`)
- `normalizer-api-migration` (`migration_refactor`)
- `ci-config-coverage-drift` (`build_ci_breakage`)
- `alert-dedupe-investigation` (`investigate_then_fix`)
- `owner-field-cross-layer` (`cross_layer_changes`)

Each family contains three concrete authored variants with:

- a broken repo and visible `AGENTS.md`
- a Dockerfile with a build-time smoke failure
- a verifier tree rooted at `verifiers/<family_id>/`
- milestone helpers under `verifiers/<family_id>/milestones/`
- verifier data and immutable test checksum manifests under `verifier_data/<family_id>/`

Regenerate the pack with:

```bash
.venv/bin/python scripts/generate_initial_codex_long_assets.py
```

Validate the pack with:

```bash
.venv/bin/python scripts/validate_codex_long_assets.py
```
