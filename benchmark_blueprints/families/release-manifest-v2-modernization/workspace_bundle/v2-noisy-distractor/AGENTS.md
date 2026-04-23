# Agent Instructions — `release-manifest-v2-modernization`

Variant: `v2-noisy-distractor`

Repair the release-path modernization end to end:

- align `.github/workflows/release.yml` with the reusable workflow contract
- finish `.github/workflows/reusable_release.yml` so the live contract exposes `artifact_manifest`
- align `release/manifest.v2.toml`
- align `codex/config.toml`
- update `docs/releases/staging_rollout.md`
- write `artifacts/release_smoke_report.json` by running `python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json`

Read-only surfaces:

- `tests/`
- `logs/`
- `config/env/`
- `deploy/`
- `scripts/`
- `release/manifest.toml`
- `release_context/` when present
- `incident_context/` when present

Rules:

- Do not modify any file outside:
  - `.github/workflows/release.yml`
  - `.github/workflows/reusable_release.yml`
  - `release/manifest.v2.toml`
  - `codex/config.toml`
  - `docs/releases/staging_rollout.md`
  - `artifacts/release_smoke_report.json`
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Preserve the comment inside `config/env/staging.toml`.
- Preserve unrelated local edits in docs or env overrides.
- Do not use the network.

Variant note:

- Ignore the archived prod-era note and log. They are read-only stale evidence, not the live contract.
