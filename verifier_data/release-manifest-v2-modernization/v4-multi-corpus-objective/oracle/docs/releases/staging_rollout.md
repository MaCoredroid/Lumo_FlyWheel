# Staging Rollout

Use the reusable workflow with `release/manifest.v2.toml`.
Set `target_environment: staging` on the live workflow call.
Run `python deploy/check_release.py --env staging --emit-json artifacts/release_smoke_report.json` after the dry-run gate.
Keep the reusable workflow and the operator doc aligned; do not fall back to `scripts/legacy_release.py`.
Check the `artifact_manifest output` before the staging smoke step so the dry-run path and the release contract stay aligned.
