# Verification

- Sanitized artifact census showed launch manifests but no end manifest, task
  output, deploy-speed artifact, or timing result.
- The failure chain terminated in the K64 block-map validation called by the
  container-side qualified binary installer.
- The host side had already issued the qualified production sidecar; teardown
  flush, ledger, and incarnation messages followed the entrypoint failure and
  are not independent causes.
- The exact original sidecar and candidate revalidated successfully from a
  non-repository working directory with the explicit committed block map.
- Focused B1/B4 installer, K64 profile, and launcher suites: `55 passed`.
- Ruff, Python compilation, launcher shell syntax, and `git diff --check`:
  passed.
- Source commit and upstream matched before artifact creation.
- The one stopped failed container exactly matched the recorded identity,
  terminal state, and zero restart count before removal. Aggregate Docker
  object count after cleanup: zero.
- No GPU run was launched. Real B1 full-step timing remains unmeasured.
