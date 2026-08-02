# Verification

- Deterministic source-manifest generation passed at exact pushed commit
  `4f1a53eef1bffb083051a292e94c2eddf631d6ea` and bound 15 files.
- The host-only readiness command accepted the canonical stock FA2 binary:
  299,183,936 bytes and SHA-256
  `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
- The host preflight requires HEAD to equal its upstream before the live runner
  may proceed and rejects a dirty tracked worktree or drifted runtime asset.
- The live validator now requires the exact host-readiness record as well as
  equal launch/end source manifests.
- The byte contract remains B1-only, K64/root1, stock-served, default-off, and
  non-acceptance. It requires exact `conv_out` and `commit_source_stage` bytes
  for 48 distinct layer identities.
- Focused SFWD launcher, descriptorless-kernel, descriptor, gate, and readiness
  tests: `31 passed`.
- Ruff, Python compilation, shell syntax validation, and `git diff --check`:
  passed.
- The branch and upstream were equal before artifact creation.
- GPU or Docker use: none. Live correctness and performance remain unmeasured.
