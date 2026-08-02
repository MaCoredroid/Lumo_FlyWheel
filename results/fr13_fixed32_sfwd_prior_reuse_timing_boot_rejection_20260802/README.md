# K64/root1 SFWD prior-reuse B1 timing boot rejection

Status: **REJECTED_PRE_TASK_CANDIDATE_BOOT**. This is not a matched timing,
TPS, acceptance, quality, or hardware-floor result.

The real SWE-Verified B1 stock arm completed. The candidate arm then failed
before health and before any candidate task began. Its private log directory
contained one `fr13_fixed32_sfwd_prior_reuse.timing.arm` marker and its
source-bound reduced gate, with no competing SFWD route marker. The ingress
route allowlist did not include that timing-arm filename, so it counted zero
routes and failed closed.

Fix commit `7e4c34f5c`:

- admits the prior-reuse timing arm as one valid SFWD route while preserving
  exact-one-route conflict rejection;
- removes every alternate and retry SFWD route marker before publishing the
  validated timing gate and arm; and
- removes an exited boot-failure container only when its immutable launch
  identity still matches, while preserving running or drifted containers.

CPU-only validation passed 117 focused tests, shell parsing, Python
compilation, fatal-source Ruff checks, changed-test Ruff checks, and
`git diff --check`. No Docker container, GPU kernel, service, task, probe, or
timing run was launched for the repair.

The rerun must use a new run root and tag, the same qualified candidate source,
the same reduced gate, the same pinned FA2 binary, K64/root1, and one real
SWE-Verified B1 diagnostic task. The resulting pair remains diagnostic-only;
it cannot satisfy B4 or hardware-floor acceptance.

This package intentionally excludes raw task/model content, requests,
responses, patches, environment values, process or container identifiers,
credentials, and logs.
