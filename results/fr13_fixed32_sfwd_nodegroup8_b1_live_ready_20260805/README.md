# Fixed32 SFWD direct nodegroup8 B1 live readiness

This artifact binds the default-off direct nodegroup8 SFWD candidate to source
commit `89f785fb560c2b7fefb7fa5a61171b5b0316fc4c` and its pushed branch
`codex/sfwd-nodegroup8-b1-live-ready-20260805`.

## Real gate command

```bash
FR13_FIXED32_SFWD_NODEGROUP8_DIRECT=1 \
FR13_FIXED32_SFWD_EMBED_GATE_CTA=0 \
RUNROOT=output/<new-runroot> \
TAG=<unique-tag> \
FORKED_FA2_SO=<pinned-qrow16-so> \
bash scripts/fr13_run_b1_sfwd_conv_postprep_gate.sh
```

Set `FR13_FIXED32_SFWD_EMBED_GATE_CTA=1` for the embedded schedule. Both
schedules use the real `astropy__astropy-12907` SWE-Verified task, Hydra27,
physical32, K64/root1, eager B1, and pinned Qrow16. The gate compares query,
key, speculative value, tree value, `g`, `beta`, and the committer source stage
at all 48 layers while always serving the incumbent tensors.

## Program geometry

| Schedule | Channel programs/request | Gate programs/request | Total programs/request |
|---|---:|---:|---:|
| standalone | 160 | 4 | 164 |
| embedded | 160 | 0 | 160 |

The source manifest changes candidate identity when the selector is enabled,
so an incumbent manifest cannot authenticate this arm. Runtime state also binds
the selector with task, batch, schedule, and source credentials.

## Verification

- 46 focused and existing CPU tests passed; one historical artifact-only test
  was deselected because its old result directory is outside this sparse
  worktree.
- Python compilation and Bash syntax checks passed.
- The pushed source commit equals the branch upstream.
- The pinned Qrow16 binary is one regular file of 299,507,792 bytes with
  SHA-256 `1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86`.
- The incumbent generated kernel function remains SHA-256
  `0384e4947e605846c9ed995bc73fa1252a6f5f815d1bc905685527fbf7f8d8ff`.

No GPU or Docker command was run in this readiness lane. This artifact contains
no latency, TPS, timing, production, or hardware-floor acceptance claim. The
next evidence is the authenticated real B1 byte gate above, followed by timing
only after the byte gate passes.
