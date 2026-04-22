# Release Gate

- The next customer rollout cannot proceed until replayed streams preserve watermark parity.
- Current blocker metric: watermark mismatch in 6 of 20 release-gate replay runs.
- Snapshot replay drift is still noted as background risk, but it is not the named gate for this launch.

This context intentionally shifts the objective relative to earlier variants.
