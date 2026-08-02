# Verification

- Pinned source, clean tracked worktree, and upstream equality: PASS at both launches.
- Prelaunch Docker and GPU ownership gates: PASS at both launches.
- Prelaunch free-space gate of at least 2 GiB: PASS at both launches.
- First attempt real-task metrics bracket: ABSENT; no timing result.
- Detached retry authenticated real-task pre/post bracket: PRESENT.
- Detached retry fixed32 engagement ratio: PASS at 31 drafts per event.
- Detached retry timer sidecars and terminal flush: PRESENT.
- Detached retry final user-facing assistant text: ABSENT.
- Detached retry runner metadata and task provenance: FAIL CLOSED.
- Aggregate reducer schema and finite-positive timer fields: PASS.
- Measurement validity and acceptance eligibility: FALSE.
- Post-run Docker and GPU ownership: CLEAN.
- Raw task, model, request, response, environment, log, and identity data: EXCLUDED.
