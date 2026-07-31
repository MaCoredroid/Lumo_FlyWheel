# FR13 fixed32 TAW real-task arm source gate

This source-only artifact records the host-side lifecycle wiring for the
current-schema fixed32 TAW native-precompute diagnostic.

The SWE orchestrator now publishes the exact marker
`swe_verified:<pinned_task_id>` only after the task's pre-flush and immediately
before agent dispatch. Publication uses a fully written, fsynced, mode-0400
temporary file plus a no-overwrite hard-link publish in the private mode-0700
logs directory. After the post-task flush, the orchestrator verifies the exact
marker and inode identity, then rotates it with another no-overwrite hard link
and removes the live path.

The launcher-to-runner path is explicit; there is no polling. It is restricted
to the pinned single-task B1 diagnostic. Both the lifecycle artifact and the
existing campaign metadata retain `gate_eligible=false` and
`floor_acceptance_eligible=false`.

This artifact contains no GPU execution, Docker execution, performance result,
hardware-floor result, or new SWE-Verified result. A real current-source B1
diagnostic run is still required to exercise the kernel live gate.
