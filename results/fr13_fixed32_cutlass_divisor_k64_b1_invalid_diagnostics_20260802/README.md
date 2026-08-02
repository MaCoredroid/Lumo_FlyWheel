# Fixed32 CUTLASS divisor K64 B1 invalid diagnostics

Status: **INVALID_DIAGNOSTICS_ONLY**.

Two attempts were made from the pinned, clean, pushed source commit. The first
reached server health but was terminated by the original foreground execution
session before a real task bracket opened. It produced no timing sample.

The detached retry opened a complete one-task real SWE-Verified metrics bracket,
engaged fixed32 speculative decoding, wrote the aggregate timer sidecars, and
completed the terminal timer flush. The task runner nevertheless rejected the
trace because its final top-level assistant response contained thinking only and
no nonempty user-facing text. Runner metadata and normal end manifests were
therefore not produced.

The aggregate counters in `aggregate_metrics.json` are retained only to guide
kernel prioritization. They are marked `measurement_valid=false`, are not an
acceptance measurement, and cannot support a behavioral, task-success, or
hardware-floor verdict. The configured floor is the B1 maximum of the modeled
weight-read and GEMM-compute lower bounds; it is not a complete full-step
hardware floor.

This package contains no raw task/model content, request, response, patch, log,
environment dump, process/container identifier, binary, PTX, SASS, credential,
or secret.
