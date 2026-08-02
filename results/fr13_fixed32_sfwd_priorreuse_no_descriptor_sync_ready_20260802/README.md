# Descriptorless SFWD host-sync removal readiness

Status: **READY_NOT_EXECUTED**.

Source commit `078bd0f23bfa8ecb4faae4a72f16553c0339a8a1` removes the
CUDA `source_flat` descriptor from the packed x-gather candidate launcher.
The prior wrapper copied all 128 descriptor values to the host once per layer
launch. The new wrapper passes the already-host tree parent list and requires
an exact match with the compiled 32-row topology before any CUDA work.

The topology input is restricted to a Python `list` or `tuple` containing
exact Python `int` values. This prevents a future CUDA tensor from introducing
an implicit synchronization through scalar conversion. Gate records, the live
pass, validator output, and launcher metadata bind the no-device-read contract.

This change does not alter the Triton kernel body or establish a runtime
speedup. The packed x-gather candidate still requires a fresh real
SWE-Verified K64/root1 B1 byte gate on both output surfaces and all 48 layers,
followed by matched real-task timing.

This reduced package excludes raw logs, model/task content, requests,
responses, patches, environment values, process/container identifiers, and
secrets.
