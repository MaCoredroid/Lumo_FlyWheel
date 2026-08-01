# FR13 fixed32 SFWD B1 eager preflight

Status: **ready for one real SWE-Verified B1 byte gate; not launched**.

Code commit `e29f73d98e2814e80ef42440d5dea366add271f8` makes the
source-exact `fixed32_sfwd_state_fusion_v1` shadow candidate runnable under the
current eager lifecycle. The route remains physical-32, full vocabulary, and
stock-served. Candidate bytes are compared across `conv_out` and
`commit_source_stage` for all 48 layers, while incumbent bytes remain served.

The task contract is the pinned one-task SWE-Verified subset
`astropy__astropy-12907` at B1/concurrency 1. Eager runs use raw per-task
`/metrics` brackets and explicitly do not use graph flush, graph census, or
graph topology/work needles. The authenticated terminal engine ingress ledger
is still required and snapshotted before container removal.

This one-task run is diagnostic-only. It is not valid for the standing exact4
or exact16 acceptance rule. The full-vocabulary mandatory-weight floor is
`153.9383846446886 ms/step`; the one-sided U95 cap is
`177.0291423413919 ms/step`. These are contract values, not measurements.

The pinned stock FA2 binary has SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`
and size `299183936` bytes. No GPU or synthetic/probe workload was used while
preparing this artifact.

Run the exact no-launch-prepared command in `prepared_command.sh` only when the
GPU host is intentionally available for the real SWE task.
