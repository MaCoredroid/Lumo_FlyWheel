# Fixed32 SFWD int32 state-address experiment

Status: **OFFLINE_CODEGEN_REJECTED_RESOURCE_REGRESSION**.

Source commit `a370d51fabb23f9efae00c6f83594df4a864a62b` narrows the
runtime state-bank element offset from int64 to int32 after adding a host check
that the full contiguous state tensor fits signed int32. The change is
address-safe under that added contract, but SM121a code generation regresses.

Relative to the selected fixed-stride parent, reported registers rise from 54
to 58, static SASS rises from 382 to 387, and encoded SASS rises from 400 to
408. LDG, STG, LDS, STS, barriers, and shared memory do not improve. Two
fresh-cache B1/B4 builds reproduce the result with no spills or calls.

The variant is rejected and was not launched or runtime-bound. No GPU,
Docker, service, task, timing, or acceptance run was used for this experiment.
The reduced package excludes raw compiler output, binaries, IR, logs, task or
model content, requests, responses, environment values, credentials, process
identifiers, container identifiers, and secrets.
