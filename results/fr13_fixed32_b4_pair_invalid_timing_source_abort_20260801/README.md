# B4 pair invalid-source abort

The real Tail23/Hydra27 K64 B4 campaign launched from source
`68336f72ada43aa1e9681329e58dc031d2a69491` was stopped after a static review
proved that its timing reducer would reject every valid v9 work census. The
source treated the mandatory terminal record as an event and required a TAW
route that the terminal schema intentionally does not contain.

Only the Tail23 all-parent stage launched. All four canonical real task traces
were recovered through the direct-file capture path as newline-framed JSONL,
but the orchestrator was interrupted before a formal production pass or byte
verdict. No M128 gate, timing arm, TPS measurement, acceptance result, or
hardware-floor result was issued. The preserved container was removed and the
GPU returned idle.

The reducer is fixed by code commit
`9f30d84dc68f97bfd871862db829b7048e921847`. It rederives persisted reports
from raw census bytes, validates the terminal, binds raw identity, and requires
one physical-work signature across both topologies and both timing arms.

This directory contains reduced metadata only. It contains no task prompts,
responses, patches, trace contents, raw logs, credentials, or runtime/process
identities.
