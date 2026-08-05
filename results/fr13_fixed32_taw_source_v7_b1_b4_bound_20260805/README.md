# Hydra27 source-v7 TAW B1/B4 binding

Status: **PRODUCTION-READY CORRECTNESS CREDENTIAL**.

The B4 side is a real SWE-Verified exact-four diagnostic on fixed Hydra27,
physical32, K64/root1. All B1, B2, B3, and B4 replay records used the measured
full-graph route and reported zero probability or product mismatches. The
reference remained served and the candidate remained shadow-only.

The independent B1 side used the canonical real task
`astropy__astropy-12907`, which resolved. Its measured full-graph replay record
also reported zero probability or product mismatches. The B1 credential and
reviewed B4 verdict bind the same source file SHA-256 and source-v7 contract.
The merged production bundle replaces the B4 campaign's B1 record with this
fresh independent B1 record while preserving its reviewed B2-B4 records.

The first B1 invocation failed before container boot because its wrapper
overrode the fixed32 phase timers to zero. Fixed32 requires all three phase
timers for explicit boundary flushes. Commit `f05693885` corrected the wrapper,
added regression coverage, and was pushed before the successful rerun. The
failed invocation performed no task or GPU work and issued no credential.

This package is correctness evidence only. It contains no timing, TPS,
acceptance, speedup, or hardware-floor claim. Raw prompts, responses, patches,
traces, logs, environment dumps, process identities, tensors, and binaries are
excluded.

