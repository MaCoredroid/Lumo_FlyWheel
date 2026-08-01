# FR13 B4 persistent-M128 rejected campaign diagnostic

The real SWE-Verified exact4 B4 run completed 320 same-process comparisons of
the persistent-M128 CUTLASS candidate against stock output. All 1,436,811,264
compared bytes were equal, and all five real projection shapes were observed.

This is not a live-gate PASS. Three tasks had valid agent terminals, but
`astropy__astropy-13398` ended with an execution error after 96 turns. The
fixed32 campaign reconciler therefore rejected the run with serve status 1.
No production selector, timing claim, acceptance claim, or hardware-floor
claim is authorized from this artifact.

The full-vocabulary run is not being repeated because deployment is fixed at
K64 with ROOT=1. The byte-clean M=128 kernel evidence remains a useful
diagnostic; the next live gate must use the K64 credential and floor ledger.

Only curated hashes and aggregate counters are published here. Raw prompts,
responses, traces, patches, environment records, and process identities are
excluded.
