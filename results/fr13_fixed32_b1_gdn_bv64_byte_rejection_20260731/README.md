# FR13 B1 GDN BV64 byte rejection

## Verdict

REJECTED on exact bytes. This is not a timing measurement or an acceptance
result.

The run used the pinned real SWE-Verified task
`astropy__astropy-12907`, batch size 1, concurrency 1, source
`cc4f420bdcf37aca39c063ec684ef739208fa92c`, and stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
Qrow16 and draft-head padding were off. TAW native-precompute and GDN BV64
were armed.

The corrected scalar-safe GDN gate reached the first real decode event, ran
distinct stock BV8 and candidate BV64 arms from identical snapshots, and
reported:

```
FR13 fixed32 GDN BV live-gate byte mismatch at record 0: ['export']
```

The gate restored the stock snapshot in its fail-closed path. No draft event
completed and no candidate output was served. GDN BV64 is rejected for
production. TAW remains unclassified because GDN aborts earlier in the replay
path. There are no valid TPS, latency, acceptance, or floor-ratio measurements.

The B1 launcher can now set `FR13_GATE_GDN_BV=0` so TAW can be gated
independently.

## Evidence

- Runroot:
  `output/fr13_b1_kernel_live_gate_taw_gdn_scalarfix_20260731T155216Z`
- Exact checksums: `checksums.sha256`
- Machine-readable verdict: `verdict.json`
