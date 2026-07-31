# FR13 B1 TAW/GDN scalar-gate rejection

## Verdict

REJECTED. This is not a timing measurement or an acceptance result.

The run used the pinned real SWE-Verified task
`astropy__astropy-12907`, batch size 1, concurrency 1, source
`e9a6061cf5a36ca81a59783ee10e9c0c6899555d`, and stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
Qrow16 and draft-head padding were off. TAW native-precompute and GDN BV64
were armed.

The engine reached the first real decode event. The GDN gate executed its
stock and candidate arms, then its raw-byte comparator failed on the scalar
`int32` counter surface:

```
RuntimeError: self.dim() cannot be 0 to view Int as Byte
```

This is a gate implementation failure, not evidence of a candidate byte
mismatch. No draft event completed, so GDN BV64 and TAW remain unclassified.
There are no valid TPS, latency, acceptance, or floor-ratio measurements.

The comparator now flattens every surface before viewing it as bytes. A scalar
`int32` regression test and the focused GDN gate suite pass before rerun.

## Evidence

- Runroot: `output/fr13_b1_kernel_live_gate_taw_gdn_20260731T154045Z`
- Exact checksums: `checksums.sha256`
- Machine-readable verdict: `verdict.json`
