# FR13 B1 non-FA2 live-gate rejection

## Verdict

REJECTED. This is not a timing measurement and not an acceptance result.

The real SWE-Verified diagnostic used only
`astropy__astropy-12907`, batch size 1, concurrency 1, source
`0fbd1a5e4cf69777661bc574f556f89dd212c512`, and stock FA2 SHA-256
`f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`.
The diagnostic is explicitly `gate_eligible=false`.

The first launch (`20260731T152308Z`) stopped before container boot because the
FA2 bind source was passed as a relative path. Docker rejected it as an invalid
volume name. No GPU kernel or task work ran.

The corrected launch (`20260731T152531Z`) booted, accepted the real task, and
failed closed at its first draft event:

```
RuntimeError: FR13 draft-head padding requires contiguous BF16
UnquantizedLinearMethod weight[65536,5120]
```

No draft event completed (`drafts=0`), so none of the TAW, draft-head, or GDN
candidate paths earned a live byte-equality pass. There are no valid TPS,
latency, acceptance, or floor-ratio measurements from either launch.

The draft-head padding candidate is retired from this campaign. It is
incompatible with the deployed head representation and is inside the excluded
draft-head/DVK optimization area. The B1 launcher now defaults that diagnostic
off. TAW and GDN must be rerun on the same pinned real task.

## Evidence

- Failed preboot runroot:
  `output/fr13_b1_kernel_live_gate_nonfa2_20260731T152308Z`
- Failed real-task runroot:
  `output/fr13_b1_kernel_live_gate_nonfa2_20260731T152531Z`
- Exact checksums: `checksums.sha256`
- Machine-readable verdict: `verdict.json`
