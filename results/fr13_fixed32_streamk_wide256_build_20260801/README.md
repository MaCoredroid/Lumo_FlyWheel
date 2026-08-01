# FR13 fixed32 forced wide256 Stream-K build

Status: compiled, default off, and ready for the one-real-task B1 byte gate.
This artifact contains no GPU timing and makes no performance or acceptance
claim. No GPU or Docker process was used to build or inspect the candidate.

## Candidate

The selector `streamk_force_wide256` is admitted only for rows
`32/64/96/128` and the five profiled projection `(N,K)` pairs. Unknown and
unset selectors stay on stock. `streamk_coop128` retains its existing tile
geometry and heuristic decomposition.

The wide candidate uses:

| Physical rows | Layout | Tile MNK | Scale granularity MNK |
| --- | --- | --- | --- |
| 32/64 | swapped A/B | 256x32x128 | 128x1x128 |
| 96/128 | normal | 128x256x128 | 1x128x128 |

Both paths use the SM120 cooperative kernel, force
`DecompositionMode::StreamK`, use deterministic reduction, keep `splits=1`,
and explicitly use two mainloop stages. The explicit stage count is required:
the first exact build with automatic staging selected one stage for the normal
wide tile and failed CUTLASS's `Stages >= 2` static assertion. The final build
keeps the requested tile and scheduler semantics and changes only the stage
count to the smallest legal value.

The normalized deployment binary is not committed because it is 113 MB. Its
immutable identity is:

```text
/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_gate_ready.abi3.so
sha256 b957cf49da2977056661443192fc2725e153adba7f21fb522c07b439c04540ee
bytes  113174464
mode   0555
```

`cuobjdump` confirms sixteen `sm_121a` cubins and one `sm_89` cubin. The four
new half/BF16 specializations have these resources:

| Tile MNK | REG | STACK | SHARED |
| --- | ---: | ---: | ---: |
| 256x32x128 swapped | 168 | 8 | 1024 |
| 128x256x128 normal | 168 | 552 | 1024 |

The 552-byte stack allocation is a material timing-regression risk. It is
reported rather than hidden or worked around by weakening the requested tile.

## Bound

For the five projection widths, logical N tiles per call change from
`272/40/40/128/64` at N-tile 128 to `136/20/20/64/32` at N-tile 256. On 48
SMs this changes the output-tile tail geometry, while forced Stream-K can also
partition K work across CTAs.

The prior B1 analytical model bounded the entire ideal Stream-K tail recovery
at 10.923627 ms/event from 112.312954 ms/event of measured CUTLASS time. That is
an optimistic ceiling before Stream-K workspace, fixup, scheduling, and the
observed stack allocation. It is not a prediction that wide256 realizes the
recovery; the candidate can regress.

## Required gate

Run `streamk_force_wide256_byte_ab` on the real SWE-Verified B1 task
`astropy__astropy-12907`. The diagnostic executes stock and candidate for all
five shapes but serves stock. Any differing byte rejects the candidate. This
one-task diagnostic is correctness evidence only, never acceptance.

Only after a clean byte gate may `streamk_force_wide256` be timed on the
standing real exact4 B1 and B4 task set. Exact16 remains the promotion gate.
