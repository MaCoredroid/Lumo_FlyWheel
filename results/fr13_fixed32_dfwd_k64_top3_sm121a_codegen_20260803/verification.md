# Verification

## Build identity

The pinned builder emitted a 159,288-byte ARM64 shared object with SHA-256
`55229a9db7364fc8c0811fe34d3eaf06bc577626a3455fcc25a0fb9990aa480b`.
Its input CUDA source SHA-256 is
`38ea96d955355bf172f534174aa2d91e6db23170144b1c84c9474016a6c05e72`,
which matches source commit
`fef06f1eb2ab17d99849bd28d99b4cfc37649e66`.

`cuobjdump --list-elf` reported exactly one ELF:
`fr13_dfwd_k64_mapped_top3_sm121a.abi3.1.sm_121a.cubin`.
`cuobjdump --dump-resource-usage` reported 30 registers, zero stack, 1,216
shared bytes, zero local bytes, and 928 constant bytes for the kernel.

## Static checks

The focused and adjacent integration suite passed 69 tests. Python compilation,
shell syntax, and whitespace checks passed. Static SASS contains zero `LDL`,
zero `STL`, and zero `CALL` instructions.

The build loaded the shared object only to verify dispatcher registration. It
did not invoke `mapped_top3_out` and the build container had no GPU device
passthrough.

## Next real gate

Run the one-task diagnostic with the exact binary and the real fixed32 K64 B1
profile:

```bash
RUNROOT=<new-output-dir> \
TAG=<unique-tag> \
FORKED_FA2_SO=<pinned-fa2-so> \
FR13_GATE_DFWD_TOP3_SO=<absolute-candidate-so> \
bash scripts/fr13_run_b1_dfwd_k64_top3.sh
```

The runner isolates this candidate, uses `hydra27_fixed32`, and requires the
runtime ready, engaged, and four-call graph markers. A clean one-task result is
diagnostic only. Any speed or acceptance decision must use the standing real
four-task set, followed by the 16-task set where required.
