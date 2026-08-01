# Fixed32 static scheduler packed-QKV correction

Status: corrected candidate compiled and preflighted; the required corrected
real SWE-Verified live gate has not run. This artifact is not acceptance
evidence and contains no timing result.

## Contract correction

The fifth real projection is packed QKV `(N,K)=(14336,5120)`, not the stale
`(8192,5120)` tuple. The correction is applied consistently to the generated
CUTLASS dispatch predicate, the B1 reducer, and the production qualification
contract. The dispatch predicate remains shared across fixed row counts
`32/64/96/128`, and the comparator hard maximum remains 320 calls.

The rebuilt candidate preserves the stock projection tile, layout, mainloop,
K iteration order, scale granularity, accumulator, and epilogue. Only complete
output-tile allocation uses CUTLASS `StaticPersistentTileScheduler100`.

## Corrected binary

`/home/mark/fr13_static_qkv14336_build/build/_C_stable_libtorch.abi3.so`

- SHA-256: `c4b47fa82726ea93db8e7e4f1d08ce39eacaa6448bbc7c70b04c6e798c3c4d32`
- Bytes/mode: `113081080` / `0555`
- Patch source SHA-256: `9b4fa368e74640ddfc2f38b65e18e83d703d76651c4c28745d528ea7e299061d`
- Generated dispatch SHA-256: `e3e9c5e54dba7485c04db24a402c86c382f2550991fcff49f8e9fba48ecf8eaf`

The corrected object has the same CUDA resource dump and the same 1,296
dynamic export names as the stale candidate. The SO identity changes because
the host dispatch predicate now contains `14336` instead of `8192`.

## Completed stale-contract run

The preserved real B1 diagnostic ran `astropy__astropy-12907`, full vocabulary,
physical row count 32, and resolved the task. Its 320 contiguous comparisons
covered 312,541,184 output bytes with 320/320 byte equality and zero differing
bytes:

| `(M,N,K)` | Calls | Mismatches |
| --- | ---: | ---: |
| `(32,5120,6144)` | 86 | 0 |
| `(32,5120,17408)` | 85 | 0 |
| `(32,16384,5120)` | 64 | 0 |
| `(32,34816,5120)` | 85 | 0 |

That run is formally rejected. The committed old reducer replay exits 4 with
`not all five real projection shapes were exercised`: the old predicate and
reducer demanded nonexistent `(8192,5120)` and excluded actual packed QKV.
The exact four-shape evidence is informative for the kernel but cannot qualify
the corrected five-shape contract. The corrected binary must run the real B1
gate again.

Raw curated evidence is included as `stale8192_*.raw.*`; no environment file,
credential, or container log is included. `stale8192_reducer_replay.json` is
the output of the reducer embedded in source commit `3f6d16394`, replayed
offline against the preserved raw run without modifying the original run tree.

## Verification

- 79 focused tests passed.
- Five shell launchers passed `bash -n`.
- Pinned vLLM `fe9c3d6c5` and CUTLASS `da5e086d` validation passed.
- Generated dispatch matched the repo patcher and passed idempotence validation.
- Candidate identity verification passed.
- Runtime-manifest preflight passed.
- No GPU or Docker command was used for this rebuild or preflight.

Run `prepared_b1_gate_command.sh` from a GPU-idle host to execute the next
required one-task, real SWE-Verified, full-vocabulary B1 byte gate.
