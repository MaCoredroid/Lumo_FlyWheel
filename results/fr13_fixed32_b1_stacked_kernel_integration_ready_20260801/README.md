# FR13 fixed32 B1 stacked kernel integration

Status: source-ready, default-off, awaiting repaired StreamK and SFWD live PASS credentials before a real one-task stacked timing run.

This source-only integration is based on `968f1150638a72938e86ade8f6ed1a173ba9e3e3` and combines:

- StreamK wide256 B1 production from the corrected eager/null-slot base.
- CFWD all-parent commit production, byte-qualified on one real SWE-Verified B1 task.
- SFWD state-fusion byte and source-bound production timing routes.

The SFWD byte gate remains exclusive. The SFWD production route can coexist only with CFWD production and either stock CUTLASS or a production-qualified B1 StreamK selector. Every selector remains default-off.

No GPU, Docker, probe, synthetic task, or performance measurement was used for this integration work. A real one-task B1 run remains diagnostic-only and cannot satisfy formal exact4/exact16 acceptance.

## Source bindings

- CFWD device source SHA-256: `8402e027b6dea8d902f86810b9e5a0fa0a01dda61e74b0f517987d2cf4c95f9a`
- SFWD qualification kernel SHA-256: `1cd112bc37f4e41237219e27fa261c902320c2204f84118e6d34862046ec5029`
- SFWD production control SHA-256: `19003a1e9d722f26fe348a4fb8f33319511a7fc6d507a8f9c066e36ae49b32a0`
- StreamK CUTLASS patch source SHA-256: `2b36f8db3835ce5bc37545f21ed77de9eee641c229f7892d64a769db18f513f4`
- StreamK qualification source commit: `968f1150638a72938e86ade8f6ed1a173ba9e3e3`

## Conflict resolutions

- Preserved full-vocabulary `K=0`, root `0`, B1 registry binding when the older CFWD runner default conflicted.
- Combined SFWD eager/credential guards with both StreamK selectors instead of replacing either path.
- Replaced CFWD ancestry-only timing validation with the exact byte-qualified device-source hash, because the functional commit is intentionally replayed into an integration history.
- Kept StreamK PASS commit validation strict by pinning it to the repaired qualification base and requiring that base to remain an ancestor of the integration source.

## Verification

See `test_results.txt`. Focused source, contract, ingress, launcher, CFWD, SFWD, and StreamK tests pass. CUDA-only tests are skipped because this task intentionally used no GPU.
