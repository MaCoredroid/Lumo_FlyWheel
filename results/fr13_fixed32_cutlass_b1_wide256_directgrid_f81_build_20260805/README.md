# Fixed32 B1 wide256 direct-grid f81 build

Status: **f81-parented source integration, SM121a link, CPU-only import, ELF
audit, and fail-closed diagnostic binding passed; GPU correctness and timing
not run**.

## Provenance

The integration branch starts at exact main commit
`f81a1c774b55a7f76d30d30ed0fac2be73665be9`. Source integration commit
`3046daafd35fd45ab72b1c96c2a43bf0f1cb9977` cherry-picks kernel commit
`6898009ccf5f557a3ca8a1bc9f3e19b1a16c2467`; the cherry-pick trailer retains
that provenance.

The binary is linked from the audited full-tile object closure with only
`scaled_mm_blockwise_sm120_fp8.cu.o` replaced by the direct-grid object. The 19
link inputs are pinned in `object_closure.tsv`.

## Host evidence

The resulting shared object is available locally at:

`/home/mark/fr13_sfwd_directgrid_f81_build/bin/_C_stable_libtorch.identity_wide256_directgrid_f81.abi3.so`

- SHA-256: `7f62f4efac0bb61d2f8367d9ff1e22a47d425ca06dcb74b4cbcd65dc701361de`
- Size/mode: `119982464` bytes, `0555`
- CPU import with `CUDA_VISIBLE_DEVICES` empty: pass
- ELF: AArch64 ELF64 shared object, build ID
  `f0538518ec10dc4b7cfc18d9fa9db120dd7696b7`
- Dynamic dependencies, RUNPATH, and all 183 undefined dynamic symbols match
  the incumbent.
- Both binaries contain 17 CUDA ELF payloads and 343 device resource records.
- The 1,327 defined dynamic-symbol count is unchanged. Exactly two old
  FP16/BF16 generic wide-scheduler symbols are replaced by the corresponding
  direct-scheduler symbols.
- The repository binary verifier accepts the object only as the K64/root
  `identity_wide256_fullgrid_b1_byte_ab` diagnostic. The older N5120 and B4
  full-tile binary pins remain unchanged.

The shared object exceeds GitHub's single-file limit and is not committed.
Its content identity, link inputs, and local path are committed here.

## Deferred real gate

Use the outer `fr13_run_b1_target_sfwd_conv_postprep_live_gate.sh` wrapper from
a clean, pushed branch after the serial GPU campaign is idle. The outer
wrapper is required because it creates and validates the f81 route sidecar;
invoking the inner CUTLASS gate directly is invalid on this source revision.

The exact inputs and deferred command are in `launch_contract.json` and
`deferred_gate_command.txt`. The launch is one authenticated real
SWE-Verified `astropy__astropy-12907` task at physical32/K64/root1. Retain the
result only if all five projection shapes appear, comparisons equal exactly
320, mismatching comparisons are zero, differing bytes are zero, and the
served result is stock.

No GPU kernel, Docker container, synthetic probe, correctness comparison,
performance timing, or hardware-floor acceptance was run for this artifact.
