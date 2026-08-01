# B1 wide256 stock-symbol-exact build

Status: compiled and statically verified; not live-qualified and not a performance result.

The two preceding wide256 binaries died during pre-health profile work before any
SWE-Verified task request. The second failure occurred in the stock caller path,
which exposed two unintended changes to the original stock template: candidate-only
template parameters changed its type identity, and the original mainloop
`StageCountAutoCarveout` policy had been replaced.

The repaired patch leaves the original `cutlass_3x_gemm_fp8_blockwise` template,
both stock stage expressions, and its three-argument `GemmUniversal` spelling
unchanged. Stream-K candidates now use a separate
`cutlass_3x_gemm_fp8_blockwise_streamk` type. A default-false trait selects the
unchanged stock caller without adding members to the stock type.

Static `cuobjdump --dump-resource-usage` comparison found all six baseline stock
device-kernel records in the rebuilt binary with exact symbol and resource matches:
`REG=168`, `STACK=0`, `SHARED=1024`, `CONSTANT[0]=2560`. There were zero missing
or changed stock records. The six candidate records have separate `_streamk`
symbols; wide256 retains `REG=168`, `STACK=8`, `SHARED=1024`, and
`CONSTANT[0]=2944`.

Candidate:

```
/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_stock_symbol_exact_gate_ready.abi3.so
sha256=f7d5c01ca79829fbfff4c93949d057bd740905165b0b6793b3c0007629add962
bytes=112481752
mode=0555
runpath=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64
```

The next valid action is the one-real-SWE B1 byte gate. No B4 measurement or
hardware-floor timing claim is made by this build artifact.
