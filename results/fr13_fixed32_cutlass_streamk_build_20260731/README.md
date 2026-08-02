# FR13 fixed32 CUTLASS Stream-K build

Status: `BUILD_PASS_ABI_ADDITIVE_GATE_READY_GPU_NOT_RUN`.

This artifact advances the source-only candidate from `a6ae2c4339` to an
actual `_C_stable_libtorch.abi3.so` build and a production-off real-SWE B1
same-process byte gate. It makes no correctness, performance, hardware-floor,
or acceptance claim. No GPU or Docker command was used for this work.

## What built

- vLLM source: `fe9c3d6c5f66c873d196800384ed6880687b9e52`, the exact
  `0.19.2rc1.dev134+gfe9c3d6c5` version pinned by fixed32.
- CUTLASS: `v4.4.2`, commit
  `da5e086dab31d63815acafdac9a9c5893b1c69e2`.
- Torch: `2.10.0+cu130`; stable target macro
  `TORCH_TARGET_VERSION=0x020A000000000000ULL`.
- CUDA: `13.0.88`; the changed TU compiled for `sm_121a`.
- Candidate selector: `streamk_coop128`.
- Diagnostic selector: `streamk_coop128_byte_ab`.
- Deployment binary SHA-256:
  `fa9395754b13de26dbed38dfc551614dbb109058764426564dcbb3c77fdd6ea9`.
- Deployment binary size: `111383840` bytes.

The binary itself is intentionally not committed. The launcher accepts only
that exact digest and size. Its RUNPATH was normalized after linking to
`/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64`, the
pinned image layout.

## Source corrections

The first source-only version was not buildable as written. The CPU build and
pre-GPU gate reviews found and corrected four issues:

1. Stable-libtorch code cannot use `TORCH_CHECK`; it now uses
   `STD_TORCH_CHECK`.
2. The `if constexpr` stock branch needed an `else`; without it, Stream-K-only
   scheduler fields instantiated for stock schedulers.
3. CUTLASS rejects cooperative `64x128x128` on SM120 because cooperative tile
   M must be at least 128. `streamk_coop64` was removed. The legal retained
   path uses `128x32x128` for swapped B1 rows and `128x128x128` otherwise.
4. The first byte comparator stopped after the first mismatch. The corrected
   comparator scans every output byte, records the total differing-byte count,
   and was rebuilt and repinned before any GPU gate.

The modeled B1 maximum recovery remains `10.923627 ms/event`; that is a model,
not measured speed. It does not by itself close the end-to-end floor gap.

## ABI audit

The stock and candidate binaries were built from the same source/toolchain and
received the same runtime RUNPATH normalization.

- Dynamic defined names/types: stock `1278`, candidate `1288`, removed `0`,
  added `10`.
- Additions: four GNU-unique `sm_count` guards, four weak CUTLASS Stream-K
  argument converters, and two weak vector destructors.
- Dynamic undefined names/types: stock `154`, candidate `179`, removed `0`,
  added `25`; additions are satisfied by the unchanged libc, libstdc++,
  libgcc, libcudart, and Torch dependencies.
- `DT_NEEDED`: identical nine entries.
- RUNPATH: identical.
- Cubin inventory: `17` in each binary, with `16 x sm_121a` and `1 x sm_89`.
- ELF format: AArch64 ELF64 shared object in both cases.

This is an additive ABI audit, not a runtime import or GPU correctness pass.
The canonical symbol inputs and empty removal diffs are under `abi/`.

## Real B1 gate

The launcher mounts and installs the candidate only for a fixed32 B1
diagnostic. The selector is also written to a read-only `/logs` file because
EngineCore does not reliably inherit arbitrary `FR13_*` variables.

`streamk_coop128_byte_ab` executes stock then Stream-K on the same CUDA stream
for up to 256 real projection calls, copies both outputs after the kernels,
compares every BF16 byte, records JSONL, and returns the stock output. The gate
requires all five real projection `(N,K)` shapes, contiguous nonzero calls, and
zero differing bytes. Unrelated qrow, TAW, draft-head-pad, and GDN candidates
are forced off. The wrapper rejects an existing `RUNROOT`, preventing stale
JSONL or attestation reuse from turning a vacuous run into a pass.

```bash
RUNROOT=output/fr13_cutlass_streamk_b1_$(date -u +%Y%m%dT%H%M%SZ) \
TAG=cutlass_streamk_b1_$(date -u +%Y%m%dT%H%M%SZ) \
FORKED_FA2_SO=/absolute/path/to/pinned_fa2.so \
CUTLASS_STREAMK_SO=/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_coop128_allbytes_gate_ready.abi3.so \
bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh
```

That command uses the existing real SWE-Verified task
`astropy__astropy-12907`. It is a diagnostic only and is invalid for
acceptance. Direct B1 timing, B4 exact4 correctness/timing, and exact16
acceptance remain unrun.
