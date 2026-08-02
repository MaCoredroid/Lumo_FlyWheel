# FR13 B1 wide256 recompute stack-zero build

Status: compiled and static-codegen qualified, but rejected for the real byte
gate because its B1 comparison cap is 256. No GPU task, byte gate, timing, or
hardware-floor claim has been made from this artifact.

The prior B1 wide256 Stream-K candidate used 168 registers and an 8-byte stack
per thread. SASS showed one local store and 15 local loads caused by keeping
`threadIdx.x % 128` live across the kernel. This candidate preserves the
deterministic CUTLASS Stream-K fixup but emits the barrier-group coordinate at
each fixup use.

## Result

- Candidate FP16 and BF16 kernels: `REG=168`, `STACK=0`, `LOCAL=0`,
  `SHARED=1024`, `CONSTANT[0]=2944`.
- Both candidate SASS bodies contain zero `LDL`, `STL`, or device-call
  instructions.
- Each candidate body has 4,952 instructions and 15 `SR_TID.X` reads. The prior
  spill-bearing body had 4,936 instructions, two `SR_TID.X` reads, 15 `LDL`,
  and one `STL`.
- All six stock device-kernel symbols and resource records exactly match the
  pinned stock-symbol-exact build: `REG=168`, `STACK=0`, `LOCAL=0`,
  `SHARED=1024`, `CONSTANT[0]=2560`.

This is a static code-generation win, not throughput evidence. The established
real B1 comparison histogram uses all 256 calls on its first four shapes
(`69 + 68 + 51 + 68`) and requires a fifth shape, so this immutable binary
cannot cover the complete five-shape byte gate. A cap-320 rebuild is required;
this binary must not consume the real SWE-Verified task.

## Identity

```text
source_commit=cbd5b8e6bf7302f03f02bfe2885dfe47b044c65e
vllm_commit=fe9c3d6c5f66c873d196800384ed6880687b9e52
cutlass_commit=da5e086dab31d63815acafdac9a9c5893b1c69e2
patch_sha256=28451424c8f78b44d967ce73df34e900b5415366f6a1b0b377899b3062a01d20
patched_dispatch_sha256=20982686b07735ea95b7f176cc6b2981e9916f0fcbc598021457f61e50a41fd2
binary_sha256=c668c584b84f97e3ff53448238a0b03e3ea610549b1dc2b7fd6664fdd22c4098
binary_bytes=113080976
binary_mode=0555
```

Archived immutable candidate:

```text
/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_recompute_stack0_k64_root1_gate_ready.abi3.so
```

The host build used CUDA 13.0.88 and GCC 13.3.0. The extension RUNPATH is
`/home/mark/fr13_streamk_build/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:`.

## Scope

`candidate_kernel_resources.tsv` and `sass_summary.tsv` contain only the two
new candidate records. `stock_kernel_resources.tsv` contains the six unchanged
stock records. No task prompt, patch, model output, token sequence, or other raw
SWE-Verified material is present. The shape counts above are aggregate gate
coverage only and contain no task content.
