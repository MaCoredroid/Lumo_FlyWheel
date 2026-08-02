# Fixed32 B4 M128 scalar-only epilogue checkpoint

Status: source-only, default off, and not acceptance-valid.

## Existing kernel evidence

The existing pre-direct-linear static M128 BF16 kernel contains 1,440 SASS
instructions, 168 registers, zero stack/local/spills, and zero device calls.
Its output path contains 64 FP32 scale multiplies followed by 32 BF16 pack
instructions. Local inspection found two generic alpha-resolution regions;
each resolves a pointer/fallback scalar and carries stride/address/control
work. Raw disassembly is intentionally not included.

This checkpoint does not remove the 64 output multiplies. It targets only the
generic scalar state and resolution work around them. No exact instruction,
register, or latency reduction is claimed until the candidate is compiled.

## Change

Only the default-off `persistent_b4_m128_static` candidate now uses a reduced
three-operation visitor tree:

1. fixed runtime scalar broadcast,
2. `Sm90Compute<cutlass::multiplies>` in FP32,
3. accumulator fetch.

The stock sourceless `LinearCombination` tree has six source-level visitor
operations and carries alpha, beta, two optional scalar pointers, and two
dynamic batch strides. The candidate has empty host callback arguments and
one FP32 device parameter. It retains the same runtime FP32 multiply and
round-to-nearest output conversion. The stock callback tree is unchanged.

The candidate reuses `Base::CollectiveMainloop`, so its stage count and
ordered K reduction are unchanged. Epilogue tile selection, destination type,
layout, alignment, TMA store path, output coordinates, and output tile count
are also unchanged.

## Alpha invariant

The dispatch constructs the epilogue thread arguments with `{}`. The custom
callback `Arguments` type is empty, so the caller cannot provide alpha. During
CUTLASS `Arguments -> Params` conversion, the custom leaf unconditionally
returns `Params{Element(1)}`. That parameter is copied into the kernel and then
into the visitor scalar; neither producer nor consumer callbacks contain an
update path. The consumer only fills the scalar fragment from that value.

This prevents alpha from drifting from one in the current dispatch while
keeping it as a runtime device parameter, so the multiply remains part of the
callback expression instead of being replaced with a direct accumulator
fetch.

## Verification

- 26 focused source tests passed.
- Python bytecode compilation and `git diff --check` passed.
- The patch applied to vLLM `fe9c3d6` with CUTLASS `da5e086`, and a second
  application was idempotent.
- Generated dispatch SHA256:
  `319bda31b05222e17eedefa65dbe328e87a9afd335301a75cf4bbb150911aedc`.

No NVCC/C++ compile, link, new SASS/resource audit, GPU execution, container
run, synthetic probe, or real-task run was performed while the B1 build was
active. Real SWE-Verified exact4 byte gates, paired timing, floor ratio, and
U95 remain pending.
