# Fixed32 B1 M32 static-linear codegen audit

Status: pinned SM121 host compile/link and offline codegen gate pass for the
canonical v3 source. The candidate remains default off and is not
acceptance-valid.

## Result

The canonical binary is built from source commit
`42269c703992ab4113cf314e04e668a930f3c8d6`. Its FP16 and BF16 M32
static-linear device bodies each have:

- 168 registers, zero stack, zero local memory, and zero `STL`/`LDL`;
- 1,024 bytes static shared memory and 2,688 bytes in `CONSTANT[0]`;
- 648 SASS instruction slots;
- 32 `QMMA`, 32 `FFMA`, 24 `FMUL`, 24 `LDSM`, 4 `STSM`, and 8
  dtype-specific output-pack instructions;
- 38 `SYNCS` instructions.

The mandatory math and data-movement counts are identical to stock and the
separate generic-static comparator. Static code size is 288 slots lower than
generic-static (648 versus 936, a 30.769% reduction) and 528 slots lower than
stock (648 versus 1,176, a 44.898% reduction). The direct scheduler has the
same 38 `SYNCS` instructions as generic-static; stock has 77.

These are offline compiler facts, not latency or throughput evidence. No GPU
kernel, task, extension import, Docker workload, synthetic probe, real
SWE-Verified byte gate, TPS measurement, or hardware-floor test was run for
this artifact.

## Spill repair

The v1 source commit `246ea98e7` compiled both dtype bodies at 664 slots with
an 8-byte stack. SASS located one `STL.64` and three `LDL.64` instructions.
The spilled value was the derived scheduler's duplicate
`direct_linear_blocks_` member.

Source commit `e36d24928` removed that member and checks the inherited
`scheduler_params.blocks_per_problem_` bound. The v2 result is 648 slots with
zero stack and no local load/store instructions. Source commit `42269c703`
then silenced the release-build unused-geometry warning. The extracted v2 and
v3 candidate SASS sections are byte-identical for each dtype.

## Canonical identity

```text
source_commit=42269c703992ab4113cf314e04e668a930f3c8d6
branch=agent/fixed32-sfwd-b1-m32-static-linear-source-20260802
vllm_commit=fe9c3d6c5f66c873d196800384ed6880687b9e52
cutlass_commit=da5e086dab31d63815acafdac9a9c5893b1c69e2
patch_source_sha256=fd1ec9a43c5bce3e1acd43072c1db1c6b0ffe2d7a21f4fba653dcefadfbe2f0e
patched_dispatch_sha256=dce378ef501f40f061a19000945f2b74fad45955248cf00f5fe954f5f832a101
binary_sha256=6404e2c2e3b9bb8b7e39be3671bcbae3f16ea7b10adf1770056beb2c98ee833f
binary_bytes=113327488
binary_mode=0555
binary_build_id=3745b4c8e4799d5abd62e6649f4a7772e4bef8da
```

The linked binary, cubin, and raw SASS are not published in Git. Their local
paths and cryptographic identities are reduced into this artifact. The
canonical linked binary is:

```text
/home/mark/fr13_m32_static_linear_build/bin/_C_stable_libtorch.m32_static_linear_v3_6404e2c2e3b9bb8b.abi3.so
```

## Required next gate

The next valid step is the authenticated real SWE-Verified B1 five-shape
raw-byte gate using the established task set, K64/root1, and the fixed physical
32-row topology. Both Tail23 and Hydra27 must resolve cleanly, every candidate
comparison must be byte-equal, and all five allowlisted projection shapes must
be covered. Only after that gate passes may paired full-step TPS and phase
breakdowns be used for hardware-floor acceptance.
