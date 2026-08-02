# Fixed32 B1 K64/root stock-tile static-persistent handoff

Status: source integration, isolated host compile/link, and static binary audit
pass. The candidate is diagnostic-only, default off, and not acceptance-valid.
No GPU kernel, Docker container, synthetic probe, or timing run was launched for
this handoff.

## Candidate

`static_persistent_stocktile` preserves the stock B1 swapped collective:

- physical rows: 32;
- tile: `128x32x128`, cluster: `1x1x1`;
- scale granularity: `128x1x128`;
- cooperative SM120 mainloop, FP32 accumulation, and stock epilogue;
- full ordered K traversal for every output tile;
- no split K, Stream-K workspace, or reduction/fixup kernel.

The only kernel-policy change is complete-output-tile assignment: CUTLASS's
dynamic persistent scheduler is replaced by `StaticPersistentTileScheduler100`.
Logical trees smaller than 32 continue to use the existing physical-32 padded
route; this kernel sees the stable physical M=32 shape.

The source selector is admitted only to the same-process stock/candidate byte
diagnostic. Direct production installation is blocked until the real K64/root
byte gate passes. The diagnostic returns stock output and stops after at most
320 comparisons.

## Host audit

The pinned extension compiled and linked for `sm_121a`. BF16 and FP16 both
report 168 registers/thread, zero stack, zero local memory, 1,024 bytes static
shared memory, and 384 threads/CTA. Static scheduling increases the parameter
bank from 1,664 to 1,792 bytes but reduces reported mbarriers from 22 to 14.

Within the same linked binary, the static BF16/FP16 kernels reduce text from
18,816 to 14,976 bytes and SASS slots from 1,176 to 936, a 20.408% reduction.
The audited projection math is unchanged per dtype: 32 QMMA, 32 FFMA, 24 FMUL,
24 LDSM, four STSM, and eight output packs. `SYNCS.*` falls from 77 to 38 and
branch instructions from 75 to 45. Neither kernel contains a local-memory
load/store or a call.

This is static code evidence only. It does not establish latency, TPS, or
distance to the 119.658015414 ms K64/root hardware floor. The one-sided 1.15x
cap is 137.6067177261 ms, but timing remains prohibited before byte PASS.

## Prior evidence correction

The earlier cooperative-128 diagnostic is not Stream-K correctness evidence.
Its 240 byte-equal comparisons covered only four shapes. Those four shapes all
fell back to CUTLASS data-parallel scheduling; the missing `14336x5120` shape
is the sole real B1 projection that would have engaged Stream-K K
reassociation. That campaign also ended with orchestrator rc=15 and zero
completed tasks.

The later wide-256 K64/root diagnostic did cover all five shapes and completed
the real task, but failed 320/320 comparisons. The present candidate avoids
both split-K reassociation and changed tile geometry, but must still prove all
five shapes byte-for-byte.

## Immutable inputs

Candidate extension:

```text
/home/mark/fr13_b1_static_persistent_k64_cap320_build_20260802/bin/_C_stable_libtorch.static_persistent_stocktile_k64_root_cap320_88c50e7d1b6060c2.abi3.so
SHA256 88c50e7d1b6060c2bcec68f50985a1db47b43d299b574edfbfc32cac1ce68742
bytes 113383800, mode 0444
```

Stock FA2 extension:

```text
/home/mark/lumoFlyWheel-b1-wide256-k64-root-profile/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
SHA256 f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
```

## Real B1 gate

Run only when the GPU campaign owner coordinates the slot. This is one real
SWE-Verified B1 diagnostic task at K64/root1, not acceptance timing:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD); TAG="b1_static_k64_cap320_$(date -u +%Y%m%dT%H%M%SZ)"; RUNROOT="/home/mark/fr13_b1_static_persistent_k64_cap320_runs/$TAG"; env RUNROOT="$RUNROOT" TAG="$TAG" FORKED_FA2_SO="/home/mark/lumoFlyWheel-b1-wide256-k64-root-profile/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so" CUTLASS_STREAMK_SO="/home/mark/fr13_b1_static_persistent_k64_cap320_build_20260802/bin/_C_stable_libtorch.static_persistent_stocktile_k64_root_cap320_88c50e7d1b6060c2.abi3.so" FR13_STREAMK_GATE_CANDIDATE=static_persistent_stocktile FR13_STREAMK_QUALIFICATION_PROFILE=k64_root bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh
```

The arm directory is
`$RUNROOT/hydra27_fixed32_k64_root_$TAG`. Required reduced inputs/outputs are:

- `logs/fr13_fixed32_cutlass_static_persistent_byte_ab.jsonl`;
- `logs/fr13_fixed32_cutlass_streamk_binary.json`;
- `cutlass_static_persistent_k64_root_byte_gate.json`;
- `container_env.txt`;
- `swe_out/verified/per_task/astropy__astropy-12907/fixed32_cutlass_streamk_real_task_arm.json`.

The gate script runs the bounded reducer inline. Replay the formal validator
against the emitted live result with:

```bash
ARMDIR="$RUNROOT/hydra27_fixed32_k64_root_$TAG"; LIVE="$ARMDIR/cutlass_static_persistent_k64_root_byte_gate.json"; LIVE_SHA=$(sha256sum "$LIVE" | awk '{print $1}'); .venv/bin/python scripts/fr13_cutlass_streamk_pass.py validate --live-result "$LIVE" --expected-live-sha256 "$LIVE_SHA" --candidate-so "/home/mark/fr13_b1_static_persistent_k64_cap320_build_20260802/bin/_C_stable_libtorch.static_persistent_stocktile_k64_root_cap320_88c50e7d1b6060c2.abi3.so" --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py --expected-source-commit "$SOURCE_COMMIT" --candidate-selector static_persistent_stocktile --qualification-profile k64_root --draft-vocab-blocks scripts/fr13_dvk_subset_blocks.json
```

Only after validator PASS, issue the diagnostic qualification sidecar as:

```bash
.venv/bin/python scripts/fr13_cutlass_streamk_pass.py issue --live-result "$LIVE" --expected-live-sha256 "$LIVE_SHA" --candidate-so "/home/mark/fr13_b1_static_persistent_k64_cap320_build_20260802/bin/_C_stable_libtorch.static_persistent_stocktile_k64_root_cap320_88c50e7d1b6060c2.abi3.so" --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py --expected-source-commit "$SOURCE_COMMIT" --candidate-selector static_persistent_stocktile --qualification-profile k64_root --draft-vocab-blocks scripts/fr13_dvk_subset_blocks.json --out "$ARMDIR/cutlass_static_persistent_k64_root_production_pass.json"
```

Even a one-task byte PASS is diagnostic, not user acceptance. Full-step timing
must use the standing real 4-task or 16-task set, and production remains off
until that subsequent qualification is explicitly wired.
