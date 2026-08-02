# FR13 fixed32 SFWD state-fusion row-group source checkpoint

Status: **source-only, default off, not compiled, not byte-qualified, and not
timing eligible**.

This checkpoint advances the existing one-launch fixed32 causal-conv/state
fusion candidate while the real SWE-Verified B4 stock/M128 pair owns the GPU.
It contains no prompts, responses, task patches, traces, raw logs, process or
container IDs, environment dumps, or secrets.

## Change

The first source candidate assigned one Triton program to one physical tree row
and one 256-channel tile. All 32 rows therefore reloaded the same four
per-channel conv-weight vectors and the same request bank-row metadata.

The row-group candidate assigns four consecutive physical rows to one program:

- exact fixed32 geometry remains 32 rows/request, width 4, state length 12;
- the program grid changes from `B*32 x ceil(C/256)` to
  `B*8 x ceil(C/256)`;
- each four-tap channel-weight vector and bank-row value is reused across four
  rows inside the program;
- the physical launch count remains one per GDN layer;
- every `(request,row)` output and commit-source row still has exactly one
  writer, and request-local row zero remains the only edge-row writer.

The candidate retains each BF16 tap product, its BF16 rounding boundary, the
left-to-right FP32 addition sequence, the existing ex2-compatible SiLU, BF16
output stores, and the `prior + x + zero` commit-source layout. No reduction,
dot product, model-weight change, drafter-quality change, storage allocation,
or recurrence change was introduced.

This is a source-level 4x reduction in program blocks and repeated conv-weight
loads. It is not a 4x latency claim. Each program now owns more live output
state and uses eight warps instead of four, so pinned SM121a codegen must prove
acceptable registers, stack, spills, and occupancy before the variant can run.

## Verification

- Python bytecode compilation passed for the kernel, patcher, and focused test.
- Ruff passed on the changed Python source and test.
- Focused SFWD state-fusion tests: `8 passed`.
- Combined SFWD/GDN schedule and launch tests: `12 passed`.
- `git diff --check` passed before the source commit.
- Source commit: `f7456b7fc83bdc292cf25b4f2d15e22a2f224363`.

## Required closure

1. After the B4 pair fully tears down, compile the row-group specialization
   with the pinned CUDA 13.0/SM121a Triton toolchain and inspect registers,
   stack, spills, local memory, launch geometry, and generated load counts.
2. Run the existing authenticated real SWE-Verified B1 reference-returning byte
   gate across all 48 GDN layers. Require exact bytes for both conv output and
   the complete commit-source stage.
3. Make a graph-safe production selector only after the eager byte gate passes;
   preserve the incumbent guarded fallback.
4. Measure candidate-only full-step TPS and SFWD/DFWD/CFWD/other breakdown on
   the standing real exact4 task set, then exact16 and one-sided U95 only after
   a material exact4 improvement.

No CUDA compile, Triton codegen, GPU command, synthetic timing, probe workload,
or real task was run for this checkpoint. It makes no speed, exactness,
acceptance, B1, B4, or hardware-floor claim.
