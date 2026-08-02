# Fixed32 B4 M128 direct-linear scheduler source checkpoint

Status: source-only, default off, and not acceptance-valid.

## Change

The existing static-persistent M128 candidate keeps CUTLASS's complete-tile
lifecycle but previously decoded every persistent worker index with the generic
batched two-dimensional scheduler. The fixed B4 projection contract makes that
mapping degenerate:

- physical M is 128 and tile M is 128, so there is exactly one M tile;
- ordinary GEMM has one L tile;
- the cluster is 1x1x1;
- only the N-tile coordinate varies.

`Fr13Fixed32M128LinearScheduler100` therefore maps each valid linear work index
directly to `(M=0, N=index, L=0)` and advances it by the unchanged physical-grid
stride. It retains the incumbent static scheduler's host parameter setup, grid
sizing, persistent worker count, mainloop, epilogue, full ordered K ownership,
and output-tile count. Unset selectors and M values other than 128 remain stock.

The five real projection shapes contain 272, 40, 40, 128, and 112 N tiles.
The focused source test enumerates the persistent grid-stride mapping and proves
that every N tile is covered exactly once without duplication for each shape.

## Verification

- 24 focused source tests passed.
- Ruff, Python bytecode compilation, and `git diff --check` passed.
- The patch applied to the pinned vLLM source and pinned CUTLASS v4.4.2 checkout,
  and a second application was idempotent.
- The generated patched dispatch SHA256 is
  `8671829b9f50a5438cf36edbed9efabca5e391e1d41112b1878ad44d342cd855`.
- An independent source review verified the active cooperative-kernel API,
  both raster orientations, exact grid-stride coverage, and fail-closed behavior.

No NVCC compile, link, SASS/resource inspection, GPU execution, container run,
synthetic performance probe, or real-task run was performed for this source
checkpoint. Compile admission, raw-byte equivalence on real SWE-Verified exact4
Tail23 and Hydra27, paired timing, floor ratio, and U95 all remain pending after
the active B4 pair tears down.
