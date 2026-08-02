# FR13 fixed32 SFWD row32/C64 prior-vector reuse

Status: **offline SM121a codegen improves the final-tap baseline; correctness
and runtime performance remain unqualified**.

This artifact audits source commit
`4f649835d42b98264ad71b46121637b12f8d9ea1` at B1 and B4 with the exact
live specialization: 32 physical rows per request, `C=10240`, width 4,
state length 34, 36 source rows, row group 32, `BLOCK_C=64`, eight warps,
and three stages. CUDA visibility was explicitly empty. No GPU kernel, Docker,
service, task, request, timing run, or acceptance run was launched.

## Change

Taps 0–2 select either prior state columns 0–2 or an ancestor `x` row from
the host-validated descriptor. The old kernel independently issued a masked
prior-state gather for each tap, then loaded the same three prior channel
vectors again for commit-stage edge writes.

The new kernel loads prior columns 0–2 once per CTA. It uses the unchanged
`source_row` descriptor to select the exact prior vector for taps 0–2, keeps
the existing ancestor-`x` path for non-prior rows, and reuses those vectors
for edge staging. BF16 product rounding, FP32 add order, int64 addressing, and
the output/current/prior/zero store order remain unchanged. These prior
columns were already unconditionally read by the exact one-rowgroup edge
writer, so the accepted memory-address scope does not expand.

Only the kernel source and its focused test changed; no timing, runner, or
gate harness file was edited. The runtime candidate ID remains
`fixed32_sfwd_state_fusion_rowgroup32_c64_xreuse_v6`; commit and function
hash bind this exact source variant.

## Result

| Metric | Final tap | Prior reuse | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers per thread | 106 | 62 | -44 |
| Allocated registers per thread | 112 | 64 | -48 |
| Allocated registers per CTA | 28672 | 16384 | -12288 |
| Static / encoded SASS | 1071 / 1088 | 993 / 1008 | -78 / -80 |
| Warp-weighted static / encoded SASS | 8568 / 8704 | 7944 / 8064 | -624 / -640 |
| LDG / warp-weighted LDG | 91 / 728 | 64 / 512 | -27 / -216 |
| STG / LDS / STS | 20 / 3 / 0 | 20 / 0 / 0 | 0 / -3 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 0 / 0 | -4096 / -1024 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 74152 | 69640 | -4512 |

Warp-weighted counts multiply static SASS instruction instances by the
unchanged eight warps; they are not measured memory transactions or runtime
latency. Achieved occupancy and latency remain unmeasured.

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and non-launch
metrics are byte-identical. A second build with a separate fresh cache and
output tree reproduced both variants byte for byte. The selected cubin SHA-256
is `45bcb87ac7b3c6f599fb4542b35f856b3c9aa68a84c9e6eba59531d5b334e7bd`;
the SASS SHA-256 is
`5b439598c82498c9f9a355465b513ae25a681f84960f5c058e93682e19fa5cb8`.

## Verification and limits

The focused suite passed 14 tests. The new test exhaustively compares the
prior-vector selection with the generic descriptor selection for every node
and each of taps 0–2. Existing tests retain coverage for the exact descriptor,
direct CPU fused indexing, final-tap specialization, B1–B4 geometry, padded
row strides, ordered BF16/FP32 math, and state length 34.

The independent verifier re-ran `nvdisasm` and `cuobjdump`, recounted SASS
classes, checked `.target sm_121a` and `.reqntid 256`, verified raw hashes,
enforced zero stack/local/spills/calls, and compared B1/B4 and fresh-cache
outputs. The embedded producer is `ptxas-blackwell` 12.9.86 from toolkit
12.9; CUDA 13 tools were used only for inspection.

Cubin identity here means two fresh builds from the same fixed worktree
metadata. Cubin debug metadata includes the canonical source file table, so
arbitrary checkout timestamps are not claimed to reproduce the container hash.
PTX, SASS, and resource identity are the code-relevant checks.

The package contains scripts and derived summaries only. It includes no
cubin, PTX, SASS, IR, model/task content, request logs, patch content,
environment dump, credential, or secret.

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row32_c64_priorreuse_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row32_c64_priorreuse_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row32_c64_priorreuse_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 4f649835d42b98264ad71b46121637b12f8d9ea1 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 4f649835d42b98264ad71b46121637b12f8d9ea1 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
