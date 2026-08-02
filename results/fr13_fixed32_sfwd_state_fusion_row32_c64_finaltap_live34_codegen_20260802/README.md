# FR13 fixed32 SFWD row32/C64 final-tap specialization

Status: **offline SM121a codegen improves x-reuse v6; correctness and runtime
performance remain unqualified**.

This artifact audits source commit
`9920370699fa11e677c510bc28bc066eed18ad88` at B1 and B4 with the exact
live specialization: 32 physical rows per request, `C=10240`, width 4,
state length 34, 36 source rows, row group 32, `BLOCK_C=64`, eight warps,
and three stages. CUDA visibility was explicitly empty. No GPU kernel, service,
task, request, timing run, or acceptance run was launched.

## Change

The host validates the complete fixed32 source descriptor before launch, and
its final tap is exactly `3 + node` for every physical row. The generic
descriptor/gather loop now handles taps 0–2 only. Tap 3 directly loads the
current `x` row, applies the same final weight and BF16 product rounding, adds
it after taps 0–2 in the same order, then reuses that value for commit-source
staging after the output store.

This removes the final descriptor load, prior/current predicates, unused prior
address path, and duplicate staging load without changing pointer width,
accepted layouts, math order, output order, CTA geometry, or launch count.
Only the kernel source and its invariant test changed; no timing, runner, or
gate harness file was edited. The runtime candidate ID remains
`fixed32_sfwd_state_fusion_rowgroup32_c64_xreuse_v6`; the exact new source is
bound by commit and function hash.

## Result

| Metric | x-reuse v6 | Final tap | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers per thread | 112 | 106 | -6 |
| Allocated registers per thread | 112 | 112 | 0 |
| Static / encoded SASS | 1163 / 1184 | 1071 / 1088 | -92 / -96 |
| Warp-weighted static / encoded SASS | 9304 / 9472 | 8568 / 8704 | -736 / -768 |
| LDG / warp-weighted LDG | 108 / 864 | 91 / 728 | -17 / -136 |
| STG / LDS / STS | 20 / 4 / 0 | 20 / 3 / 0 | 0 / -1 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 | 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 81288 | 74152 | -7136 |

The 106-register report rounds to the same 112 registers per thread as the
v6 baseline under the 256-register-per-warp allocation quantum. Achieved
occupancy and runtime latency were not measured.

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and non-launch
metrics are byte-identical. A second build with a separate fresh cache and
output tree reproduced both variants byte for byte. The selected cubin SHA-256
is `1671c3005c76dd8932a6b52529e692947964be476814f1f755bd6ef53c7841a6`;
the SASS SHA-256 is
`50c0262eb9275c415705b8e43022092eba3572688c2aaff1488f9e0cf6ec537b`.

## Verification and limits

The focused suite passed 13 tests. It proves the exact final-tap descriptor
mapping, checks the specialized loop and direct current-row path, and retains
coverage for B1–B4 geometry, padded row strides, ordered BF16/FP32 math,
state length 34, and reference-returning control.

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
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row32_c64_finaltap_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row32_c64_finaltap_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row32_c64_finaltap_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 9920370699fa11e677c510bc28bc066eed18ad88 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 9920370699fa11e677c510bc28bc066eed18ad88 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
