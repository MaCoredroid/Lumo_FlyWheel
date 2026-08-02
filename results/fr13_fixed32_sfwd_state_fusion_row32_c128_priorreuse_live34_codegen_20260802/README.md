# FR13 fixed32 SFWD row32/C128 prior-reuse probe

Status: **offline launch-total static win with material occupancy risk; retain
C64 until a GPU correctness and timing gate**.

This artifact compiles source commit
`4f649835d42b98264ad71b46121637b12f8d9ea1` at B1 and B4 with the exact
prior-reuse kernel and a probed `BLOCK_C=128` schedule: 32 physical rows per
request, `C=10240`, width 4, state length 34, 36 source rows, row group 32,
eight warps, and three stages. CUDA visibility was explicitly empty. No source
schedule, production selector, harness, Docker container, GPU kernel, task,
request, timing run, or acceptance run was changed or launched.

## Classification

C128 halves channel CTAs from 160 to 80 per request. Although the per-CTA body
is larger, launch-total warp-weighted static SASS, encoded SASS, LDG, and STG
counts all fall by 6–10%. It has zero shared memory, spills, local memory, and
calls.

The tradeoff is occupancy risk. Allocated registers rise from 16384 to 20480
per CTA. Under the SM121 64K-register budget, register-limited residency falls
from four to three CTAs per SM, or 32 to 24 warps. The project hardware ledger
records 48 GB10 SMs, so the B1 grid also falls from an average 3.33 to 1.67
CTAs per SM. Greater per-warp instruction-level parallelism may offset some
latency hiding loss, but static codegen cannot establish that.

Therefore this is an **artifact-only offline win and future timing candidate**,
not a selected schedule. The checked-in kernel remains C64.

## Result

Per-CTA resources:

| Metric | C64 | C128 probe | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 80 | -80 |
| B1 / B4 CTAs per launch | 160 / 640 | 80 / 320 | -80 / -320 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers per thread | 62 | 80 | +18 |
| Allocated registers per CTA | 16384 | 20480 | +4096 |
| Static / encoded SASS | 993 / 1008 | 1858 / 1872 | +865 / +864 |
| LDG / STG | 64 / 20 | 120 / 36 | +56 / +16 |
| LDS / STS / launch shared | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 69640 | 125072 | +55432 |

Launch-total warp-weighted work per request:

| Metric | C64 | C128 probe | Delta |
|---|---:|---:|---:|
| Static SASS | 1271040 | 1189120 | -81920 (-6.45%) |
| Encoded SASS | 1290240 | 1198080 | -92160 (-7.14%) |
| LDG | 81920 | 76800 | -5120 (-6.25%) |
| STG | 25600 | 23040 | -2560 (-10.00%) |

Most instruction classes improve or remain equal after CTA weighting. Two
small classes regress: PRMT rises 1.61% and MOV rises 12.5%. Launch-total FADD,
FMUL, MUFU, and SEL counts remain unchanged. These SASS counts are static
issued-work proxies, not measured memory transactions or latency.

## Correctness and determinism

`BLOCK_C=128` exactly divides 10240 channels, so the probe adds no channel
mask or tail path. It compiles the identical prior-reuse function with the same
row mapping, int64 addressing, ordered BF16-product/FP32-add math, and
output/current/prior/zero stores. The focused source suite passed 14 tests.

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and non-launch
metrics are byte-identical. A second build with a separate fresh cache and
output tree reproduced both variants byte for byte. The cubin SHA-256 is
`be2b9bf13d3e0763334db397f562c566e1e08d42f522ff66a42ba5b1ebb64d7c`;
the SASS SHA-256 is
`d9046acd19cb8c421c639145a625494ac0c99d826e8f53ea86a16ff8fcd85aa1`.

The independent verifier re-ran `nvdisasm` and `cuobjdump`, recounted SASS
classes, checked `.target sm_121a` and `.reqntid 256`, verified raw hashes,
enforced zero stack/local/spills/calls, checked launch-total improvement, and
compared B1/B4 and fresh-cache outputs. The embedded producer is
`ptxas-blackwell` 12.9.86 from toolkit 12.9; CUDA 13 tools were used only
for inspection.

Cubin identity here means two fresh builds from the same fixed worktree
metadata. Cubin debug metadata includes the canonical source file table, so
arbitrary checkout timestamps are not claimed to reproduce the container hash.

The package contains scripts and derived summaries only. It includes no
cubin, PTX, SASS, IR, model/task content, request logs, patch content,
environment dump, credential, or secret.

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row32_c128_priorreuse_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row32_c128_priorreuse_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row32_c128_priorreuse_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 4f649835d42b98264ad71b46121637b12f8d9ea1 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 32 --block-c 128 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 4f649835d42b98264ad71b46121637b12f8d9ea1 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 32 --block-c 128 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
