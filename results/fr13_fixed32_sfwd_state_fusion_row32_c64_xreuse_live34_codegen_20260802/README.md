# FR13 fixed32 SFWD row32/C64 current-x reuse codegen

Status: **offline SM121a codegen improves the immediate row32/C64 baseline;
correctness and performance remain unqualified**.

This artifact audits source commit
`3d268dda7ba60cec7ef430445820602794dbe13c` at the exact live
specialization: B1 and B4, 32 physical rows per request, `C=10240`, width 4,
state length 34, 36 source rows, row group 32, `BLOCK_C=64`, eight warps, and
three stages. CUDA visibility was explicitly empty. No kernel, service, task,
request, correctness gate, timing run, or acceptance run was launched.

## Change

The validated fixed32 source descriptor always maps tap 3 to the current tree
node. The kernel now reuses that already-loaded BF16 `x_value` for persistent
commit-source staging after the output store, instead of loading the same
`x` tile a second time. The host still validates every descriptor value
before launch, pointer/index arithmetic stays int64, math and store order are
unchanged, and the row32/C64 grid stays at 160 CTAs per request.

## Result

| Metric | row32/C64 v5 | x-reuse v6 | Delta |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Warps / threads per CTA | 8 / 256 | 8 / 256 | 0 / 0 |
| Reported registers per thread | 109 | 112 | +3 |
| Allocated registers per thread | 112 | 112 | 0 |
| Static / encoded SASS | 1215 / 1240 | 1163 / 1184 | -52 / -56 |
| Warp-weighted static / encoded SASS | 9720 / 9920 | 9304 / 9472 | -416 / -448 |
| LDG / warp-weighted LDG | 116 / 928 | 108 / 864 | -8 / -64 |
| STG / LDS / STS | 20 / 4 / 0 | 20 / 4 / 0 | 0 / 0 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 | 0 / 0 |
| Stack / local / LDL / STL / calls | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 |
| Cubin bytes | 83480 | 81288 | -2192 |

The 109- and 112-register reports both round to 112 registers per thread under
the 256-register-per-warp allocation quantum. This avoids a register-allocation
step regression, but achieved occupancy was not measured.

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and non-launch
metrics are byte-identical. A second build with a separate fresh cache and
output tree reproduced both variants byte for byte. The selected cubin SHA-256
is `26a9352b132e57e04c64e4daa751e7722b6470bb7246233c695cec42473e4c94`;
the SASS SHA-256 is
`9ba8abd312dc81de3769b780456e227c5e2fbaf1d6076acbd6e85d0c50214cb8`.

## Rejected 16-warp schedule

The initially explored 16-warp schedule reduced per-thread registers and
static code size, but doubled the executing warp count. Against row32/C64 v5,
warp-weighted static SASS rose `9720 -> 11120`, encoded SASS rose
`9920 -> 11520`, and LDG rose `928 -> 1024`. It also increased allocated
registers per CTA from 28672 to 32768 and used a 512-thread CTA. With no runtime
or occupancy evidence to justify those costs, it was rejected.

## Verification and limits

The focused source suite passed 13 tests. It now proves every descriptor's
final tap is its current node and checks that the redundant post-loop load is
absent. The independent verifier re-ran `nvdisasm` and `cuobjdump`,
recounted instruction classes, checked `.target sm_121a` and
`.reqntid 256`, verified raw hashes, enforced zero stack/local/spills/calls,
and compared B1/B4 and fresh-cache outputs.

The embedded producer is `ptxas-blackwell` 12.9.86 from toolkit 12.9;
system CUDA 13 tools were used only for inspection. Cubin identity here means
two fresh builds from the same fixed worktree metadata. Cubin debug metadata
includes the canonical source file table, so arbitrary checkout timestamps are
not claimed to reproduce the container hash; PTX, SASS, and resource identity
are the code-relevant checks.

The package is reduced and non-sensitive. It contains scripts and derived
summaries only, with no cubin, PTX, SASS, IR, model/task content, request logs,
patch content, environment dump, credential, or secret.

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row32_c64_xreuse_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row32_c64_xreuse_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row32_c64_xreuse_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 3d268dda7ba60cec7ef430445820602794dbe13c \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 3d268dda7ba60cec7ef430445820602794dbe13c \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
