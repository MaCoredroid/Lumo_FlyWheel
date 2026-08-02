# FR13 fixed32 SFWD row16/C128 live-shape offline codegen

Status: **offline SM121a codegen is viable; correctness and performance remain
unqualified**.

This artifact audits source commit
`d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e` at the exact live specialization:
`B=1` and `B=4`, 32 physical rows per request, `C=10240`, width 4, state
length 34, 36 source rows, row group 16, `BLOCK_C=128`, eight warps, and
three stages. CUDA visibility was explicitly empty. Compilation, cubin
inspection, and disassembly were offline only; no kernel or task was launched.

The package is reduced and non-sensitive. It contains scripts, derived JSON
and Markdown, source hashes, and package checksums. It contains no cubin, PTX,
SASS, model or task content, prompt or patch content, raw request log,
environment dump, credential, or secret.

## Result

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and all non-launch
metrics are byte-identical. A second compile using a separate fresh cache and
output tree reproduced both variants byte for byte.

| Metric | B1 | B4 |
|---|---:|---:|
| CTAs per request | 160 | 160 |
| CTAs per launch | 160 | 640 |
| Registers per thread | 105 | 105 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 |
| Static / encoded SASS instructions | 1260 / 1288 | 1260 / 1288 |
| LDG / STG | 116 / 20 | 116 / 20 |
| LDS / STS | 32 / 16 | 32 / 16 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 86248 | 86248 |

The cubin SHA-256 is
`401b87321eb74a15b1c2ee56ad70785f35a0d301d176f91f5801ce316137bf46`.
The SASS SHA-256 is
`79a919be96a54046f76feb9d2d1cecabf9d2c366107057757dab33fd84a0df9c`.

The independent verifier re-ran `nvdisasm` and `cuobjdump`, recounted every
instruction class, checked `.target sm_121a` and `.reqntid 256`, verified file
hashes and B1/B4 identity, and enforced zero stack, local memory, spill
loads/stores, and calls. The compiled body requests 105 registers per thread,
256 threads per CTA, and 4096 launch-shared bytes. These pass the verifier's
conservative per-thread and per-CTA viability limits. This is not an occupancy
or runtime-performance claim.

The cubin ELF note identifies the actual Triton backend producer as
`ptxas-blackwell`, CUDA toolkit 12.9, version 12.9.86, targeting `sm_121a`.
Torch reports CUDA 13.0. The system `nvdisasm` and `cuobjdump` used only for
inspection are version 13.0.85. System `nvcc` 13.0.88 was present but was not
invoked and is not attributed as the cubin producer.

## Rowgroup8 comparison

The exact live34 rowgroup8/C256 baseline is source revision
`8b1786a34612f789a99159ea19ffe6dea4f75dc3`. Its extracted kernel function has
the same SHA-256 as this candidate, so this comparison isolates the compile
schedule.

The baseline's published `160/640` counter is the B1/B4 launch grid, despite
its earlier `ctas_per_request` label. The table below normalizes that as CTAs
per launch and reports the per-request value separately.

| Metric | Row8 / C256 | Row16 / C128 | Change |
|---|---:|---:|---:|
| CTAs per request | 160 | 160 | 0 |
| B1 / B4 CTAs per launch | 160 / 640 | 160 / 640 | 0 / 0 |
| Registers | 111 | 105 | -6 |
| Static SASS instructions | 1219 | 1260 | +41 |
| Encoded SASS instructions | 1240 | 1288 | +48 |
| Cubin bytes | 84016 | 86248 | +2232 |
| LDG / STG | 116 / 20 | 116 / 20 | 0 / 0 |
| LDS / STS | 0 / 0 | 32 / 16 | +32 / +16 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 | 0 / 0 |

Row16/C128 reduces reported registers by six, but does not reduce CTAs or
global-memory instruction counts and has a larger static body, cubin, and
shared-memory instruction footprint. Offline codegen therefore establishes
viability, not a likely latency win.

## Scope and risk

No spills, calls, or unsafe static resource usage were observed, so the
candidate passes the requested codegen viability gate. No GPU execution,
byte-equivalence check, real task, timing run, acceptance run, or production
selection was performed. Correctness, achieved occupancy, scheduling effects,
and runtime performance remain unknown. In particular, the larger instruction
body and additional LDS/STS traffic could offset the lower register count.

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row16_c128_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row16_c128_live34_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row16_c128_live34_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" \
  "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 16 --block-c 128 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" \
  "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 16 --block-c 128 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
