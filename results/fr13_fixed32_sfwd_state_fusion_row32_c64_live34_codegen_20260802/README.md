# FR13 fixed32 SFWD row32/C64 live-shape offline codegen

Status: **offline SM121a codegen passes the viability and code-size gates;
correctness and performance remain unqualified**.

This artifact audits source commit
`019eb811c4704b127ffed158a06f8741421ab528` at the exact live
specialization: `B=1` and `B=4`, 32 physical rows per request,
`C=10240`, width 4, state length 34, 36 source rows, row group 32,
`BLOCK_C=64`, eight warps, and three stages. CUDA visibility was explicitly
empty. Compilation, cubin inspection, and disassembly were offline only; no
kernel, service, task, request, correctness gate, timing run, or acceptance run
was launched.

The package is reduced and non-sensitive. It contains scripts, derived JSON,
Markdown, TSV, source hashes, and checksums. It contains no cubin, PTX, SASS,
IR, bytecode, model or task content, prompt or patch content, request or process
data, environment dump, credential, or secret.

## Result

B1 and B4 have different Triton compile hashes because batch is part of the
compile contract. Their cubin, PTX, SASS, resource report, and all non-launch
metrics are byte-identical. A second compile using a separate fresh cache and
output tree reproduced both variants byte for byte.

| Metric | B1 | B4 |
|---|---:|---:|
| CTAs per request | 160 | 160 |
| CTAs per launch | 160 | 640 |
| Registers per thread | 109 | 109 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 |
| Static / encoded SASS instructions | 1215 / 1240 | 1215 / 1240 |
| LDG / STG | 116 / 20 | 116 / 20 |
| LDS / STS | 4 / 0 | 4 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 83480 | 83480 |

The cubin SHA-256 is
`a83a5fa657766e46586756f3187b9087a99142ccb5f376f0f4d17c3e43db0397`.
The SASS SHA-256 is
`e064644a42653aef191240beb7c53a05b02bbca17aa9d266f4f1d99dd673be34`.

The independent verifier re-ran `nvdisasm` and `cuobjdump`, recounted
instruction classes, checked `.target sm_121a` and `.reqntid 256`,
verified raw-output hashes and B1/B4 identity, and enforced zero stack, local
memory, spill loads/stores, and calls. The compiled body requests 109 registers
per thread, 256 threads per CTA, and 4096 launch-shared bytes. Its 1215 static
and 1240 encoded instructions and 83480-byte cubin do not exceed the corrected
row8/C256 baseline. These checks establish offline viability, not occupancy or
runtime performance.

The cubin ELF note identifies the actual Triton backend producer as
`ptxas-blackwell`, CUDA toolkit 12.9, version 12.9.86, targeting
`sm_121a`. Torch reports CUDA 13.0. The system `nvdisasm` and
`cuobjdump` used only for inspection are version 13.0.85. System `nvcc`
13.0.88 was inspected with `--version`; it was not used to compile this
kernel and is not attributed as the cubin producer.

## Schedule comparison

All three schedules keep 2048 elements per program, 160 CTAs per request, and
160/640 CTAs per B1/B4 launch.

| Metric | Row8/C256 | Row16/C128 | Row32/C64 |
|---|---:|---:|---:|
| Registers | 111 | 105 | 109 |
| Static SASS instructions | 1219 | 1260 | 1215 |
| Encoded SASS instructions | 1240 | 1288 | 1240 |
| Cubin bytes | 84016 | 86248 | 83480 |
| LDG / STG | 116 / 20 | 116 / 20 | 116 / 20 |
| LDS / STS | 0 / 0 | 32 / 16 | 4 / 0 |

Relative to row16/C128, row32/C64 uses four more registers but removes 45
static and 48 encoded SASS instructions, reduces the cubin by 2768 bytes, and
reduces LDS/STS by 28/16. Relative to corrected row8/C256, it uses two fewer
registers, four fewer static instructions, the same encoded count, a
536-byte-smaller cubin, and four additional LDS instructions. Static codegen
does not establish which schedule is faster.

## Scope and risk

The candidate passes the requested offline codegen/resource gate. Focused
source tests cover the fixed32 contract, padded x row-stride handling,
state-length 34 enforcement, K64/root1 provenance, reference-returning shadow
behavior, default-off production control, and preserved eager lifecycle.

No GPU execution or byte-equivalence run was performed. Correctness, achieved
occupancy, scheduler behavior, and runtime performance remain unknown. The
higher register count than row16/C128 and extra LDS traffic relative to
row8/C256 may matter at runtime despite the smaller instruction body.

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion/results/fr13_fixed32_sfwd_state_fusion_row32_c64_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
PRIMARY=$(mktemp -d /tmp/fr13_sfwd_row32_c64_live34_primary.XXXXXX)
REBUILD=$(mktemp -d /tmp/fr13_sfwd_row32_c64_live34_rebuild.XXXXXX)

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$PRIMARY/cache" \
PYTHONPYCACHEPREFIX="$PRIMARY/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 019eb811c4704b127ffed158a06f8741421ab528 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$PRIMARY/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR="$REBUILD/cache" \
PYTHONPYCACHEPREFIX="$REBUILD/pycache" "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision 019eb811c4704b127ffed158a06f8741421ab528 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output "$REBUILD/output" --rows-per-program 32 --block-c 64 \
  --state-len 34 --num-warps 8 --batches 1 4

"$PY" "$ART/verify_codegen_outputs.py" \
  --primary "$PRIMARY/output" --rebuild "$REBUILD/output"
```
