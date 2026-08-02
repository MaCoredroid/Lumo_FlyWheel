# FR13 fixed32 SFWD row-group-8 offline codegen

Status: **offline SM121a codegen passed and is viable for the next B1 byte
gate; this is not a latency claim**.

This artifact audits the row-group-8 fused conv/state kernel at the exact
deployed Qwen fixed32 specialization and reconstructs the published row-group-4
codegen as its comparison. It contains no cubins, PTX, SASS, task text, model
outputs, timing samples, traces, environment dumps, or secrets.

## Exact compile contract

- Model config SHA256:
  `f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df`
- `C=10240`: `2 * (16 key heads * 128) + (48 value heads * 128)`
- `N=32`, conv width `4`, state length `12`, source rows `36`
- BF16 activations, state, weights, output, source stage, and disabled-bias
  placeholder
- int32 state indices; int64 source descriptor; bias disabled
- `BLOCK_C=256`, row-group `8`, eight warps, three stages
- Triton 3.6.0, Torch 2.10.0+cu130, CUDA 13.0 target `sm_121a`
- independently compiled `B=1` and `B=4`

The `37e88f2be` launcher closes `C` at 10240 before removing channel-tail
masks. `N=32` is also closed and divisible by eight. These facts make the
offline specialization internally consistent; they do not establish runtime
byte equality.

## Result

The final row-group-8 cubin compiles at 88 registers with zero stack, local
memory, spill instructions, or calls. Triton reports 4096 bytes of launch
shared memory while the ELF resource report exposes 1024 bytes. B1 and B4
cubins are byte-identical because `B` remains a compile-time contract value but
does not change the generated body.

| Metric | Row-group 4 | Row-group 8 (`37e88f2be`) | Change |
|---|---:|---:|---:|
| B1 / B4 CTAs | 320 / 1280 | 160 / 640 | 0.5x |
| Registers | 64 | 88 | +24 |
| Launch shared bytes | 2048 | 4096 | +2048 |
| ELF shared bytes | 1024 | 1024 | 0 |
| Static SASS body instructions | 633 | 1083 | +450 |
| Encoded SASS instructions | 664 | 1104 | +440 |
| LDG / STG | 64 / 12 | 116 / 20 | +52 / +8 |
| LDS / STS | 16 / 8 | 0 / 0 | -16 / -8 |

The static-body count follows the row-group-4 artifact convention and excludes
`NOP`, `BAR`, `BRA`, and `EXIT`; the encoded count includes them. `LDL`, `STL`,
and `CALL` counts are zero in every compiled revision.

The pure row-group-8 revision `b540aa20d` and redundant-row-mask revision
`75d678613` produce identical code-only SASS: 86 registers, 1094 static body
instructions, and 1128 encoded instructions. The final channel specialization
reduces that to 1083 / 1104 and cuts static branches from 16 to 4, at the cost
of two additional registers.

## Verdict

Row-group 8 is **viable offline**: exact B1/B4 SM121a compilation passes, the
CTA count halves relative to row-group 4, and stack/local/spill/call gates stay
zero. The trade is higher register pressure, twice the launch shared-memory
request, and a substantially larger static body. This is not a performance or
latency result. The candidate remains default-off, not byte-qualified, not
timing-eligible, and not production-eligible.

The next allowed GPU action is the authenticated real SWE-Verified B1
reference-returning byte gate across all 48 GDN layers, requiring exact conv
output and full commit-source-stage bytes while serving incumbent bytes.

## Reproduction

The checked-in script refuses to run unless CUDA visibility is explicitly
empty. It invokes `triton.compile` with an explicit `GPUTarget("cuda", 121,
32)` and audits the resulting cubin with `nvdisasm` and `cuobjdump`; it never
launches a kernel.

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8/results/fr13_fixed32_sfwd_state_fusion_rowgroup8_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
export CUDA_VISIBLE_DEVICES=

TRITON_CACHE_DIR=/tmp/fr13_sfwd_repro_row4_cache "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision f7456b7fc83bdc292cf25b4f2d15e22a2f224363 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output /tmp/fr13_sfwd_repro_row4 --rows-per-program 4 --num-warps 8 \
  --batches 1 4

TRITON_CACHE_DIR=/tmp/fr13_sfwd_repro_row8_cache "$PY" "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8 \
  --revision 37e88f2be00373b25adbc63c6e923068442f7089 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output /tmp/fr13_sfwd_repro_row8 --rows-per-program 8 --num-warps 8 \
  --batches 1 4
```

The canonical source paths are intentional because compiler line information is
part of the cubin bytes. Intermediate revisions can be reproduced by replacing
the row-group-8 revision and output/cache names with `b540aa20d482dadf215575922189925a5f5a6ed0`
or `75d6786138e146580a4c5b8df9d669037210e7af`.
