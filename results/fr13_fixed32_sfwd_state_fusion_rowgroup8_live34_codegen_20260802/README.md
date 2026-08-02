# FR13 fixed32 SFWD row-group-8 live-geometry codegen

Status: **offline SM121a codegen passes at the corrected live geometry; a real
SWE-Verified B1 byte gate is still required**.

This successor artifact supersedes the deployability claim in
`../fr13_fixed32_sfwd_state_fusion_rowgroup8_codegen_20260802/`. That earlier
artifact compiled the historical 12-row convolution-state contract. The live
runtime uses state length 34 and may present padded input rows, so this audit
adds the scalar input row stride and compiles the exact live specialization.
The earlier artifact remains useful only as historical codegen evidence.

The package contains no cubin, PTX, SASS, task text, model output, patch,
request log, environment dump, process identity, credential, or secret.

## Bound contract

- Source revision: `8b1786a34612f789a99159ea19ffe6dea4f75dc3`
- Candidate: `fixed32_sfwd_state_fusion_rowgroup8_v3`
- Kernel source SHA-256:
  `c3036ae4775553e3aeb2131e8b3609c852a22ab86493f7d9843d4aeaed825a70`
- `B=1` and `B=4`, `N=32`, `C=10240`, width `4`, state length `34`
- source rows `36`, `BLOCK_C=256`, row group `8`, eight warps, three stages
- BF16 data surfaces, int32 state indices, int64 source descriptor, no bias
- Triton 3.6.0, Torch 2.10.0+cu130, CUDA 13.0 target `sm_121a`

The audit reads the kernel through `git show` at the bound revision. It runs
with CUDA visibility explicitly empty and calls the offline Triton compiler;
it never launches a GPU kernel.

## Result

B1 and B4 produce byte-identical cubins. Their compile hashes differ because
batch remains part of the compile contract, while the generated body is the
same.

| Metric | B1 | B4 |
|---|---:|---:|
| CTAs per request | 160 | 640 |
| Registers | 111 | 111 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| Launch / ELF shared bytes | 4096 / 1024 | 4096 / 1024 |
| Static / encoded SASS instructions | 1219 / 1240 | 1219 / 1240 |
| LDG / STG | 116 / 20 | 116 / 20 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Cubin bytes | 84016 | 84016 |

Cubin SHA-256 is
`53abe443d9f7b8dd2816afac88804df46a3a48c2e4ecb797b6a97358b1f51257`;
SASS SHA-256 is
`b1fe1a7dec4311a4ab4aeec27660be7308cd4108983c39934b5570a2f8d92e5d`.
A second compile from a fresh cache reproduced both summaries and both cubins
byte for byte.

Relative to the obsolete 12-state-row report, the corrected specialization
uses 23 more registers, 136 more static instructions, 136 more encoded
instructions, and a 5,920-byte larger cubin. CTA count and global-load/store
counts are unchanged. Zero stack, local memory, spill loads/stores, and calls
remain the important viability gates.

## Verdict

The exact live-shape candidate is **viable for the next real B1 correctness
gate**. This is static codegen evidence only: it is not byte-equivalence,
acceptance, latency, TPS, production, or hardware-floor evidence. The
candidate remains shadow-only and default-off until the authenticated real
SWE-Verified B1 gate covers all 48 layers and returns incumbent bytes.

## Verification

```text
bash syntax: pass
python bytecode compilation: pass
focused pytest: 149 passed in 2.75s
fresh-cache B1/B4 rebuild: byte-identical summaries and cubins
git diff --check: pass
```

## Reproduction

```bash
ART=/home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8/results/fr13_fixed32_sfwd_state_fusion_rowgroup8_live34_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
export CUDA_VISIBLE_DEVICES=

TRITON_CACHE_DIR=/tmp/fr13_sfwd_row8_live34_repro_cache "$PY" \
  "$ART/offline_codegen_audit.py" \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8 \
  --revision 8b1786a34612f789a99159ea19ffe6dea4f75dc3 \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output /tmp/fr13_sfwd_row8_live34_repro \
  --rows-per-program 8 --block-c 256 --state-len 34 \
  --num-warps 8 --batches 1 4
```
