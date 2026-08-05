# Fixed32 B4 FA2 qrow32 GQA-pair SM121a build

Status: **offline static build pass; real B4 byte gates pending**.

This artifact records the CUDA 13 `sm_121a` build, codegen audit, ABI
comparison, and offline load check for the default-off B4 qrow32 GQA-pair
candidate. It is not a performance result. No GPU device was visible or used,
no real or synthetic task was run, and no output or LSE parity claim is made.

## Built source

- Repository base: `dbf46dc58b05a16fe245167e8e1cd498c813daf6`.
- FA2 source: `29210221863736a08f71a866459e368ad1ac4a95`.
- Patcher flags: `--tree-bias-tile-earlyout --fixed32-query-gqa-pair32`.
- Translation unit: `flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu`.
- The six-file patched FA2 source used by the build byte-matches regeneration
  from repository base `dbf46dc58`.

The private launcher maps two adjacent query heads to one BM64, four-warp CTA.
Its canonical B4 grid is `3 x 4 x 4 = 48` CTAs per layer, with no split-K or
combine launch. This is a source/codegen statement, not measured speedup.

## Static qualification

- Exactly one cubin: `sm_121a`.
- Target kernel: 243 registers, zero stack, zero static local memory, zero
  spill stores, and zero spill loads.
- Shared memory: 1,024 bytes static in the cubin and 98,304 bytes dynamic,
  fixed by a source assertion.
- Target SASS: zero `LDL`, `STL`, and `CALL` instructions.
- Public defined symbols, undefined symbols, `DT_NEEDED`, and `RUNPATH` match
  the qrow16 reference shared object exactly.
- The GQA-pair launcher is `LOCAL`, not part of the dynamic symbol table.
- Offline `torch.ops.load_library` succeeded and registered `varlen_fwd` and
  `varlen_fwd_tree_bias`.

Compilation and linking used the pinned vLLM image with `--network none`, one
CPU, `NVIDIA_VISIBLE_DEVICES=void`, empty `CUDA_VISIBLE_DEVICES`, and
`CUDA_CACHE_DISABLE=1`. The image contains `cuobjdump` but not `nvdisasm`, so
the matching CUDA 13 host `nvdisasm` binary was bind-mounted read-only into
the same offline container for SASS decoding. Tool build IDs are recorded in
`verification.txt`.

## Admission boundary

This candidate remains default-off and is neither timing-eligible nor
production-eligible. The next required evidence is retained-operand raw-byte
comparison on the canonical real SWE-Verified exact4 B4 set for both Tail23
and Hydra27, covering BF16 output and FP32 LSE for all 16 tree-attention
layers. Clean B4 timing can begin only if both byte gates pass.

The shared object and intermediate objects are intentionally not committed.
Their hashes and sizes are recorded in `manifest.json`.
