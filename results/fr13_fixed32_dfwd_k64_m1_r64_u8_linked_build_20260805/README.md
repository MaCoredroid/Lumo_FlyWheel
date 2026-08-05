# Fixed32 DFWD K64 M1 R64-U8 linked build

Status: **reproducible linked build, default off and runtime unwired**. No GPU
kernel execution, SWE-Verified task, byte gate, timing, acceptance, or
production admission was performed.

The reviewed source was linked twice from empty build directories in the
pinned `vllm/vllm-openai:cu130-nightly` environment with PyTorch
`2.11.0+cu130`, CUDA 13.0, and target `sm_121a`. The build uses nvcc
`--frandom-seed=fr13_bf16_k64_m1_r64_u8`; both shared objects and both
extracted cubins are byte-identical.

## Build result

- Source SHA-256: `af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b`
- Shared-object SHA-256: `8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e`
- Shared-object bytes: `117904`
- Extracted cubin SHA-256: `ae3e8ac1fbdf88d0ca9d8c8f8fb971512d0622b02f1c425132dffb7cd696709b`
- Registers/thread: `29`
- Stack/local/shared bytes: `0/0/0`
- Registered op: `fr13_bf16_k64_head::gemvx_m1_shuffle_r64_u8_out`

The linked shared object is deliberately omitted from Git. The retained local
gate input is
`/home/mark/fr13_dfwd_u8_linked_build_3bdd984c2/det-primary-bin/fr13_bf16_k64_m1_r64_u8.abi3.so`.

## Qualification boundary

This package proves only that the exact source reproducibly builds and
registers in the deployed Torch/CUDA ABI. The next valid step is a default-off
real SWE-Verified B1 shadow gate comparing all 65,536 BF16 logits at root and
MTP depths 1 through 4 while serving the incumbent. Only after that gate may
the exact4 and exact16 acceptance/TPS campaign measure this candidate.
