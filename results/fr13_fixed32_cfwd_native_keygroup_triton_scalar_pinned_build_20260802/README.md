# Fixed32 CFWD Triton-scalar pinned-image build

Status: **full extension build and source binding pass; default off; real-task
byte qualification pending**.

The v4 Triton-scalar repair was rebuilt into the full vLLM `_C.abi3.so`
extension from the pinned serving image toolchain. The binding gate forced the
candidate CUDA object to rebuild from the patched source and then forced the
full extension to relink. A host-side verification subsequently matched the
binary, candidate source, patcher source, patched vLLM files, build graph, and
pinned vLLM base commit.

## Bound outputs

- Candidate: `fixed32_cfwd_native_keygroup_triton_scalar_cuda_v4`
- Architecture: `sm_121a`
- Full extension: 201,386,728 bytes, SHA256
  `c9647856777d17dd3cde07a1d0e0b4c87ea1fb823bebd65f45ded1d461f8af26`
- Candidate object: 6,229,472 bytes, SHA256
  `7d405783dea70b04e2bae89ffd6a0ae2779119b689884a5f36d1de3f80d31546`
- Candidate source SHA256:
  `5699ab062624bd2f6368143c48068bfccf1f9c3b5629e243d92616b94359bc54`
- Pinned vLLM base commit: `fe9c3d6c5f66c873d196800384ed6880687b9e52`

The extension has no post-GLIBC-2.35 dynamic symbol reference, and it contains
the expected registered operator and v4 tensor signature.

## SM121a codegen

The exact object in the full extension build passed the frozen checker:

| Resource | Result |
| --- | ---: |
| Registers/thread | 64 |
| Stack/local bytes | 0 / 0 |
| Reported shared bytes | 7,592 |
| `LDL` / `STL` / `CALL` | 0 / 0 / 0 |
| `MUFU.EX2` | 0 |
| `MUFU.RSQ` / `MUFU.RCP` | 3 / 1 |
| `SHFL.BFLY` / `SHFL.IDX` | 202 / 16 |
| `FFMA` | 70 |

## Qualification boundary

This is build and static-codegen evidence only. It supplies no timing result,
hardware-floor claim, production authorization, or byte-equivalence result.
The candidate remains default-off and timing-ineligible until the same
authenticated real SWE-Verified B1 all-depth byte gate passes. B4 and timing
remain out of scope until that gate closes.

This directory excludes tasks, prompts, responses, patches, raw logs,
environment dumps, process/container identities, credentials, binaries,
objects, PTX/SASS dumps, and timing samples.
