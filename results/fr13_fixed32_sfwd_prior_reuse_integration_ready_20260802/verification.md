# Verification record

Date: 2026-08-02 UTC

## Source checkpoint

- Source commit: `259d050211a8a721feba34d1c4eeab768bd61535`.
- Campaign-tip parent: `2771cf78b86c73b30ef3fcd92a954192196f8483`.
- Source commit was pushed to
  `agent/fixed32-sfwd-priorreuse-integration-20260802`.
- The legacy qualified kernel source is unchanged at SHA-256 `c3036ae4...`.
- The committed launch source manifest generated successfully and hashes to
  `c2a536344925ca27961565e2b3a9e25de8612477d8c633baaf5f8ff52c3088cd`.

Static verification:

```text
Focused regression suite: 158 passed in 1.70s
New-file Ruff checks: pass
Modified-file fatal Ruff checks: pass
Python compilation: pass
Shell parsing: pass
git diff --check: pass
Old-kernel source identity: pass
Read-only lifecycle/source review: no concrete blocker
```

The remaining non-blocking test gap is a fully synthetic artifact/tamper test
for the complex validator. The validator itself was reviewed fail-closed and
no false-PASS route was found.

## Offline builds

The exact committed function `_fr13_fixed32_sfwd_prior_reuse_kernel` was
compiled with `CUDA_VISIBLE_DEVICES=` for SM121a, B1 and B4, row group 32,
`BLOCK_C=64`, eight warps, and three stages. A second build used separate
output, Triton cache, and Python cache directories.

Both builds produced:

```text
status: pass
CTAs per request: 160
CTAs per launch, B1/B4: 160/640
B1/B4 cubin/PTX/SASS/resource identity: true
fresh-cache binary identity: true
fresh disassembly identity: true
registers per thread: 62
allocated registers per CTA: 16384
threads per CTA: 256
launch/ELF shared bytes: 0/0
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS: 993/1008
LDG/STG/LDS/STS: 64/20/0/0
cubin bytes: 69088
```

The independent verifier re-ran `nvdisasm` and `cuobjdump`, recounted SASS
classes, checked `.target sm_121a` and 256 threads per CTA, enforced resource
limits, compared B1 with B4, and compared the primary build with the fresh
rebuild.

## Explicitly not run

- GPU kernel execution or live byte-equivalence gate
- Docker or service/server launch
- Synthetic probe or real SWE-Verified task/request
- B1/B4 runtime timing, TPS, or hardware-floor acceptance
- Production selection

The live next step is deliberately B1-only. B4 is compile-verified here but
cannot be qualified until the B1 two-surface byte gate passes and a separate
real-task B4 gate is defined.
