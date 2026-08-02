# Verification record

Date: 2026-08-02 UTC

## Source checkpoint

- Source parent: `0482413f190d8a4c4541eecb54c235c33d53fdab`.
- Final-tap source commit:
  `9920370699fa11e677c510bc28bc066eed18ad88`.
- The remote branch resolved to the same source commit before packaging.
- Only the SFWD kernel source and its focused test changed.
- Isolated Python compilation: pass.
- `git diff --check`: pass.

Focused source test:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
13 passed in 0.89s
```

The suite checks that every fixed32 descriptor's final tap is its current node,
that the generic loop stops before the specialized tap, and that the direct
current-row weight/product/add path is present. Existing geometry, stride,
math-order, state-length, and control tests remain active.

## Improvement gate

| Metric | x-reuse v6 | Final tap | Verdict |
|---|---:|---:|---|
| CTAs per request | 160 | 160 | unchanged |
| Allocated registers/thread | 112 | 112 | unchanged |
| Static / encoded SASS | 1163 / 1184 | 1071 / 1088 | improved |
| Warp-weighted static / encoded SASS | 9304 / 9472 | 8568 / 8704 | improved |
| LDG / warp-weighted LDG | 108 / 864 | 91 / 728 | improved |
| LDS | 4 | 3 | improved |
| Stack / local / spills / calls | 0 | 0 | unchanged |

## Offline builds

The audit refused `CUDA_VISIBLE_DEVICES=0` with exit code 1, as required.

Primary:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_finaltap_primary.Ztyl86/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_finaltap_primary.Ztyl86/pycache
--output /tmp/fr13_sfwd_row32_c64_finaltap_primary.Ztyl86/output
--revision 9920370699fa11e677c510bc28bc066eed18ad88
--rows-per-program 32 --block-c 64 --state-len 34
--num-warps 8 --batches 1 4
```

Fresh rebuild:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_finaltap_rebuild.kwJnHW/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_finaltap_rebuild.kwJnHW/pycache
--output /tmp/fr13_sfwd_row32_c64_finaltap_rebuild.kwJnHW/output
```

Both B1 and B4 builds passed without a visible GPU or kernel launch.

## Independent verification

```text
status: pass
target: sm_121a
backend producer: ptxas-blackwell 12.9.86 (CUDA toolkit 12.9)
CTAs per request: 160
CTAs per launch, B1/B4: 160/640
B1/B4 binary identity: true
fresh-cache binary identity: true
fresh disassembly identity: true
registers per thread, v6/final-tap: 112/106
allocated registers per thread, v6/final-tap: 112/112
threads per CTA: 256
launch shared bytes: 4096
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS: 1071/1088
warp-weighted static/encoded SASS: 8568/8704
LDG / warp-weighted LDG: 91/728
LDS: 3
cubin bytes: 74152
improvement gate: pass
```

The verifier checked raw output hashes, independently re-ran `nvdisasm` and
`cuobjdump`, recounted SASS classes, checked PTX target/thread metadata,
enforced resource gates, compared B1 with B4, and compared the primary build
with the fresh rebuild.

## Explicitly not run

- GPU kernel execution or byte-equivalence gate
- GPU service/server launch
- Synthetic probe or real SWE-Verified task/request
- B1/B4 runtime timing or floor acceptance
- Production selection

Raw compiler outputs remain under `/tmp` and are not packaged.
