# Verification record

Date: 2026-08-02 UTC

## Source checkpoint

- Source parent: `d01f7cbe57bdd266e47c8d0826c15a7072dfcf53`.
- Prior-vector reuse source commit:
  `4f649835d42b98264ad71b46121637b12f8d9ea1`.
- The remote branch resolved to the same source commit before packaging.
- Only the SFWD kernel source and its focused test changed.
- Isolated Python compilation: pass.
- `git diff --check`: pass.

Focused source test:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
14 passed in 1.70s
```

The new CPU invariant exhaustively compares generic and reused prior-vector
selection for all 32 nodes and taps 0–2. Existing exact descriptor, direct
fused indexing, final-tap, geometry, stride, math-order, and control checks
remain active.

## Improvement gate

| Metric | Final tap | Prior reuse | Verdict |
|---|---:|---:|---|
| CTAs per request | 160 | 160 | unchanged |
| Allocated registers/thread | 112 | 64 | improved |
| Allocated registers/CTA | 28672 | 16384 | improved |
| Launch shared bytes | 4096 | 0 | improved |
| Static / encoded SASS | 1071 / 1088 | 993 / 1008 | improved |
| Warp-weighted static / encoded SASS | 8568 / 8704 | 7944 / 8064 | improved |
| LDG / warp-weighted LDG | 91 / 728 | 64 / 512 | improved |
| LDS | 3 | 0 | improved |
| Stack / local / spills / calls | 0 | 0 | unchanged |

## Offline builds

The audit refused `CUDA_VISIBLE_DEVICES=0` with exit code 1, as required.

Primary:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_priorreuse_primary.MPglgF/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_priorreuse_primary.MPglgF/pycache
--output /tmp/fr13_sfwd_row32_c64_priorreuse_primary.MPglgF/output
--revision 4f649835d42b98264ad71b46121637b12f8d9ea1
--rows-per-program 32 --block-c 64 --state-len 34
--num-warps 8 --batches 1 4
```

Fresh rebuild:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_priorreuse_rebuild.11wTes/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_priorreuse_rebuild.11wTes/pycache
--output /tmp/fr13_sfwd_row32_c64_priorreuse_rebuild.11wTes/output
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
registers per thread, final/prior-reuse: 106/62
allocated registers per thread, final/prior-reuse: 112/64
allocated registers per CTA, final/prior-reuse: 28672/16384
threads per CTA: 256
launch shared bytes: 0
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS: 993/1008
warp-weighted static/encoded SASS: 7944/8064
LDG / warp-weighted LDG: 64/512
LDS: 0
cubin bytes: 69640
improvement gate: pass
```

The verifier checked raw output hashes, independently re-ran `nvdisasm` and
`cuobjdump`, recounted SASS classes, checked PTX target/thread metadata,
enforced resource gates, compared B1 with B4, and compared the primary build
with the fresh rebuild.

## Explicitly not run

- GPU kernel execution or byte-equivalence gate
- Docker or GPU service/server launch
- Synthetic probe or real SWE-Verified task/request
- B1/B4 runtime timing or floor acceptance
- Production selection

Raw compiler outputs remain under `/tmp` and are not packaged.
