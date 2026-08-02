# Verification record

Date: 2026-08-02 UTC

## Scope

- Compiled source:
  `4f649835d42b98264ad71b46121637b12f8d9ea1`.
- Artifact parent:
  `89bc742698d6c42f7e2f7dcdf6a7e8d27207b11a`.
- Probed schedule: row32/C128, eight warps, 80 CTAs per request.
- Checked-in source schedule remains row32/C64, 160 CTAs per request.
- No source, timing harness, runner, gate, Docker, GPU, or task state changed.
- Isolated Python compilation and `git diff --check`: pass.

Focused source test retained from the exact compiled source:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
14 passed in 1.51s
```

## Static decision

C128 is classified as an offline launch-total win with runtime occupancy risk.

| Metric | C64 | C128 | Verdict |
|---|---:|---:|---|
| CTAs per request | 160 | 80 | improved |
| Launch-total warp static SASS | 1271040 | 1189120 | improved 6.45% |
| Launch-total warp encoded SASS | 1290240 | 1198080 | improved 7.14% |
| Launch-total warp LDG | 81920 | 76800 | improved 6.25% |
| Launch-total warp STG | 25600 | 23040 | improved 10.00% |
| Allocated registers per CTA | 16384 | 20480 | risk, +25% |
| Register-limited CTAs / warps per SM | 4 / 32 | 3 / 24 | risk |
| B1 mean grid CTAs per 48-SM GB10 | 3.33 | 1.67 | risk |
| Shared / spills / local / calls | 0 | 0 | unchanged |

The source and SASS evidence cannot resolve whether fewer CTAs and instructions
outweigh lower latency-hiding capacity. C128 is not selected without GPU
correctness and timing.

## Offline builds

The audit refused `CUDA_VISIBLE_DEVICES=0` with exit code 1, as required.

Primary:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c128_priorreuse_primary.O0e7Lb/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c128_priorreuse_primary.O0e7Lb/pycache
--output /tmp/fr13_sfwd_row32_c128_priorreuse_primary.O0e7Lb/output
--revision 4f649835d42b98264ad71b46121637b12f8d9ea1
--rows-per-program 32 --block-c 128 --state-len 34
--num-warps 8 --batches 1 4
```

Fresh rebuild:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c128_priorreuse_rebuild.d0cA71/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c128_priorreuse_rebuild.d0cA71/pycache
--output /tmp/fr13_sfwd_row32_c128_priorreuse_rebuild.d0cA71/output
```

Both B1 and B4 builds passed without a visible GPU or kernel launch.

## Independent verification

```text
status: pass
target: sm_121a
backend producer: ptxas-blackwell 12.9.86 (CUDA toolkit 12.9)
CTAs per request: 80
CTAs per launch, B1/B4: 80/320
B1/B4 binary identity: true
fresh-cache binary identity: true
fresh disassembly identity: true
registers per thread: 80
allocated registers per CTA: 20480
register-limited CTAs/warps per SM: 3/24
threads per CTA: 256
launch shared bytes: 0
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS per CTA: 1858/1872
LDG/STG per CTA: 120/36
cubin bytes: 125072
launch-total improvement gate: pass
occupancy risk: runtime required
```

The verifier checked raw output hashes, independently re-ran `nvdisasm` and
`cuobjdump`, recounted SASS classes, checked PTX target/thread metadata,
enforced resource safety, compared launch-total work with C64, compared B1
with B4, and compared the primary build with the fresh rebuild.

## Explicitly not run

- GPU kernel execution or byte-equivalence gate
- Docker or GPU service/server launch
- Synthetic probe or real SWE-Verified task/request
- B1/B4 runtime timing or floor acceptance
- Source schedule or production selection

Raw compiler outputs remain under `/tmp` and are not packaged.
