# Verification record

Date: 2026-08-02 UTC

## Source checkpoint

- Source parent: `adc586533837feb0c793d451db77076a553380e3`.
- Current-x reuse source commit:
  `3d268dda7ba60cec7ef430445820602794dbe13c`.
- The remote branch resolved to the same source commit before codegen.
- `bash -n scripts/fr13_run_b1_sfwd_state_fusion_gate.sh`: pass.
- Isolated Python compilation: pass.
- `git diff --check`: pass.

Focused source test:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
13 passed in 1.44s
```

## Schedule decision

The 16-warp exploration was rejected after warp weighting:

| Metric | w8 v5 | w16 exploration |
|---|---:|---:|
| Warp-weighted static SASS | 9720 | 11120 |
| Warp-weighted encoded SASS | 9920 | 11520 |
| Warp-weighted LDG | 928 | 1024 |
| Allocated registers per CTA | 28672 | 32768 |

An int32 source-descriptor cast was also rejected. Combined with x reuse it
changed registers `112 -> 92`, but increased static/encoded SASS
`1163/1184 -> 1279/1304` and cubin bytes `81288 -> 85840`, while LDG
remained 108. Keeping int64 also avoids expanding the address-range proof.

The selected eight-warp x-reuse source improves the immediate v5 baseline:
`LDG 116 -> 108`, static/encoded SASS `1215/1240 -> 1163/1184`, and
cubin bytes `83480 -> 81288`. Reported registers rise `109 -> 112`, but
both use the same 112-register allocation quantum.

## Offline builds

The audit refused `CUDA_VISIBLE_DEVICES=0` with exit code 1, as required.

Primary:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_xreuse_primary.ZnC881/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_xreuse_primary.ZnC881/pycache
--output /tmp/fr13_sfwd_row32_c64_xreuse_primary.ZnC881/output
--revision 3d268dda7ba60cec7ef430445820602794dbe13c
--rows-per-program 32 --block-c 64 --state-len 34
--num-warps 8 --batches 1 4
```

Fresh rebuild:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_xreuse_rebuild.CGIA0K/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_xreuse_rebuild.CGIA0K/pycache
--output /tmp/fr13_sfwd_row32_c64_xreuse_rebuild.CGIA0K/output
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
registers per thread, v5/x-reuse: 109/112
allocated registers per thread, v5/x-reuse: 112/112
threads per CTA: 256
launch shared bytes: 4096
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS: 1163/1184
warp-weighted static/encoded SASS: 9304/9472
LDG / warp-weighted LDG: 108/864
cubin bytes: 81288
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
