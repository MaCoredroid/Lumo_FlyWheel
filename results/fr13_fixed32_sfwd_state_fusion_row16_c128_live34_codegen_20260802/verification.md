# Verification record

Date: 2026-08-02 UTC

## Source and syntax

- `git status --short --branch`: clean before edits; branch
  `agent/fixed32-sfwd-state-fusion-row16-c128-20260802` at
  `d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e`.
- `git show d83b5aa40:<path> | sha256sum` for the kernel, changed gate script,
  and focused test: pass; hashes are in `source_checksums.sha256`.
- `/home/mark/fr13_streamk_build/venv/bin/python -m py_compile` for the kernel,
  focused test, audit script, and verifier: pass with bytecode directed to
  isolated `/tmp` directories.
- `bash -n scripts/fr13_run_b1_sfwd_state_fusion_gate.sh`: pass.
- `git diff --check`: pass before packaging.

The compiler venv does not contain pytest (`No module named pytest`), so the
focused suite used the repository test venv:

```text
.venv/bin/python -m pytest -q tests/test_fr13_fixed32_sfwd_state_fusion.py
13 passed in 1.50s
```

## CUDA visibility guard

```text
CUDA_VISIBLE_DEVICES=0 /home/mark/fr13_streamk_build/venv/bin/python \
  offline_codegen_audit.py --help
refusing to run unless CUDA_VISIBLE_DEVICES is explicitly empty
```

Expected refusal: pass.

## Primary compile

```bash
CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row16_c128_live34_primary_v2.PToX7I/cache \
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row16_c128_live34_primary_v2.PToX7I/pycache \
/home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_sfwd_state_fusion_row16_c128_live34_codegen_20260802/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-sfwd-state-fusion \
  --revision d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e \
  --canonical-path /home/mark/lumoFlyWheel-sfwd-state-fusion/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --output /tmp/fr13_sfwd_row16_c128_live34_primary_v2.PToX7I/output \
  --rows-per-program 16 --block-c 128 --state-len 34 \
  --num-warps 8 --batches 1 4
```

Result: pass for B1 and B4. No GPU was visible and no kernel was launched.

## Fresh rebuild

The same command was run with separate fresh paths:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row16_c128_live34_rebuild_v2.0mxsI0/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row16_c128_live34_rebuild_v2.0mxsI0/pycache
--output /tmp/fr13_sfwd_row16_c128_live34_rebuild_v2.0mxsI0/output
```

Result: pass for B1 and B4.

## Independent verification

```bash
/home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_sfwd_state_fusion_row16_c128_live34_codegen_20260802/verify_codegen_outputs.py \
  --primary /tmp/fr13_sfwd_row16_c128_live34_primary_v2.PToX7I/output \
  --rebuild /tmp/fr13_sfwd_row16_c128_live34_rebuild_v2.0mxsI0/output
```

```text
status: pass
target: sm_121a
backend producer: ptxas-blackwell 12.9.86 (CUDA toolkit 12.9)
CTAs per request: 160
CTAs per launch, B1/B4: 160/640
B1/B4 binary identity: true
fresh-cache binary identity: true
fresh disassembly identity: true
registers per thread: 105
threads per CTA: 256
registers per CTA, unrounded: 26880
launch shared bytes: 4096
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
```

The verifier read the raw temporary cubin/PTX/SASS/resource/ELF outputs, checked
their hashes against each audit summary, independently re-ran `nvdisasm` and
`cuobjdump`, recounted SASS operations, checked PTX target/thread metadata,
enforced resource gates, compared B1 with B4, and compared the primary compile
with the fresh rebuild.

## Explicitly not run

- GPU kernel execution
- GPU service or server launch
- real task or request
- B1/B4 correctness gate
- timing or acceptance run
- production selection

Raw compiler outputs remained under `/tmp` and are not included in this
artifact.
