# Verification record

Date: 2026-08-02 UTC

## Source and syntax

- The branch was clean before edits at source parent
  `d83b5aa40a876e3a9ee3bc667a2b3e814ba9e42e`.
- The row32/C64 source change was committed as
  `019eb811c4704b127ffed158a06f8741421ab528` and the matching remote branch
  head was verified before the audit.
- Source SHA-256 and Git blob bindings are recorded in `manifest.json` and
  `source_checksums.sha256`.
- Python compilation passed for the kernel, focused test, audit script, and
  verifier with bytecode directed to isolated `/tmp` directories.
- `bash -n scripts/fr13_run_b1_sfwd_state_fusion_gate.sh`: pass.
- `git diff --check`: pass.

The focused source suite used the repository test environment with CUDA hidden,
bytecode disabled, and the pytest cache provider disabled:

```text
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_fr13_fixed32_sfwd_state_fusion.py
13 passed in 1.31s
```

The suite qualifies the fixed32 launch contract, exact source mapping and CPU
reference indexing, padded x row-stride acceptance, state length 34, K64/root1
provenance, source-bound default-off production control, reference-returning
shadow behavior, and the inherited eager lifecycle.

## CUDA visibility guard

```text
CUDA_VISIBLE_DEVICES=0 .../offline_codegen_audit.py --help
refusing to run unless CUDA_VISIBLE_DEVICES is explicitly empty
exit_code=1
```

Expected refusal: pass.

## Primary compile

```text
CUDA_VISIBLE_DEVICES=
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_live34_primary.qQGwJn/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_live34_primary.qQGwJn/pycache
--output /tmp/fr13_sfwd_row32_c64_live34_primary.qQGwJn/output
--revision 019eb811c4704b127ffed158a06f8741421ab528
--rows-per-program 32 --block-c 64 --state-len 34
--num-warps 8 --batches 1 4
```

Result: pass for B1 and B4. No GPU was visible and no kernel was launched.

## Fresh rebuild

The exact compile was repeated with separate fresh locations:

```text
TRITON_CACHE_DIR=/tmp/fr13_sfwd_row32_c64_live34_rebuild.bo79nA/cache
PYTHONPYCACHEPREFIX=/tmp/fr13_sfwd_row32_c64_live34_rebuild.bo79nA/pycache
--output /tmp/fr13_sfwd_row32_c64_live34_rebuild.bo79nA/output
```

Result: pass for B1 and B4.

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
registers per thread: 109
threads per CTA: 256
registers per CTA, unrounded: 27904
launch shared bytes: 4096
stack/local bytes: 0/0
LDL/STL/CALL: 0/0/0
static/encoded SASS instructions: 1215/1240
cubin bytes: 83480
corrected row8 code-size ceiling: pass
```

The verifier read the raw temporary cubin/PTX/SASS/resource/ELF outputs,
checked their hashes against each audit summary, independently re-ran
`nvdisasm` and `cuobjdump`, recounted SASS operations, checked PTX
target/thread metadata, enforced resource and code-size gates, compared B1 with
B4, and compared the primary compile with the fresh rebuild.

The embedded cubin producer was `ptxas-blackwell` 12.9.86 from toolkit
12.9. Torch reported CUDA 13.0. System `nvdisasm` and `cuobjdump` 13.0.85
were inspection tools only. System `nvcc` 13.0.88 was queried for its version
but did not compile the kernel.

## Explicitly not run

- GPU kernel execution
- GPU service or server launch
- synthetic probe or real task/request
- B1/B4 correctness gate
- timing or acceptance run
- production selection

Raw compiler outputs remained under `/tmp` and are not included in this
artifact.
