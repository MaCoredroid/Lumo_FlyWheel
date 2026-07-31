# Fixed32 B1 real-task gate bundle

Status: **source-only, default off, and not deployable**. No GPU was used and
the qrow16 FA2 candidate was not built on this branch.

This branch integrates the qrow16 source, native-precompute TAW byte gate, and
strict DFWD padded-row selector on top of `e8739aa8b`. The older M32-only DFWD
flags are superseded by:

- `FR13_DRAFT_HEAD_PAD_ROWS=0|32|64|128`;
- `FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0|1`.

The launcher now also validates and forwards
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0|1`. All three selectors default to zero.

## Executable live gate

The currently executable one-task live gate covers TAW and all three DFWD row
shapes. Apply this overlay to the existing held fixed32 B1 diagnostic lifecycle
and send exactly the real SWE-Verified task `astropy__astropy-12907` at
concurrency one:

```bash
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1
export FR13_DRAFT_HEAD_PAD_ROWS=0
export FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=1

.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$ONE_REAL_ASTROPY_12907_SUBSET" \
  --out-root "$RUNROOT/swe_out" \
  --dataset-tag verified \
  --concurrency 1 \
  --agent-host alienware \
  --agent-endpoint http://127.0.0.1:8023/v1 \
  --eval-host alienware
```

`ONE_REAL_ASTROPY_12907_SUBSET` must be a normal SWE-Verified subset containing
only `astropy__astropy-12907`. This is a B1 diagnostic, not exact4 or exact16
acceptance. Do not send warmup, synthetic, replay, or probe requests.

Require the container environment to contain the three values above. At an
uncaptured root boundary, require:

```text
[FR13_FIXED32_TAW_NATIVE_PRECOMPUTE] PASS ... probability_mismatches=0 product_mismatches=0 reference_returned=1
[FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB] PASS ... full_logit_bit_mismatches=0
```

The TAW gate always returns the existing exact products. The DFWD gate compares
M32, M64, and M128 complete BF16 logit rows with the existing `gemvx` result and
always returns that reference result. Diagnostic wall time is not a candidate
performance measurement.

## Qrow boundary

The three-gate live claim is **blocked**. The existing
`scripts/fr13_fa2_qrow16_byte_ab.py` checker is a useful compile/preflight tool,
but it is not the required live gate. A dense-KV capture, dense-to-paged
repacking, a synthetic request, or a later replay process is not acceptance
evidence.

Before qrow16 can join the one-task bundle, its output and FP32 LSE must be
compared bitwise against the stock geometry in the same EngineCore process and
CUDA boot, using the actual paged `key_cache`, `value_cache`, `block_table`,
query, sequence lengths, and tree bias from the live
`astropy__astropy-12907` dispatch. The served output must remain the stock
reference output. That runtime gate is not implemented here because adding it
would expand this CPU-only integration task into a serving/capture harness
change.

## Static verification

```bash
.venv/bin/pytest -q \
  tests/test_fr13_b1_gate_bundle.py \
  tests/test_fr13_draft_head_pad_rows.py \
  tests/test_fr13_fixed32_taw_native_precompute.py \
  tests/test_fr13_fa2_tree_kernel_candidates.py
python3 -m py_compile \
  scripts/fr10_phase4_patch_vllm_tree_gdn.py \
  scripts/fr13_device_multidraft_kernel.py \
  scripts/fr13_fa2_qrow16_byte_ab.py
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
```

