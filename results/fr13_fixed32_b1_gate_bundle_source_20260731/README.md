# Fixed32 B1 real-task gate bundle

Status: **source-ready, default off, and not yet GPU-validated**. No GPU was
used and the qrow16 FA2 candidate was not built on this branch.

This branch integrates the qrow16 source, native-precompute TAW byte gate, and
strict DFWD padded-row selector on top of `e8739aa8b`. The older M32-only DFWD
flags are superseded by:

- `FR13_DRAFT_HEAD_PAD_ROWS=0|32|64|128`;
- `FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0|1`.

The launcher validates and forwards the TAW, DFWD, and qrow live-gate selectors.
All candidate selectors default to zero.

## Real B1 live gate

After building the qrow16 SO, apply this overlay to the fixed32 B1 lifecycle
and send exactly one real SWE-Verified task at concurrency one:

```bash
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1
export FR13_DRAFT_HEAD_PAD_ROWS=0
export FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=1
export FORKED_FA2_SO=/absolute/path/to/qrow16/_vllm_fa2_C.abi3.so
export FR13_FA2_QROW16_SO_SHA256=<sha256-of-that-exact-so>
export FR13_FA2_QROW16_LIVE_PAGED_AB=1
export FR13_FA2_QROW16_LIVE_PAGED_AB_INSTANCE_ID=astropy__astropy-12907

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

Require all three live PASS records:

```text
[FR13_FIXED32_TAW_NATIVE_PRECOMPUTE] PASS ... probability_mismatches=0 product_mismatches=0 reference_returned=1
[FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB] PASS ... full_logit_bit_mismatches=0
[FR13_FA2_QROW16_LIVE_PAGED_AB] PASS ... output_byte_mismatches=0 lse_byte_mismatches=0 stock_served=1
```

The TAW gate always returns the existing exact products. The DFWD gate compares
M32, M64, and M128 complete BF16 logit rows with the existing `gemvx` result and
always returns that reference result. Diagnostic wall time is not a candidate
performance measurement.

## Qrow live boundary

The final B1 FULL graph is captured with stock FA2. During capture, the first
tree-attention layer retains references to its exact query, paged K/V cache,
block table, sequence tensors, and FP32 tree bias, keyed by CUDA graph identity.
Immediately after the first real fixed32 observed B1 replay, the same EngineCore
process recalls stock and qrow16 on those live tensors and compares raw BF16
output bytes and raw FP32 LSE bytes.

The graph's `entry.output` is never replaced, so the request serves the stock
result. The private qrow selector is absent during capture, set only around the
candidate recall, and restored in `finally`. Its C++ dispatch raises unless all
production geometry predicates match, preventing a stock-vs-stock false pass.
Any byte mismatch raises after writing the FAIL artifact. Dense, repacked, and
offline replay inputs are not part of this gate.

`scripts/fr13_fa2_qrow16_byte_ab.py` remains compile preflight only. The live
JSON result is `/logs/fr13_fa2_qrow16_live_paged_ab.json` by default.

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
