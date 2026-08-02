# Qrow16 live paged A/B source gate

Status: source-complete, default-off, GPU/build pending.

The gate runs inside the EngineCore and CUDA boot that serves one real
SWE-Verified B1 task. The final FULL graph is captured with stock FA2. Its
first tree-attention invocation retains the exact live query, paged K/V cache,
block table, sequence metadata, and FP32 tree bias under the CUDA graph id.

After the first real fixed32 observed replay, the replay hook runs stock and
qrow16 recalls on those retained tensors. It compares every BF16 output byte
and every FP32 LSE byte. The graph's stock `entry.output` is not modified, so
stock is served. Any candidate dispatch geometry drift or byte mismatch raises.

The qrow selector is private and default-off. The launcher rejects an inherited
selector, requires an exact candidate-SO SHA-256 for the live gate, and only
installs the replay hook when `FR13_FA2_QROW16_LIVE_PAGED_AB=1`.

This source artifact contains no performance or correctness result. A candidate
SO must still be built, then the gate must pass on one real SWE-Verified B1 task.
Synthetic inputs, dense/repacked KV, and later-process replay are invalid.

## Static verification

```bash
python3 -m py_compile \
  scripts/fr13_patch_fa2_tree_bias.py \
  scripts/fr13_fa2_qrow16_byte_ab.py
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
pytest -q \
  tests/test_fr13_fa2_tree_kernel_candidates.py \
  tests/test_fr13_b1_gate_bundle.py
```
