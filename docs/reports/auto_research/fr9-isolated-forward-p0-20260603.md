# FR9 Isolated Forward P0 Result - 2026-06-03

Branch: `fr9-spine2-lossless-winner`

## Scope

Phase 0.5 upgrade check was verified locally against the pulled vLLM 0.22.0
source before building:

- `vllm/v1/attention/backend.py` still defaults
  `AttentionBackend.supports_batch_invariance()` to `False`.
- GDN/Mamba/SSM backends (`gdn_attn.py`, `mamba1_attn.py`,
  `mamba2_attn.py`, `short_conv_attn.py`, `linear_attn.py`) do not override
  `supports_batch_invariance`.
- Full-attention backends such as FlashAttention/Triton attention do override
  it. This confirms the upgrade path does not provide batch-invariant GDN.

## Built

Added an opt-in vLLM 0.19 diagnostic patch:

- `FR9IsolatedForwardProbe` helper module in
  `vllm/v1/worker/fr9_isolated_forward_probe.py`.
- `GPUModelRunner.sample_tokens` hook, disabled unless
  `LUMO_FR9_ISOLATED_FORWARD_PROBE=1`.
- `GPUWorker.lumo_fr9_isolated_forward_probe(...)` collective RPC entry point.
- Host launcher forwarding for `LUMO_FR9_ISOLATED_FORWARD_*` envs.
- Relaunch helper image override via `LUMO_VLLM_IMAGE`, default unchanged.

Production remains fail-closed: the probe is disabled by default, and the
existing independent-row `spines>1` lossless-policy guard is unchanged.

## Verification

Host/unit checks:

- `pytest -q tests/test_fr9_isolated_forward_patch.py` -> `3 passed`
- `python3 -m py_compile docker/patches/apply_fr9_isolated_forward_probe.py
  src/lumo_flywheel_serving/model_server.py
  scripts/swe_x86_helpers/relaunch_qwen36_round.py` -> passed
- Diagnostic image build succeeded:
  `lumo-flywheel-vllm:26.01-py3-v0.19.0-fr9iso`

Runtime attempts used config Fb, MTP=5, row-mode independent, spines=1,
`VLLM_BATCH_INVARIANT=1`, and the Qwen3.6-27B FP8 GDN-hybrid model.

## Runtime Result

P0 is **not proven**.

The runner hook correctly activated and reached state hashing/isolated-forward
setup, but vLLM 0.19 did not expose a runner-local tracked scratch-block
allocator:

1. Short requests have only the active current block allocated per KV/GDN group.
   The first safe implementation failed with:
   `RuntimeError:no tracked scratch block for group 3; len=1 current=0`.
2. A diagnostic untracked scratch-block experiment selected a block ID not used
   by active requests and copied public cache into it. That bypassed vLLM's
   scheduler/block-pool ownership and was not safe: the first B=1 request hit
   `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
   inside FlashAttention and killed `EngineCore`.

After that crash, the helper was changed back to fail closed unless
`LUMO_FR9_ISOLATED_FORWARD_UNTRACKED_SCRATCH=1` is explicitly set. That env is
intentionally not part of the normal forwarded launcher envs.

## Verdict

The P0 primitive is not feasible as a pure `GPUModelRunner` source-edit on vLLM
0.19 with the current public APIs. A correct implementation needs one of:

- a scheduler/block-pool-integrated scratch allocation path that gives the
  isolated hidden request real tracked blocks, or
- a custom kernel/state path that snapshots and restores GDN/Mamba state without
  reusing unowned cache blocks.

Do not build the selector on top of this probe. No losslessness claim and no
speed claim are supported by this result.
