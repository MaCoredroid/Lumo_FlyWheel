> **CORRECTION 2026-09-18 (independent Codex check, see F8 in STATIC_FINDINGS.md):** the kernel-path conclusion in F2/F7 (and the 'likely why' in RESULT.md / DRAFT_COMMENT.md) is WRONG. The startup line 'GDN decode kernel: cuda' is configuration only; the fused CUDA MTP path additionally requires speculative metadata and 8:1 V/K heads, and this run had neither (speculative_config=None; 48:16 = 3:1). Default decode-only batches use the packed Triton path; mixed decode/prefill batches can call #54146's readout. No per-batch kernel trace was captured. The L20-D bypass prediction is withdrawn. The negative result itself stands. The evidence tarball is the pristine agent output and still contains the uncorrected text.

# Static findings — stock vLLM 0.28.0 wheel (read-only; no source patched)

Wheel sha256 `817b8181f7f61b4a62dc1d5d9ab39f2bfb60a6cb86c29879a78a147b85756787`.
Paths are relative to `.venv/lib/python3.12/site-packages/vllm/`.
Items marked **[CORRECTION]** revise a claim made in CARD.md §2, which was written
before the server was launched. The card is left unedited on purpose.

## F1. PR #54146 is still unmerged, and its target line is unchanged in 0.28.0

`third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:225`

    o = q.new_empty(NK, *v.shape)          # output materialized in q.dtype

PR #54146 would make this `q.new_empty(NK, *v.shape, dtype=o_dtype)` with
`o_dtype = float32` when `initial_state.dtype == float32`. The PR is **open**, not
merged. `gh api repos/vllm-project/vllm/commits?path=...fused_sigmoid_gating.py`
shows the file's most recent commit is `626c90b2d` (2026-07-16, the "move fla to
third party" refactor) — so this line is unchanged on main today as well.

## F2. [CORRECTION] On stock 0.28.0 that Triton readout is NOT the decode hot path

CARD §2 claim 2 said GDN decode goes through `fused_sigmoid_gating_delta_rule_update`.
The server log from this run says otherwise:

    qwen_gdn_linear_attn.py:505  GDN decode kernel: cuda
    qwen_gdn_linear_attn.py:158  Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)

Mechanism (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:492-533`,
`envs.py:130`): `VLLM_GDN_DECODE_KERNEL` **defaults to `"cuda"`** in 0.28.0. The layer
takes the fused CUDA kernel `torch.ops._C.fused_gdn_decode_post_conv_mtp` unless
`_fused_gdn_decode_unsupported_reason()` objects, which it does only for an
interleaved-GQA layout, head_k/head_v != 128, non-SiLU gating, a non-bf16 model or
conv cache, a recurrent state outside `FUSED_GDN_STATE_DTYPES = (float32, bfloat16)`,
or compute capability < 8.0. This model on this GPU satisfies every condition
(Qwen3.5 non-interleaved, K=V=128, SiLU, bf16 model + bf16 conv cache, **fp32**
recurrent state, sm_121), so it falls through to `"cuda"` — as the log confirms.

**Consequence: the code PR #54146 patches is not on the decode path here.** The Triton
kernel remains in use for prefill chunks and as a documented fallback
(`VLLM_GDN_DECODE_KERNEL=triton`, or any unsupported-reason hit). That is a material
difference between the reporter's 0.21.0 and stock 0.28.0, and it is the most likely
reason a 0.21.0-era GDN decode collapse would not reappear on 0.28.0.

## F3. `mamba_ssm_cache_dtype=auto` still resolves to float32 for this model

`model_executor/models/config.py:781-806` (`Qwen3_5ForConditionalGenerationConfig`)
copies HF `text_config.mamba_ssm_dtype` into `cache_config.mamba_ssm_cache_dtype`
whenever the latter is `auto`; the pinned config has `mamba_ssm_dtype: "float32"`.
So sizzlecar's reading of 0.21.0 still holds on 0.28.0. **Answers issue Question 2.**

## F4. The fp16-overflow story does not apply to this model even on the Triton path

PR #54146's failure mode is fp16 saturation at ~65504. The pinned config sets
`text_config.dtype: "bfloat16"`, so `q.dtype` is bfloat16, whose exponent range matches
fp32 (~3.4e38). Downcasting a large fp32 state to bf16 loses mantissa precision; it
does not overflow to `inf`. The fused CUDA path likewise allocates
`core_attn_out = torch.zeros(..., dtype=hidden_states.dtype)` = bf16
(`qwen_gdn_linear_attn.py:896-899`). So for a bf16 Qwen3.6 the inf→NaN mechanism
should not fire on either kernel. **Bears on issue Question 3.**

## F5. `KVBlockZeroer` still skips Mamba — but new GDN state is zeroed elsewhere

Two halves, and the second one matters:

- `v1/worker/utils.py:100+` — docstring verbatim: *"Only AttentionSpec layers are
  processed; Mamba layers are skipped."* The constructor loop does
  `if not isinstance(spec, AttentionSpec): continue`. Likewise
  `v1/core/single_type_kv_cache_manager.py:86` only records new block ids when the
  spec `isinstance(kv_cache_spec, AttentionSpec)`. And `needs_kv_cache_zeroing`
  (`v1/kv_cache_interface.py:997`) is switched on *by the presence of Mamba layers*
  yet the zeroing it enables is applied only to attention blocks. So the reporter's
  structural observation (suspicion #2) is still literally true on 0.28.0.
- **But it does not imply stale-state reuse.** The GDN recurrent state is addressed by
  a per-request state index, not from the prefix-cache block pool, and
  `qwen_gdn_linear_attn.py` (prefill branch, ~line 1502) explicitly zeroes it for any
  sequence that has no initial state:

        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0

  A new request therefore starts from a zeroed GDN state regardless of `KVBlockZeroer`.
  **Answers issue Questions 4 and 5 for 0.28.0:** a contaminated state would have to
  arrive by a path that skips this prefill zeroing — which is precisely the bug class
  of #51483 (merged 2026-09-16) and its still-open sibling #51562, where a stateless
  first chunk is misclassified as a decode and so never goes through the prefill branch.

## F6. Upstream status of the neighbours

- #54308 (open) — same symptom, Qwen3-Next, and its title scopes it to **non-Blackwell**.
- #54146 (open, unmerged) — the readout fix; one outside comment (JartX) calls it
  incomplete for Qwen3.5-arch models.
- #51483 (**merged/closed 2026-09-16**) — "Do not classify a stateless first chunk as a
  decode". #51562 (open) — the same root cause in the GatedDeltaNet metadata builder.
- No one has reproduced #55291: no cross-reference on its timeline, no linked PR, no
  comment claiming a repro or a fix.

## F7. The kernel switch in F2 is **not** Blackwell-specific — it would apply on the reporter's L20-D too

This matters for how far the negative result generalises. Of the conditions in
`_fused_gdn_decode_unsupported_reason()`, exactly one is about the GPU:
`current_platform.has_device_capability(80)` — compute capability ≥ 8.0. Every other
condition is a property of the model and its cache dtypes (non-interleaved GQA,
head_k = head_v = 128, SiLU gating, bf16 model, bf16 conv cache, recurrent state in
`(float32, bfloat16)`).

The reporter's NVIDIA L20-D is Ada, compute capability 8.9, so it clears that bar. With
the same model and the same `auto` cache settings, **0.28.0 on 2 × L20-D would also take
the fused CUDA GDN decode kernel and also bypass the Triton readout** that #54146
patches. The GB10/Blackwell part of this rig is therefore not what makes the Triton path
unreachable here — the vLLM version is.

Two caveats kept explicit:
- TP=2 is untested here. The condition list contains nothing TP-dependent, but the
  reporter's failure could involve TP-specific state handling this run cannot see.
- If anything on their setup trips one of the unsupported reasons, the layer logs
  `Falling back to the Triton GDN decode path: <reason>` and the unfixed readout is
  back in play. That single log line is the cheapest thing the reporter can check, and
  it is worth asking them for.

## F8. [CORRECTION of F2 and F7, 2026-09-18] The fused CUDA decode kernel was not selected in this run

Independent source check (Codex, against the installed 0.28.0 wheel in the rig venv and live main `2bbdfcfc`), no GPU:

- `qwen_gdn_linear_attn.py:1798–1811` (0.28.0 wheel): the fused CUDA MTP decode path requires speculative-decoding metadata **and** an 8:1 V/K head ratio in addition to the conditions F2 listed. This run had `speculative_config=None` and 48 V / 16 K heads (3:1). So the startup line `GDN decode kernel: cuda` records the configured preference, not the executed kernel.
- `_forward_core` falls through to the packed Triton path for default decode-only batches (`:1271`, `:1672`); a mixed decode/prefill batch can call the readout that #54146 patches (`:1472`). No per-batch kernel trace was captured, so which path each batch took is unknown.
- Live main `2bbdfcfc` accepts a 3:1 ratio but still requires speculative metadata; #54146 remains OPEN and `fused_sigmoid_gating.py:225` is unchanged.
- Consequences: F7's claim that the reporter's L20-D would take the fused kernel on 0.28.0 is withdrawn. F4 holds only for the specific FP16 65504 threshold (BF16 activations), not for every overflow/NaN mechanism. F5: conditional prefill zeroes the gathered initial SSM state, not the complete recycled page; this does not establish universal recycling safety or #51483/#51562 as the only remaining cause.
- Unchanged: F1, F3 (`auto` → float32, pinned `mamba_ssm_dtype`), F6, and the headline counts listed below (recomputed independently from `runs/soak/records.jsonl`: 298 records, 249,544 generated tokens, 0 collapses/errors, longest `!` run 1, max repeated-token run 4, peak 36,496; traffic 23:14:24–02:30:58 UTC). Exposure detail: the 20-worker phase lasted 49m58s, P3 used four workers, P4 sent no requests; all 26 canaries avoided collapse but exhausted their 100 tokens inside reasoning (greedy/seed settings were added to the issue's probe, so requests were not verbatim).
