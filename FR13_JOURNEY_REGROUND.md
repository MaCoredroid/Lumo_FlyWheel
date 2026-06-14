# FR13 — Journey Re-Ground Ledger (2026-06-14, CPU read-only)

**Mandate (user GROUNDING RULE):** the carrier-hunt binds cite vLLM line-numbers that are **0.19.0-keyed**
and OFF by tens-to-hundreds of lines (causal_conv1d alone drifted 123 lines). A stale `/tmp/vllm_live_019`
(vLLM 0.19.0) burned us once. This workflow OWNS the COMPLETE re-ground of EVERY remaining load-bearing
journey code-read against the **REAL running image**:

> `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
> = `cu130-nightly` = **vLLM 0.19.2rc1.dev134+gfe9c3d6c5**.

All re-reads via `scripts/vllm_src.sh <relpath>` (cats one file fresh from the pinned image). NO `/tmp` cache
trusted. Each REAL file:line below was read on 2026-06-14.

---

## TOP-LINE — DID ANY CONCLUSION SHIFT?

**NO carrier-hunt conclusion SHIFTED. All deltas are line#-drift (0.19.0→0.19.2 keying) + two
naming-slips and three precision refinements that do NOT re-open any carrier closure.** The version diff
moved line numbers and (in two files) renamed/relocated the construct; in every case the *logic the bind
depends on* is present and identical on the real 0.19.2 source. NOT AN ESCALATION.

The three precision refinements are flagged so binds don't over-claim, but none re-opens a carrier search:
1. **fp8 M-invariance is config-DEFAULT-dependent**, not unconditional — holds for GB10 because no shipped
   per-device JSON matches GB10's device-name (default `BLOCK_SIZE_M=64` applies). (§5)
2. **gate `ROWS_PER_BLOCK=1` is COMPUTED per-M, not a hardcoded constexpr** — `=1` only in the small-M
   decode/verify regime (which is the deployed regime). (§6)
3. **`fused_post_conv_prep` EXISTS live** (audit said "line doesn't exist") — it is real, on the PREFILL
   path, NOT the verify path; the verify-path conclusion (sequential sigmoid-gating) is unchanged. (§4)

**Bug-class playbook rows in scope (FR13_BUG_CLASS_PLAYBOOK):**
- **#10 — Shared-source ≠ shared-SASS (codegen identity):** "the bar is the RUNNING image's bytes, not a
  same-named cache; re-confirm structure on the deployed source, not by name." This whole exercise is #10
  discipline applied to citations: a `/tmp` cache "looked like vLLM" but compiled from different bytes.
- **#11 — Batch-composition / version-skew:** "near-identical files masked the skew" — the 0.19.0 cache was
  byte-adjacent to 0.19.2, so off-by-N line citations passed visual inspection while pointing at the wrong
  region. (NOTE: the prompt glosses #11 as "naming-slip/version-skew"; the LITERAL playbook #11 is
  "Batch-composition/BI-flag sensitivity". The version-skew failure mode is the #11 *flavor* invoked here
  + #10. Two genuine **naming-slips** surfaced below — `prepare_mamba`→`preprocess_mamba`,
  `spec_masks`→`spec_sequence_masks` — those are the #11 "naming-slip" gloss in action.)

---

## ALREADY RE-GROUNDED BY THE STALENESS AUDIT (w3hax7wlb, FR13_VERSION_STALENESS_AUDIT.md) — cited HELD

Do NOT redo; cited here as authoritative-HELD for completeness:

| # | construct | real 0.19.2 basis | verdict |
|---|---|---|---|
| A1 | conv FIXED+CLOSED, 3-tap FIFO, spec offset = num_accepted−1 | `causal_conv1d.py` L852-866 (offset), L1184-1186 (state_len), L1234 (IS_SPEC_DECODING) | **HELD** |
| A2 | fused_recurrent packed-decode = recurrent rank-1, 5 ops, num_warps=1 num_stages=3 | `fla/ops/fused_recurrent.py` def L256, ops L327-331, L438-439 | **HELD** |
| A3 | fused_sigmoid_gating = sequential rank-1 verify dispatch | `fla/ops/fused_sigmoid_gating.py` def L24, ops L144-153, warps=4 stages=3 L211-212 | **HELD** |
| A4 | commit 7441fc43 native-ref imports the LIVE kernel | `scripts/fr13_native_packed_decode_ref.py:94-96` import; kwargs match wrapper L339-350 | **HELD** |

(I re-touched A2/A4 below where the scan-align seams and the GDN dispatch depend on them — they re-confirm.)

---

## §1 — FA2 / TREE_ATTN decode path (FR13_FA2_FORK_IS_DECODE_KERNEL_CORRECTION)

**Cited claim:** decode dispatch where `FR13_FA2_TREE_BIAS=1` routes to `flash_attn_varlen_func(..., tree_bias=)`
vs the `unified_attention` Triton fallback; "the fork is the deployed decode kernel at the 0.0039 floor;
full-attn NOT the carrier."

**Patch target:** `scripts/fr13_patch_fa2_tree_bias.py::_patch_tree_attn` edits the vLLM file
`vllm/v1/attention/backends/tree_attn.py`; the C++ tree-bias op is patched into the FlashAttention csrc
(`flash.h`, `flash_fwd_kernel.h`, `flash_api.cpp`, `flash_api_torch_lib.cpp`) + `flash_attn_interface.py`.

**Re-ground (REAL 0.19.2):**
- `tree_attn.py` exists; `TreeAttentionImpl` def **L294**, `forward` def **L362**.
- The patcher's decode **anchor** (the multi-line `unified_attention(... qq_bias=decode_meta.tree_attn_bias ...)`
  block) matches the REAL source **byte-for-byte at L421-441** (`if decode_meta := attn_metadata.decode_metadata:`
  L421, `unified_attention(` L422, `qq_bias=decode_meta.tree_attn_bias,` **L434**).
- Import needle `from vllm.v1.attention.ops.triton_unified_attention import unified_attention` present at **L26**.
- The fork target `flash_attn_varlen_func` is exported from `vllm/v1/attention/backends/fa_utils.py`
  **L18-23** (`if current_platform.is_cuda(): from vllm.vllm_flash_attn import (flash_attn_varlen_func, ...)`).
  The `tree_bias=` kwarg is NOT in the stock signature — it is ADDED by the C++/interface patch (expected;
  it IS our fork).
- `decode_meta.max_query_len` is set at `tree_attn.py` **L140** (`int(q_seqlens.max().item())`); the
  swapped-mode guard `max_query_len > 1` gating the fork is real (patcher L618; decode tree = tree_len > 1).

**Verdict: HOLDS.** Real file:line `vllm/v1/attention/backends/tree_attn.py:421-441` (decode anchor),
`:26` (import needle), `vllm/v1/attention/backends/fa_utils.py:18-23` (flash_attn_varlen_func export).
The "fork IS the deployed decode kernel; 0.00195 is the moot Triton fallback; full-attn is NOT the carrier
(L0-GDN is upstream)" conclusion is structurally unchanged — the patch's anchors all exist on 0.19.2.

---

## §2 — mamba_utils cross-step state contract (FR10 keystone, feedback_read_vllm_source_first)

**Cited claim:** `vllm/v1/worker/mamba_utils.py` ~L224-254 — `curr_state_idx = num_blocks − 1 −
num_speculative_blocks`; accepted-state copy uses **LINEAR** bias `num_accepted_tokens_cpu[i] − 1`;
copy fires only on block migration (`prev_state_idx != curr_state_idx`) so the **in-place case (prev==curr)
has NO stock copy**; functions `prepare_mamba`/`postprocess_mamba`.

**Re-ground (REAL 0.19.2 `v1/worker/mamba_utils.py`, file present):**
- `curr_state_idx = num_blocks - 1 - num_speculative_blocks` → **L204** (verbatim).
- linear bias `input_batch.num_accepted_tokens_cpu[i] - 1` passed to the copy → **L214** (verbatim).
- migration-gated copy `if prev_state_idx != -1 and prev_state_idx != curr_state_idx:` → **L206**; the
  in-place case (prev==curr) skips `collect_mamba_copy_meta` ⇒ NO stock copy. **L206-218.**
- the EXTRA in-place guard `if src_block_idx == dest_block_idx and accept_token_bias == 0: return` →
  `collect_mamba_copy_meta` **L108-109**; the `accept_token_bias + 1` passed into the per-state copy func →
  **L125**.
- block-layout corner-case diagram (Block0 `[A,B,C,draft1]` / Block1 running-state) → comment **L195-203**.

**NAMING-SLIP (bug-class #11 gloss):** the function is **`preprocess_mamba`** (def **L147**), NOT
`prepare_mamba` as the keystone bind says; `postprocess_mamba` def **L222** (correct). The 0.19.2
`num_accepted_tokens_cpu[i] = 1` reset is at **L218** (preprocess) and **L272** (postprocess).

**Also note a DIFFERENT file with a similar name:** `model_executor/layers/mamba/mamba_utils.py` (376 L) holds
the per-state `get_conv_copy_spec` (offset = `num_accepted_tokens - 1`, **L314**) / `get_temporal_copy_spec`
(**L341**) that the WORKER `collect_mamba_copy_meta` dispatches to via `MambaStateCopyFunc`. The contract is
split across the two files; the WORKER file is the one the keystone bind means.

**Verdict: HOLDS.** Real file:line `vllm/v1/worker/mamba_utils.py:204` (curr_state_idx), `:214` (linear bias),
`:206` (migration gate / in-place no-copy), `:108-109,125` (in-place guard). The read-base / in-place /
linear-accept-bias contract our committer + replay depend on is present byte-for-byte. Only the function name
(`prepare`→`preprocess`) drifted.

---

## §3 — gdn_linear_attn.py forward dispatch (fr13_native_packed_decode_ref + scan-align)

**Cited claim:** `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`-gated `_forward_core` with `num_prefills==0 +
num_decodes>0 + spec_masks None` → `_forward_core_decode_non_spec` → `packed_decode`.

**Re-ground (REAL 0.19.2 `model_executor/layers/mamba/gdn_linear_attn.py`, 1211 L):**
- flag → `self.enable_packed_recurrent_decode = envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` → **L384-386**.
- gate in `_forward_core` (def **L786**): `if (self.enable_packed_recurrent_decode and
  attn_metadata.spec_sequence_masks is None and attn_metadata.num_prefills == 0 and
  attn_metadata.num_decodes > 0):` → **L806-811**, returns `_forward_core_decode_non_spec(...)` **L812**.
- `_forward_core_decode_non_spec` def **L1045** → `fused_recurrent_gated_delta_rule_packed_decode(...)` call
  **L1085** (kwargs `mixed_qkv, a, b, A_log, dt_bias, scale=head_k_dim**-0.5, initial_state=ssm_state, out,
  ssm_state_indices, use_qk_l2norm_in_kernel=True`).
- spec/verify path (`spec_sequence_masks is not None`) instead dispatches
  `fused_sigmoid_gating_delta_rule_update(...)` at **L959** and **L1009** (the sequential rank-1 verify
  kernel = audit A3). Import block **L26-30**.

**NAMING-SLIP (bug-class #11 gloss):** the bind says "spec_masks None"; the REAL attribute is
**`spec_sequence_masks`** (`attn_metadata.spec_sequence_masks is None`, L807).

**Verdict: HOLDS.** Real file:line `gdn_linear_attn.py:384-386` (flag), `:806-811` (gate), `:1045` /
`:1085` (decode→packed_decode), `:959,1009` (spec→sigmoid-gating). The dispatch `fr13_native_packed_decode_ref`
and the scan-align fix rely on is intact (matches audit Part 2D exactly).

---

## §4 — GDN scan / SUBOP path (causal_conv1d_update spec-path + the real verify FLA op)

**Cited claims:** (a) `causal_conv1d_update` spec-path `state_len = width − 1 + (seqlen − 1)` (the 5x-assert
root); (b) the audit-copy named `fused_post_conv_prep` at a line "that does not exist live — re-confirm the
REAL FLA op the verify path calls."

**Re-ground (REAL 0.19.2):**
- (a) `causal_conv1d.py`: `causal_conv1d_update` def **L1071**; spec-path
  `if num_accepted_tokens is not None: state_len = width - 1 + (seqlen - 1)` → **L1183-1184**;
  else `state_len = width - 1` → **L1186**; `IS_SPEC_DECODING=num_accepted_tokens is not None` → **L1234**.
  Spec-decode update-kernel FIFO read: `conv_state_token_offset = tl.load(num_accepted_tokens_ptr + ...) - 1`
  **L852-853**, non-spec `= 0` **L856**, `prior_tokens = conv_states_base + conv_state_token_offset*stride`
  **L866**. (This is audit A1; **HELD**, re-confirmed.)
- (b) **CORRECTION to the audit's "line doesn't exist" note:** `fused_post_conv_prep` **DOES exist live** —
  def `model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py:152`; imported `gdn_linear_attn.py:27`;
  called at `gdn_linear_attn.py:727` (autotune warmup) and **L930** (the `num_prefills > 0` PREFILL branch,
  guarded `if attn_metadata.num_prefills > 0:` L913). It is a **PREFILL** post-conv prep, NOT on the
  decode/verify path. **The REAL op the SPEC-VERIFY path calls is `fused_sigmoid_gating_delta_rule_update`**
  (`gdn_linear_attn.py:959,1009`; kernel `fla/ops/fused_sigmoid_gating.py:24`, sequential rank-1 = audit A3);
  the DECODE-non-spec path calls `fused_recurrent_gated_delta_rule_packed_decode`
  (`gdn_linear_attn.py:1085`; kernel `fla/ops/fused_recurrent.py:256`).

**Verdict: HOLDS (with a factual correction to the audit's existence note).** The spec-path conv state_len
conclusion is intact (`causal_conv1d.py:1183-1186,852-866,1234`). The "verify path does not run
`fused_post_conv_prep`" CONCLUSION is correct — but the construct EXISTS (on prefill), it just is not the
verify op. No carrier closure re-opens. Real verify op = `fused_sigmoid_gating_delta_rule_update`
(`fla/ops/fused_sigmoid_gating.py:24`).

### §4b — scan-align seams (FR13_SCAN_ALIGNMENT_MATH, re-confirm the named native ops)
The scan-align fix aligns our `_gdn_node_step` to the native packed-decode kernel. The two load-bearing
SEAMS, on REAL 0.19.2 `fla/ops/fused_recurrent.py`:
- **Seam-d (l2norm):** native `b_q = b_q / tl.sqrt(tl.sum(b_q*b_q) + 1e-6)` → **L314** (k at L315);
  ours uses `rsqrt`. Seam real; eps 1e-6 matches. (bind cited `:313-314`, off-by-one.)
- **Seam-e (beta bf16 roundtrip):** native `beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)`
  → **L326**; ours stays fp32. Seam real. (bind cited `:324`.)
- 5-op recurrent body → **L328-332**; `SOFTPLUS_THRESHOLD=20.0` → **L473**; `num_warps=1` → **L439**
  (bind cited `:438` = `num_stages=3`; the same off-by-one the audit flagged). g/softplus → **L321-324**.
**Both seams HOLD on the real source; line numbers drift a few.** The scan-align math (codegen-alignable,
NOT chunk-vs-recurrent irreducible) is unchanged.

---

## §5 — fp8 GEMM M-invariance (in_proj_qkvz / o_proj, w8a8_triton_block_scaled_mm)

**Cited claim:** `w8a8_triton_block_scaled_mm` `BLOCK_SIZE_M=64` constexpr → "M≤64 = one M-tile →
M-invariant" (in_proj/o_proj fp8 GEMMs do not depend on batch M).

**Re-ground (REAL 0.19.2 `model_executor/layers/quantization/utils/fp8_utils.py`):**
- kernel `_w8a8_triton_block_scaled_mm` (def near **L575**): `BLOCK_SIZE_M: tl.constexpr` **L581**;
  `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)` **L592**; `offs_am = (pid_m*BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M))
  % M` **L601**; fp32 accumulator over K **L611-612**.
- launcher `w8a8_triton_block_scaled_mm` def **L678**: default config `"BLOCK_SIZE_M": 64` **L726**;
  grid `triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, ...)` **L735**. So for M ≤ 64,
  `cdiv(M,64) = 1` → ONE M-tile → tiling is M-invariant.

**PRECISION REFINEMENT (flag, conclusion intact):** the default config is only used when NO per-device tuned
JSON matches. The selection is `configs = get_w8a8_block_fp8_configs(N,K,...)` **L717**; if a JSON exists,
`config = configs[min(...,key=|x-M|)]` **L720** with a possibly-different `BLOCK_SIZE_M`. Device-name resolved
via `current_platform.get_device_name().replace(" ","_")` **L653**. Shipped JSONs cover A100/H100/B200/
MI300X/L20/H20 etc. — **NONE matches GB10's device name** (verified: no `GB10`/`GB200`/`B200`-for-Spark/
`sm_121`/`Spark` JSON resolves for GB10). ⇒ on GB10 the DEFAULT config (`BLOCK_SIZE_M=64`) applies, so the
M-invariance claim holds — but it is **config-default-dependent, not unconditional**. (If a GB10-named fp8
JSON ever ships with `BLOCK_SIZE_M<64`, re-check.)

**Verdict: HOLDS (on GB10, by default-config).** Real file:line `fp8_utils.py:581,592,601` (kernel
M-tiling), `:726` (default BLOCK_SIZE_M=64), `:735` (grid), `:717-720,653` (config-selection caveat).

---

## §6 — gate (RMSNormGated ROWS_PER_BLOCK) + rejection_sampler forward target-constraint

**Cited claims:** (a) RMSNormGated gate kernel `ROWS_PER_BLOCK=1` per-row; (b) `rejection_sampler.py forward`
target-constraint line the committer composes with (keystone bind: "target constraint lives in the unmodified
stock `rejection_sampler.py` forward (line 135), before the injected tree branch").

**Re-ground (REAL 0.19.2):**
- (a) `RMSNormGated` (CustomOp) def `model_executor/layers/layernorm.py:406`; `forward_cuda` imports
  `from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn` **L508** → returns
  `rmsnorm_fn(...)` **L510**. The kernel `_layer_norm_fwd_1pass_kernel` carries `ROWS_PER_BLOCK: tl.constexpr`
  `fla/ops/layernorm_guard.py:82`; per-row indexing `row_start = tl.program_id(0)*ROWS_PER_BLOCK` **L90**,
  `rows = row_start + tl.arange(0, ROWS_PER_BLOCK)` **L94**; passed `ROWS_PER_BLOCK=rows_per_block` **L242**.
- (b) `rejection_sampler.py` `forward` def **L60**; `target_logits = apply_sampling_constraints(target_logits,
  metadata.cu_num_draft_tokens, sampling_metadata)` → **L135** (EXACTLY the bind's "line 135"); then
  `rejection_sample(...)` → **L141**; `apply_logits_processors` (temp/penalties) runs FIRST at **L129**.
  ⇒ target_logits IS temp/top_p/top_k-constrained BEFORE the rejection sample / any injected tree branch —
  confirming the keystone resolution (codex right: `constraint_count=1, before_tree=True`; Claude was wrong
  reading only the patcher).

**PRECISION REFINEMENT (flag, conclusion intact):** `ROWS_PER_BLOCK` is **NOT a hardcoded `=1` constexpr** —
it is COMPUTED: `calc_rows_per_block(M, device)` = `min(next_power_of_2(cdiv(M, 2*sm_count)), 4)`
(`layernorm_guard.py:171-175`, called L224). For the deployed small-M decode/verify regime (a few rows over
GB10's many SMs) `cdiv(M, 2*sm_count)=1 → rows_per_block=1`, so the per-row claim holds THERE; but for large M
(prefill) it can be 2/4. The "per-row, M-independent gate" claim is valid in the verify regime, with this
basis correction.

**Verdict: HOLDS.** Real file:line `layernorm.py:406,508,510` (RMSNormGated→guard kernel),
`fla/ops/layernorm_guard.py:82,90,94,171-175,242` (ROWS_PER_BLOCK, computed=1 at small M);
`v1/sample/rejection_sampler.py:60,129,135,141` (forward, constraint at L135 before rejection_sample).

---

## CONSOLIDATED RE-GROUND LEDGER (binds can adopt)

| § | journey code-read | cited (0.19.0-keyed / bind) | REAL 0.19.2 file:line | verdict |
|---|---|---|---|---|
| §1 | FA2-fork = deployed decode kernel; full-attn not carrier | tree_attn decode branch / flash_attn_varlen_func(tree_bias=) | `v1/attention/backends/tree_attn.py:421-441`, `:26`; `fa_utils.py:18-23` | **HOLDS** |
| §2 | mamba cross-step contract (curr_state_idx / linear bias / in-place no-copy) | `v1/worker/mamba_utils.py:~224-254`, `prepare_mamba` | `v1/worker/mamba_utils.py:204,214,206,108-109,125`; `preprocess_mamba`=L147 | **HOLDS** (name `prepare→preprocess`) |
| §3 | GDN packed-decode dispatch | flag + `_forward_core` num_prefills==0/num_decodes>0/spec_masks None → packed_decode | `gdn_linear_attn.py:384-386,806-811,1045,1085`; spec→`:959,1009` | **HOLDS** (attr `spec_sequence_masks`) |
| §4a | conv spec-path state_len (5x-assert root) | state_len=width-1+(seqlen-1) | `causal_conv1d.py:1183-1186,852-866,1234` | **HOLDS** (= audit A1) |
| §4b | real verify FLA op (fused_post_conv_prep mismatch) | "fused_post_conv_prep line doesn't exist" | EXISTS `fused_gdn_prefill_post_conv.py:152` (PREFILL); verify op = `fused_sigmoid_gating.py:24` | **HOLDS** (audit existence-note corrected) |
| §4c | scan-align seams (l2norm rsqrt-vs-sqrt; beta bf16) | fused_recurrent.py:313-314 / :324 | `fla/ops/fused_recurrent.py:314,326,328-332,439,473` | **HOLDS** (off-by-one) |
| §5 | fp8 M-invariance (in_proj/o_proj) | w8a8_triton_block_scaled_mm BLOCK_SIZE_M=64 constexpr | `fp8_utils.py:581,592,601,726,735`; caveat `:717-720,653` | **HOLDS** (config-default on GB10) |
| §6a | gate ROWS_PER_BLOCK=1 per-row | ROWS_PER_BLOCK=1 | `layernorm.py:406,508`; `layernorm_guard.py:82,90,171-175` (computed=1 small-M) | **HOLDS** (computed, not constexpr) |
| §6b | rejection_sampler target-constraint before tree | rejection_sampler.py forward L135 | `v1/sample/rejection_sampler.py:60,129,135,141` | **HOLDS** (L135 exact) |
| A1-A4 | conv/recurrent/sigmoid-gating/native-ref | (audit) | (audit, see HELD table above) | **HELD** |

---

## SHIFTED LIST

**NONE — all line#-drift (+ 2 naming-slips + 3 precision refinements), conclusions intact.** No
carrier-search closure re-opened. The FA2-fork-is-decode-kernel closure (§1), the L0-GDN-cross-event-replay
carrier hypothesis (§1/§4), the codegen-alignable-scan finding (§3/§4b), the committer's in-place/linear-bias
contract (§2), and the M-invariant fp8 + per-row gate + before-tree constraint composition (§5/§6) all HOLD
on the REAL 0.19.2 image.

## CORRECTED-CITATION TABLE (annotate binds with these)

| bind / construct | stale (0.19.0) cite | corrected REAL 0.19.2 cite |
|---|---|---|
| keystone mamba contract | `v1/worker/mamba_utils.py:224-254`, `prepare_mamba` | `:204` curr_state_idx, `:214` linear bias, `:206` migration gate, `:108-109,125` in-place guard; fn `preprocess_mamba`=`:147` |
| GDN packed-decode gate | (0.19.0 L845-855 / spec_masks) | `gdn_linear_attn.py:806-811` (attr `spec_sequence_masks`), `:1085` packed_decode |
| scan-align num_warps | `fused_recurrent.py:438` | `fla/ops/fused_recurrent.py:439` (L438 = num_stages=3) |
| scan-align l2norm seam-d | `:313-314` | `fla/ops/fused_recurrent.py:314` |
| scan-align beta seam-e | `:324` | `fla/ops/fused_recurrent.py:326` |
| FA2 decode anchor | (0.19.0 lines) | `v1/attention/backends/tree_attn.py:421-441` |
| conv spec state_len | (0.19.0 ~565/749/845-863/156) | `causal_conv1d.py:1183-1186,852-866,1234` |
| rejection target-constraint | `rejection_sampler.py:135` | `v1/sample/rejection_sampler.py:135` (UNCHANGED — exact) |
| fp8 BLOCK_SIZE_M | (constexpr 64) | `fp8_utils.py:726` default; M-invariance config-default-dependent on GB10 |
| gate ROWS_PER_BLOCK | (=1) | `layernorm_guard.py:171-175` computed (=1 small-M only) |
| verify FLA op | "fused_post_conv_prep" (verify) | verify = `fused_sigmoid_gating.py:24`; post_conv_prep = `fused_gdn_prefill_post_conv.py:152` (PREFILL) |

---

## PROVENANCE
All re-reads via `scripts/vllm_src.sh <relpath>` against the pinned image
`vllm/vllm-openai@sha256:3dbe092e…` (`0.19.2rc1.dev134+gfe9c3d6c5`), 2026-06-14, CPU read-only (a GPU
verify-boot ran concurrently; this workflow edited ONLY this file). Files read:
`v1/attention/backends/tree_attn.py`, `v1/attention/backends/fa_utils.py`, `v1/worker/mamba_utils.py`,
`model_executor/layers/mamba/mamba_utils.py`, `model_executor/layers/mamba/gdn_linear_attn.py`,
`model_executor/layers/mamba/ops/causal_conv1d.py`, `model_executor/layers/fla/ops/fused_recurrent.py`,
`model_executor/layers/fla/ops/fused_sigmoid_gating.py`, `model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`,
`model_executor/layers/quantization/utils/fp8_utils.py`, `model_executor/layers/layernorm.py`,
`model_executor/layers/fla/ops/layernorm_guard.py`, `v1/sample/rejection_sampler.py`.
