# FR-13 — forked-FA2 TREE_ATTN prefill drift: concrete record + fix spec (UNFIXED)

Concrete record (not hand-wavey). Status as of 2026-06-08.

## The drift — exact location + magnitude
- **Where it first appears:** the forked-FA2 `TREE_ATTN` **prefill** path diverges from native `FLASH_ATTN` prefill at **layer 7 `attn_out_raw`** (full_attention layer). Localized in commit `2f8ef1ea` (artifact `output/.../prefill_full_attn_replay.json`).
- **Downstream:** that L7 prefill divergence feeds the GDN **layer 8 `h0_state_in` ≈ 7.2e-4** (inherited recurrent-state seed; artifact `prefill_gdn_state_replay.json`), which is one contributor to the **1.9 max_abs final-logits drift** on the spine ladder (`output/fr13_ex2_live_ladder_20260608T021853Z/gateA_spine_ladder.json`).
- **NOT a kernel-math bug:** the forked FA2 kernel is byte-exact 0.0 in **decode** (Gate-2, `d2f1ba18`: stock-vs-fork no-bias = 0.0 fp16+bf16). So the divergence is a **WIRING difference in the prefill path**, not the CUTLASS kernel.

## Root cause (wiring)
The `TreeAttentionImpl` **prefill** branch routes through a **different helper** (`unified_attention(...)`) than native `FlashAttentionImpl` prefill, which calls `flash_attn_varlen_func(...)` with its full extras — `scheduler_metadata`, `q_descale`, `k_descale` (this is an **fp8** model), `cu_seqlens`, `max_seqlen`, `causal=True`, `softmax_scale`, window. Different helper / missing extras ⟹ different numerics in prefill. codex flagged this directly: *"The stock FlashAttention backend calls the same helper with scheduler/descale extras."*

## Status: FIX WRITTEN (not yet applied to the live patcher; not yet GPU-verified)
Two earlier agents (`codex_fr14`, `codex_fr15`) **hung before writing it** (announced the edit, produced none, ~8–48 min idle). An **isolated worktree agent (2026-06-08) wrote the fix** — saved as **`patches/fr13_fa2_prefill_native.patch`** (`git apply --check` = clean). It pinned the exact bug in live source (`/tmp/vllm_live_019/.../tree_attn.py:403-422`): tree prefill passes `q_descale=None` and `descale_shape=(query_start_loc.shape[0]-1, key.shape[1])`, vs native `flash_attn.py:806-863` which passes `q_descale=layer._q_scale.expand(...)`, `descale_shape=(..., self.num_kv_heads)`, plus `scheduler_metadata`/`fa_version`/`num_splits`/`s_aux`.

**NOT applied to the live `scripts/fr13_patch_fa2_tree_bias.py` yet, on purpose:** the patch adds a mandatory anchor-match (`raise` if not found) to the patcher that runs at every forked-FA2 server launch. Applying mid-e2e risks breaking codex's launch if the container's installed `tree_attn.py` doesn't match the anchor byte-for-byte. Apply + verify AFTER the in-flight e2e (`output/fr13_argmax_e2e_20260608T055851Z`).

**Verdict-workflow cross-check (`wefeg4ngz`, 2026-06-08):** independently CONFIRMED the spine is argmax-lossless 6/6 (smallest margin 5.875; identity map gives only 4/6) and that 1.77 is stale (FR10 pre-fork; forked-FA2 e2e never measured). Recommendation RUN_E2E_NOW (medium confidence). Caveats: N=1/B=1-eager/spine-only; prior TREE_ATTN-Triton build missed 2.8×; branch losslessness needs the path-rerun oracle.

## Impact under the user-ACCEPTED gate (argmax-lossless)
The spine argmax-lossless check (`FR13_SPINE_ARGMAX_LOSSLESS.md`, commit `16660de9`) was run on the build **WITH this prefill drift present** and the spine still came out **6/6 argmax-matched** (drift 1.9 ≪ native top-1 gap 7–12). So this drift **does not flip a spine output token** ⟹ it does **not** break the accepted argmax-lossless gate. It is still **real raw drift** and is the known cheap lever to (a) reach literal-0 and (b) tighten temp-0.6 rejection-sampling acceptance if the e2e shows any loss.

## The fix (spec) — flag-gated, default OFF
Route `TreeAttentionImpl` **prefill** through `flash_attn_varlen_func(...)` with **NO tree bias**, matching native `FlashAttentionImpl` prefill **call-for-call** (same `scheduler_metadata` / `q_descale` / `k_descale` / `cu_seqlens` / `max_seqlen` / `softmax_scale` / `causal` / window). Implement in `scripts/fr13_patch_fa2_tree_bias.py`, mirroring the existing **decode** patch's mechanism in that file. **Flag-gate** behind `FR13_FA2_PREFILL_NATIVE` (default **off**) so the current e2e build (which the 6/6 argmax finding was measured on) is **unchanged**; turn the flag ON to validate/deploy the fix. Must NOT touch the tree **decode** bias path or **regular decode** (Gate-2 must stay 0.0).

## Verification (pending the one-GPU window, AFTER codex's e2e)
Offline prefill replay (`scripts/fr13_prefill_full_attn_replay.py` + the GDN-state reducer): with `FR13_FA2_PREFILL_NATIVE=1`, confirm **L7 `attn_out_raw` = 0.0** and **GDN L8 `h0_state_in` = 0.0** vs native; re-confirm Gate-2 regular-decode still 0.0; bind to `FR13_LADDER_LOG.md`. Do NOT boot a server while another process holds the single GPU.
