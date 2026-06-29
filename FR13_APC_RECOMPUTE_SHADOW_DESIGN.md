# FR13 APC per-carrier in-forward file-based shadow — DESIGN (2026-06-28)

Goal: on the LIVE derailing cache-ON SWE task (astropy-12907, model's own output drives turns),
capture to FILES — on every cache-hit forward — both the RESTORED carrier state a hit reads and a
same-forward NO-CACHE RECOMPUTE of that state, for all 4 carriers (GDN conv, GDN ssm, full-attn KV,
position), diffed OFFLINE with the proven per-token-argmax reducers. No logger, no blocking GPU sync
of state values in the hot path, no perturbation of the served trajectory.

## DECISIVE FINDING
The full-attn KV restore capture ALREADY EXISTS (`FR13_FLASH_ATTN_OP_CAPTURE`, patcher L15190-15297:
reads `dense_key/value` from `attn_metadata.block_table` at the `flash_attn_varlen_func` call = the
restored prefix KV a hit feeds the kernel). The GDN ssm recompute ALSO exists (`_fr13_post`, the
recurrent-exact no-cache state, patcher L5609-5623 in the HIT_RECURRENT_SUFFIX block). Missing =
only the same-forward recompute TWIN dump + offline diff.

## CARRIER SCORECARD (3/4 faithful on the deterministic replay; KV last)
- GDN conv restore  ✓ 48/48 faithful
- GDN ssm restore   ✓ 48/48 faithful
- full-attn position ✓ faithful (RoPE base = true prefix len, not stale)  [REFUTED as carrier]
- full-attn KV cache = LAST UNMEASURED (mamba-cache & kv-cache use different block tables)

## HOOK-POINT MAP (scripts/fr10_phase4_patch_vllm_tree_gdn.py, verified this session)
- GDN conv restore read:        L5473-5516 (prefill_conv_replacement); conv_restore capture L5485-5497 (FR13_APC_CONV_RESTORE_CAPTURE); pre_conv L5475; conv_out L5512
- GDN ssm restore + in-forward recompute: L5539-5720 (prefill_scan_replacement / HIT_RECURRENT_SUFFIX); hit rows via has_initial_state L5587; restored seed initial_state[hit] L5602; recurrent-exact recompute fused_sigmoid_gating_delta_rule_update L5609-5623 (_fr13_post)
- GDN file capture:             L5738-5864 (FR13_PREFILL_GDN_CAPTURE); initial_state L5842, conv_restore L5824, final_state L5844, core_out L5843, has_initial_state L5799-5803; CUDA-graph-safe skip L5764-5766
- GDN ssm commit (post-fwd):    L5865-5868 (ssm_state[non_spec_state_indices]=last_recurrent_state); spec-decode commit _fr10_commit_handoff L4327-4436
- FULL-ATTN KV restore read:    L15190-15297 (_fr13_flash_attn_op_capture); dense_key/value from block_table L15246-15250; key_input/value_input L15279-15284; query/output/block_table/seq_lens/query_start_loc; gate FR13_FLASH_ATTN_OP_CAPTURE; CUDA-graph-safe L15206; wired at FlashAttentionImpl.forward anchor L15305-15319 (vLLM flash_attn.py L793-815). KV write reshape_and_cache_flash flash_attn.py L869-878. (TreeAttn capture L14974-15170 fires on DECODE only, NOT the cache-hit re-prefill -> FLASH_ATTN hook is the right one.)
- full-attn stage capture (recompute source): _patch_qwen_full_attn_capture L15957-16180; k/v_after_rope L16156-16157, positions L16155 (FR12_FULL_ATTN_CAPTURE)
- write-side shadow (DO NOT reuse for verdict; failure-mode 5 + uses .item()): _fr13_apc_shadow_log L11754-11900
- env scaffold: FR13_APC_SHADOW taken (L1350-1356) -> use NEW flag FR13_APC_RECOMPUTE_SHADOW
- vLLM source readable: /tmp/vllm_cu130_src/vllm/v1/attention/backends/flash_attn.py (context_len=seq_len-query_len L479-480), tree_attn.py, worker/mamba_utils.py (collect_mamba_copy_meta L97-135, postprocess_mamba L222-267, accept_token_bias L257)

## REUSE (proven instruments)
- scripts/fr13_apc_hit_first_divergence.py = the diff brain (conv _conv_window_match L208-238, ssm min-dist L175-189, core_out per-pos argmax _argmax_mismatch L192-205, verdict ratio thr 3.0). Keep file schema payload-compatible so it runs unchanged on the restore arm.
- scripts/fr13_apc_restore_confound_diff.py = trustworthy SSM diff metric ma() L26-83.
- scripts/fr13_full_attn_multilayer_table.py = FA2 path-rerun oracle _fa2_tree_path_rows L122-180, _dense_spine_math L183-228 (template for KV-restore-vs-recompute flash-attn output argmax).
- scripts/fr13_apc_cacherow_diff.py = layer parsing L47-65 only (its write-frame produced the "92.7% stale" artifact — do NOT reuse the frame).

## FULL-ATTN KV MEASURE (cheapest, no patcher edit): one cache-ON boot, two existing flags
Form A: FR13_FLASH_ATTN_OP_CAPTURE (restore dense_key[:context_len]) + FR12_FULL_ATTN_CAPTURE
(k_after_rope = same-forward no-cache recompute of the suffix-overlap rows). Diff offline with a NEW
reducer scripts/fr13_apc_fakv_restore_diff.py: max_abs + flash-attn OUTPUT-row argmax rerun (scalar-
blind guard). Overlap = the boundary region (cached_tokens..seq_len) where the num_accepted bug lives.
Form B (fallback, full prefix): side-CUDA-stream re-projection of the cached prefix tokens; never sync.

## TWO MODES (same captures + reducers)
- MEASURE (FR13_APC_RECOMPUTE_SHADOW=1, cache drives the forward): reproduces the REAL derail on the
  LIVE task, localizes which carrier is stale on the failing turn. RUN FIRST.
- CORRECT (FR13_APC_RECOMPUTE_SHADOW=use_recompute / existing FR13_APC_HIT_RECURRENT_SUFFIX=1 for ssm):
  substitute the recompute into the forward; if the task then resolves, that carrier is causally THE
  cause. Extend per-carrier (conv-leaf substitute; FAKV write recompute K/V into cache rows pre-kernel).

## RUN RECIPE
Vehicle: scripts/fr13_bigdenom_swe_serve_variant.sh <arm> cat6root subset_astropy12907.json (boots the
forked cat6root tree server, runs the codex SWE agent). OFFLOAD_CODEX=0 (measurement, not behavior change
-> reproduces the real derail). ENFORCE_EAGER=1 (diagnostics). Bound to ~15 turns (derail ~turn 12).
Restore-arm flags: FR13_PREFILL_GDN_CAPTURE, FR13_APC_CONV_RESTORE_CAPTURE=1, FR13_FLASH_ATTN_OP_CAPTURE,
FR12_FULL_ATTN_CAPTURE; per-layer LIMITs low (8/64) to bound eager+capture turn time (~22min/turn full on)
+ disk. Key payloads by a per-new-turn monotone _FR13_APC_TURN_ID (NOT req_id — the cross-turn-reqkey trap).

## IMPLEMENTATION SCOPING (refined 2026-06-28, found the MEASURE-mode gating subtlety)
The existing GDN recurrent-exact recompute (_fr13_post, patcher L5626; restored seed _fr13_h0=initial_state[_fr13_hit_idx] L5602; hit rows _fr13_hit_rows L5583/5594; kernel fused_sigmoid_gating_delta_rule_update L5609) is INSIDE the block gated by FR13_APC_HIT_RECURRENT_SUFFIX==1 (L5549) — and that flag is the CORRECT/SUBSTITUTE mode (it writes the recompute back into the forward, changing the trajectory). So the agent's 8a "just insert a dump" is WRONG for MEASURE mode. MEASURE needs a SEPARATE measurement-only gated path (FR13_APC_RECOMPUTE_SHADOW==1) that: (a) detects cache-hit rows (has_initial_state), (b) runs the SAME suffix recompute kernel to get _fr13_post, (c) dumps initial_state[hit] (restore) vs _fr13_post (recompute) to file, (d) does NOT write _fr13_post back (forward continues on the cache-restored state = reproduces the real derail). Implementation = clone the recompute portion (L5583-5629) into a new measurement-only branch, strip the write-back (L5665-5713), add the dump. Verify byte-identical-when-OFF via ast.parse + a same-boot default-path check.
PIECES, in order: (1) GDN ssm measure-only recompute+dump (above); (2) GDN conv measure dump (conv_restore already captured L5485; recompute = trailing K-1 cols of pre_conv at boundary, offline); (3) full-attn KV: restore already captured (FR13_FLASH_ATTN_OP_CAPTURE dense_key/value); recompute = Form B side-stream re-projection of cached prefix tokens (heavier patcher edit) OR a same-boot cache-OFF FR13_FLASH_ATTN_OP_CAPTURE arm + flash-attn output-argmax diff (cheaper, mild confound); (4) offline reducers (reuse fr13_apc_hit_first_divergence.py for GDN; new fr13_apc_fakv_restore_diff.py for KV); (5) live-SWE MEASURE run on astropy-12907 (~15 turns, ENFORCE_EAGER, low per-layer LIMITs), find first stale carrier on the failing turn; (6) CORRECT-mode causal confirm.
NOTE the prompt_token_ids hard problem (teacher-forced gate) is SIDESTEPPED by the in-server file-based twin (no external re-tokenization).

## VERIFIED FLAG INTERFACE (red-teamed 2026-06-28, agent line#s were ~135 off; main=17235 lines)
- FLASH_ATTN op capture (cache-hit RE-PREFILL path, the full-attn KV restore): `_fr13_flash_attn_op_capture` patcher L15191 (patch fn _patch_flash_attn_op_capture L15175). Env: FR13_FLASH_ATTN_OP_CAPTURE=<path> (enables), FR13_FLASH_ATTN_OP_CAPTURE_LAYER (default layers.3.self_attn, "*"=all 16 full-attn), _SKIP, _LIMIT. Reads key_cache[block_id,:take]/value_cache[block_id,:take] over the block table (L15054-15058 equivalent in this fn) -> payload dense_key/dense_value (lists per seq), block_table, seq_lens, query_start_loc, query, key_input/value_input, output, scale, num_heads/num_kv_heads. CUDA-graph-safe (L15206). DISTINCT from the TreeAttn capture (tree_attn_op_capture.v1, TreeAttentionImpl.forward L14990-15104) which fires on DECODE not the re-prefill — use the FLASH_ATTN one for the KV-restore carrier.
- CHEAP full-attn-KV measure (patcher-edit-FREE, scorecard-completing, likely confirms faithful): same-boot 2-pass cache-ON vs cache-OFF(reset-each) with FR13_FLASH_ATTN_OP_CAPTURE=path LAYER=* on a cache-hit turn; offline diff dense_key[:context_len] restore(passA) vs cold(passB) per layer + flash-attn OUTPUT argmax rerun (mild cross-pass autotune confound, but argmax robust to gross-stale). Need NEW reducer fr13_apc_fakv_restore_diff.py. RUN THIS FIRST next session (cheap), THEN the live shadow.

## RECOMMENDED CHEAP DECISIVE PATH (converged 2026-06-28 after the full-attn-KV backend block)
The confound-robust e2e test that distinguishes "single stale carrier" from "diffuse compounding" WITHOUT
the in-server recompute-twin OR the prompt_token_ids hard problem:
TEACHER-FORCED per-token argmax on the REPLAY's cache-hit turns.
- Feed BOTH arms (cache-ON warm vs cache-OFF reset, SAME boot) the SAME continuation tokens (the recorded
  response, or any fixed continuation) -> no free generation -> NO autotune-warming amplification -> argmax
  flips ONLY on a real per-position divergence (the binding instrument per reference_scalar_metric_per_token_blindspot).
- The REPLAY uses the EXACT recorded /v1/responses requests (the dumps) -> the prompt_token_ids hard problem
  (host-render tools-drop) DISAPPEARS: the server tokenizes the recorded request itself.
- Same boot (reset_prefix_cache between arms) -> no cross-boot confound. Teacher-forcing -> no cross-pass
  generation-divergence (the flaw that made my same-boot greedy free-gen read turn-0 divergence = autotune).
VERDICT: argmax FAITHFUL on all replay cache-hit turns -> per-turn restore is argmax-lossless -> the SWE
derail is DIFFUSE compounding (within-floor state diffs accumulating over the long agentic episode) or
agentic/sampling, NOT a single-turn flip -> then the only remaining instrument is the LIVE per-token argmax.
argmax FLIPS on a replay cache-hit turn -> a real per-turn divergence -> drill the carrier at that turn.
This subsumes the full-attn KV carrier (logits are downstream of KV). Check scripts/fr13_apc_teacher_forced_logit_gate.py
can take a recorded request + teacher-force its continuation; the gate exists but its prompt-source path
needs the recorded-request route wired (it had a host-render tools-drop bug). RUN THIS next session (cheap, 1 boot).

## AVOIDS THE 5 DOCUMENTED FAILURE MODES
1 replay-doesn't-derail -> runs LIVE; 2 blocking-sync-perturbs -> .detach().cpu().clone() only, .item()
on CPU index ints only, Form-B on side stream; 3 logger-dropped -> file-based torch.save; 4 scalar-blind
-> per-token/per-(token,head) argmax gate; 5 write-frame-artifact -> reads the RESTORE frame (state a hit
actually consumes), never the collect_mamba_copy_meta write source.
