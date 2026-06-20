# APC residual-carrier lossless DRILL-DOWN design (2026-06-20)

Workflow wgzk90y3h (4 agents, source-verified). The conv whole-row snapshot (dec50857,
FR13_APC_CONV_SNAPSHOT=1) is a PARTIAL fix — metrics improved but tasks still give up empty.
This pins the residual carrier + a fast one-task instrumented drill to confirm it, per the
user's steer (one task is enough). Branch fr13-prefix-cache.

## Residual-carrier hypothesis (source-confirmed)
FIX-1/FIX-2 only touched `get_conv_copy_spec`. **`get_temporal_copy_spec` — the SSM/recurrent
align snapshot — was NEVER made tree-aware.** Stock (`[vLLM] mamba_utils.py:336-341`) reads
`state[block_ids[cur_block_idx + num_accepted-1]]`: a clean WHOLE-ROW copy, but of a row chosen by
APC/bias arithmetic, **NOT** the committed-leaf NODE row our tree committer wrote at
`fr10_phase4_patch_vllm_tree_gdn.py:5139-5142` (`spec_state_indices_tensor[b,:tree_n]`). At
`num_accepted>1` those row-spaces disagree → the align cache snapshots an uncommitted/branch-loser
recurrent row → corrupt GDN state restored on the next-turn cache hit → garbled runaway →
empty-patch give-up. **Same wrong-row WIRING class as the conv FIX-1, on the copy we never
overrode** (NOT a position-shift, NOT an fp8 round-trip — fp32 ssm cache is byte-clean at the copy).

## The drill-down (one task, ~15-25 min, mostly EXISTING taps)
- **Task:** `astropy__astropy-13033` (gave up empty in 208s under the snapshot fix = fast + reliably
  poisons; 13236 ran the full wall ~1933s under the snapshot so it's slower for the drill).
- **Boot env:** FR13_ENABLE_APC=1, FR13_APC_CONV_FIX=1, FR13_APC_CONV_SNAPSHOT=1 (conv already fixed
  → the residual is what's left), **ENFORCE_EAGER=1** (taps are eager-only — `_fr13_boundary_emit`
  raises under CUDA-graph capture, fr10_phase4:1008-1019), **FR13_REPLAY_BOUNDARY_LOG=1**,
  FR13_REPLAY_BOUNDARY_LAYERS=layers.0.linear_attn, MAMBA_BLOCK_SIZE=1024, ssm float32, AGENT_WALL_S=900.
- **The detector already exists:** Tap C (`fr10_phase4:10977-11003`) fires on every align copy-op
  with `copy_func ∈ {get_conv_copy_spec, get_temporal_copy_spec}`, `phase ∈ {preprocess=restore,
  postprocess=snapshot}`, src/dest row, AND — only for get_temporal_copy_spec — the **`stale_read`**
  verdict (`:10968-10976` = `src_row not in committer's last_written_rows`, published :6397). That's
  the direct wrong-row detector. Tap B_JOIN (`:4625-4745`) gives the SSM h0 producer→consumer
  BYTE_EQUAL/BYTE_DIFF verdict. Tap C_bias (`:10746-10781`) shows the bias translation.
- **Localize:** at a garbling `num_accepted>1` boundary → Tap C `stale_read==true` on
  `get_temporal_copy_spec` (while conv clean = FIX-2 working) = SSM align snapshot is the carrier.
  Tap B_JOIN BYTE_DIFF corroborates. The taps ALSO reveal WHICH row is the committed leaf (the fix
  target). Optional ~20-line `C_TEMPORAL_JOIN` tap for snapshot↔restore↔oracle byte confirmation.
- **GATE (replaces the 4-task garble count):** at every num_accepted>1 align boundary the SSM state
  restored from the APC cache must be BYTE-EQUAL (fp32 → exact) to the committed-leaf recurrent row.
  Single-task, decisive, not the coarse garble scan.

## The fix (DECISION POINT: determine the row from tap data, do NOT assume)
Mirror the conv override: extend `_patch_mamba_state_utils_tree_conv_node_copy` to ALSO override
`get_temporal_copy_spec`, gated `FR13_APC_SSM_SNAPSHOT` (default 0, inert; **MUST be added to the
docker -e list** — the conv-flag wiring bug). Preprocess/restore reads the committed-leaf node row;
postprocess/snapshot falls through to stock. WHOLE-ROW copy, no offset slice (temporal is wrong-row,
not shifted). **CRITICAL:** the correct restore row may NOT be `block_ids[cur_block_idx+num_accepted-1]`
— if the tap data shows the committed leaf is at `spec_state_indices[b, accepted_len-1]` (a NODE bank
row, not in block_ids), the fix must read THAT row. Resolve the exact source from the drill-down tap
output BEFORE writing the fix value.

## Iterate loop (~minutes/iter vs ~90min)
instrument(once) → boot one give-up task eager+taps (~15-25min) → read tap C stale_read/B_JOIN →
flip FR13_APC_SSM_SNAPSHOT=1 with the tap-derived correct row → re-drill → confirm stale_read==false
/ BYTE_EQUAL at every boundary + no empty give-up → bind lossless (fr13_b1_lossless_prescore.sh on
the one task, clear-margin flip within E5 floor). Per [[feedback_no_cross_boot_byte_gate]]: don't
conclude "fixed" from one non-give-up eager boot (autotune floor); re-run/confirm.

## Risks
- Multiple simultaneous carriers (conv residual + ssm) — taps separate them per copy_func.
- Garble in the SSM VALUE not the row index — B_JOIN byte-compare catches value drift too.
- fp8 ssm round-trip — mitigated by --mamba-ssm-cache-dtype float32 (byte-clean at the copy).
- 13033 resolves instead of giving up (autotune nondeterminism) → re-run via /reset_prefix_cache or
  use 13236 (full-wall, slower but robust).

Source: workflow wgzk90y3h output. Links apc_armA_poisoning_finding, vllm_43559_rootcause,
sglang_mamba_radix_cache_design.

---
## DRILL RESULT 2026-06-20: SSM carrier CONFIRMED (cat9_apc_tap, 13033, eager+taps, 10060 tap records)
Tap C stale_read (computed only for get_temporal_copy_spec):
- **get_temporal_copy_spec: postprocess 480/480 STALE (100%), preprocess 464/2400 STALE (19%)** — reads a row NOT in the committer's written rows = the carrier. B_JOIN corroborates (READ_ROW_NOT_WRITTEN 10, NO_PRIOR_WRITE 21). e.g. snapshots src_row=69, committed=[35,70,37,75,39] -> 69 ∉ committed.
- get_conv_copy_spec: stale_read NOT computed (tap only sets it for temporal) so can't isolate via taps; but the garble scan (conv snapshot 12 vs 6-8, no reduction) is the evidence conv isn't the dominant carrier.
THE CORRECT ROW (Tap A committer producer): for accepted_path [1,2,4,6,8] (len 5) the committer writes per-depth SSM states to consecutive rows 849..853 (dst_rows col0..col4); the **accepted-LEAF = the last = row 853 = spec_state_indices[b, accepted_len-1]**. The snapshot must read THAT, not the bias-chosen row.
WHY temporal lands wrong: get_temporal_copy_spec reads block_ids[cur_block_idx + num_accepted-1] where num_accepted = bias+1; in postprocess the conv FIX-2 (FR13_APC_CONV_FIX) returns the RAW alignment remainder as the bias (so conv falls through to whole-row), and the temporal copy inherits that raw remainder -> block_ids[cur+remainder] = wrong row. FIX: tree-aware get_temporal_copy_spec override gated FR13_APC_SSM_SNAPSHOT that reads the committed accepted-leaf row (NOT the bias-chosen one), in BOTH phases. Needs the committer's accepted-leaf row source (spec_state_indices[b,accepted_len-1] / the published last_written_rows leaf), since block_ids[cur+raw_remainder] is wrong. Implementation = careful (the override must reach the committed-leaf row; mirror how the conv override at :11048 derives its tree row but use the LEAF specifically).

---
## CORRECTED FIX PLAN 2026-06-20 (design's path[-1] was WRONG — red-team caught it)
RED-TEAM: the fix-design (wg435jh8y) published path[-1] as the leaf row. Tap proof: path[-1]==committer rows[-1] in **0/4350** records. path[-1] = _gdn_path[-1] = (leaf_node_id)+1 (a NODE value); the actual committed leaf ROW = the bank row the committer WROTE = spec_idx[b][accepted_len-1] = _rows_written[-1]. `_LUMO_FA_LAST_ACCEPTED_TREE_ROWS` = accepted_gdn_rows = _gdn_path[-1] = SAME wrong node value (not the written row). `accepted_rows` = per-batch count/index, not the bank row.
THE CORRECT LEAF SOURCE: `_fr13_layer._fr13_replay_spec_idx[b][int(accepted_len)-1]` (CPU int), available at the COMMIT site fr10_phase4:8075-8092 (the _fr13_replay_launch call; spec_state_indices=_fr13_layer._fr13_replay_spec_idx, lens=_fr13_replay_lens). Also a 2nd commit site at ~8719-8743. The Tap A producer (_fr13_boundary_replay_post :6348) computes _rows_written[-1] = THIS, and publishes _FR13_BOUNDARY_LAST_WRITTEN_BY_REQ[req_id]['rows'][-1] — but it's heavy (torch.cuda.synchronize) + gated under FR13_REPLAY_BOUNDARY_LOG.
3-SITE FIX (all gated FR13_APC_SSM_SNAPSHOT default 0; ADD flag to launcher docker -e list :305-306):
 1. COMMIT site (after _fr13_replay_launch, both 8092 + 8743): publish per-req leaf -> gdn_mod._FR13_APC_SSM_LEAF_BY_REQ[req_id] = int(spec_idx[b][alen-1]); req_ids from gdn _LUMO_FA_SAMPLER_ROW_REQ_IDS; lens from _fr13_replay_lens. (one .tolist() sync/step, B=1 OK; optimize later.)
 2. BIAS chokepoint (_fr10_tree_accept_token_bias, has req_id; extend the _patch_mamba_utils_preprocess_context_flag injection): read gdn._FR13_APC_SSM_LEAF_BY_REQ.get(req_id) -> publish _fr13_mecopy._FR13_CUR_SSM_LEAF_ROW (per-call) + _FR13_CUR_SSM_LEAF_REQ=req_id; clear if None.
 3. OVERRIDE (extend _patch_mamba_state_utils_tree_conv_node_copy to also replace get_temporal_copy_spec): if FR13_APC_SSM_SNAPSHOT + tree mode + num_accepted>1 + _FR13_CUR_SSM_LEAF_ROW not None: src_state = state[_FR13_CUR_SSM_LEAF_ROW] (whole row, no slice); else stock. Add _FR13_CUR_SSM_LEAF_ROW=None to the replacement header globals.
GATE: re-drill (eager+taps + FR13_APC_SSM_SNAPSHOT=1, verify reaches container) on 13033 -> Tap C stale_read==false on get_temporal_copy_spec at EVERY num_accepted>1 boundary + no garble. Then per-token rescore within E5 floor vs no-spec recurrent oracle.
