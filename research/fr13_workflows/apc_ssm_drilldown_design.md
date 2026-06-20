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
