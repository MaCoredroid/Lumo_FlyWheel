# FR13 Branch Seam CPU Prep Bind

Date: 2026-06-09

HEAD before bind: `785f3b12`

Scope: CPU-only artifact inventory for the genuine branch seam identified by the all-8 branch oracle. No server was booted and no GPU work was run.

## Target Seam

Source artifact: `output/fr13_decisive_final_20260609T182455Z/all8_branch_oracle.json`

First real mismatch:

- prompt `0`, sample `0`
- event index `13`
- served prefix length `45`
- path `[0,1,3,5,8]`
- node `8`
- depth `4`
- check kind `parent_target`
- path prefix token IDs `[638,4381,283,1727]`
- tree parent-target token `198`
- native-on-branch next token `1358`
- tree accepts draft: `true`
- native accepts draft: `false`

Tree trace row:

- `output/fr13_decisive_final_20260609T182455Z/tree_b1_greedy_branch/logs/tree_path_lcp.jsonl`, line `14`
- `accepted_node_ids=[0,1,3,5,8]`
- `winner_path=[0,1,3,5,8]`
- `emitted_tokens=[638,4381,283,1727,198,262]`
- `draft_token_ids=[638,4381,13,283,4922,1727,2168,363,198]`
- `parent_target_ids=[638,4381,4381,283,283,1727,1727,198,198]`
- `self_target_ids=[4381,283,1727,1727,283,198,198,4071,262]`

## Existing Capture Inventory

The exact all-8 run root has no layer/subkernel `.pt` capture for this seam:

- present: `all8_branch_oracle.json`
- present: tree `tree_path_lcp.jsonl`, `tree_sampler_debug.jsonl`, `fr10_mtp_draft_trace.jsonl`
- present: native oracle server logs
- absent: `gdn_l*_subkernel*.pt`
- absent: `full_attn_*.pt`
- absent: `tree_layer_hidden*.pt` / `native_layer_hidden*.pt`
- absent: final-logit `.pt` captures

Checked nearby B1 capture roots:

- `output/fr13_gdn_substate_prompt0_20260609T061732Z`: has L0 subkernel `.pt` calls `0..2` and layer-hidden calls `0..3`; it does not contain the `[638,4381,283,1727]` seam event.
- `output/fr13_pos16_substate_20260609T081638Z`: has L0 subkernel `.pt` calls `0..6`; it does not contain the `[638,4381,283,1727]` seam event.
- `output/fr13_conv_fix_same8_greedy_token_20260609T074753Z`: has a nearby prefix in `tree_path_lcp.jsonl` but no `.pt` substate capture; it is not the all-8 oracle mismatch and lacks the native-on-branch paired target.
- `output/fr13_branch_node_redteam_20260609T090932Z`: has tree LCP/sampler traces and native probes, but no `.pt` substate capture.

Conclusion: there is no existing B1 substate capture for prompt0/event13/node8 that can localize where tree `198` vs native-on-branch `1358` is born. Existing artifacts identify the branch seam but cannot pin a layer or sub-op.

## Prepared Future Capture

Do not run this until the CPU workflow gives the next GPU decision. If needed, the minimal targeted boot should reproduce the all-8 prompt0 B1 greedy branch event and capture only the seam event:

Tree side:

- Use the same B1 greedy tree branch probe shape that produced `output/fr13_decisive_final_20260609T182455Z/tree_b1_greedy_branch`.
- Enable `FR12_SUBKERNEL_CAPTURE=/logs/gdn_l0_subkernel.pt`.
- Start at `language_model.model.layers.0.linear_attn`; repeat across candidate GDN layers only if L0 is clean.
- Use `FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=10`.
- Use `FR12_SUBKERNEL_CAPTURE_SKIP=13` and `FR12_SUBKERNEL_CAPTURE_LIMIT=1` to hit the tree event with `tree_path_lcp.jsonl` line `14` / event index `13`.
- Capture all rows for that event; node8's parent target is sourced through the tree target-row mapping, so do not pre-filter to row8 only.
- Add `FR13_FINAL_LOGIT_CAPTURE` with rows covering the target-row source and node8/self rows if the subkernel is clean through `o_proj_out`.

Native-on-branch side:

- Use the existing branch-oracle path construction, not an ad hoc prompt.
- Native request prefix must be the real prompt0 served prefix of length `45` plus path prefix token IDs `[638,4381,283,1727]`.
- Enable `FR12_SUBKERNEL_CAPTURE=/logs/gdn_l0_subkernel.pt` with `FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=6`, `FR12_SUBKERNEL_CAPTURE_SKIP=0`, `FR12_SUBKERNEL_CAPTURE_LIMIT=1`.
- Capture final logits for the native next-token row to confirm `1358` is the same config-stable oracle target.

Reducer:

- Use `scripts/fr13_gdn_subop_diff.py` for the first captured GDN layer.
- If GDN is clean through `o_proj_out`, escalate to `FR12_FULL_ATTN_CAPTURE_LAYER_PREFIX` / `FR13_TREE_ATTN_OP_CAPTURE` for the first downstream full-attention layer on the same tree row and native-on-branch row.
