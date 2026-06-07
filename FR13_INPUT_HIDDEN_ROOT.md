# FR-13 ROOT (Claude, 2026-06-07): tree verify input-hidden = MTP fc-fusion prev_hidden threaded wrong

codex's per-layer diff pinned the first divergence to the verifier INPUT hidden (pre-layer-0): input max_abs by depth `[0.0, 0.326, 0.387, 0.299, 0.107]` (depth 0 matches, depths 1-4 diverge), with positions/tokens/RoPE all matching, propagating to final_norm drift ~25 (committed 4c858a29).

## The mechanism (source, live qwen3_5_mtp.py:112-128)
`Qwen3_5MultiTokenPredictor.forward` does EAGLE-style fusion:
```
inputs_embeds = pre_fc_norm_embedding(inputs_embeds)
hidden_states = pre_fc_norm_hidden(hidden_states)      # previous-position hidden
hidden_states = cat([inputs_embeds, hidden_states], -1)
hidden_states = fc(hidden_states)                      # -> decoder-layer input
```
Verify input = `fc(cat(norm(embed(token)), norm(prev_hidden)))`. Embeds match (tokens match), so the depth-0-matches / depths-1-4-differ pattern is EXACTLY the `prev_hidden` (previous-position hidden) input differing: depth 0 prev_hidden = prefix (matches); depths 1-4 prev_hidden = previous SPINE node hidden (tree threads wrong).

## The bug
The tree feeds the WRONG `hidden_states` (prev_hidden) into the MTP fc-fusion for spine depths >=1. For the linear spine, the correct prev_hidden for node d = node d-1's VERIFY hidden. Candidates: tree feeds the DRAFTER's proposal hidden instead of the verify hidden; a stale/un-threaded hidden buffer; or the wrong tree node's hidden (parent-vs-sequential, though for the caterpillar spine parent==sequential).

## Fix + verify
Thread the correct previous-spine-node verify hidden into the fc-fusion so the tree spine input_hidden == native MTP-5 (depths 1-4 -> 0.0). Then: per-layer input drift 0 -> final logits match -> spine-only accept 0.83 -> ~2.7, bag_TV vs E5 -> floor. Verify spine AND branch (branch nodes thread their parent-path hidden). WIRING fix in our tree verify input construction, NOT a kernel.
