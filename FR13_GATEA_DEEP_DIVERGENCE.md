# FR-13 GATE A — deep-layer spine-vs-NATIVE divergence (under investigation, NOT a pass)

Monitor red-team, 2026-06-07, from `output/fr13_postfork_gate_20260607T165548Z/gateA_spine_full_attn.json` (post-fork tree-vs-native top-down ladder, spine rows tree `[0,1,2,4,6]` → native `[0,1,2,3,4]`).

## Observation
The post-fork spine-vs-**native** ladder is **clean 0.0 through full_attn layer 23**, then **diverges from layer 27 onward**:
- layer 23: all depths 0.0.
- layer 27: `input_hidden` depth0=0.0, depth1=0.125, depth2=0.25, depth3=0.047, depth4=0.034; `attn_out_raw` worst 0.625.
- layer 31: `input_hidden` all depths 0.125–0.5; growing each full_attn layer to **`attn_out_raw` 1.875 by layer 51**, `o_proj_out` 1.625 at L59–63.
- `first_nonzero_stage = input_hidden` for L27+ → the divergence is **inherited** (the residual feeding the layer is already off), originating in the **GDN (linear_attn) layers 24–26** between full_attn 23 and 27, then compounding.

## Why this is NOT the grouping floor and NOT a token mismatch
- Magnitude 0.25→1.875 ≫ the 2-ULP floor (0.0039).
- **Input tokens MATCH native**: layer-3 `input_hidden` = 0.0 on all 5 spine depths (the spine = the linear MTP-5 chain = native's chain; embeddings identical). A token mismatch would diverge from layer 3, not appear fresh at layer 27.

## Why the workflow's "14/16 byte-exact" did NOT catch it
The workflow checked `tree_attn_op` (forked FA2) vs an FA2-on-path **oracle built from the tree's OWN captured q/k/v** — a *kernel* check. It structurally cannot see the hidden state diverging from **native**. This is exactly why the user mandated the full ladder-vs-native; it surfaced a divergence the op-check masked.

## Two hypotheses (root-cause TOP-DOWN — decisive test pending)
- **(A) no-copy GDN shared-state contamination of the spine** — path0 (spine) degraded by branch tokens sharing the recurrent state in GDN layers 24–26 (the FR10/FR11 core issue: `project_gdn_tree_superset_routes`, `project_fr10_nocopy_costgate_conclusion`). The FA2 fork fixes full_attn but does **NOT** address GDN shared state. If true, the tree spine is **not** byte-exact/lossless to native → a real losslessness finding.
- **(B) deep-layer row-alignment / capture artifact** in the in-progress reducer (native MTP-5 row order at depth).

**DECISIVE TEST (directed to codex):** run **spine-only (branches OFF / spines=1)** tree vs native. If the deep divergence **vanishes** → (A) shared-state contamination. If it **persists** → (B) alignment/capture. Also re-verify the deep-layer native row mapping.

## Status
GATE A is **NOT passed** and must not be bound as passing. `gateA_spine_ladder.json` final hidden/logits are still **empty** (`passed: False`) — the final-spine-logits-vs-native number (the losslessness-critical one) is not yet computed. If (A) holds, the deep-layer divergence will flip final-spine-logit argmaxes beyond the E5 floor → the e2e bag-TV would be lossy. Surfaced to the user; no self-declared pass.
