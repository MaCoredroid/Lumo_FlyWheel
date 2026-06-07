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

## Code read (monitor, alongside codex) — what the drift data + source say
**Positions confirm the contamination geometry.** tree event positions = `[13,14,15,15,16,16,17,17,18,18]` (spine rows 0,1,2,4,6,8 at pos 13–18; branch rows 3,5,7,9 duplicate-positioned at 15/16/17/18, interleaved *between* spine tokens in sequence order). native = `[13,14,15,16,17,18]` (clean chain). So a spine token's GDN conv-window / recurrent-scan neighbors include the interleaved branch token **unless tree-masked**.

**Both GDN cross-token ops ARE tree-masked (live source):**
- Scan: `fr10_gdn_tree_kernel.py:_tree_gdn_kernel` (L278–289) replays each node's ancestor path gated by `visible_mask[i,j]` — "keeps spine results independent of sibling rows." Spine scan excludes branches by construction.
- Conv: `fr10_phase4_patch_vllm_tree_gdn.py` `use_fr10_tree_conv` + `fr10_tree_conv_source_indices` (source-by-width ancestor indices) — tree-aware conv windows.

**Therefore:** simple branch→spine contamination is *guarded*, and the guard holds through layer 23 (= 0.0). The divergence **magnitude (0.25→1.875, not ULP-scale)** rules out the **2-ULP grouping floor** (0.0039, a separate, real, accepted phenomenon — do NOT conflate) and rules out a small kernel fp-order diff (~1e-7). 0.25–1.875 is **structural**: the spine genuinely sees different state/inputs at deep layers. Refined hypotheses:
- **(A1)** a deep-layer **leak in the tree mask / state-bank (`h0`) indexing** — the conv/scan isolation or the per-layer recurrent-state-column selection (`h0_indices`/`h0_num_accepted_tokens`) goes wrong past a certain depth/layer → spine state picks up branch/foreign contribution. REAL, fixable.
- **(A2)** tree-GDN kernel (`_tree_gdn_kernel` ancestor-replay) vs native FLA chunked kernel diverge for the spine — but magnitude argues against (fp-order is ULP-scale, not 1.875).
- **(B)** deep-layer row-alignment/capture artifact in the reducer (same mapping is correct early, so weaker).

**DECISIVE TEST (codex running, `fr13-spine-tree` server):** spine-only (branches OFF) tree vs native. Vanishes ⟹ branch-driven (A1). Persists ⟹ kernel (A2) or alignment (B). Next: per-GDN-layer localization (which of layers 24/25/26 first diverges) to pin the exact op.

## Status
GATE A is **NOT passed** and must not be bound as passing. `gateA_spine_ladder.json` final hidden/logits are still **empty** (`passed: False`) — the final-spine-logits-vs-native number (the losslessness-critical one) is not yet computed. If the divergence is real (A1), it will flip final-spine-logit argmaxes far beyond the E5 floor → the e2e bag-TV would be lossy → a genuine no-copy-GDN losslessness finding to fix at root (the mask/state-indexing leak), NOT to wave through. **Keep the 2-ULP floor separate**: that is the accepted irreducible no-copy grouping floor; THIS (0.25–1.875) is a distinct structural bug. Surfaced to the user; no self-declared pass. Fix once root cause is confirmed by the spine-only test + per-GDN-layer localization.
