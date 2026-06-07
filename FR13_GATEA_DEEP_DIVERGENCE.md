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

## Per-layer curve (all 64 layers, tree-spine vs native, fp32) — answers "special layer vs compounding"
Captured `tree/native_layer_hidden.pt` hold every layer's `hidden`+`residual`. Diff at spine rows `[0,1,2,4,6,8]`→native `[0,1,2,3,4,5]` (pos 13–18):
- **Layers 0–23: EXACTLY 0.0 (bit-identical, fp32).** Not "tiny" — zero. So this is NOT gradual accumulation from layer 0.
- **Layer 24 (linear_attn/GDN, first GDN after full_attn 23): FIRST nonzero** — hidden 0.035, residual 0.25. Discrete step injection.
- Layers 25→63: compounds → 0.05, 0.125, … 1.9 (L58), **5.25 (L63)**.
⟹ **Verdict: a discrete onset at GDN layer 24 + downstream compounding — NOT a smooth ramp.** The originating layer is 24.

**Onset is NOT branch-aligned (flips the leading hypothesis):** at L24 the largest diff is **pos 14 (spine row 1)**, which has NO branch in its conv window or scan ancestors (branches start at pos 15); root pos 13 diverges *later* (~L28). That is the OPPOSITE of branch-contamination ordering. So (A1) branch-into-spine contamination is now UNLIKELY. New leading hypothesis: a **layer-24+ GDN kernel/state issue** (tree `_tree_gdn_kernel` vs native FLA, or an `h0`/recurrent-state-bank column selection that is bit-exact for the first 23 GDN layers then diverges), amplified by the gate (~32× 1/rms, FR12) and compounded. The spine-only test should still diverge at L24/pos14 if branches are irrelevant (expected).

## Status
GATE A is **NOT passed** and must not be bound as passing. `gateA_spine_ladder.json` final hidden/logits are still **empty** (`passed: False`) — the final-spine-logits-vs-native number (the losslessness-critical one) is not yet computed. If the divergence is real (A1), it will flip final-spine-logit argmaxes far beyond the E5 floor → the e2e bag-TV would be lossy → a genuine no-copy-GDN losslessness finding to fix at root (the mask/state-indexing leak), NOT to wave through. **Keep the 2-ULP floor separate**: that is the accepted irreducible no-copy grouping floor; THIS (0.25–1.875) is a distinct structural bug. Surfaced to the user; no self-declared pass. Fix once root cause is confirmed by the spine-only test + per-GDN-layer localization.

## Spine-only decisive test result — 2026-06-07
Run dir: `output/fr13_spine_only_decisive_20260607T171840Z`.

Strict spine-only TREE_ATTN:
- `TREE=[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0)]`
- `FR13_FA2_TREE_BIAS=1`
- `FR10_ALLOW_LINEAR_FALLBACK` unset by `scripts/fr13_launch_forked_fa2_tree_server.sh`
- `--enforce-eager`, `--gpu-memory-utilization 0.4`, B=1

Matched native:
- `--attention-backend FLASH_ATTN`
- `fr10_decode_mode=naive_mtp`
- 5-token MTP, B=1, eager

Row mapping was re-verified on the captured first verifier event:
- scheduled token IDs equal: `[271, 71093, 12305, 198, 727, 884]`
- positions equal: `[13, 14, 15, 16, 17, 18]` for all three mRoPE rows
- hidden rows equal: tree `[0,1,2,3,4,5]` vs native `[0,1,2,3,4,5]`
- logits rows equal: tree `[0,1,2,3,4,5]` vs native `[0,1,2,3,4,5]`
- sampler metadata equal on the verifier rows: `logits_indices=[0,1,2,3,4,5]`, `target_logits_indices=[0,1,2,3,4]`, `bonus_logits_indices=[5]`, `sampled_token_ids=[271,71093,12305,198,727,884]`

Spine-only ladder result (`spine_only_ladder.json`, threshold `0.00390625`):
- input max_abs: `0.0`
- first nonzero: **layer 45 linear_attention**, hidden `0.01953125`, residual `0.015625`
- layer 43 and 44 remain exactly `0.0`
- final_norm max_abs: `1.0`
- final logits max_abs: `0.59375`

Interpretation: the divergence **persists with branches OFF**, so branch-state contamination is not the sole explanation for Gate A failure. The row mapping check did not support a deep-layer alignment artifact. Current localization is a spine-only GDN/tree-kernel mismatch beginning at layer 45 on this single-spine run, distinct from the earlier branched-tree layer-24 onset and far above the accepted FA2 2-ULP grouping floor.
