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

## DEFINITIVE: the flip is in the GDN (linear_attention), NOT the attention layer
First-diverging layer's `layer_type`, both runs (fp32 hidden vs native):
- **Branched:** L22 linear=0.0, **L23 full_attention=0.0**, **L24 linear_attention=0.035 ← first nonzero**.
- **Spine-only:** **L43 full_attention=0.0**, L44 linear_attention=0.0, **L45 linear_attention=0.0195 ← first nonzero**.

In BOTH runs the full_attention layer immediately before the onset is **exactly 0.0**, and the first nonzero is a **linear_attention (GDN)** layer. Post-onset full_attn layers only carry/amplify an inherited divergence (their `first_nonzero_stage`=`input_hidden`, not their own attn op). ⟹ **The FA2 fork succeeded (full_attn byte-exact); the remaining divergence is the GDN tree-kernel.** The flip is value/state-dependent (branched = 1st GDN after full_attn; spine-only = 2nd) = a ~1-ULP rounding crossing that compounds via the recurrent state. **Fix belongs in the GDN tree-kernel, not the attention.** Sub-op localization (h0/state-load → conv → scan → gate → o_proj) at the onset layer in progress (codex).

## "Look closely" — it's a data-dependent EDGE CASE, not a layer bug (user insight, confirmed)
GDN code is identical per layer, yet the onset `(row, layer)` differs by run: **spine-only = row 0 / layer 45**, **branched = row 1 / layer 24**. Same code, different trigger ⟹ a **value-dependent numerical boundary crossing**, not a layer-specific defect.

Close look at the spine-only onset (L45 hidden, fp32):
- L44 = 0 nonzero (clean). L45 = nonzero **entirely on row 0** (pos 13): ~4587/5120 channels shift up to ~0.02; **rows 1–5 still exactly 0.0**. L46 spreads to all rows (recurrent compounding).
- **Row 0 is the anchor/carried token** (MTP-5 verify: row0 = previously-accepted token whose recurrent state h0 is carried in; rows 1–5 = drafts). Row 0 diverging first ⟹ points at the **carried-state (h0) handoff or row-0 state-readout**, NOT the tree mask (no branches here anyway). The edge case is in the per-node readout/state for the anchor; the recurrent state then compounds it (this is WHY it's a losslessness blocker unlike the non-compounding FA2 floor).
- Magnitude on row 0 is ~0.02 (multi-ULP, not a single bit) across ~90% of channels ⟹ a single upstream state/gate value off, spread by the o_proj/MLP matmul. codex sub-op capture (state-in/h0 → conv → scan → gate → o_proj at L44/45) will pin which.

## ROOT CAUSE PINNED (offline per-stage diff, no GPU) — the causal conv1d for the anchor row
`scripts/fr13_gdn_subop_diff.py` on the L45 sub-op capture (tree vs native), row 0:
- `input_hidden` 0.0, `pre_conv` 0.0 (conv INPUT bit-exact) → **`conv1d_out` = 0.000977 (1 bf16 ULP) = FIRST NONZERO** → `h0_state_in` 0.0 (same bank row [7]) → `gdn_scan_out` 1e-6 → `gate_out` 0.000488 → `o_proj_out` 0.00195. L44 predecessor: all 0.0.
- **The divergence is the conv1d operation**: same conv input (`pre_conv`=0.0), 1-ULP-different conv output, only on row 0 (the anchor/decode token, whose conv path differs from the drafts'). **State-handoff RULED OUT** (`h0_state_in` bit-exact, same bank row). 
- Mechanism: 1-ULP conv seed → `gate_out` amplifies (~16×, the anchor's gate is large) → o_proj 0.00195 → **recurrent state compounds it over L46→63 to 0.59 final-logit drift** (this compounding is why even 1 ULP is a losslessness blocker, unlike the non-compounding FA2 floor).
- **FIX = conv1d bit-exact alignment for the anchor/decode row** (tree causal-conv → native `causal_conv1d_update`; same class as the FR12 "conv bf16-taps" fix). Alignable reduction/rounding-order edge case, NOT a structural bug or state issue. Found by pure data+math (offline diff), no GPU.

## EXPANDED DRIFT GATE (user 2026-06-07): eager-B1 + eager-B4 + CUDA-graphed-B4
The conv 1-ULP is a **value-dependent edge case**, so a fix that zeroes eager-B1 drift is NOT proven at deployment. GATE A (drift→0) must be confirmed across THREE regimes; the conv fix must be **batch-invariant** (handle the #42960-class batch-dependence) to hold at B=4:
1. **Eager B=1 + hooks** — sub-op/per-layer diff (`fr13_gdn_subop_diff.py` + the layer-hidden ladder). Pinpoints the op + proves the fix → 0.0 at the op level. (current regime)
2. **Eager B=4 + hooks** — per-layer hidden ladder at B=4. Confirms the fix holds under B=4 co-residency and that B=4's different values don't surface the edge case at a different (row,layer). Same capture tooling, B=4.
3. **CUDA-graphed B=4, hooks OFF** — the deployment regime. **Constraint: the per-layer/sub-op capture HOOKS crash under CUDA-graph capture** (the FR12 Dynamo/graph-capture instrumentation issue, not the kernel) — so graphed-B4 drift is verified **capture-free**, via the served output: **bag-TV vs E5 (≤ floor) + accept/event (≥ E5)** = the e2e deliverable gate. (Per superset-by-math this is *determined* once drift=0 holds + the fix is batch-invariant — but it MUST be measured here, not assumed.)
A fix is only GATE-A-complete when all three are clean (op-level 0.0 eager-B1, no regression eager-B4, e2e within-floor+superset graphed-B4). Reuse `reference_modelserver_host_memory_recovery` + ONE-GPU between regimes.

## FIX ATTEMPT 1 — fp32 conv taps = WRONG DIRECTION (ruled out, monitor red-team)
Matched-token comparison (positions equal [13–18], tree vs native):
- **bf16-taps (current):** `conv1d_out` 0.000977 (1 elem) + `input_hidden` 0.0 (bit-exact thru L44) — 1-ULP close.
- **fp32-taps:** `conv1d_out` 0.0625 (57k elems) + `input_hidden` 0.3125 (diverges globally) — far worse.
⟹ native `causal_conv1d_update` is ≈ **bf16-taps**; fp32 taps **overshoot** (move the conv AWAY from native at every layer, compounding to 0.31). **fp32-taps reverted.**

**Correct fix direction:** stay bf16-taps; the remaining 1-ULP at L45 is **NOT the tap dtype** — it's an op-order detail. **Read native `causal_conv1d_update` source** (exact tap multiply-accumulate ORDER + bias add + silu/activation rounding) and align the manual tree conv op-by-op to bit-exact — not guess-and-test dtypes (that burns GPU). codex redirected.

## FIX ATTEMPT 2 — silu/activation tie-break (PTX-aligned) = fixed L45 but REGRESSED layer 0
codex aligned the manual conv to native `causal_conv1d_update` PTX (mul.bf16 → fp32 accumulate → ex2.approx sigmoid → bf16 store). **Boot-free replay fixed the saved L45 element, but the LIVE strict tree-vs-native test regressed at layer 0 (max_abs 0.00195).** Reverted, not committed (honest, no false pass).

**Refined root cause:** the manual conv's **silu activation ≠ native's `ex2.approx` silu**, and the mismatch is **per-element whack-a-mole** (a rule that fixes L45 breaks L0). It is NOT the tap dtype (attempt 1) and NOT a single tie-break (attempt 2).

**Smarter plan (avoid live whack-a-mole / GPU thrash):** iterate the manual-conv alignment **fully OFFLINE across MULTIPLE layers at once** (capture conv inputs `pre_conv`+`conv_state`+`conv_weights`+`bias` for L0 + L45 + one branch row in ONE capture; boot-free replay; drive conv1d_out → 0.0 vs native for **ALL** captured layers/rows simultaneously — this catches the L0 regression offline). Replicate native's EXACT op sequence (tap-mul dtype, fp32 accumulate order, bias, silu `ex2.approx`, bf16 store). Only after offline=0.0 for all, ONE live full-ladder test (all spine rows + branches + logits + gate-2). **If the silu proves un-matchable in torch, STOP and bring the option of routing the conv through native `causal_conv1d_update` to the user (possible banned reroute) — do NOT reroute unilaterally.**

## METHODOLOGY CATCH — the multi-layer offline replay was CONTAMINATED at deep layers
codex's `conv_replay_multilayer.json` `input_alignment` (the `_detail_alignment` it added) revealed the **conv INPUT (`pre_conv`) itself diverges** at the captured deep layers: L0=0.0 (clean), **L24=0.152, L45=0.242, L62=0.375**. Cause: in the *live* convml capture, by those layers the tree+native runs had **already drifted** (the bug we're fixing), so the captured conv inputs differ. ⟹ comparing tree-conv-out to native-conv-out there tests **input divergence, not the conv kernel**; only **L0 is a clean test (conv already 0.0)**. The `aggregate_best_variant` (0.125) was tuned against this contamination = invalid.

**Corrected methodology (directed):** tune the conv variant ONLY on layers where `pre_conv` is bit-exact (`input_alignment==0.0`): **L0** (clean) + the **fresh `fr13_gdn_l45_fullstate` L45 capture** (pre_conv=0.0, conv1d_out=1 bf16 ULP — the real edge case). The test must be **SAME-INPUT**: run the manual conv AND native `causal_conv1d_update` on the **identical** captured `pre_conv`+state+weights → isolates the kernel. Drive conv=0.0 on the clean-input layers (fix the L45 edge WITHOUT regressing L0). **Encouraging:** at clean input (L0) the manual conv is *already bit-exact* — so it CAN match native; the issue is the value-dependent edge at the onset input, which is tractable (find the variant that handles it without breaking L0).

## WALL + USER DECISION (2026-06-08): grind the FULL ex2.approx silu replica (no reroute)
Clean-input result (corrected methodology): L0 conv=0.0; fresh L45 conv=1 bf16 ULP. **No torch rounding rule satisfies both** (`tie_positive_down` fixes L45 but regresses L0 to 0.125) — native's CUDA `ex2.approx` silu is not matchable by a tie-break rule. Residual compounds to ~0.59 logits ⟹ cannot accept.
**User chose:** GRIND the **full ex2.approx replica** (replicate native's exact 2^y hardware silu bit-for-bit in OUR manual conv), NOT the `causal_conv1d_update` reroute (banned reward-hack per FR-12). Plan: read native `causal_conv1d_update` CUDA source for the exact silu op-sequence; replicate the ex2.approx algorithm (exponent/fraction split + exact polynomial + rounding) in torch; iterate OFFLINE on clean-input layers (L0 + fresh L45) via same-input manual-vs-native test until conv1d_out=0.0 on BOTH (an exact replica resolves the conflict a rounding rule can't); then ONE live full-ladder test (spine+branches+logits+gate-2). codex_fr14 directed.

## Status
GATE A is **NOT passed** and must not be bound as passing. `gateA_spine_ladder.json` final hidden/logits are still **empty** (`passed: False`) — the final-spine-logits-vs-native number (the losslessness-critical one) is not yet computed. If the divergence is real (A1), it will flip final-spine-logit argmaxes far beyond the E5 floor → the e2e bag-TV would be lossy → a genuine no-copy-GDN losslessness finding to fix at root (the mask/state-indexing leak), NOT to wave through. **Keep the 2-ULP floor separate**: that is the accepted irreducible no-copy grouping floor; THIS (0.25–1.875) is a distinct structural bug. Surfaced to the user; no self-declared pass. Fix once root cause is confirmed by the spine-only test + per-GDN-layer localization.

## Post-ex2 live ladder — 2026-06-08

Run dir: `output/fr13_ex2_live_ladder_20260608T021853Z`.

Code under test:

- server commit `42d49580`
- ex2 helper code commit `ed0390df`
- strict tree run: `TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset, B=1 eager, `GPU_UTIL=0.4`, `MAX_MODEL_LEN=65536`

The offline clean-input conv replay for L0+fresh L45 is fixed (`max_abs=0.0`),
but the live strict top-down ladder still fails:

| stage | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| layer 3 full_attention hidden | 0.0 |
| layer 7 full_attention hidden | 0.0 |
| **layer 8 linear_attention hidden** | **0.00390625** |
| final_norm_hidden | 1.65625 |
| final logits | 1.90625 |

Full-attn/branch reduction on the same run shows the forked FA2 tree path is
not the primary source: `tree_vs_fa2_branch=0.0` for 15/16 full-attn layers and
`0.00048828125` at layer 55, while the large later full-attn drift enters as
`input_hidden`. The next localization target is therefore the GDN sub-op at
layer 8 on the strict tree/native matched event.

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

## ex2 replica LIVE FAILURE at L8 + broad-test redirect (monitor, 2026-06-08)
The ex2 silu replica passed offline at L0+L45 (0.0) but the LIVE full-ladder FAILED at **GDN layer 8** (committed `7f950694`). L8 was 0.0 pre-fix; only the conv silu changed ⟹ **the replica REGRESSED L8** — it matches L0/L45 but NOT L8, so it is NOT bit-exact to native's FULL op sequence at all input values. The `ex2.approx` *instruction* is right; the surrounding sequence (bf16 cast points, the `-acc*0x3FB8AA3B` argument, `+1`, `div.full.f32`, `cvt.rn.bf16`) must match exactly. The offline 2-layer pass was necessary but insufficient.
**Redirect (no L8-by-L8 whack-a-mole):** ONE spine-only capture of conv sub-op inputs (pre_conv+state+weights) for a SPREAD of CLEAN layers (L0,L4,L8,L12,L24,L36,L44 — all clean since spine-only onset=L45); iterate the replica OFFLINE vs native until conv1d_out=0.0 for EVERY clean layer, verifying intermediates (ex2 out / +1 / div / bf16 cvt). **This is the decisive go/no-go for the "grind our kernel bit-exact" path:** if ONE replica matches ALL clean layers → fix; if not → the ex2.approx is effectively un-replicable in our kernel → re-escalate (reroute decision, which the user declined, vs reconsider).

## Spine-only spread capture + offline conv replay — 2026-06-08

Run dir: `output/fr13_conv_spread_20260608T025907Z`.

Config:
- strict spine-only tree (`TREE=[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0)]`)
- tree arm: `TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset, B=1 eager, `GPU_UTIL=0.4`
- native arm: `FLASH_ATTN`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`
- capture prefixes: GDN layers `0,4,8,12,24,36,44`
- replay tool: `scripts/fr13_conv_spread_ex2_replay.py`, artifact `conv_spread_ex2_replay.json`

Offline replay result (tree PTX-style bf16 taps -> f32 adds -> `-acc*0x3FB8AA3B` -> `exp2` -> `+1` -> div -> bf16 store, compared to captured native `conv1d_out`):

| GDN layer | clean `pre_conv` | `pre_conv` max_abs | captured tree conv vs native | PTX replay vs native |
| ---: | --- | ---: | ---: | ---: |
| 0 | yes | 0.0 | 0.0 | 0.0 |
| 4 | yes | 0.0 | 0.0 | 0.0 |
| 8 | yes | 0.0 | 0.0 | 0.0 |
| 12 | no | 0.0751953125 | 0.012451171875 | 0.012451171875 |
| 24 | no | 0.265625 | 0.046875 | 0.046875 |
| 36 | no | 0.40625 | 0.06640625 | 0.06640625 |
| 44 | no | 0.4296875 | 0.125 | 0.125 |

This invalidates the assumption that all requested spread layers remain clean under the current post-ex2 live run. L12/L24/L36/L44 are already contaminated by upstream hidden/state drift, so they cannot be used to tune or reject the conv kernel. On the clean layers that were actually clean in this run (L0/L4/L8), the conv output is already 0.0 vs native.

Sub-op diff on the same spread capture changes the root-cause target:

| layer | first diverging stage | `input_hidden` | `pre_conv` | `conv1d_out` | `h0_state_in` | downstream |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | none | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | `h0_state_in` | 0.0 | 0.0 | 0.0 | 0.0007215589284896851 | `o_proj_out` 0.003906 |
| 12 | `input_hidden` | 0.09375 | 0.075195 | 0.012451 | 0.0021439790725708008 | `o_proj_out` 0.046875 |

L8 uses the same captured state row/column geometry as clean L4 (`h0_rows=[1]`, `h0_cols=[0]`, `spec_state_indices_tensor=[[1,2,3,4,5,6]]`, same source indices), but the state contents differ: `h0_state_in` has max_abs `0.0007215589284896851` over 748,835 elements while `conv1d_out` is 0.0. Therefore this spread capture does **not** support an L8 conv/SILU mismatch. The next root-cause target is the recurrent state content/write path feeding L8 (`h0` bank row 1), not more L8-specific conv activation tuning.

Limitation: the native CUDA kernel does not expose its internal `ex2`, `+1`, div, or bf16-cvt registers in the capture. The replay reconstructs that sequence and verifies final bf16 output against native `conv1d_out`; internal register comparison is reconstruction-only, not a native-register proof.

## CONV FRONT DONE; NEXT FRONT = scan-state h0 (user: continue GDN grind, 2026-06-08)
Spread sub-op diff vindicates the conv: at L8 `input_hidden=pre_conv=conv1d_out=0.0` (ex2 conv replica is bit-exact at the conv level), and the captured-conv == ex2-conv at contaminated layers (input-driven). **The conv silu grind SUCCEEDED.** The live L8 first-divergence is a DIFFERENT sub-op: **`h0_state_in` (GDN recurrent scan state) = 0.0007** — a separate value-dependent ~1-ULP. So the GDN tree-kernel has multiple bit-exact fronts vs native FLA (conv ✓, scan-state next, possibly gate/o_proj). FA2 attention is byte-exact; all remaining drift is GDN.
**User decision (2026-06-08): CONTINUE the GDN grind** (over reroute/re-evaluate). codex directed to localize h0_state_in: (a) ex2 conv during PREFILL diverging at some prompt-token values (conv not bit-exact at ALL values → contaminates prefill state) vs (b) the SCAN kernel (`_tree_gdn_kernel`) diverging from native FLA. Offline-iterate to 0.0 across all values, then one live full-ladder test. Fronts are FINITE (conv/scan/gate/o_proj/state) → convergent-in-principle.
