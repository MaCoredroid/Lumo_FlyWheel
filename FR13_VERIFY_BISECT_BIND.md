# FR13_VERIFY_BISECT_BIND — deep-row verify-gap bisect across speed-fix combos (DRAFT, not committed)

Date 2026-06-13 UTC. Tree boot `fr13-forked-fa2-tree` cat9 / TREE_ATTN / num_spec=9,
**ENFORCE_EAGER=1**, FORKED FA2 .so, all the canonical FR13 verify-path flags ON
(`FR13_CONV_COMMITTED_PATH=1 FR13_REPLAY_ROUTE=1 FR11/FR12 bf16 taps`). Probes =
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 pinned), greedy seed 1313 top_p 1.0
max_tokens 128. Instrument = `scripts/fr13_verify_bisect_probe.py` (per-served-position
CLEAN max_tokens=1 teacher-force on the byte-identical served prefix, on the SAME tree
server; flag CLEAR-MARGIN flip iff served_id != clean_argmax_id AND deviation > 1.0 nat).
Artifacts `output/fr13_verify_bisect/c{0..4}_{capture,classify}.json`.

## VERDICT: **DEEP-GAP.** Not a regression from any speed fix.
The clear-margin deep-row verify flips are the PRE-EXISTING deep TREE_ATTN/fp8/GDN-tree-scan
verify-forward numerics gap (the gap the FR13 per-layer-0.0 lineage drove to max_abs
~0.00195 = within-floor; a clear-margin ARGMAX FLIP at the deepest spine-tail row exceeds
that within-floor band). FIX-2 (eager-pack) and FIX-3 (conv-fusion) are EXONERATED.

## BISECT TABLE (per-combo clear-margin flip counts, threshold 1.0 nat)
| combo | flags changed vs default-ON | total flips | per-prompt [p0,p1,p2,p3] | p2 `code`vs`files` deep flip | p3 codefence deep flip | within-boot det | accepted/drafts |
|---|---|---:|---|---|---|---|---|
| C0 | none (all FIX ON) | **22** | [3, 8, 4, 7] | **PRESENT** | **PRESENT** | PASS | 708 / 226 |
| C1 | `FR13_TREE_CONV_FUSED=0` | **22** | [3, 8, 4, 7] | PRESENT | PRESENT | PASS | 708 / 226 |
| C2 | `FR13_EAGER_PACK=0` | **22** | [3, 8, 4, 7] | PRESENT | PRESENT | PASS | 708 / 226 |
| C3 | both speed-kernel OFF (`EAGER_PACK=0 TREE_CONV_FUSED=0`) | **22** | [3, 8, 4, 7] | PRESENT | PRESENT | PASS | 708 / 226 |
| C4 | + `FR13_TREE_SAMPLE_ROW=0` (FIX-A off too) | **21** | [3, 11, 4, 3] | PRESENT | (moved†) | PASS | 714 / 322 |

C0 = C1 = C2 = C3 are **byte-identical**: identical 22-flip set, identical positions,
identical deviations, identical spec-decode metrics (708 accepted / 2034 draft-tok / 226
drafts). Toggling conv-fusion and/or eager-pack changes NOTHING in the served stream or the
verify-forward gap — confirming both fixes are semantics/bit-exact-preserving by
construction (their byte-A/B was correct; they do not touch the verify forward). No
eager-pack↔conv-fusion dependency forced both off — C2 (eager-pack OFF, conv-fusion ON) and
C1 (conv-fusion OFF, eager-pack ON) both booted independently (flags read separately at
module scope; FIX-3 preconditions = CONV_COMMITTED_PATH + REPLAY_ROUTE, NOT eager-pack).

†C4 changes the DRAFTER's sampling row (FIX-A1), so the drafted candidates → accept pattern
→ served stream all change (metrics: 714 accepted / 322 drafts). The flip POSITIONS move
(exactly the class-11/GB10 warning: a flag that changes the served stream moves the flip),
but the flip CLASS persists: p2 pos21 ` code` vs ` files` deep-row flip STILL PRESENT; the
code-fence/template-boundary `Let`-vs-`\`\`\`` flip just relocates (C4: p0 pos67 `Let` vs
`\`\`\`` dev 8.375; p1 pos124 `<tool_call>` vs `\`\`\``; p3 still has code-fence flips). 21
clear-margin flips persist with EVERY speed fix off → the gap is not in any fix.

## THE FIRST COMBO WHERE FLIPS DROP TO ZERO: there is NONE.
The bisect decision rule (first combo where clear-margin deep-row flips → 0 names the
regressing fix) yields NO such combo. Flips persist at full strength through C3 (both
speed-kernel fixes off) and persist (count 21, same class) through C4 (all speed + drafter
FIX-A off). ⇒ REGRESSION is REFUTED; DEEP-GAP is the verdict.

## NAMED FIX TARGET: the tree VERIFY-FORWARD numerics at the deepest tree row.
Not a speed fix; not the committer (exonerated 0/944 ch1, FR13_COMMIT_ARGMAX_BIND.md). The
target is the tree verify forward (TREE_ATTN deep-row ancestry-mask / online-softmax + fp8
GEMM + conv committed-prior window + GDN recurrent scan) at flat row `start + node 7` (the
depth-4/5 spine tail of `best_path=[0,1,3,5,7]`), which computes argmax-flipping logits vs a
clean single forward at structural/template boundaries (code fences, `<tool_call>`,
`<|im_end|>`). This is class-9 (verify-forward losslessness) needing a per-layer ladder at
node 7, NOT a flag flip. Reconciles with FR13 lineage: per-layer max_abs was driven to
~0.00195 (within-floor) but a clear-margin argmax flip at the deepest row exceeds that — the
within-floor max_abs gate was the coarse check that MISSED this, same way the agentic
within-floor SWE check did.

## WITHIN-BOOT DETERMINISM (class 8): PASS, all 5 combos
`within_boot_det_rep1_eq_rep2 = [true,true,true,true]` on every combo (served streams
byte-identical rep1==rep2 on all 4 prompts). GB10 within-boot greedy determinism holds; the
clear-margin flips are reproducible within-boot.

## ENGAGEMENT (class 9/12): PASS
- Flags in container env verified per combo via `docker exec ... env | grep FR13_`:
  C0[1,1,1,1] C1[conv=0] C2[pack=0] C3[pack=0,conv=0] C4[+sample=0] (drafter-single-logits
  ON throughout). Boot logs: `enforce_eager: True`, `TREE_ATTN backend`,
  `num_speculative_tokens: 9`, `Application startup complete`.
- Clean reference validated by the close workflow (wf_3c6f5c0a): native E5 teacher-forced
  argmax == served 4/4 (asymmetry proof); the tree server's own max_tokens=1 forward gives
  the clean argmax (e.g. `\`\`\`` code-fence at the p3 boundary). C0 reproduces the 2 banked
  clear-margin flips (p2 pos21 ` code` vs ` files`; p3 ` Let` vs `\`\`\`` code-fence) =
  non-vacuous control. The probe sweeps ALL served positions (not just the first
  tree-vs-native fork), so it surfaces the full clear-margin set (22) at threshold 1.0 nat —
  a strict superset of the close workflow's 2 first-fork flips.
- Counters labeled raw (class 12): spec_decode accepted/draft/drafts per combo in the table.

## FIRST-DIVERGENT-LAYER: NOT yet laddered (deferred follow-up; GPU-serialized budget).
This run spent 5 serialized boots (each ~5-6 min cold boot + per-position probe sweep) on
the bisect. The per-layer ladder is the documented decisive next step but needs a fresh
capture boot. Plan below.

### LADDER PLAN (top-down per-layer hidden/logit capture at flat row `start+node 7`)
Goal: first nonzero layer at the node-7 row for the p2/p3 flip contexts vs a clean
teacher-forced forward = the root sub-op.
1. Boot ONE eager tree server (C0 config, all FIX ON — they don't move the gap, keep
   deployed config) with the FR10/FR13 layer-capture envs aimed at the node-7 row:
   `FR10_LAYER_HIDDEN_CAPTURE`, `FR10_LAYER_HIDDEN_CAPTURE_ROWS=<flat start+7>`,
   `FR13_FINAL_LOGIT_CAPTURE` + `_ROWS`, plus the per-backend op captures already wired in
   the launcher: `FR13_TREE_ATTN_OP_CAPTURE` / `FR13_FLASH_ATTN_OP_CAPTURE`
   (layer prefix `language_model.model.layers.{0..N}.self_attn` for the 16 full-attn layers),
   `FR12_SUBKERNEL_CAPTURE` (GDN scan/gate/o_proj per GDN layer),
   `FR13_PREFILL_GDN_CAPTURE` / conv committed-prior taps.
2. Drive the tree server to the exact step that serves p2 pos21 and p3's code-fence
   boundary (replay the served prefix as the request; the node-7 row is the deepest spine
   tail of best_path [0,1,3,5,7] at those accept events). Capture the per-layer hidden at
   that flat row.
3. Clean reference: a max_tokens=1 teacher-force on the byte-identical served prefix
   (the same instrument as this bisect) with the SAME capture envs, comparing layer-by-layer
   hidden at the corresponding row.
4. First layer whose node-7-row hidden diverges beyond bf16 noise = root. Candidate ranking
   from FR13 lineage + this gap's concentration at structural boundaries:
   (a) **TREE_ATTN deep-row attention** (ancestry-mask accumulation / online-softmax rescale
       at the deepest path row) vs FLASH_ATTN clean — the prime suspect (full-attn layers;
       FR11 flagged "not tree-exact in fp8"; node-7 is the most-accumulated attention row).
   (b) **conv committed-prior window** feeding the GDN scan at the deep node (wrong prior
       column at num_accepted>1 → corrupt deep-tail hidden; cf FR13_conv_priorwindow_root).
   (c) **GDN recurrent-scan accumulation depth** (row-7 state most-accumulated).
   (d) **fp8 GEMM bucket crossing** at the deep row (within-floor per-stage but argmax-
       flipping after lm-head).
   Per [[feedback_top_down_per_layer_lossless_gate]] + [[feedback_math_correct_vs_bitexact]]:
   run the input→layer0→…→logits ladder at the node-7 row; first nonzero layer is the root;
   then locate the exact divergent op (alignable→0.0 = wiring/numerics fix, or real
   algorithmic backend diff → BUILD OUR kernel + plug). Do NOT reward-hack/splice.

## Artifacts (`output/fr13_verify_bisect/`)
- `c{0..4}_capture.json` — served streams rep1+rep2 per combo (within-boot det source).
- `c{0..4}_classify.json` — per-position clear-margin flip records + counts + known-flip tags.
- `logs/c{0..4}_{boot,probe}.log`, `logs/c{0..4}_serverlogs/` — boot + probe logs.
- Instrument: `scripts/fr13_verify_bisect_probe.py` (bisect = capture + per-position classify).
```

## RECONCILES WITH PRIOR FR13 BINDS
- FR13_COMMIT_ARGMAX_BIND.md (committer exonerated, CHANNEL 2 verify gap): this bisect
  shows that CHANNEL-2 gap is INVARIANT to the speed fixes ⇒ it is the deep verify-forward
  numerics, not introduced by FIX-2/FIX-3. The named seam there (deep node-7 verify forward)
  is confirmed as the fix target.
- FR13_B1_SWE_GOLD_BIND.md (B=1 did not graduate; clear-margin flips at p2/p3): C0
  reproduces those exact 2 banked flips as a control; the gap is pre-existing, so the fix is
  the verify-forward numerics ladder at node 7, not any flag.
