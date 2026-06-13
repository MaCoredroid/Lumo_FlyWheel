# FR13 Gold-Gate Probe MARGIN bind — the open red-team item, resolved

Date 2026-06-13 UTC. Resolves the ONE open red-team item from
`FR13_B1_SWE_GOLD_BIND.md`: are the 128-tok rider-probe greedy forks (tree
vs native E5) NEAR-TIE (lossless backend tip) or CLEAR-MARGIN (real loss)?

**VERDICT: CLEAR-MARGIN present — a REAL, reproducible serving-path divergence
localized in the TREE spec-decode commit path (NOT a backend near-tie, NOT a
logprob artifact).** The tree's served greedy stream serves a token that the
tree's OWN clean greedy target distribution clearly rejects (up to 5+ logprob).

## Method (GPU, serialized; both arms B=1 greedy, seed 1313, BI=0, metrics=0)
- Tree = `fr13_launch_forked_fa2_tree_server.sh` (cat9, TREE_ATTN, num_spec=9,
  all FIX default ON). Engagement: **tok/draft == 9.0** (254 drafts / 2286 draft tok).
- Native E5 = `fr10_launch_speed_server.sh` with
  `SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
  ATTENTION_BACKEND=FLASH_ATTN FR10_DECODE_MODE_DEFAULT=naive_mtp` — the
  launcher DEFAULT TREE is the cat9 9-tree (would give num_spec=9, WRONG);
  the E5 SPEC_CONFIG override is mandatory. Engagement: **tok/draft == 5.0**
  (258 drafts / 1290 draft tok).
- 4 pinned prompts `output/fr13_acceptance_ladder/prompts_swe4.json`, raw
  `/v1/completions`, max_tokens=128, temp 0, top_p 1.0, top-20 logprobs,
  return_token_ids, `vllm_xargs.fr10_decode_mode` matched per arm.
- Within-boot determinism: rep1==rep2 served ids on ALL 4 prompts, BOTH arms.
- Fork cross-check: this fresh cross-boot pair forks at [17,11,21,68] vs banked
  task1 greedy [17,15,21,61]; prompt-0 early fork (17) and prompt-2 (21)
  reproduce EXACTLY; 1/3 differ within the cross-boot floor; mechanism identical.

## Key instrument correction (do NOT trust streaming top_logprobs on the tree arm)
The TREE arm's streamed `top_logprobs` array is MISALIGNED at spec-decode accept
positions (~12/128 positions show served-token != reported-argmax). This is a
vLLM logprob-REPORTING quirk, NOT a model error: teacher-forcing the tree on the
byte-identical prefix at internal positions reproduces the served token as the
clean argmax (pos 9/10/11/15/16/18 all MATCH). Native arm has ZERO such
misalignments. **All margins below come from CLEAN max_tokens=1 teacher-forced
distributions on the byte-identical shared prefix** (native teacher-force
reproduces native's streamed token at every fork = method validated faithful).

## Per-fork classification (CLEAN teacher-forced)
Threshold: NEAR-TIE iff served T_t is within **1.0 logprob** of the tree's OWN
clean argmax (`deviation_gap = lp(tree argmax) - lp(served T_t) <= 1.0`); a 1.0
logprob gap = served token >=~37% as probable as the argmax = genuine near-tie
band where fp-noise can flip the choice. Beyond 1.0 = the tree's own target
clearly prefers another token = CLEAR-MARGIN.

| p | fork | T_t (tree served) | T_n (native served) | tree clean argmax | nat clean argmax | backends agree argmax | dev_gap (tree argmax−served) | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 17 | ` and` | ` structure` | ` structure` (−0.52) | ` structure` (−0.47) | YES | 0.375 | NEAR-TIE |
| 1 | 11 | ` workspace` | ` repository` | ` workspace` (−0.64) | ` repository` (−0.64) | no (true tie) | 0.000 | NEAR-TIE |
| 2 | 21 | ` code` | ` files` | ` files` (−0.12) | ` files` (−0.16) | YES | 2.125 | CLEAR-MARGIN |
| 3 | 68 | `Let` | ` ``` ` | ` ``` ` (−0.011) | ` ``` ` (−0.001) | YES | 5.125 | CLEAR-MARGIN |

Cross-arm margins (clean): p0 margin_native(Tn−Tt)=0.50; p1 both 0.125; p2
margin_native=1.75; p3 margin_native=6.75 (native is ~99.9% on ` ``` `, tree
served `Let` at ~0.6%).

## What this means
- p1 is a textbook lossless backend NEAR-TIE (two backends tip ` workspace` vs
  ` repository`, each ranks the other rank-2 at 0.125 — fp-noise band). p0 is a
  near-tie tip (0.375).
- **p2, p3 are NOT backend disagreements.** Both backends AGREE on the clean
  argmax (` files`, ` ``` `). The tree's spec-decode SERVED stream serves a
  rank-2 token its OWN clean greedy clearly rejects. Native E5's served stream
  is greedy-EXACT (0 deviations); the tree's is not.

## Localization (for follow-up fix — NOT fixed here)
Full per-position teacher-force of prompt 0 (82 tok): the tree stream deviates
from its clean greedy argmax at **5/82 positions**, gaps {0.375, 5.688, 3.875,
and two with served-token absent from clean top-5 (gap >> 5)}. Deviations
cluster at structural/template boundaries (code-fence `bash`/`cmd`/`|` at pos
34/36/37; `<|im_end|>` stop region at pos 81). A 5+ logprob swing that flips a
99%-confident argmax to a <1% token is NOT a fp/numeric seam between two forward
passes — it is a logic/commit-path defect in the TREE spec-decode SERVE path
(the committed token != the verified-greedy token at accept boundaries).
- Defect class: tree spec-decode COMMITTER serves a non-argmax (draft/bonus or
  off-by-one row) token at a subset of accept boundaries. Candidate seams:
  tree-sample-row (FIX-A) bonus/self handling at accept-run ends; the eager-pack
  replay row mapping; conv-fusion committed-path row selection at accept>1.
- Decisive next test (do NOT do here): per-accept-boundary committer-row gate —
  assert the committed token id == argmax(verify-forward logits at that row) on
  the tree server, in-process, over the pinned prompts; first divergent boundary
  = the row-mapping bug.

## Engagement / health
- logprobs ACTUALLY captured top-20/position both arms (top_lens == served_lens).
- drafts: tree tok/draft==9.0, native tok/draft==5.0 (E5 asserted).
- within-boot determinism: rep1==rep2 served ids, BOTH arms, all 4 prompts;
  teacher-force within_boot_det==True all forks both arms.

## Artifacts (output/fr13_gold_margin/)
- tree_capture.json / native_capture.json — streamed served ids + top-20.
- tree_teacher_force.json / native_teacher_force.json — CLEAN per-fork dists.
- margin_reduce_clean.json — the classification table above.
- margin_reduce.json — initial (streaming-logprob) reduce, SUPERSEDED by clean.

## Bearing on the gold gate
The `FR13_B1_SWE_GOLD_BIND.md` BINDING verdict (SWE served-stream within-floor,
both tasks) is UNCHANGED — that gate is about the agentic SWE stream and held.
This margin probe resolves the OPEN rider-probe item: it is NOT a benign backend
near-tie everywhere. Two forks (p0,p1) are near-tie/lossless; two (p2,p3) are a
real tree-committer serving deviation. greedy-lossless-within-backend-floor is
**NOT** confirmed for the rider probes; a localized committer-row divergence
remains for a follow-up fix. NO fix attempted here (per task).
