# FR13_VERIFY_DECISIVE_BIND — per-path-oracle + substitution + tree-vs-native flip-rate (DRAFT, not committed)

Date 2026-06-13 UTC. Decisive test of the deep-row verify argmax flip (the prior
`FR13_NODE7_LADDER_BIND` localization HELD=FALSE on causation + reference confound).
Tree boot `fr13-forked-fa2-tree` cat9 / TREE_ATTN / num_spec=9, ENFORCE_EAGER=1, C0
(all FIX default ON), forked FA2. Native E5 = `fr10-native-e5` num_spec=5 linear MTP,
FLASH_ATTN, eager. Recurrent oracle = `fr10-nospec-recur` (`FR12_NO_SPECULATIVE_CONFIG=1`,
speculative_config=None = pure target decode = non-MTP ground truth), FLASH_ATTN, eager.
Probes = `output/fr13_acceptance_ladder/prompts_swe4.json` (4 pinned), greedy seed 1313.
Artifacts `output/fr13_verify_decisive/`. Deep-row event: p2 served pos21 = 1970 (` code`)
vs clean 3425 (` files`), margin 0.5; this boot's flip verify forward = num_tokens=10
forward, forward_row 6 (position 849, input = served[20]=9468), reproduced every boot.

## WITHIN-BOOT DETERMINISM (class 8): PASS everywhere
- tree det_gate `within_boot_det = [T,T,T,T]` (2 boots); tree capture [T,T,T,T];
  native E5 capture [T,T,T,T]; recurrent oracle reproduced served[:21] byte-identical.
- p2 pos21 = 1970 reproduced on every tree boot; the substitution arms are deterministic.

## Q1 — PER-PATH ORACLE (op-divergence vs input-drift): the L0 "carrier" was INPUT-DRIFT
Tree deep-row (forward_row 6, pos 849) per-layer hidden vs TWO references; input_hidden
byte-exact 0.0 vs both (same input token 9468):
| layer | vs CHUNKED prefill (node7-ladder ref) | vs RECURRENT non-MTP ORACLE (true per-path) |
|---|---:|---:|
| **L0 GDN (first nonzero)** | **0.0078** | **0.000854 (~1 bf16 ULP)** |
| L1 GDN | 0.0029 | 0.0017 |
| L58 GDN | 1.8125 | 0.1875 |
| L59 full_attn | 0.795 (cos .969) | 1.309 (cos .969) |
| L63 full_attn | 35.75 (cos .950) | 33.75 (cos .945) |
| final_norm | 3.375 (cos .988) | 3.125 (cos .986) |
- **The node7-ladder's L0 GDN 0.0078 was ~89% INPUT-DRIFT**: vs the TRUE recurrent
  (non-MTP) per-path oracle it drops ~9× to **0.000854 = ~1 bf16 ULP** — exactly the
  FR13_GATEA per-op floor (conv ~0.000977 / scan-state ~0.0007). The chunked prefill
  (one length-850 forward) builds GDN state via chunked-scan; the tree builds it
  recurrently across decode events — the ~6e-5-class chunk-vs-recurrent state seam was
  most of the 0.0078. **L0 GDN is NOT a fixable op** (it's at the bf16-ULP floor vs the
  correct reference). STree Eq.4-6 / SpecInfer Def 4.1: the recurrent oracle is the
  definitional per-path reference; the chunked prefill is a confounded approximation.
- **BUT the divergence is REAL vs the oracle**: it still accumulates diffusely (L58 ~0.19)
  and explodes at deep full-attn (L59 1.31, L63 33.75, final_norm 3.125 cos 0.986), and
  the argmax STILL flips (recurrent oracle = 3425; tree = 1970). The per-layer magnitude
  profile is SIMILAR vs both references past L2 — the flip is diffuse fp-accumulation, not
  a single seam. So: **L0 fix is illusory (input-drift), the real divergence is diffuse.**

## Q2 — SUBSTITUTION (causal carrier): DIFFUSE L0-L58, NOT the deep full-attn
Flag-gated in-process hook `FR13_HIDDEN_SUBSTITUTE` (added to
`scripts/fr10_phase4_patch_vllm_tree_gdn.py`, default OFF; overwrites one forward-row's
residual-stream {hidden,residual} after a target layer, then real layers + norm + lm_head
finish). Oracle = chunked-prefill row-849 per-layer hidden (the per-path token reference).
Read = the spliced forward's lm-head argmax at forward_row 6.
| arm | splice | forward_row6 argmax | reverts? | meaning |
|---|---|---|---|---|
| baseline | none | **1970** (flip), margin 0.5 | — | the flip |
| **A** | oracle @ **L58** | **3425** (clean), margin 0.25 | **YES** | tree L59-L63 on a CLEAN L58 give the CLEAN answer ⇒ **L0-L58 accumulation IS the carrier** |
| C | oracle @ **L0 only** | **1970** (flip), margin 0.5 | **NO** | zeroing the L0 seed alone does NOT fix it ⇒ drift **re-accumulates** L1-L58, **DIFFUSE** |
| B (control) | oracle @ **L63** | **3425** (clean), margin 1.875 | YES | mechanism validated (oracle output → clean argmax) |
- **REFUTES the node7-ladder claim that the L59/L63 deep full-attn is the flip CARRIER.**
  Feeding the oracle's L58 into the tree's real L59-L63 → CLEAN argmax. The deep full-attn
  is an AMPLIFIER of the L0-L58 seed, not an independent carrier.
- **Carrier = the diffuse GDN fp-accumulation across L0-L58** (no single layer; L0 alone
  insufficient). Not localizable to one op/wiring seam.

## Q3 — DELIVERABLE FLIP-RATE: the tree is WORSE than native baseline (NOT within floor)
Clear-margin flips (served_id != clean_argmax_id AND deviation > 1.0 nat) over the 4 pinned
probes, BOTH arms classified vs the SAME non-MTP recurrent oracle (the lossless-def "non-MTP
ground truth" reference), teacher-forced max_tokens=1 on the no-spec server.
| arm | total clear-margin flips | per-prompt [p0,p1,p2,p3] | within-boot det |
|---|---:|---|---|
| **TREE** (cat9 TREE_ATTN num_spec=9) | **22** | [5, 7, 4, 6] | [T,T,T,T] |
| **NATIVE E5** (num_spec=5 MTP-5 FLASH_ATTN) | **3** | [0, 1, 1, 1] | [T,T,T,T] |
- **Tree is ~7× WORSE than native E5** vs the identical per-path/non-MTP oracle.
- Native E5 is NOT zero (3 flips) — even MTP-5 diverges from non-MTP at a few high-entropy
  template boundaries (p1 pos94 / p3 pos68 `Let` vs `` ``` `` code-fence; p2 pos33 quote
  style). This IS the genuine two-tier floor — but it is SMALL (3) for native. The tree's
  22 flips (all at structural/template boundaries: code fences, newlines, `<tool_call>`,
  `<|im_end|>`, bash/python/cd command choices; incl the banked p2 ` code` vs ` files`)
  far EXCEED it ⇒ the tree's deep-row divergence is **not within the two-tier floor**; it is
  a real excess loss attributable to the tree verify forward numerics.

## VERDICT: IRREDUCIBLE_DIFFUSE_FLOOR
Synthesis: (Q1) the L0-GDN "first divergence" the prior ladder named is **input-drift**
(bf16-ULP vs the true recurrent oracle) — NOT a fixable op; (Q2) the flip carrier is the
**diffuse L0-L58 GDN fp-accumulation** (L0-alone splice does not revert; L58 splice does),
amplified by — not carried by — deep full-attn; (Q3) the tree flips **7× more** than native
E5 vs the same non-MTP ground truth ⇒ **worse than baseline, not the two-tier floor**.
There is NO single op/wiring seam to fix (the FA2-fork attn is already deployed and
byte-exact ~0.0039; conv/scan/gate/o_proj are at the bf16-ULP floor per FR13_GATEA). The
excess is diffuse fp-non-associativity / fp8-bucket / reduction-order drift compounding over
64 layers in the tree's TREE_ATTN + fp8-GEMM verify forward, that the FLASH_ATTN+MTP-5
native path does not incur to the same degree. Removing it requires **batch-invariant /
non-fp8 numerics** on the tree verify forward (the FR-13 "Method A" direction) at a known
SPEED COST (BI GEMMs are slow, OFF for speed) — i.e. a **user pass/fail decision** (lossless
within native floor at a speed price, vs ship-with-floor). This is NOT a self-declarable
pass/fail; bring the table to the user.

### named fix / floor
- **No single named op fix exists** (L0/conv/scan/gate/o_proj at bf16-ULP floor; FA2-fork
  attn deployed+byte-exact). The only lossless route is **batch-invariant / non-fp8 tree
  verify numerics** (FR-13 Method A: `VLLM_BATCH_INVARIANT` + `FR13_BI_TREE_ATTN`, double-
  gated, inert by default) — pins bf16 lm_head/logits GEMM, softmax, RMSNorm, reductions —
  at a speed cost. Speed-vs-lossless is the user decision the verdict surfaces.
- The deliverable gate the gold-SWE check missed: a clear-margin argmax flip at a structural
  boundary is below the hidden-state within-floor max_abs gate but ABOVE the token bar; the
  tree exceeds native here.

## METHOD / FIDELITY notes
- Recurrent oracle = no-spec server (speculative_config=None) → every decode is recurrent
  non-MTP target; the pos-849 decode hidden (num_tokens=1, row 0) is the true per-path
  recurrent reference. non_mtp mode on the TREE server is DEAD (pre-existing stock
  `propose_tree` `EagleProposer.positions` crash, FR13_B1_FIX*_GATE binds) → the recurrent
  oracle MUST come from a no-spec boot; the chunked prefill on the tree server is a
  confounded (chunk-vs-recurrent) approximation, which is exactly the Q1 confound.
- Substitution = in-process residual-stream overwrite (NOT a reroute/copy of the deploy path
  — a flag-gated diagnostic; control arm B validates it propagates oracle→clean correctly).
- Q3 classify reference = no-spec max_tokens=1 teacher-force per served position = non-MTP
  ground truth; chunk-vs-recurrent ~6e-5 is far below the 1-nat clear-margin threshold, so
  it does not affect flip counts; tree and native classified identically.

## Artifacts (`output/fr13_verify_decisive/`)
- `q1_recur_vs_chunked.json`, `q1_summary.json` — Q1 per-layer ladder vs both references.
- `sub_summary.json`, `sub_arm*.json`, `sub_result_arm*.json` — Q2 substitution arms.
- `q3_tree_classify.json` (22), `q3_native_classify.json` (3) — Q3 flip counts + detail.
- `q3_tree_capture.json`, `q3_native_capture.json` — served streams (within-boot det).
- `nospec_recur_p2.json` — recurrent oracle reproduces served[:21], pos21 argmax 3425.
- `cap/` (tree flip forward call6 + oracle), `cap_boot4/recur_oracle_pos849.pt`, `nospec/cap/`.
- Hook: `FR13_HIDDEN_SUBSTITUTE` in `scripts/fr10_phase4_patch_vllm_tree_gdn.py` +
  `-e FR13_HIDDEN_SUBSTITUTE` in `scripts/fr13_launch_forked_fa2_tree_server.sh` (default OFF).
