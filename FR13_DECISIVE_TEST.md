# FR13 DECISIVE TEST (user-approved 2026-06-09) — settle gate-artifact-lossless vs real-kernel-seam

## Verified verdict (TWO independent workflows wxqn40d17 + whivm67j3, BOTH adversarial-verify-passed)
- **Committer is LOSSLESS-BY-DESIGN at greedy.** `greedy_tree_lcp_max` (fr10_phase4_patch_vllm_tree_gdn.py:3421-3580, :3496) is temp-0 SpecInfer VerifyGreedy: at temp0 only ONE child matches each node's target argmax, so the longest-LCP path = the UNIQUE verify-argmax chain = native greedy (the drafter "spine"/path0 can deviate early; the committed branch IS native greedy). whivm67j3's "lossy committer" claim was REFUTED by its own verify — it rested on the pos18 native self-noise artifact (10278 vs 52589 = native b1-vs-b4, NOT a tree loss). NO committer bug.
- **FR7 co-resident contamination ENGINEERED OUT** of the live kernel (h_cache + strict_mask, zero cross-sibling; fr10_gdn_tree_kernel.py:277-353). The OLD single-shared-state mechanism is gone. (The conv-state-remap WAS a real contamination seam pre-fix: conv1d_out 18.375 → 0.0 via 3a9039cc.)
- **287 = FEW-ROOTS-CASCADING (~8), inflated by native self-noise** (85 of 368 are native b1-vs-b4; native-vs-native only 4/8 exact). First TRUE outside-self-noise losses: pid0 pos46, pid1 pos28, pid2 pos35, pid3 pos10, pid4/5/6 pos1 (token 846, degenerate early-stop), pid7 pos15.
- **NOT a pass either (current e2e FAILS):** temp0.6 same8 bag-TV 0.250 (4.2x over 0.0593), accept/event 2.05 < native 3.05. But on a build with OPEN seams + spine-equality (not superset) comparator → cannot bind a no-go.
- **OPEN, finite path.** NOT no-go, NOT pass.

## SPEED reality (note before building)
At GREEDY a lossless tree = the unique native argmax chain → accept/event capped at native MTP-5 **3.076** (branches give NO lossless greedy speedup). The no-copy tree's SUPERSET speedup is a **temp>0** phenomenon (rejection-sampling accepts more candidates losslessly). So: GREEDY test = LOSSLESS proof; TEMP-0.6 test = SPEED (superset accept/event ≥ 3.076).

## The decisive test (execute in stages, commit each)
**Seam 1 — uniform conv write-back (replace the layer≥4 BAND-AID).** The conv fix 3a9039cc is gated `_fr10_use_rolled_tail_prior = (layer_idx >= 4)`. Replace with the uniform write-back root fix (store native's exact [history…,accepted…] ordering so ALL layers read native's tail convention). GATE: substate gdn_l0_subkernel conv1d_out row0 = 0.0 at ALL calls AND across GDN layers, no regression (call0/1 stay 0.0), spine+branch.

**Seam 2 — forked-FA2 PREFILL → native byte-exact.** The forked FA2 prefill diverges at full-attn L7 (gate-2-prefill gap, WIRING). Route TreeAttentionImpl prefill through flash_attn_varlen no-bias (mirror the decode patch). EXTEND gate-2 (regular-decode==pristine) TO PREFILL: forked-FA2 no-bias prefill == pristine stock FA2 prefill = 0.0 every layer.

**Then the decisive measurement (B=4, CUDA-graph-captured, deployed regime):**
1. Fresh PAIRED build (both seams closed). Run native self-noise baseline (native b1 + native b4) to get the self-noise mask.
2. Run the native-on-branch-path oracle (scripts/fr13_branch_token_oracle.py, real /v1/completions) for ALL 8 first-losses (pid0 pos46 … pid7 pos15) — NOT just event4/pid0.
3. e2e same8 vs E5, compare the committed token to BOTH (i) native-on-the-selected-branch-path (SUPERSET gate) and (ii) paired native greedy (spine gate), self-noise-corrected.

## DECISIVE OUTCOMES (binds the close/pass-fail)
- **LOSSLESS+SUPERSET CONFIRMED** (synthesis vindicated): ALL 8 served flips == native-on-their-branch-path AND spine per-depth argmax matches native AND temp0.6 accept/event ≥ 3.076 + bag-TV within self-noise floor.
- **REAL KERNEL SEAM = LOSSY** (no-go-leaning): ANY served flip differs from BOTH native-spine AND native-on-its-branch-path while that node's INPUT hidden is bit-exact (0.0) — i.e. a downstream sub-op diverges on clean input (like spine-only L45 / the pre-fix call2 conv-slot). That is a live verify/kernel seam fed clean input → real loss.
- Isolate the pos-1 structural flips (pid 1/2/3/5/6, native `<think>`/control vs tree prose) SEPARATELY — may be a distinct earlier seam from the mid-sequence branch-selection flips.

## Discipline
ONE GPU, recover between arms, commit+bind each seam + the final test to FR13_LADDER_LOG.md. NO copy/dense/splice. Both strict gates (verify-path + regular-decode==pristine incl prefill). The real gate = B=4 + CUDA-captured + per-request, superset comparator, self-noise-corrected — NOT B1/eager/16tok/spine-equality.
