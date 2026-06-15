# FR13 — CARRIER RE-OPEN: the 23-vs-3 gap is a TRAJECTORY-FORK-DOMINATED hybrid, not a single per-forward kernel seam

Date 2026-06-15. CPU-only, READ-ONLY re-analysis of the banked recurrent-oracle flip data
(`output/fr13_scan_align_rerun/logs/{off,recompute,native}_recur_flips.json`, `probe_us_{off,recompute}.json`)
after the GDN SCAN state-feed was ruled out e2e (FR13_SCAN_NOT_E2E_CARRIER_BIND.md, run w7wr68z06). No GPU.
Oracle = `fr13_recurrent_decode_oracle` single-step `_forward_core_decode_non_spec`, teacher-forced on the
served prefix (forces served[i], advances conv/ssm state) — the SAME deployment-correct recurrent oracle on all
3 arms. Binding metric = per-token clear-margin (deviation_nat ≥ 1.0) argmax-vs-oracle flip, per
[[reference_scalar_metric_per_token_blindspot]]. De-cascade discipline = FR13_PLUS2_DECASCADE (identical rule
all arms). Playbook rows quoted: #12 (measurement traps / per-pos counters / non-like-for-like trajectories),
#9 (vacuous — already cleared by the bind's triple non-vacuity proof), #10 (codegen — n/a here, no kernel A/B).

## Headline numbers (raw, de-cascaded, per-token-rate)
| arm | raw clear | per-prompt | positions | rate/1000 | de-cascaded indep (gap≤2) | indep rate/1000 |
|---|---|---|---|---|---|---|
| **native-E5 (BAR)** | **3** | [0,0,2,1] | 512 | 5.9 | **3** | 5.9 |
| cat9 OFF (deployed) | 23 | [5,4,5,9] | 435 | 52.9 | **18** | 41.4 |
| cat9 RECOMPUTE | 32 | [10,9,7,6] | 512 | 62.5 | 23 | 44.9 |

De-cascading collapses 23→18 (OFF) and 32→23 (recompute) but does NOT close the gap to native 3. The
remaining ~7x per-token excess is a REAL per-forward divergence — NOT pure cascade inflation. native-E5=3 is the
existence proof it is not irreducible.

---

## STEP 1 — TRAJECTORY-FORK vs PER-FORWARD-SPREAD: it is a HYBRID, fork-dominated

### Flip structure (off_recur_flips.json)
- **First-flip position per prompt** (OFF): p0=35, p1=24, p2=21, p3=61. All flips begin LATE (after the prompt
  is well underway), at format/codefence/tool-call boundaries — not a single boot-time fork.
- **Re-convergence**: EVERY OFF cluster re-converges to oracle dev≈0 within 1–2 positions (0/18 sustained
  forks). BUT this is partly BY CONSTRUCTION — the oracle is teacher-forced on the served prefix, so the
  SEQUENTIAL cascade is de-cascaded automatically. Re-convergence-in-1 therefore does NOT by itself prove
  per-forward independence (a class-#12 trap: the instrument's own teacher-forcing manufactures the snap-back).
- **Inter-flip gaps**: large and irregular (OFF p2 gaps 38/43/2/9; p1 gaps 23/15/40). Mostly isolated, NOT one
  contiguous cluster — argues AGAINST a single early fork that cascades for the whole stream.

### The decisive fork found in prompt 3 (the ctrl-degenerate basin)
Positions 95–108 of OFF prompt 3 carry 6 of the 9 clear flips. Decoded:
- pos 93–94 served `` ``` `` `\n\n` (dev 0.000, on-oracle).
- **pos 95 served `<` (id 27), oracle argmax `` ``` `` (id 71093), dev 9.500, served OUT of oracle top-k** —
  a HARD trajectory fork: cat9 opened a `<ctrl:begin><ctrl:token>0<ctrl:part>thought…` GARBAGE block instead
  of the codefence/prose continuation native would emit.
- pos 96,99,102,104,108 = clear flips INSIDE that degenerate `<ctrl:…>` basin. The oracle (teacher-forced on
  the served `<ctrl:…` prefix) AGREES at the format-fixed slots (97,98,100,101,103,105,106,107,109 all dev=0)
  and only flips at the high-entropy ctrl-token CONTENT slots. These 5 downstream clears EXIST ONLY because
  pos 95 forked into a basin native never entered. The teacher-forced oracle de-cascades the SEQUENTIAL
  dependence but cannot un-create the basin → these read as "independent" yet are fork progeny.

### Reconciled de-cascade count (3 views)
| view | rule | cat9 OFF count |
|---|---|---|
| raw clear | deviation ≥ 1.0 | 23 |
| sequential-decascade | gap ≤ 2 = cascade tail | 18 |
| ROOT-FORK-aware | collapse the p3 ctrl-basin (95–108) to its ONE root fork at pos 95 | **~14–15** |

VERDICT step 1: **HYBRID, trajectory-fork-DOMINATED.** ~5 of the 23 are downstream of ONE hard fork (p3 ctrl
basin) and 5/23 are served-out-of-topk hard divergences (the basin entries). The remaining ~13–15 are in-topk
rank-1/2 near-tie crossings at the SAME high-entropy format boundaries native crosses (codefence ` ``` `=71093,
prose `Let`=9764, tool-call) — i.e. genuine per-forward near-tie crossings, but only ~13–15 of them, not 23.
This is the same structure as chain3 in FR13_PLUS2 (dispersed real crossings) PLUS one degenerate fork.

## STEP 1 quantitative gap structure (per prompt)
- p0 (len76, EOS@75): clears 35,37,67,74,75 → 3 clusters; 74/75 are the EOS itself (`.`+`<|im_end|>`).
- p1 (len103, EOS@102): clears 24,47,62,102 → 4 isolated; 102 = EOS.
- p2 (len128): clears 21,59,102,104,113 → 4 clusters; 59/102 at `` ``` ``/`Let` codefence-prose boundary.
- p3 (len128): clears 61,69,77,95,96,99,102,104,108 → 61/69/77 isolated (grep/sort/explore), then the
  95–108 ctrl-basin = ONE fork + 5 progeny.

---

## STEP 2 — the RECOMPUTE-WORSE clue: it is a COMMITTED-PATH (trajectory) shift + a length artifact, NOT a
## co-residency-removal failure

Recompute makes the scan STATE bit-exact (int-view 0.0) AND aligns to native geometry (removes leaf
co-residency: each node replayed from the spine independently), yet clear flips ROSE 23→32. Reconciled:

1. **LENGTH / denominator artifact (class #12)**: OFF EOS'd EARLY on p0 (76) and p1 (103); RECOMPUTE ran all
   four to 128. That exposes 52+25 = 77 EXTRA scored positions. Per-1000-token rate: OFF 52.9 vs RECOMPUTE 62.5
   — the rate rose only 1.18x, not 32/23=1.39x. The bind's common-prefix-normalized recompute=25 (vs 23) is the
   honest like-for-like: a SMALL real rise, heavily inflated by length in the raw count.

2. **COMMITTED-PATH (trajectory) shift — the mechanism**: OFF↔RECOMPUTE served streams fork early (LCP p0=17,
   p1=15, p2=31, p3=69; ~369 token diffs). Recompute changes the per-node VERIFY LOGITS (different scan
   state-feed path → different argmax at some nodes) → the LCP-max committer (`drafts[node] vs
   parent_targets[node]`, longest-LCP path wins, commits `drafts[node]`+correction) selects a DIFFERENT path
   and/or a different bonus/correction token at the fork → a DIFFERENT served prefix → a DIFFERENT downstream
   trajectory with its OWN high-entropy boundaries. So "removing co-residency" did not reduce the per-forward
   crossing RATE; it merely re-rolled WHICH trajectory is walked. This directly answers the prompt: removing
   co-residency CHANGES THE ACCEPTED PATH → a different trajectory → comparable-or-more flips. The accept
   mechanism is the LCP committer (below), and it is deterministic-greedy, so a logit change at a near-tie
   flips the committed token and forks the stream.

3. **What it says about the banked "+17 leaf co-residency" decomposition** (FR13_WIDTH_CARRIER_INPROJ_BA,
   FR13_22flip_carrier_l0gdn): CHALLENGED. If +17 were the dominant carrier, removing co-residency would have
   dropped the count toward native; instead it ROSE (or held flat per-rate). Co-residency is NOT the dominant
   e2e flip carrier — it is at most a SECOND-order trajectory perturbation. Notably RECOMPUTE had FEWER
   out-of-topk hard forks (3 vs 5) and NO ctrl-degenerate basin — so the recompute trajectory was actually
   "cleaner" per-event but longer, netting more raw flips. This is fully consistent with scan-ruled-out: the
   carrier is the trajectory the committer walks, not the per-node state realization.

---

## STEP 3 — ACCEPT/COMMIT vs DRAFTER vs TOPOLOGY: why cat9 commits a different token than its no-spec greedy at
## the flip positions, and native does not

### The two accept mechanisms (read from source)
- **native MTP-5 greedy** (`vllm_src.sh v1/sample/rejection_sampler.py` `rejection_greedy_sample_kernel`,
  L653): LINEAR chain of 5 draft tokens. For each pos accept iff `draft_token_id == target_argmax_id`, and
  STORE `target_argmax_id` (the verify-forward argmax, NOT the draft). First mismatch → stop, commit the
  target_argmax at the reject pos. Bonus token appended iff all accepted. ⇒ native ALWAYS commits the verify
  argmax of its single spine forward.
- **cat9 committer** (`fr10_phase4_patch_vllm_tree_gdn.py:6610` `_lumo_tree_path_lcp_max_greedy_sample`):
  9-node CATERPILLAR (spine 0-1-3-5-7, leaves 2,4,6,8). For each root-to-leaf path, lcp = consecutive nodes
  with `drafts[node] == parent_targets[node]` where `parent_targets = target_logits.argmax` (L8370). Longest-LCP
  path wins (stable earliest-leaf tie-break). Commit = accepted `drafts[node]` (== verify argmax on the accept
  prefix) + one correction (`reject_parent_target` / `tree_self_target` / `root_parent_target`, L6876-6896).

### WHY cat9 flips at 23 and native at 3 — structurally
Both commit the verify argmax on the accepted prefix, so per-token a single forward is argmax-equivalent. The
gap is NOT a committer-row bug (FR13_COMMIT_ARGMAX_GATE already gates channel-1 row mapping). It is the
COMBINATION of:
1. **Verify-forward logit divergence at near-ties (the per-forward residual)** — cat9's verify forward
   (forked-FA2 tree + GDN tree-scan) differs from native MTP-5's FLASH_ATTN spine forward by the documented
   floors: FA2-fork 2-ULP MMA-grouping floor (14/16 calls 0.0, 2 single-ULP/~1M, max 0.0039, ~15x below E5
   noise, NO depth growth — [[project_fr13_fa2_fork_nocopy_floor]]) + diffuse per-layer ~1-bf16-ULP GDN
   accumulation over ~48 layers ([[reference_diffuse_gdn_accumulation_explained]]). At a near-tie boundary
   (served in oracle top-k rank 1-2, dev just over 1.0) a sub-ULP realization diff flips the argmax. ~13-15 of
   the 23 are exactly this in-topk near-tie class. This is the SAME class native crosses (native's 3 are all
   in-topk rank-1 at ` ``` `=71093 / quote-style) — cat9 just crosses it ~4-5x more often because it has TWO
   extra divergent kernels (forked-FA2 tree-bias + GDN tree-scan path) vs native's single-spine FA2.
2. **Trajectory amplification via the tree → degenerate basins**: cat9 verifies a TREE and walks a DIFFERENT
   prefix than native's MTP-5 spine. Once cat9's near-tie crossing diverges the prefix, it can walk into a
   degenerate basin (the p3 `<ctrl:token>` block) native's spine never enters, manufacturing a cluster of
   downstream crossings (the fork-dominance of step 1). native's single linear spine has fewer near-ties (no
   branch-induced extra forwards) and no tree to amplify a crossing into a basin → 3.

So at the 23 flip positions cat9 commits a different token than its no-spec greedy because (a) for ~13-15: the
cat9 VERIFY forward's argmax itself diverged from the clean recurrent argmax at a near-tie (verify-logit
divergence, the per-forward floor); (b) for ~5: the prefix already FORKED upstream (pos-95-class) and the
committer is faithfully serving the diverged-prefix's greedy token. (a) is the carrier; (b) is fork progeny.

---

## STEP 4 — NEXT-BEST HYPOTHESIS + a CHEAP test

### Hypothesis (consistent with scan-ruled-out + recompute-worse + the de-cascade)
**H-FORK-AMPLIFICATION:** The 23-vs-3 gap is NOT one fixable per-forward kernel seam. It is ~13-15 genuine
per-forward NEAR-TIE crossings (driven by the residual two-kernel verify-forward floor: FA2-fork 2-ULP +
diffuse-GDN ~1-ULP-over-48-layers, NEITHER a single paddable op) that the TREE TOPOLOGY then AMPLIFIES into the
raw 23 by (i) giving cat9 more divergent forwards per step than native's single spine, and (ii) letting a
crossing fork into a degenerate basin (ctrl-token) that spawns a downstream cluster. The dominant, ACTIONABLE
lever is TOPOLOGY, not a kernel seam: a shallower / root-sibling tree (per
[[project_fr13_tree_reshape_unifying_lever]]) reduces the depth-accumulation that turns the ~1-ULP floor into a
flip AND reduces the basin-amplification — exactly the chain5-vs-chain3 result (deep 5-spine de-cascaded to 2 ≤
native; shallow 3-tree stayed at 5 dispersed). The carrier is "verify-forward near-tie floor × tree
amplification," and the cheapest win is reshaping the tree, with the binding arbiter being e2e accept/event vs
E5 — NOT the raw flip count (which is fork-inflated and length-sensitive, class #12).

This is explicitly NOT "irreducible" (native-E5=3 disproves it; research-before-deadend honored): the per-forward
floor is real-but-small and the amplification is topology-controllable.

### CHEAP test (CPU de-cascade already done here; ONE targeted single-GPU A/B to confirm)
**CPU (free, partly done):** the de-cascade above already refutes "pure cascade" (18/23 survive teacher-forcing)
and "pure per-forward spread" (the p3 ctrl basin is one fork + 5 progeny; 5/23 out-of-topk). CONFIRM-able further
on banked data with zero GPU: re-score the SAME 4 served streams but classify each clear flip as {near-tie
in-topk rank≤2} vs {hard out-of-topk fork} vs {fork-progeny inside a degenerate basin} — predict ~13-15 / ~3 /
~5. (Numbers above already give this split; a reviewer can re-run the cell.)

**Single-GPU A/B (one boot, decisive, cheap):** run cat9 with the tree RESHAPED to a SHALLOWER root-sibling tree
(e.g. depth-3 spine + 2 root siblings, the chain3/cat-shallow shape) at temp 0.0 seed 1313 on the SAME 4 prompts,
re-score against the SAME recurrent oracle. PREDICT (H-FORK-AMPLIFICATION): de-cascaded independent flips drop
toward native (chain5→2 precedent) AND e2e accept/event holds or improves. If the shallow tree's flip rate does
NOT drop, H is refuted and the carrier is a depth-independent per-forward seam (re-open the FA2-fork floor as
deterministic, contra the banked "probabilistic tie-break, no depth growth" finding). Cost = 1 GPU boot,
B=1, 4 prompts, ~minutes — far cheaper than the full B=4 SWE-4 gate. This is the non-vacuous instrument: it
varies ONLY topology, holding kernels/seed/prompts fixed, so a flip-rate drop isolates amplification.

### What this REOPEN rules IN / OUT
- RULED OUT as dominant e2e carrier: GDN scan state-feed (bind), leaf co-residency "+17" (recompute-worse),
  any single per-forward op (scan was the strongest, dead).
- RULED IN (prime): tree-topology amplification of a small two-kernel verify-forward near-tie floor; degenerate
  basin forks (p3 ctrl). Actionable lever = tree reshape; arbiter = e2e accept/event vs E5.
- DO NOT bake recompute (different deterministic stream, 369 tok diffs, NOT lossless, and WORSE).

## Reward-hack / hygiene
CLEAN: pure read of banked artifacts + committed source via vllm_src.sh; no GPU boot; no served-path splice; no
new kernel; this doc is the only write. Recurrent oracle is the bind's A/B oracle (no adoption — adoption would
be the FR13_ORACLE_FRAME reward-hack signature). The single-GPU A/B proposed varies ONLY topology (no
copy/dense/forced-spine). Bring the e2e accept/event table to the user; no self-declared pass.

Pairs with FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_PLUS2_DECASCADE.md,
[[reference_scalar_metric_per_token_blindspot]], [[reference_diffuse_gdn_accumulation_explained]],
[[project_fr13_fa2_fork_nocopy_floor]], [[project_fr13_tree_reshape_unifying_lever]].
