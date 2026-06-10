# FR-13 — Direction & Numbers (2026-06-10)

Contract: **workflows are the worker** (codex stood down); Claude red-teams/binds/commits. Raw research: `research/fr13_workflows/INDEX.md`. Running now: **Method-A BI campaign** (`wuubgvyw0`, prep phase).

## WHAT FR-13 IS (user, 2026-06-10)
**A prototype of the fast + lossless general-tree VERIFIER for a later suffix-decoding ⊕ MTP fusion drafter**: the production plan is MTP-only for a short spine (~MTP-3) with **suffix decoding composing the branches and the deeper tail**. Consequences:
- The deliverable is the **verifier+committer, drafter-agnostic** (consumes a standard candidate-tree descriptor + q — `project_fr10_drafter_verifier_interface`): lossless for ANY q, any topology. The current caterpillar is just the prototype tree.
- **The speed target is the depth-matched SUPERSET claim**: the 9-node caterpillar must beat the **5-node top spine** (same drafter, same depth — native E5-class 3.076/per-shape). Beating the deployed K=9 LINEAR chain was **never the goal** (that comparison informs drafter design later; suffix decoding is what buys depth/branches in production).
- Therefore "tree shape can't beat K=9 chain at equal budget" is NOT a blocker — tree shapes get rich when suffix decoding arrives; what must be true NOW is: verify a tree **fast** (per-forward tax → ~native, Half-2) and **lossless** (Half-1), for arbitrary trees.

## LOSSLESS, PRECISELY (the nuance — binding definition from prior decisions + memory)
- **B=1, greedy: byte-exact, PROVEN.** Input → all 64 layers → final logits = 0.0, spine AND branch; branch correctness judged vs the **native-on-the-branch's-path-to-root oracle** (SpecInfer Def 4.1 / STree Eq.4-6) — native MTP has no branch counterpart, so the oracle is the reference. Committer is lossless-by-design at temp-0 (longest-LCP = the unique verify-argmax chain = native greedy).
- **The e2e gate is TOKEN-LEVEL, not hidden-state max_abs**: per-depth **argmax match** (greedy) — hidden drift below the native top1-top2 margin cannot flip a token (math-correct ≠ bit-exact, but the bar is the token). Per-node **marginal**, not joint (Traversal 2505.12398).
- **temp>0 (deployed 0.6): lossless by THEOREM** (SpecInfer Thm 4.2 MSS; Multi-Draft 2410.18234 Thm 1 — multi-candidate rejection sampling preserves the target distribution for any q). The gate is **distributional equivalence to native**, NOT byte-exact tokens.
- **B=4 deployed regime: byte-exact is IMPOSSIBLE** — native itself is non-deterministic (measured same-seed floor bag-TV **0.113**). So the operational bar = **within native's own same-seed self-noise floor**: self-noise-corrected real-loss-rate ≈ 0, bag-TV ≤ floor (multi-sample p95 floor preferred over any single draw — the old 0.0593 was one draw), **accept/event ≥ same-shape native** (superset principle: the tree contains native's chain, so sub-native accept = our bug, never "drafter quality").
- **Two-tier vs non-MTP ground truth**: the reference is equivalence to NON-MTP ground truth, not byte-exact-vs-MTP-baseline — MTP-5 itself sits ~6e-5 (chunk-vs-recurrent) from non-MTP; our kernel may legitimately be CLOSER (7.45e-9). Fix the kernel only if it diverges MORE than baseline.
- **Verifier-only proof**: regular decode (no tree/spec/bias) with our forked .so == pristine stock, 0.0 every layer — our changes must not touch the non-spec path.
- **Verify-drift ≠ sampling noise**: when attributing B=4 drift, isolate the kernel/verify channel (forced-decode logit-KL/TV) from path/sampling non-determinism; the irreducible co-residency component is absorbed into the floor.
- Per-layer 0.0 ladders are DEV checks; **the verdict is always e2e vs E5 at the deployed regime** (B=4, CUDA-captured, SWE prompts).

## THE BAR
- **Lossless** = tree within the *measured* native same-seed B=4 floor, NOT byte-exact (native itself is non-deterministic at B=4): **bag-TV ≤ 0.113** (seed1313v1313, FULL-capture, 64-tok shape; bind `FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md`). The old `0.0593` floor was a single different-seed draw — superseded. Seed-robust different-seed budget ≈ 0.0986-0.1099.
- **Superset/speed signal** = tree accept/event must climb to **≥ native at the same shape** (the tree is a strict superset of the same MTP drafter: spine ≡ native chain + top-2 branches ⇒ accept < native is IMPOSSIBLE for a correct verify — sub-native = our contamination, never "drafter quality").
- Native accept/event is **shape-specific** — compare within-shape only:
  | shape | native accept/event |
  |---|---|
  | B=4, 128-tok, spp=4 (diag 190931Z) | 3.794 |
  | B=4, 128-tok gate arm (194841Z) | 3.619 |
  | B=4, 64-tok, spp=1 (floor arms) | 3.203 / 3.125 |
  | B=1 (E5 reference) | 3.076 |

## HALF 1 — LOSSLESS (the B=4 carrier)
**Status: B=1 byte-exact lossless PROVEN** (per-layer ladder 0.0; committer lossless-by-design; conv 3a9039cc + FA2-prefill c8704a5a seams byte-exact). **B=4 is lossy beyond floor**:
| metric (B=4 captured, 128-tok) | tree | bar |
|---|---|---|
| bag-TV | **0.2335** | 0.113 floor (~2× above) |
| real-loss outside self-noise | **0.4751** (105/221) | ≤0.05 |
| accept/event | **2.024** | 3.6-3.8 (same-shape native) |
| root-reject (same single candidate as native) | **36.6%** | ~0 (= verify contamination) |
| same-seed tree-vs-native, confound-free | 165/256 (64.5%) | — |
| tree run-to-run non-det | 189/256, failing rows MOVE | native 137/256 (diff-seed) |

**Carrier elimination ladder (all committed):** GDN scan EXONERATED (batch-invariant by construction: beec984a/e4a6a2f2/45519178; no batch axis, static_range, the only atomic is a gated diagnostic) → FA2 attention-reduction RULED OUT (num_splits inert by construction: `set_params_splitkv` needs `max_seqlen_q==1`, tree has >1; 0≡1 non-split, no combine — 397319ab) → **remaining candidates: {BI-coverable numerics, non-BI-covered channel, accept-pattern state/bank-row wiring}**.

**Direction = Method A (RUNNING):** flag-gated `VLLM_BATCH_INVARIANT` enablement for TREE_ATTN (justified by the num_splits proof; double-gated `FR13_BI_TREE_ATTN=1`, inert by default; rename REJECTED as reward-hack) + 5 serialized BI-on boots (T1/T2 tree same-seed cross-boot, N1/N2 native, Nn noise) + corruption gates + determinism compares.
- **Key reframe (verified):** BI does NOT swap the fp8 GEMMs on GB10 (same `cutlass_scaled_mm` BI-on/off; the FR12 "inherently invariant" probe was a different op binding — still a candidate). BI pins the SURROUNDING numerics: **bf16 lm_head/logits GEMM** (the L0c drift seam), softmax, RMSNorm, bmm, cuBLAS split-k/workspace, TF32, reduced-precision reductions.
- **Decision tree:** CASE-1 (excess → floor + accept/event into the native band) = BI-coverable-numerics carrier confirmed, lossless achievable, production bar stays relative BI-off (BI is slow; native's floor contains the same variance). CASE-2 (excess remains) = non-BI-covered channel (incl. live fp8 cutlass GEMM, conv/h0 bank-row wiring) — **but the system is then DETERMINISTIC, unblocking the fixed-row localization that previously failed because failing rows moved (e263a45b)**. Either branch advances.

## HALF 2 — SPEED (the HBM state tax) — separable from Half 1
**Measured wall gap** (decode_seconds basis, not TPS÷accept): tree **2.336×** slower = 1.432× more forwards (accept gap → fixed by Half 1) × 1.632× per-forward (≈42%/58% split).

**Re-based state-traffic accounting** (rows = 3.146 MB each; 48 GDN layers × B=4; verified w78aq6xum):
| configuration | rows/layer/req (a=4) | GB/forward | vs actual native |
|---|---|---|---|
| actual native E5 (1R+6W) | 7 | 4.23 | 1.0× |
| CURRENT tree (scratch 9W + publish 9R+9W + remap 2a + h0 1R) | 36 | 21.74 | **5.14×** |
| stage-1 accept-only publish (PARKED) | 26 | 15.70 | 3.71× (**only 1.385× cut**; net-negative at a=5+spill) |
| **FULL route: activation-store + replay** | 6 | 3.62 | **0.86×** (below native) |
| full + single-row commit (needs consumer audit) | 3 | 1.81 | 0.43× |
- Replay compute is FREE (<1% of the 99 ms weight-bandwidth floor; 27 GB/273 GB/s). Per-node activations ~16.2 KiB bf16 vs 3 MiB state row (~190×). Bonus: deletes the 201 MB/layer capture-blocking `tree_state_all` alloc; the replay kernel is spill-free at any tree size (no N_PAD=16 wall).
- **Bit-exactness mechanism:** replay RE-EXECUTES the identical fp32 instruction sequence on identical bytes (vs rejected delta-only which REASSOCIATES) — conditional on codegen identity: one shared `@triton.jit` body + a one-time byte A/B gate (**NOT yet run**); plus the ±0.0 handoff normalization and softplus/sigmoid/l2norm recompute-identical obligations.

**Accept-only post-mortem (bind `FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md`):** offline gates 1-3 PASSED (scan 0.0 @ N_PAD=1/16, accepted-rows torch_equal, regular-decode==pristine) but **LIVE gate-4 FAILED** (accept 2.024→1.521, real-loss 0.7315, bag-TV 0.5347). Root causes identified: (1) **stale row-0 on zero-accept events** (h0 clamps accepted_len-1→col 0; path-only publish never refreshes it); (2) **non-graph-stable pending dict under FULL capture** → silent wrong-buffer publish (+~9.4 GiB unprofiled pin). Patch parked on `fr13-accept-only-wip` (1a566d41); main untouched. **Lesson bound: offline single-forward bit-identical ≠ live multi-step bit-identical.**

**Direction (queued behind Method A — one repo-mutating workflow at a time):** staged replay-route build: fix stage-1's three blockers on the branch (persistent staging bank, pop-on-publish, zero-accept row-0 path) → STORE_NODE_STATES=False + activation ring + chain-replay kernel (Option 1 eager → Option 2 capture-fused), gated by byte A/B + durable-state diff=0 + live gate-4-class campaign.

**WY: in the plan as LAST-RESORT FALLBACK (user 2026-06-10) — not battle-tested lossless, so never the primary.**
Half-2 ordering:
1. **PRIMARY = replay route** (above): the battle-tested byte-exact, batch-invariant-by-construction kernel, tax removed by activation-store + identical-op-order replay (0.86× native).
2. **FALLBACK = WY**, triggered ONLY if the primary fails a hard gate: (a) the codegen-identity byte A/B fails irrecoverably (shared `@triton.jit` body can't force bit-identical replay), or (b) the replay route fails its live gate-4-class campaign for structural reasons, or (c) an unfixable spill/capture wall.
   WY's standing (verified wox8pnjx8): within-floor ~1 bf16 ULP, **not byte-exact** (different summation tree — provably can never be 0.0 vs the sequential incumbent); 4.19e-9 vs non-MTP ground truth (closer than shipping MTP-5's own ~6e-5 chunk gap); batch-invariance argued by construction for the self-authored static kernel but **must be re-proven live**; the "6/6 spine argmax" evidence is contested ("single-event coincidence", b8747d23). If triggered, WY runs the mapped path: re-plug archived `tree-gdn-wy-kernel` (branch `fr13-wy-archive`, c0448bd7) **with the 8a975837 state fix** (the pre-fix 1.1989 e2e is NOT a verdict against it) → B=1 ladder spine AND branch (native-on-branch-path oracle) → B=4 CUDA-capture confirm → decisive B=4 SWE e2e vs E5 at the **within-argmax/within-floor bar** (bag-TV ≤ floor + accept/event ≥ same-shape native). ~3-5 boots.
   WY also reaches the same ~1.0× HBM endpoint (accept-only state commit is native to its design) — the fallback loses byte-exactness, not the speed target.

## Convergence principle (user's superset logic)
Fixing the Half-1 carrier should pull accept/event 2.024 → ≥ same-shape native (spine recovered) **+ branch bonus**; the Half-2 route cuts the per-forward tax below native. Lossless and faster are ONE outcome, not a trade.

## Open unknowns (honest)
1. **Which carrier** — Method A resolves (running).
2. **Codegen-identity byte A/B** — pending, gates the full replay route.
3. **Root-cause confirmation** — the two gate-4 mechanisms are source-level inferences; the branch retry confirms empirically.
4. **Branch/superset upside — RESOLVED (wlhtzqvib, verify-passed): comparator-split.**
   - **Topology vindicated**: the deployed tree IS the intended caterpillar (`FR10_CATERPILLAR_NATIVE_SPINE_TOP2`, gate-verified 193/193, genuine per-position top-2; the two-chains note above was STALE — patched out for all post-06-04 runs). One structural gap: **no sibling at depth 1** (root top-2 is dead code) while **62% of reject-steps land at step-0** — the largest reject bucket has no branch rescue.
   - **"Branches added 0" was partly a measurement artifact**: the per-pos counter indexes accepted-path-length (pos5-8 ≡ 0 for depth-5); the source-index trace shows **+0.21/event gross**; the committer does follow branches (48.1% conditional survival). Net-zero = contamination swamping it (spine loses ~0.75-0.85/event; root-reject 36.6% on a byte-identical candidate).
   - **ENDGAME (re-anchored per user — see "WHAT FR-13 IS"):** the target is the **depth-matched superset claim**: 9-node caterpillar > 5-spine native (3.076-class). That is **plausible**: bonus +0.17..+0.36 (+5-12%), CONDITIONAL (p2 was measured on corrupted runs; a ~+0 healthy floor cannot be excluded — the paired boot pins it). The K=9-linear comparison (tree can't match it at equal budget) is **informative for the future drafter, NOT a target**: in production, suffix decoding (⊕ MTP-3) composes the branches/deeper tail — richer trees than this prototype. The depth-1-sibling gap (62% of rejects at step-0, root sibling +0.08-0.15) is a cheap topology improvement worth taking regardless.
   - **Cheapest decisive measurement:** ONE paired boot post-Method-A (healthy tree + native, pinned same-8 prompts, existing `tree_path_lcp`/`spec_trace` logging) pins healthy a_d, healthy E[p2], adjacency, root-reject — replaces every assumption.
5. Robust multi-sample p95 floor (drift-tracker design) — built after Method A if its 2-seed floors prove noisy.
