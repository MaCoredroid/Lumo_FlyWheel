# FR13 — TOTAL DRIFT REANALYSIS: actionable leads (bind)

Date 2026-06-14. Distilled from `FR13_TOTAL_DRIFT_REANALYSIS.md` (committed `473cbda0`, fresh skeptical
CPU re-derivation of the 21 baked flips). My red-team: HOLDS — appropriately hedged (2-3 un-aligned seams +
cascade inflation, NOT a single magic carrier like the prior overstated FA2-tile / BV-warps / width-H1
hypotheses). Two load-bearing claims CODE-CONFIRMED this tick:
- ~~**Decode backend = TREE_ATTN, FA2 fork = PREFILL-only**~~ **CORRECTED 2026-06-14 — WRONG (see
  FR13_FA2_FORK_IS_DECODE_KERNEL_CORRECTION.md).** `fr13_launch_locked.sh:24` exports `FR13_FA2_TREE_BIAS=1`
  (missed in the original grep) ⇒ the **FA2-fork IS the deployed decode kernel** for cat9 (max_query_len=9>1
  routes to `flash_attn_varlen_func(tree_bias=...)`). The 0.00195 is the `unified_attention` EXP2-Triton
  FALLBACK residual, shadowed OFF. The live full-attn decode is at the fork's 0.0039 floor (lossless,
  ~15× below E5). Full-attn is NOT the carrier ⇒ reinforces the replay (L0-GDN-upstream) pivot.
- **Replay route always-on** (`if True: # FR13_REPLAY_ROUTE baked ON`); its byte-A/B was replay-vs-OUR-scan,
  never vs native MTP's `fused_sigmoid_gating_delta_rule_update`.

## The reframed accounting (de-cascaded)
Baked 21 raw ≈ native 3 + ~2 spine-intrinsic + ~9-13 co-residency. **Raw OVERSTATES the per-forward defect
by ~1.5×** via cascade/trajectory inflation (bug class #12) → ~11-13 INDEPENDENT events, 3 of them native
floor. Tree flips land at boundaries **near-disjoint from native's 3** (1/18 position overlap) = a SUPERSET
of crossings, not "more of the same diffuse set." Arbiter accept/event 3.0-3.15 ~ native 3.076 ⇒ residual is
currently **sub-deployment-impact**, but the headline "21" is NOT the irreducible floor it was presented as.

## NEW LIVE LEADS (priority order) — what the prior accounting SKIPPED
The prior "BI EXHAUSTED at in_proj_ba" was scoped to the GDN layers only and skipped these LIVE channels:

1. **PRIME — replay-route cross-event durable-state vs NATIVE MTP (never A/B'd vs native).** The committer
   re-executes the accepted chain from h0 via `_tree_gdn_replay_kernel` (a rank-1 Triton kernel) and writes
   the durable next-event state; native MTP writes its durable state with `fused_sigmoid_gating_delta_rule_update`
   (a DIFFERENT kernel). The "passed" byte-A/B compared replay-vs-our-own-scan, NOT vs native. A ~1-ULP/event
   durable-state difference ACCUMULATES across verify events — the only mechanism that turns per-forward-bit-
   exact kernels into ~11-14 e2e flips. Corroborated by STree (2505.14969): recurrent-state replay is the
   bf16-sensitive path for GDN/Mamba hybrids, not the per-forward scan. **Decisive A/B = replay-durable-state
   vs native-MTP-durable-state, cross-event.** Bug class #10 (codegen-identity not spec-guaranteed) + #12.

2. **TREE_ATTN full-attn query/KV M-invariance (never re-tested POST-bake).** The 16 full-attn layers run a
   SEPARATE Triton TREE_ATTN kernel (exp2-softmax, reversed KV iter) — not native's CUDA FLASH_ATTN. The
   FA2-QPAD branch (`9ad6793f`) MEASURED the forked query tile as M_DEPENDENT (L31 3.9e-3 → 0.0 query-padded);
   that QPAD fix was OVERTURNED (`8b7684dd`) PRE-bake on the "first-nonzero is L0 GDN" ladder. Whether padding
   the TREE_ATTN query/KV tile (analogous to the in_proj_ba pad) moves e2e flips POST-bake was NEVER tested.
   Note: the 0.00195 TREE_ATTN-vs-FLASH residual is the E5-deliverable axis (cancels in the same-backend flip
   measurement); the LIVE question is M-dependence between the M=10 tree-verify forward and the M=1 decode
   oracle. Bug class #12 (co-residency / M-keying).

3. **(methodological — RAISE at close, do NOT unilaterally adopt) the lossless REFERENCE.** Lossless-specdec
   literature: lossless = "served == verify-time argmax," and output can still differ from pure AR via
   **numerical dispatch divergence**. The 21 is cat9-tree-verify vs the no-spec DECODE oracle (a different
   dispatch). Part of the 21 is the EXPECTED tree-verify-vs-sequential-decode gap, not a kernel bug. The
   academically-correct gate is tree-verify-vs-native-MTP (same dispatch class). BUT the user's ACTIVE GOAL
   explicitly sets the no-spec oracle as the bar ("per-token argmax vs the same no-spec oracle") — so this is
   a close/pass-fail reframe to SURFACE to the user, NOT to adopt mid-chase.

## Every per-forward kernel ruling still HOLDS
scan (geometry-invariant K-reduction), conv (per-row), fp8 qkvz/o_proj (M≤64 one-tile), gate (per-row rms),
FA2-fork (prefill 2-ULP floor), reshape, BI — all re-verified from kernel code. Two FLAGS, both about the
REFERENCE/LIVE-PATH not the ruling: FA2=prefill-only; the 22-flip reference is the no-spec decode oracle.

## Strategic pivot for the GPU front
The blocked L0-GDN sub-op A/B (one-more-fix ww3yd22ry, conv/scan M10-vs-M5) is predicted ~0 — and the
reanalysis says even a clean ~0 CONFIRMS the pivot (L0-GDN conv/scan is NOT the carrier), it does not close
the question. The HIGHER-value GPU A/B is **replay-durable-state vs native-MTP-durable-state (cross-event)**.
Pending the one-more-fix Verdict (does fe0af022 finally capture?), the next GPU front should be the replay-
vs-native durable-state A/B, then the TREE_ATTN query-pad e2e flip test — both as proper observe-only A/Bs
with adversarial verify (no reroute / splice / reward-hack; `feedback_no_reroute_reward_hacking`).
