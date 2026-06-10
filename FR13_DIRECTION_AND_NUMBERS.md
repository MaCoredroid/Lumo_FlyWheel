# FR-13 — Direction & Numbers (2026-06-10)

Contract: **workflows are the worker** (codex stood down); Claude red-teams/binds/commits. Raw research: `research/fr13_workflows/INDEX.md`. Running now: **Method-A BI campaign** (`wuubgvyw0`, prep phase).

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

**WY: PARKED-NOT-DEAD, dominated** (within-floor ~1-ULP but not byte-exact; FLOP advantage worthless bandwidth-bound; chunking risks reintroducing batch-variance). Only revisit if the replay route fails its gates.

## Convergence principle (user's superset logic)
Fixing the Half-1 carrier should pull accept/event 2.024 → ≥ same-shape native (spine recovered) **+ branch bonus**; the Half-2 route cuts the per-forward tax below native. Lossless and faster are ONE outcome, not a trade.

## Open unknowns (honest)
1. **Which carrier** — Method A resolves (running).
2. **Codegen-identity byte A/B** — pending, gates the full replay route.
3. **Root-cause confirmation** — the two gate-4 mechanisms are source-level inferences; the branch retry confirms empirically.
4. **Branch/superset upside** — branches added ~0 net accepts in EVERY measured run; topology memory says deployed propose_tree builds 2 parallel chains, not the intended caterpillar. Whether post-carrier-fix branches can actually deliver accept/event > native (the whole point of the tree) **has never been quantified** → research workflow launched (see below).
5. Robust multi-sample p95 floor (drift-tracker design) — built after Method A if its 2-seed floors prove noisy.
