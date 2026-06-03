# FR9 Spine-2 Copy-Based (Selector-Off) Lossless Investigation — CLOSEOUT

**Date:** 2026-06-03
**Branch:** `fr9-spine2-lossless-winner` (off `main@c22a0859`)
**Status:** **CLOSED.** Selector-off copy-based spine-2 is **borderline / not cleanly
lossless**; spines=1 remains the only lossless production default. The fail-closed
safety infra ships. A clean lossless-and-fast spine-2 has **no plausibly-cheap path
on vLLM 0.19 + Qwen3.6 Gated DeltaNet** — every route hits the same GDN root cause.
**No speed-win is claimed. No losslessness is claimed for spines>1.**

## 0. Objective
Make selector-off `spines=2` (copy-recurrent-state multi-spine, `--config Fb
--row-mode independent`) **statistically lossless** w.r.t. `spines=1` on the real
B=4 SWE-Verified workload at temp 0.6, **before** any speed work. "Lossless" here =
statistical (preserve served-model quality/distribution within the model's own
run-to-run numeric noise), not bitwise.

## 1. Measured numbers (quick verify)
Setup (identical across arms): config Fb, mtp=5, independent rows, lowmem088, **B=4**,
**temp 0.6**, 4-task SWE-Verified subset (`swe-bench-agentic-b4-four-verified`),
1800 s per-task wall+eval gates, `--no-commit`. Arms run on separate ModelServer
relaunches.

| Arm | Tasks (R/F) | accept/event | accept/draft |
|---|---|---:|---:|
| s1a (spines=1) | 2R/2F | **3.3355** | 0.6671 |
| s1b (spines=1) | 1R/3F | **3.2628** | 0.6526 |
| s2 (spines=2 selector-off) | matches s1b on 3/4 | **3.2364** (lowest of 3) | 0.6473 |

Per-task verdicts (the self-distance is real):

| Task | s1a | s1b | s2 |
|---|---|---|---|
| astropy-12907 | resolved | resolved | resolved |
| astropy-13033 | failed | failed | failed |
| astropy-13236 | **resolved** | **failed** | empty-patch retry (confound) |
| astropy-13398 | failed (1530 s) | failed (1600 s) | failed |

**Acceptance-length distribution TV (the rigorous metric):**

| Comparison | TV (accept-count dist 0–5) | vs floor |
|---|---:|---|
| s1a vs s1b (self-noise FLOOR) | **0.0188** | — |
| s2 vs s1b | **0.0097** | within ✓ |
| s2 vs s1a | **0.0283** | beyond ✗ |

Spine accounting (s2): `policy=lossless` all rows, `non_lossless_public_stream_events=0`,
`winner_spines={"0": N}` (every public commit = spine 0), and
**`lossless_suppressed_superset_events=831`** (831 events where the hidden spine
accepted a longer prefix but was suppressed for losslessness).

Wall-hit validity gate (operator heuristic: most tasks should hit the 1800 s wall):
3/4 hit the wall; `13398` is a consistent early-fail across BOTH s1 runs
(1530 s / 1600 s) → carried as flagged, not excluded; `13236` is high-variance
(R/F/empty across arms) → the dominant self-distance contributor.

## 2. The losslessness issue
Independently reproduced (monitor computed the TVs from `agentic_summary.acceptance`,
matched codex exactly). Verdict: **BORDERLINE, not clean.**

- **s2 is "worst of both":** the superset is SUPPRESSED (831 longer-prefix wins
  thrown away because selector-off publishes spine-0 only) **and** path0 is
  DEGRADED — `accept/event(s2)=3.236` is the **lowest of all three**, a small but
  directional reduction. This matches the prior FR9 finding: *"token-tree on
  GDN-hybrid gives only an INTERNAL superset; path0 is degraded by shared recurrent
  state."*
- **Root cause:** the GDN/SSM recurrent kernels are **batch-composition-dependent**.
  vLLM batch-invariance covers only attention + GEMM, **not** the GDN scan
  (vLLM **#42960** OPEN; GDN backends inherit `supports_batch_invariance()==False`,
  confirmed through latest 0.22.0 and `main`). So co-scheduling the hidden spine in
  the same GDN batch perturbs spine-0's logits.
- **Statistical read:** with 1 run/arm × 4 tasks the test is **underpowered**.
  `TV(s2,s1b)=0.0097` is within the `0.0188` self-floor, but `TV(s2,s1a)=0.0283` is
  beyond it, and accept/event is lowest of three. Honest conclusion: **"no
  statistically-detectable loss at quick-verify power, but a real small directional
  perturbation is present"** — NOT a clean lossless proof. (The greedy temp=0 probe
  earlier also showed a batch-4-only token flip, consistent with the same numeric
  perturbation.)

## 3. Investigation — the path to a *clean* lossless spine-2
The correct fix (operator-specified) = **state isolation** (canonical spine-0 +
checkpoint/replay; public state produced ONLY by canonical spine-0 execution, hidden
branches read checkpoints, never write public state) **co-designed with** the
distribution-preserving **multi-draft selector** (MDSP 2410.18234 / SpecHub
2411.05289) to publish the suppressed superset losslessly. Cost-gated, primitive-first.

**Phase 0 (feasibility, vLLM 0.19):** vLLM has recurrent-state device-to-device
**copy** among scheduler-owned cache blocks, but **no exposed primitive** to restore
an arbitrary checkpoint and run an **isolated `num_reqs=1` single-step forward**
outside normal batch construction. `collective_rpc`/`apply_model` insufficient
(passes the model module, not the live `GPUModelRunner` caches/request state).
→ report `fr9-p0-gdn-state-isolation-feasibility-20260603.md` (commit `953e60b5`).

**Upgrade check:** latest vLLM **0.22.0 and `main` still lack** batch-invariant GDN
(#42960) **and** the isolated-forward primitive. The upgrade is *cheap* on this box
(prebuilt `vllm/vllm-openai:v0.22.0-cu130` multi-arch arm64 image; `+cu130` release
wheels exist — the earlier "needs source build" concern was wrong) but **useless for
the objective.** → `/tmp/fr9_vllm_upgrade_feasibility.md`, `/tmp/fr9_vllm_cu130_dgxspark.md`.

**Build attempt (isolated-forward primitive):** implemented `FR9IsolatedForwardProbe`
/ `run_worker_probe` as a vLLM patch in a **separate diagnostic image** (production
untouched), 64×2 bit-reproducibility probes via ModelServer. Result (commit
`a80e9554`, report `fr9-isolated-forward-p0-20260603.md`):
- ✅ **Public-state isolation works** — `public_cache_unchanged=true` (sha256 before==after).
- ❌ **The isolated forward does not run cleanly:** tracked-scratch path → `no tracked
  scratch block for group 3` (a single short isolated request has no extra
  block-pool-owned scratch); the untracked-scratch shortcut → **`CUDA illegal memory
  access` inside FlashAttention** (the forward is entangled with block-pool/cache
  ownership). Left **fail-closed**.
- **Verdict:** the primitive is **NOT feasible as a pure `GPUModelRunner` source
  edit on vLLM 0.19**; it needs scheduler/block-pool-integrated scratch allocation
  **or** a custom recurrent-state kernel.

**Tree-route reconsideration (the natural pivot):** the repo already MERGED a
**lossless GDN tree-delta verifier** in Round F (`9fc08ae7`, Triton kernel
`scripts/round_f_tree_delta_triton_validate.py`, flag-gated). It was **NO-SHIP on
SPEED**, not losslessness: per-node GDN tree path 243.1 ms/event vs 242.9 ms budget,
6.4 tps vs E3's 16.7. And critically, the F_a closeout already found **STree does not
rescue this stack**: STree is a *diagonal*-SSM tree scan, but **Qwen3.6 Gated
DeltaNet = diagonal gating + a rank-1 delta update**; the rank-1 delta term is
order-dependent and has **no cheap published tree kernel** — correctness needs
explicit per-node parent-state copy (too slow) or a **novel chunked tree-delta
kernel**. (Research on whether that delta-tree kernel is derivable/published is in
flight: `/tmp/fr9_stree_deltanet_tree_kernel.md`.)

## 4. Conclusion
1. **Copy-based selector-off spine-2 is NOT cleanly lossless.** It is borderline (a
   real, small, GDN-co-residency perturbation that drops accept/event below
   spines=1), and selector-off discards the superset anyway. **spines=1 remains the
   only lossless production default.**
2. **What ships (correct + honest):** the fail-closed safety infra — the lossy
   best-of-spines public commit is deleted, `policy=lossless` is the only public
   policy, and `spines>1` **fail-closes before launch** (only an explicit
   `LUMO_IR_DIAGNOSTIC_UNISOLATED=1` override allows the controlled experiment).
   Production default is unchanged and safe.
3. **A clean lossless spine-2 has no plausibly-cheap path on this stack.** Every
   route dead-ends on the same root — **Qwen3.6's Gated DeltaNet is non-diagonal and
   vLLM 0.19 has no batch-invariant GDN / isolated-forward primitive**:
   - copy-based selector-off → co-residency perturbation (this report);
   - isolated `num_reqs=1` forward → needs block-pool/scheduler surgery or a custom kernel;
   - tree verification → lossless but NO-SHIP on speed; **STree only covers the
     diagonal gate, not the rank-1 delta** (needs a novel tree-delta kernel);
   - separate-engine isolation → lossless by construction but ~2× memory + cross-engine coordination.
4. **Recommendation:** keep **spines=1** as the lossless default; spine-2 stays
   **diagnostic-only (fail-closed)**. The clean-lossless-AND-faster spine-2 is a
   research-frontier **DeltaNet tree-delta kernel** project — pursue only if that
   kernel is shown derivable (pending research); otherwise accept this CLOSED result.

## 5. Honesty notes
- No "lossless" claim for spines>1; no speed-win claim. `mtp5_s1 lowmem088`
  (~39.91 TPS earlier baseline) remains the only accepted speed-win baseline.
- All TVs were reproduced independently by the monitor and matched the agent's.
- Two earlier infra failures were diagnosed honestly, not faked: in-container CUDA
  decode-kernel tracing (CUPTI not injected into a pre-existing process → 0 kernel
  rows) was abandoned for `dgx_steptrace`; a codex `image_generation`/`gpt-image-2`
  tool crash-loop was fixed by disabling that feature.
- Artifacts/reports on branch `fr9-spine2-lossless-winner`:
  `fr9-spine2-distribution-safe-winner-spec`, `...-stat-lossless-quick-verify`,
  `fr9-p0-gdn-state-isolation-feasibility`, `fr9-isolated-forward-p0`, this report.

## 6. Post-closeout research update — the tree-delta kernel IS derivable (revises §3/§4)
Research (`/tmp/fr9_stree_deltanet_tree_kernel.md`, landed after the initial close)
revises the tree-route assessment: a no-copy tree-delta kernel for Gated DeltaNet is
**NOT published but IS derivable**. The prior F_a premise ("the rank-1 delta term is
order-dependent, so it has no tree form") is **wrong**.
- Key fact: the within-chunk delta operator `T=(I+tril(diag(β)KKᵀ,−1))⁻¹diag(β)` is
  **strictly lower-triangular (causal)**; row t of `W=TK`,`U=TV` depends only on tokens
  ≤ t. Appending a leaf at position m+1 adds one bottom row and **cannot change trunk
  rows 1..m** — exactly the property a tree needs. Compute the shared trunk WY factors +
  end-state ONCE, then extend each leaf as a cheap masked rank-1 step
  (`w_ℓ = β_ℓk_ℓ − β_ℓ·W_trunkᵀ(K_trunk·k_ℓ)`) reusing trunk factors — no re-solve, no
  state copy. The scalar gate `α` commutes (cumulative `γ`). This is the delta-rule
  analogue of STree's `A_tree=L·A_log`; the only delta-specific extra is one matvec
  against precomputed trunk factors. STree confirmed diagonal-only; no 2025-26 paper
  ships this (closest: Aurora arXiv:2602.06932 tree-attends only the attention layers).
- **The real gate is NOT the algebra — it is vLLM-0.19 CUDA-graph capture of a custom
  GDN/tree backend** (the exact Round F wall).
- **Revised recommendation (risk-first spike, future):** (1) FIRST settle whether
  vLLM 0.19 can CUDA-graph-capture a custom GDN/tree backend — decide BEFORE writing the
  kernel; (2) if capture is viable, derive + microbench the trunk-share/leaf-append
  Triton kernel vs the per-node-copy baseline on ONE GDN layer; (3) only then wire into
  the multi-spine verifier. Do NOT re-shelve as "delta has no tree form" (false). Shelve
  ONLY if (1) shows vLLM 0.19 still cannot capture the GDN/tree backend = the Round F
  capture ceiling (kernel cost then moot).
- Net: lossless-AND-faster spine-2 moves from "unknown if possible" to "**derivable,
  gated on the vLLM-0.19 capture question**" — a bounded, risk-first spike if/when
  spine-2 speed is wanted. Still no speed/lossless claim made now.
