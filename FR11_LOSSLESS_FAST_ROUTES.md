# FR11 — Lossless + Fast Route Through the Tree-on-GDN Divergence

**Date:** 2026-06-05 · Deep-research workflow (18 agents: 5 map + 12 routes + 6 adversarial cost-gates + source-verified synthesis + adversarial review). Grounded in our code + vLLM 0.22 source + 2026 SOTA. Companion to `FR11_CLOSEOUT.md` (why no-copy is banked) and `FR10_MULTISPINE_STATE_ISOLATION_DESIGN.md` (the build).

---

## Verdict: ONE candidate survives — copy-recurrent multi-spine — as a **CONDITIONAL GO behind two gates**, not a clean win

Of 12 generated routes, **2 survived cost-gating** and they're the same substrate: **copy-recurrent multi-spine**, fixed-shape K=2 prototype → K=4 throughput optimum, with **spine-A ≡ native MTP-5 on a native-layout linear chain**. This is the Option-B design already scoped in `FR10_MULTISPINE_STATE_ISOLATION_DESIGN.md`. Everything else is killed (table below).

The honest framing is **conditional GO**, for three reasons the adversarial pass verified at source:

### 1. NOT patcher-cheap — bounded state-isolation surgery
The stock per-row GDN kernel (`fused_sigmoid_gating.py:24-178`) already runs each `(req, spine)` as an independent linear chain → **the compute path is free**. But vLLM 0.22 does **not** give per-spine isolation through metadata alone (`FR10_STATUS.md:627`): spines co-reside in one request's Mamba row + one `mamba_state_idx[req]`. Physical isolation needs: block-pool allocation-tail surgery (`kv_cache_interface.py:591`, init-time, feeds block-pool + CUDA-graph capture shapes) + widening **five `num_spec`-sized graph buffers** (`gdn_attn.py:122-143`) pre-capture + a commit-source redirect. **Scope: 2-4 day fixed-shape K=2 prototype; 1-2 week robust dynamic-B4.**

### 2. NOT lossless-by-construction — a measured open risk with a hard gate
Spine-A is *mathematically* a native linear chain (per-token recurrence = native). **But that's not the bit-identity this stack demands.** Adding a co-resident spine-B changes the GDN kernel's launch row-count `N`; **GDN is not batch-invariant** (vLLM #42960, OPEN through 0.22 — verified `supports_batch_invariance()==False`, no GDN override at `backend.py:240`); reduction order shifts. This is the **same row-count→non-associative-reduction mechanism that drives the no-copy drift** (a 48-row K=2 co-resident launch vs a 24-row native launch). And **FR9 already measured it**: copy-based independent-rows `spines=2` was **borderline-not-cleanly-lossless** (TV(s2,s1a)=0.0283 > 0.0188 floor; 831 suppressed-superset events). So losslessness needs a **co-resident spine-A parity gate run-and-passed inside the prototype** — it can still fail.

### 3. Prize is modest + entirely drafter-gated + not yet demonstrated
Net throughput is **modeled +7-12% realistic / ~+28% idealized — NOT ~2x**. **No isolated-spine run has beaten 3.076 yet.** K *identical* spines = 3.076 exactly = **zero gain**, +9%/spine traffic. The whole payoff hinges on one **unmeasured** number: β, the realized 2nd-candidate accept rate of a *diverse* drafter that doesn't exist yet.

> The caterpillar table (K=2→3.56, K=4→4.29) is an **analytic model of an unmeasured β**, not a measurement. On bandwidth-bound GB10, per-spine state *traffic* is ~9%/spine (sub-dominant), so K is a real-but-not-fatal cost lever; K=4 is the modeled interior optimum.

---

## The cheapest next step: a **one-GPU-run β experiment** (de-risk the prize before any surgery)

The entire prize reduces to one unknown — β. The existing native captures have **zero logits** (only a 2.2KB summary + 3.1KB metrics), so a zero-GPU replay is impossible. The cheap decisive experiment:

1. Run native MTP-5 **once** on the same 8 prompts / B=4 / temp0.6 / top_p0.95 with **`logprobs≥2`** enabled (a sampler flag — zero isolation code, zero tree forward, zero drift).
2. Offline, compute the rejection-sampler accept prob of the **2nd-ranked token at each request's FIRST-REJECT depth** (the adaptive low-confidence position where marginal accept lives — *not* static depth-1, which `FR10_STATUS.md:508` proved is the worst spot). For vLLM's one-hot MTP draft, β = `p_target[runner-up]`.
3. Feed measured β into the caterpillar model.

**Cost:** ~0.5 day + one short decode. **Bar:** β ≥ ~0.3 → **GO** on the K=2 prototype (modeled E[accepted] ≥ ~3.4); β ≈ 0 → **STOP**, ship `spines=1`.

**Scope limit (important):** this de-risks the **prize** axis only. It does **not** retire the **losslessness gate** (the #42960 co-residency perturbation) — that is paid inside the 2-4 day prototype, where it can still fail and force a fallback (separate native-M launch = re-stream weights; or a powered chi-square GOF accepting a bounded perturbation). Two independent unknowns.

---

## Routes killed (and why)

| Route | Verdict | What kills it |
|---|---|---|
| **Copy-recurrent multi-spine (K=2→K=4)** | **CONDITIONAL GO** | the winner; gated on β + co-resident parity; bounded surgery |
| Diverse-spine drafter (target top-2) | = winner's drafter half | not a distinct build — the near-free drafter layer on the isolation substrate |
| SuffixDecoding diverse spine | FUTURE | highest modeled ceiling (+28% on agentic) but needs the substrate + its gate first |
| **Layer-class hybrid** (tree-exact full-attn + isolated GDN) | **NO-GO** | full-attn is **not** tree-exact in fp8/branched (amp 1.072×/layer); forcing it perfect leaves ~33% drift; and it's *more* surgery |
| Bit-identical separate native-M launch | **NO-GO (slow)** | re-streams in_proj/out_proj fp8 weights twice → **1.5-1.8× slower** (this is also the parity-gate-fail fallback — a real cost) |
| M-invariant batch-invariance on projections | **NO-GO (slow)** | BI-tax on the dominant weight stream; `BATCH_INVARIANT=1 did not lift the drift` (it's row-count, not within-batch order) |
| Leaf-only acceptance, no isolation | **NO-GO (lossy)** | verifies the leaf against the spine's own logits = the shared-state error; lossless leaf accept *requires* the isolated chain it refuses to build |

---

## Literature stance (unchanged, confirms the framing)
No theoretical no-go. STree (2505.14969) is diagonal-Mamba2-only — no exactness claim for the rank-1 `(I−βkkᵀ)` gated-delta term; GatedDeltaNet-2 is *more* non-diagonal; Component-Aware (2605.01106) names STree+drafting as future work only. **Per-candidate isolated recurrent state (copy-recurrent multi-spine) is the only known *construction* with a path to lossless for non-diagonal GDN** — but "construction exists" ≠ "bit-identity proven on a non-batch-invariant kernel."

## Bottom line
There is exactly **one** lossless+fast candidate: fixed-shape copy-recurrent multi-spine. It is **plausibly lossless** (spine-A is native on a linear layout, sidestepping the *branched-tree* component of the drift) but **not proven** (co-residency perturbs the non-batch-invariant GDN launch; FR9 measured this borderline), it is **bounded surgery not patcher-cheap**, and its **modest modeled prize (+7-12%) is fully gated on an unmeasured β**. **Cost-gate the prize with the one-GPU-run β experiment before spending a GPU-day on isolation; then run-and-pass the co-resident parity gate inside the prototype before trusting any throughput number.** Else `spines=1` native MTP-5 remains the clean lossless default.
