# Draft-Vocab Reduction Brief — GB10 Qwen3.6-27B MTP Drafter

Tags: [LIT]=literature-backed (cited) · [OURS]=our measured data · [INF]=my inference. All numbers assume the pipeline facts you gave (fp8 head 1.27GB/248,320 rows ≈ 5.11MB per 1k rows; 273GB/s; graph-captured drafter; verifier full-head → lossless by construction).

---

## 0. Two constraints that decide everything (read first)

**(a) Graph-capture shape-stability is the gating hardware constraint, not accept rate.** The drafter is CUDA-graph-captured, so any dynamic token universe must be a **fixed-shape dense buffer of size K_max** whose *contents* change per event but whose *shape* does not. This single fact re-ranks the whole field: it kills data-dependent variable-size routed gathers and confidence-fallback branches, and favors NanoSpec's exact design — "gathered on a separate copy CUDA stream, packed into a pre-allocated dense buffer" [LIT: NanoSpec arXiv:2605.26444 §3.3].

**(b) The drafter-span delta may not convert to e2e wall.** Your fresh numbers show full-head drafter 94.9ms → gather-64k 56.3ms (Δ38.6ms) [OURS]. But pure-HBM predicts only Δ13.8ms for 248k→64k rows (3.77GB×saving /273GB/s) [INF]. The measured Δ is ~3× the HBM prediction, which means one of: the draft lm_head is partly **compute-bound at tree width** (it's a GEMM [M≈21 × d]@[d × V], not a GEMV — cutting V cuts flops too, so capping helps *more* than bytes predict) [INF]; **or** part of that 38.6ms is a stream-drain span artifact, exactly the failure mode in our prior finding that "drafter propose = 0.3ms post-loop; dfwd span = stream-drain artifact; verify −120..150 is the gap." **Do not bank the 38.6ms until it's shown on `measured_tps_fullstep_wall`.** This is Exp 0 and it's free.

---

## 1. What the literature says about draft-vocab reduction + the fat-draft-tail gap

**The static-frequency family is what we already built, and it is the weakest modern option.** FR-Spec [LIT: arXiv:2502.14856, ACL 2025], VocabTrim [LIT: arXiv:2506.22694], and Balancing-Coverage [LIT: arXiv:2603.05210] all take a static top-K by **corpus** frequency, keep the verifier full-head (lossless), and treat corpus-occurrence as a proxy for draft-relevance. The field's accept knee is ~25% of vocab; for our 248k that's ~62k, which is exactly why our gather-64k is near-lossless (5.552/5.733 = 96.8% [OURS], matching FR-Spec's 64k=97.7% [LIT]). SGLang's FR-Spec uses a **gather** (freq .pt), never a contiguous id-slice, because BPE id-order ≈ freq-order only approximately — this confirms our contig-128k is a confounded read and **contiguous slice is not a valid frequency cap** [LIT: SGLang docs + thunlp/FR-Spec].

**Does anyone correct for the fat-draft-tail? No one corrects for the exact effect; three papers correct in the right *direction*.**
- **Nobody quantifies "committed coverage (99.74%) overestimates per-position draft-time coverage (~95%), compounding per depth at temp>0."** This is a genuine gap in the literature and is our contribution [LIT dead-ends, all five searches agree].
- The **formal backing** exists: acceptance ceiling α ≤ Σ_{v∈V_tr} p(v) (target mass inside the kept slice), independent of drafter quality; at greedy it collapses to 0 when the target argmax falls outside the slice [LIT: SlimSpec arXiv:2605.10453 §3.1]. The rejection mass is the "drafter kernel" k(x)=max(0,q−p), α=1−‖k‖₁ [LIT: RDK arXiv:2506.03206]. Per-depth compounding is E[L]=Σ_d Π_k α_k [LIT: OPT-Tree TACL, PosS arXiv:2506.03566] — so a 0.95 vs 0.9974 per-position gap enters multiplicatively across the depth-5 spine. (Caveat: PosS studied autoregressive EAGLE/HASS, not MTP; "MTP decays faster" was flagged unsourced.)
- **The temp=0 vs temp=0.6 crux directly explains our greedy-vs-sampled discrepancy.** llama.cpp #25187 trims **our exact geometry** (native MTP, 248,320 rows → top-32,768) and gets **byte-identical output and zero accept loss — but only at temp=0** (215/251 accepted, mean len 3.56, both arms) [LIT, verified against the live issue]. At greedy, committed-argmax ≈ what the drafter needs so coverage looks perfect; at temp 0.6 the sampler reaches into the tail and the fat rejected tail bites. This is the cleanest external statement of *why* our per-position coverage < committed coverage.
- **Directional correction — three independent votes for generation-based over corpus-based:** VocabTrim's calibration ablation (target-gen > draft-gen > raw-text; code task has the *worst* drop, 5.6%, on general-english calibration) [LIT]; Balancing-Coverage computes coverage over assistant-response tokens only [LIT]; DynaSpec/NanoSpec use context+generation [LIT]. NMT precedent is a decade old (Domhan NAACL 2022: static alignment coverage looks fine on aggregate recall but fails on the tail that matters; cure = context-conditioned neural selection) [LIT].
- **Nobody builds the subset from the DRAFT's own PROPOSAL distribution** (including rejected continuations) — the precise fat-tail correction. EvoSpec adapts on *target* misses (committed side) [LIT: arXiv:2605.27390]; RDK *redistributes* mass rather than recounts [LIT]. This specific recipe is unbuilt → our opening.
- **Our fp8 block-scale-aligned gather has no external prior art** — FR-Spec/VocabTrim are bf16; llama.cpp tested Q6_K and explicitly excludes block-alignment. That piece is ours [LIT dead-ends].

**Important correction to propagate accurately:** RDK does **not** sample OOV tokens from the verifier's full-vocab distribution (that was a confabulated summary). It reshapes the **drafter's own** proposal mass toward high-overlap in-subset tokens via a precomputed token-affinity matrix M, with **zero extra forward latency**, and its reported result is **held acceptance-rate** (≈26.7% under >75% pruning), **not** a 1.5–2× speedup [LIT: verified against arXiv:2506.03206 PDF].

---

## 2. Dynamic-K schemes ranked for OUR setting

Cost-gate all of them first with SlimSpec's formula (paper, no GPU): ρ_TPS = ρ_τ·(1+κ)/(1+ν·κ), where κ=T_head_full/T_nonhead, ν=cheap-head latency/full-head latency, ρ_τ=accept preservation. A cheap head wins iff ρ_τ > (1+νκ)/(1+κ) [LIT: SlimSpec §3.2]. Our temp 0.6 and huge 27B verify forward both *lower* κ (hurt the case) — which is why Exp 0/1 must run before any build.

### #1 — Context-union fixed-K_max vocab (NanoSpec-style). RECOMMENDED.
Active set per event = union of {prompt ids, prefill top-K_pre, **all draft-tree tokens from prior events incl. rejected**, verify top-K_ver}, recency-windowed; gathered into a **fixed K_max dense fp8 buffer** on a copy stream.
- **Why #1:** it is the only scheme that *is* the fat-draft-tail correction — unioning all draft-tree tokens (accepted or not) captures the fatter rejected tail directly, and NanoSpec's ablation proves the generation-derived set contributes *more* than prompt context (351 vs 313 tok/s isolated) [LIT: NanoSpec Table 5]. It **matches full-vocab accept length at <3k tokens** (4.25 = full on EAGLE-3), beating FR-Spec-32k (3.96) and DynaSpec-27k (4.08), with LM-head 2.330ms→0.237ms (9.8×) [LIT: NanoSpec Tables 1,4]. On code specifically, context-adaptive kills the static loss (DynaSpec: 3.31→3.86) — that's our workload [LIT: DynaSpec arXiv:2510.13847].
- **Fixed shape → graph-safe** by construction (pre-allocated dense buffer) [LIT: NanoSpec §3.3]. Our fp8 block-scale-aligned gather infra already exists.
- **GB10 cost:** K_max=4096 → 4096×5.11MB/1k ≈ 21MB/read ×4 iters = **84MB/step vs 5.08GB full** (~60× fewer head bytes) [INF from your 5.1MB/1k]. Free add-on: DynaSpec's **position-aware budget** — larger K for MTP iter-0, decaying to iter-3, since accept loss compounds per depth [LIT: DynaSpec §4.2].
- **The one real risk [INF, must measure]:** NanoSpec's "gather fully hidden on copy stream" is on H20 (far higher BW). On GB10's 273GB/s unified memory the copy competes with compute for the *same* bandwidth, so the gather may not hide. This is Exp 4's gate.

### #2 — Session-accumulated draft-proposal-frequency subset (rejection-feedback online adaptation). NOVEL, our contribution.
Accumulate **draft-proposal** frequency (incl. rejected) across the session, periodically rebuild the fixed K_max set (EvoSpec-style expansion on target-miss as a safety valve).
- **Why #2:** this is the unbuilt recipe from §1 — no one counts the draft's own proposals; EvoSpec only corrects the committed side [LIT dead-ends]. It directly targets the effect we measured. Rebuild is periodic (not per-step) → graph-safe between rebuilds, cheap.
- **Relationship to #1:** this is the session-persistent version of #1's event-local window; **they compose** (union feeds the counter). Practically, ship #1 first and let #2 be the counter that seeds the union. Risk: adaptation lag on topic shift within a long SWE session [INF].

### #3 — Uncertainty-triggered full-head fallback (our original idea). DEPRIORITIZE.
Small head, read max-prob, fall back to full 248k head on low confidence.
- **Verdict: not a published vocab-head technique** (CALM is the layer-axis precedent; CSV-Decode arXiv:2511.21702 is a geometric-certificate fallback-to-full) [LIT, verified]. Two structural strikes for *us*: (1) the fallback fraction f pays the **full 1.27GB read**, so expected bytes = (1−f)·small + f·full — unless f is tiny it loses to a persistent small head [INF]; (2) the **data-dependent branch breaks CUDA-graph capture** — our hard constraint from §0(a). Rank last.

### Off-axis alternative that may dominate all three — flag, don't skip.
**Dense low-rank head** reads r/d bytes *every* step, preserves *full vocab* (zero coverage gap, ρ_τ≈0.99 at r=d/8), is a dense GEMM (graph-safe, no gather) — SlimSpec explicitly argues dense low-rank beats routed gathers on GPU, which matches our contig-128k gather pain (72.1ms) [LIT: SlimSpec §2, arXiv:2605.10453]. SlimSpec needs a retrained head (conflicts with training-free MTP), **but SVD-Softmax [LIT: NeurIPS 2017] is training-free** — SVD our existing MTP head once, preview at width d/8 over all rows, refine top-N. Caveat: the preview still touches all V rows (only width shrinks) and top-K is approximate. If Exp 4 shows the gather is HBM-bound on GB10, **this is the fallback** because it converts bytes deterministically with no data-dependent shape.

Stackable accelerant on any of the above: **RDK** to lift accept at a fixed small K with zero added read (drafter-mass redistribution via precomputed M) — graph-safe, offline preprocessing [LIT: arXiv:2506.03206].

---

## 3. Concrete next experiments, cheapest-first, each with a gate

**Exp 0 — Does the head-read saving convert to e2e? (free, gating, do first).**
A/B existing full-head vs existing gather-64k on `measured_tps_fullstep_wall`, clean run, eps-matched, same SWE-Verified subset, temp 0.6, **graph mode**, same-seed. No new code (both modes exist).
- **Gate:** gather-64k e2e TPS beats full by more than the ρ_τ=0.968 accept drag ⇒ head read is on the critical path, proceed. If not ⇒ the 38.6ms drafter-span is drained/hidden; **STOP the draft-vocab line, redirect to verify row-work** (the standing campaign). This is the decisive test of the §0(b) caveat.

**Exp 1 — SlimSpec cost-gate arithmetic (paper, concurrent with Exp 0).**
Plug measured T_head, T_nonhead, ν, and temp-0.6 κ into ρ_TPS=ρ_τ(1+κ)/(1+νκ).
- **Gate:** ρ_τ > (1+νκ)/(1+κ) for a plausible cheap-head ν. If the inequality fails at ν→0 (ideal head), no vocab scheme can win and we stop before building [LIT: SlimSpec §3.2]. Also resolves the §0(b) 38.6ms-vs-13.8ms discrepancy by pinning whether the draft head is compute- or HBM-bound at tree width.

**Exp 2 — Context-union coverage on OUR traces (offline, no serving).**
From captured SWE traces, build the union set (prompt + all draft-tree tokens incl. rejected + verify top-K) and measure **per-position draft-time coverage** of a fixed K_max=4096 union vs current static gather-64k, at temp 0.6.
- **Gate:** union coverage ≥ static-64k coverage at ≤1/15 the rows (the NanoSpec claim, re-tested on our data/temp). Pass ⇒ the accept ceiling α≤Σp(v) is higher for the union ⇒ green-light Exp 4.

**Exp 3 — RDK redistribution on gather-64k (offline accept sim).**
Precompute token-affinity M from an SWE/agent-log corpus; redistribute drafter mass; simulate accept vs plain TLI gather.
- **Gate:** accept recovery toward full (5.552 → toward 5.733) at zero added read, graph-safe. Cheap upside independent of the dynamic-vocab build.

**Exp 4 — Live context-union fixed-K_max with copy-stream gather (GPU, serialized, only if Exp 0+2 pass).**
Wire K_max=4096 union + copy-stream gather into the captured drafter graph.
- **Gates (all three):** (a) **lossless** — in-process same-boot byte-identical vs full-head verify, temp 0.6, same-seed (per our no-cross-boot rule); (b) **e2e** — beats gather-64k on `measured_tps_fullstep_wall`, eps-matched; (c) **gather hidden** — measured gather ms is a small fraction of the drafter step (tests whether NanoSpec's copy-stream hiding survives GB10's 273GB/s). If (c) fails, pivot to the training-free SVD-Softmax dense low-rank head (deterministic bytes, no gather).

**Bottom line:** The literature validates our design and confirms our fat-draft-tail finding is novel and correctly directional; the modern frontier says static frequency capping (what we built) is the weakest option and context+generation-derived union (NanoSpec) is the strongest, with our draft-proposal-recount as an unbuilt extension. But two of our own constraints dominate the paper numbers: graph-capture forces a fixed-K_max dense buffer (rules out fallback-branch and variable routed gather), and the drafter-span may be a stream-drain artifact — so **Exp 0 (free, e2e) gates the entire line** before any build, and every downstream gate is `measured_tps_fullstep_wall`, not the drafter-span delta.
