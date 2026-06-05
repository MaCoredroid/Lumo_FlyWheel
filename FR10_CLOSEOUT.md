# FR10 Close-Out — No-Copy GDN Token-Tree Speculative Verifier

**Date:** 2026-06-05 · **Stack:** vLLM cu130-nightly (0.22) / DGX Spark GB10 · **Model:** Qwen3.6-27B-fp8 (48 GDN linear_attn + 16 full_attention layers, M-RoPE) · **Regime:** B=4, temp=0.6, top_p=0.95, mtp=5, spines=1, gpu_mem 0.88.

---

## TL;DR verdict

The no-copy GDN token-tree verifier is **concluded NO-GO on this stack**, on evidence, not fatigue. The tree **loses to native MTP-5** (1.77 vs 3.076 accepted/event; strict-win bootstrap CI [−1.67, −1.22], entirely negative). The cause is **diffuse per-layer numerical drift** in the live verify forward — *not* a fixable single bug, and *not* the suspected GDN shared-recurrent-state limit (contamination proven 0). The proven-lossless, contamination-free GDN tree kernel is **banked**.

The forward path where lossless+speed is physically favorable on GB10 is **copy-recurrent multi-spine** — designed, red-teamed, lossless-by-construction, gated only on bounded per-spine state-isolation engineering. See `FR10_MULTISPINE_STATE_ISOLATION_DESIGN.md`.

---

## Objective

Build a **lossless, CUDA-only, CUDA-graph-capturable** GDN/STree token-tree verify kernel for the Qwen3.6 GDN-hybrid that **beats native MTP-5** (E5 ≈ 3.076 accepted/event). Lossless first, then speed.

## Outcome (measured)

- Best no-copy tree: **accept/event ≈ 1.77** (3 measurement bases agree: /metrics, committer log, per-position summary).
- Native MTP-5 baseline (same 8 prompts): **3.076**.
- Superset gates on paired traces: **Gate 2 (path0==native) FAIL** (tree path0 1.54 vs native 3.08, first_diff event 3); **Gate 3 (total ≥ native) FAIL** (delta −1.31); **Gate 4 (strict-win, bootstrap CI lower-bound > 0) FAIL** (CI [−1.67, −1.22]). **Gate 1 (internal superset, winner ≥ path0) PASS** (viol=0).

## What is PROVEN GOOD (banked)

1. **GDN tree kernel byte-native** to the serial reference (scan-output replay gate, per-node out_vs_native ~0, max 1.5e-5, powered negative control 0.47). CUDA-graph capturable (FULL + PIECEWISE).
2. **Leaf contamination = 0.0** — full-tree spine GDN output == path0-only (state delta 3.8e-6). The kernel correctly isolates paths; this **refutes the STree shared-recurrent-state "wall"** for the kernel.
3. **Committer correct** — canonical multidraft accepts identically to native linear on the *same* logits (delta 1e-8); does not dilute the spine.
4. **Cross-step state handoff** proven byte-native (src_native gate: ssm 2.86e-6, conv 0.0, powered control).
5. **A real bug found and fixed**: vLLM assigns **flat arange RoPE positions** (`num_computed + query_pos`) to tree nodes; the branched caterpillar needs **depth-based** positions (siblings share depth). Fix = override spec-row positions to `num_computed-1 + tree_depth[node]` *after* `compute_slot_mapping` (KV slots stay flat). Recovered **0.84 → 1.86**.
6. Lossless-spec-decode **gate methodology** (internal hard superset + path0==native + strict-win bootstrap CI), wired to consume live logs.

## Hypotheses ELIMINATED (each by measurement or source-read, not guessing)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Drafter degraded | NO | GATE-D1 32/32; drafter == native MTP greedy |
| Committer dilution | NO | accepts identically on same logits, delta 1e-8 |
| temp/top_p not applied | NO | stock `rejection_sampler.py:135` constrains target before the tree branch |
| GDN shared-state contamination | NO | full-tree spine == path0-only, delta 0.0 |
| fp8 batch-nondeterminism | NO | BATCH_INVARIANT=1 did not lift depth-0 (got *worse*) |
| Measurement basis | NO | 3 bases agree at 1.77; "2.4–2.5" was a moving-window log artifact |
| Full-attn tree mask dropped | PARTIAL | FLASH_ATTN drops `qq_bias`; TREE_ATTN applies it but **regressed** (mask was already correct; not the deeper lever) |
| Flat RoPE positions | **YES (fixed)** | depth-position remap recovered 0.84→1.86 |

## The decisive diagnostic (conclusion driver)

A contradiction remained: every component proven correct, yet logits degraded. Resolved by a **per-layer spine hidden-state comparison, tree TREE_ATTN vs native MTP-5, at decode-event-0** (shared prefill h0; input embeddings match **exactly 0.0**; 12 token+position-aligned spine rows):

```
GDN layer 0   max_abs 0.0156   ← already nonzero right after embeddings match exactly
layer 2       0.094
layers 44–51  0.6 – 1.25
layer 62      3.40
full_attn 63  53.0
final_norm    5.0
```

**No clean single-layer locus.** Divergence is nonzero at layer 0 and **compounds monotonically**. Conclusion: a tiny pervasive per-layer drift (FR10 tree GDN vs the stock GDN the native run uses, a few % at layer 0) **accumulates over 64 layers** into the logit gap → the acceptance deficit. The byte-native proofs held on captured/good-event inputs; the live spine carries a small **diffuse** drift across all layers that no single proof covered, and there is no fixable seam.

## Why no-copy is closed and multi-spine is the forward path

- **No-copy**: diffuse accumulation, no locus. Of the 3 possible per-layer outcomes, 2 would have reopened it; it came back diffuse → no cheap fix.
- **Multi-spine (recommended)**: each spine is an isolated **linear** GDN row cloned from the native pre-spec state → spine A **is** native MTP-5 by construction → sidesteps the diffuse tree-verify drift entirely. Lossless by construction.
  - **GB10 cost is favorable**: decode is **weight-bandwidth-bound** (~27 GB fp8 weights streamed per forward); an extra spine's recurrent state is tens of MB (<1%). The "GDN linear verify tax" is real on compute-bound GPUs but negligible here. Spine-2 needs only ~0.03 extra accepted tokens/event to beat native.
  - **Blocker = per-spine state isolation engineering.** vLLM 0.22 does **not** provide it for free (verified by source-read: `batch_expansion` removed; GDN/Mamba state allocated per-request via `block_table_tensor[mask, :num_spec+1]`, not per-spine). FR9's co-residency problem persists on 0.22.
  - **Design** (`FR10_MULTISPINE_STATE_ISOLATION_DESIGN.md`): allocate per-spine physical Mamba rows → clone canonical pre-spec state into each (stock GDN copy funcs) → isolated linear GDN forward per spine → commit winner's final state to the canonical row. Scope: **2–4 day fixed-shape prototype; 1–2 week robust graph-safe B4** (crosses Mamba allocation + GDN metadata + commit semantics; not patcher-trivial).

## Honest meta-state

A *cheap* lossless tree/multi-spine win over native MTP-5 does not exist on vLLM 0.19/0.22 + GB10: no-copy tree → diffuse drift; copy multi-spine → needs the state-isolation build. **spines=1 (native MTP-5) is the clean lossless default.** The decision is binary and informed: invest the bounded multi-spine isolation engineering (favorable GB10 payoff), or close at spines=1 and bank the proven kernel + design.

## SOTA / forward references (2026)

- **Component-Aware Self-Speculative Decoding in Hybrid LMs** (arXiv 2605.01106) — spec-decode tailored to softmax+SSM/GDN hybrids; relevant to the verify-cost structure.
- **Gated DeltaNet-2** (NVIDIA, arXiv 2605.22791) — newer GDN variant (decoupled erase/write); architecture is evolving.
- **STree** (arXiv 2505.14969) — the token-tree-on-linear-attention algebra we proved lossless at the kernel level (`A_tree = L·A_log`).
- Inherent constraint: linear attention's fixed-size recurrent state makes verify **sequential** (can't parallelize across draft positions like softmax attention) — the structural reason GDN spec-decode is hard, manifested here as *correctness* (diffuse drift) for the tree and as *cost* (negligible on bandwidth-bound GB10) for multi-spine.

## Key commits

`bfc596d7`/`e6be5306` depth-RoPE position fix · scan-output gate + `fr10_scan_output_replay_gate.py` · `0a7ebc94`/`5f286033`/`76f946cb`/`4daf2475` cost-gate verdict + per-layer evidence · `49f2280b` 0.22 isolation scope · `f63276e8` multi-spine isolation design. Branch: `fr10-gdn-tree-kernel`.

## Recommendation

**GO (de-risked): build the 2–4 day fixed-shape spine-2 prototype, measure accept/event + decode TPS vs E5, then decide the robust version** — the one route where lossless+speed is physically favorable on GB10. Else close at spines=1 with the kernel + design banked.
