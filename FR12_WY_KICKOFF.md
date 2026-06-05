# FR-12 — WY-Tree Kernel (STree-no-copy): make no-copy GDN tree verify EXACT + FAST

**Branch:** `fr12-wy-tree-kernel` · **Driver:** codex gpt-5.5-high in tmux `codex_fr11` · **Red-team + loop:** Claude (Opus), 10-min loop. **ONE GPU — strictly serial GPU jobs.**

## The single path (do not pivot)
Build on STree's **no-copy single-shared-state** idea, but fix the exact thing STree can't do: STree's `A_tree = L·A_log` is **diagonal-only**; our Qwen3.6 Gated DeltaNet transition is `g_t·(I − β_t k_t k_tᵀ)` — a scalar gate (commutes, easy) times a **rank-1 non-diagonal reflector** (the part STree drops). Make the no-copy tree verify EXACT for that rank-1 term via the **compact-WY** structure, and FAST via its low-rank form.

## What is already validated (build on it — scripts persist in `output/gdn_novel_research/`)
1. **Compact-WY identity CONFIRMED numerically:** `∏_{t}(I − β_t k_t k_tᵀ) = I − K T Kᵀ`, K=[k_1..k_n], T∈R^{n×n} from a triangular recurrence (`wy_foundation_validate.py`, `wy_gated_delta_foundation.py`). The gates factor out; the rank-1 product is the only obstruction and it is low-rank.
2. **KEY: vLLM's FLA delta kernel ALREADY computes the WY T-matrix intra-chunk** — `solve_tril` returns the WY T-factor of the rank-1 reflector product (the "UT-transform" at `fla/ops/chunk.py:40`). **So the WY-tree kernel is a GENERALIZATION of existing, proven code**, not a from-scratch build: extend the chunked-delta kernel's **inter-chunk** accumulation from *causal* to **tree-ancestry**. Start from `wy_tree_fused_probe.py` + the real FLA `chunk.py`/`chunk_delta_h.py`/`fused_recurrent.py`.

## HARD CONSTRAINTS (read every line)
- **ONE GPU JOB AT A TIME.** There is one GB10. NEVER run concurrent `docker run --gpus all`. Serialize every GPU experiment; kill the container when done (`docker kill`); `torch.cuda.empty_cache()` between runs. No contamination of correctness OR timing.
- **You MAY REPLACE any part of vLLM** — write the kernel and swap the GDN verify/backend path. Don't patch *around* vLLM invariants (that's what killed Round-F/FR9). Read LIVE source first: `/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/fla/ops/` + `/tmp/vllm_live_019`, and the consumer in `models/qwen3_next.py` / `v1/attention/backends/gdn_attn.py`.
- **READ CODE + PRIOR COMMITS to learn from our errors.** `git show`/closeouts for: FR9 co-residency perturbation (#42960) + FR9 isolated-forward CUDA-illegal-access (`a80e9554`); Round-F `InputBatch.condense()` clone-death (`5bfdcc96`/`d4806933`); FR10 dense tree solve 243ms-too-slow; FR10/FR11 byte-exact-vs-MTP5 = WRONG bar. Do not repeat any of these.
- **Ablation = the only success gate.** (a) **LOSSLESS = vs NO-MTP** native decode (the true target dist; NOT vs the MTP-5 baseline — MTP-5 itself diverges ~6e-5 from no-MTP). Statistically within the no-MTP run-to-run self-noise floor, byte-exact-greedy where achievable. (b) **SUPERSET to naive MTP-5** = accept/event ≥ **3.076** (never fewer than MTP-5). Native MTP-5 same-harness reference exists: `output/fr10_native_mtp5_same8_*`. Use `scripts/fr10_quick_decode_tps_probe.py` (`--modes naive_mtp tree_mtp` + a no-MTP mode), `--require-tree-engagement`, metrics-OFF.
- **DO NOT close the investigation (pass or fail) without asking the user.** Surface numbers; the user decides closeout. **Do NOT pivot to batch-invariant-scan / multi-spine** without asking.
- **Commit + push every step** to `fr12-wy-tree-kernel`; numbers go in committed docs (`output/` is gitignored).

## Phased plan
1. **Confirm the foundation** (cheap, mostly CPU + one short GPU): re-run/extend `wy_*` — the compact-WY identity, the **tree-ancestry T recurrence** (node inherits parent's T, appends its reflector), and verify the FLA `solve_tril` output IS that WY T-factor on a real chunk. Each node's output must equal the serial per-path recurrence (oracle: `scripts/fr10_gdn_tree_algebra_reference.py`, vLLM `recurrent_gated_delta_rule.py`) to <1e-8.
2. **Build the WY-tree Triton kernel** (one GPU run at a time): generalize the FLA chunked-delta **inter-chunk** step to tree-ancestry using the compact-WY low-rank form. Validate on GB10: EXACT vs serial reference (≤ bf16-ULP floor) AND **faster-or-equal to FLA's ~135 µs/event** (the dead dense solve was 243 µs). CUDA-graph capturable.
3. **Replace the vLLM verify path** with the WY-tree kernel (flag-gated).
4. **Ablate end-to-end** (serial GPU): lossless vs no-MTP + superset vs MTP-5 (≥3.076), B=4 temp0.6 top_p0.95 mtp5, metrics-OFF, tree-engagement asserted. **Report numbers; ASK before any closeout.**

## Definition of done
Steps 1-2 give a kernel that is exact + ≤135µs on GB10; step 4 gives the two ablation numbers. Bring both to the user before declaring anything.
