# #58718 second-GB10 measurement comment (funded item L) — v2 (Codex replacement verbatim; agent draft NO-GO on '90/90' and the 2.24B-element claim; AWAITING MARK GO — one ordinary PR comment, no branch)
> Codex L_review.md: ratios recomputed from raw p50s; sm120 baseline mapping verified (batch_invariant_configs.py:492–524 main; head :719–754 selects sm121); isolated BLOCK_K 128→64 covers 2,201,511,168 elements on 12 shapes; build-scoped question, not a policy claim.

---

Second GB10 measurement at `b40edf4`: both the small-M win and large-M penalty reproduce. Mean per-shape tuned/default p50 ratios over 15 BF16 `weight.t()` GEMMs (both timing orders agree within the stated 3% direction threshold):
| M | 1 | 32 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| tuned/default | 0.721 | 0.754 | 0.929 | 1.027 | 1.263 | 1.202 |
The large-M penalty is larger here: torch 2.13.0+cu132, Triton 3.7.1, ptxas 13.1, driver 590.48.01; your report lists CUDA 13.0. Against main’s sm120 table, normalized ratios are 0.89–0.91 at measured M≤256 and ~1.16 at M=1024/2048 (direct timing ratios ~1.15). Is that the intended tradeoff? All 15 shapes pass the M-invariance hash check. Isolated BLOCK_K 128→64 swaps on the 12 affected shapes changed zero of 2.20 billion BF16 elements at M={1,2048}, across rand/randn inputs; split-K and cuBLAS controls did change outputs. This is build-specific, not a reduction-order guarantee. Does your build differ, or would a validated default-tile fallback at large M be worth exploring? [Evidence](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/db66544e84131c7d009053dd1bca8923ce5c270b/results/upstream/58718).
*Investigated with AI assistance; these are isolated-GEMM measurements, not serving results.*
