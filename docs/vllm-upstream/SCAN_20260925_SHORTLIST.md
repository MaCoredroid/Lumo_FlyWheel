# Scan refresh 2026-09-25 — reconciled shortlist (Codex + agent, independent; window Sep 17–25)
Overlap: #58718, #57908, #57605 appear in both lists. Codex-only: #58400, #58737. Agent-only: #58624/#58681, #57266/#57278.

| # | Item | What we supply | Cost | Engagement | In both? |
|---|---|---|---|---|---|
| 1 | #58400 (njhill) V2 one-token prompt tails on FULL graphs | model-free state oracle: committed conv/SSM state vs eager with poisoned rejected slots + fresh-prompt control (P8/P9 playbook) | CPU prep + GPU ≤1 day | high (maintainer-owned; njhill just merged #58434/#58462 in this region) | Codex #1 |
| 2 | #57908 (cbertucci33) CoW for aligned prefix-cache snapshots | discrimination matrix of its two new manager tests at merge-base vs head; reachability trace | CPU only | low (no reviewer, author unproven) | both |
| 3 | #58718 (hclsys) GB10 batch-invariant matmul table | second-GB10 timing + bitwise M-invariance at PR head; compiled-path dispatch evidence | GPU ≤½ day | high (author + LioEinaudi replying within 30 min) | both (agent #1) |
| 4 | #57605 (jsolman) Mamba align-mode lookahead allocation | two-case model-free MambaManager test (boundary±1, lookahead 0/non-0; running-state column) discriminating merge-base vs head; check composition with #57050 | CPU only | medium-low | both |
| 5 | P9 test → main (plan change) | if main now carries idx_mapping (#58434/#58462 merged), turn tests/v1/worker/test_mamba_aligned_state_indices.py into a test-only PR on main in njhill's active region | zero-GPU feasibility check first; then CPU/GPU ≤½ day | medium (our own PR; needs a maintainer) | agent plan-change (b) |
Deferred: #58737 fused align gather (overlaps active #52297); #58624/#58681 DeepGEMM alignment (unverified whether our pinned model hits grouped-MoE path); #57266/#57278 conv_ssm_forward crash (no author/maintainer activity); #56792 (unchanged: SGLang cross-ref only).

Direction signals: prefer V2 (ZJY0516); keep diffs <100 LOC (yewentao256); LucasWilkinson RFC #58638 sets hybrid KV-grouping direction for spec drafters (Qwen3.6 + DFlash named) — independent geometry PRs risk being superseded; the align-seed cluster still has no maintainer verdict and now shares our P8 oracle (Karl0007/vllm:oracle/53142-adapted-arms).


**Correction 2026-09-26 (item N):** main @379e9a1ea8 does NOT carry idx_mapping in compute_aligned_state_indices (mamba_utils.py:1092-1096); #58434 changed one is_decode expression and #58462 a dummy-tensor dtype. Both scans misread those merges. A reduced, order-independent 4-case port of P9 is feasible as an own test-only PR (see N_P9MAIN_REPORT_agent.md).
