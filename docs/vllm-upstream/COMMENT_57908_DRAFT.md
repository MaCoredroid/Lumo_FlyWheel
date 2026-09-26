# #57908 review comment (funded item K) — v2 (Codex replacement verbatim; agent draft NO-GO on "PREMISE UNREACHABLE" framing; AWAITING MARK GO — on GO: push k-57908-cow-premise @ea240654f0758e82abeb18a041aacf83f00779c5 to the fork, verify compare link, resolve {{BRANCH_LINK}}, post ONE ordinary PR comment)
> **POSTED 2026-09-26T20:05:32Z (Mark GO):** https://github.com/vllm-project/vllm/pull/57908#issuecomment-5849447492; branch k-57908-cow-premise @ea240654f0 pushed
> Codex K_review.md: independent CPU sweep confirms exactly the 7 head-only failures (base 5F/755P; head 10F/750P); premise probe base/main 12P, head 9P/3F; should-fix applied: M4 docstring replaced (branch ea240654f0758e82abeb18a041aacf83f00779c5, prose only). Framing: test incompatibility, not corruption; question which path writes the shared snapshot in place. Evidence pinned a5b9fa53 (verified).
> {{BRANCH_LINK}} = https://github.com/vllm-project/vllm/compare/9317100d2f837c5584cbf8504ddbf4ddc688bac3...MaCoredroid:vllm:k-57908-cow-premise

---

Seven existing tests pass at merge base `0ff0477` but fail at head `9317100` in my CPU rerun of `tests/v1/core` (serving/E2E files excluded). These include both parameters of `test_dcp_partial_hit_resumes_on_replicated_mamba_snapshot`, whose assertion rejects copying the aligned Mamba snapshot. These failures need reconciliation with the intended ownership change; they do not establish state corruption.

For a correctly seeded full aligned hit `C=kB`, V1's `preprocess_mamba` selects source `k-1` and destination `ceil((C+q)/B)-1 >= k` for `q>0`. V2's fused preprocess uses the same destination rule. The bounded producer probe also migrates. Which supported path writes the shared aligned snapshot in place despite this preprocessing?

Your two new tests pass at head; at base they stop at reservation-count assertions, before reaching ownership checks. I can offer a 12-case CPU planner/manager probe (base: 12 pass; head: 9 pass, 3 ownership/budget assertions fail): {{BRANCH_LINK}}. This characterizes behavior without executing GPU copies or forwards. [Pinned evidence](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/a5b9fa53c6b3ceb6f56d3f520d9e437d888b3f95/results/upstream/57908).

*Investigated with AI assistance.*
