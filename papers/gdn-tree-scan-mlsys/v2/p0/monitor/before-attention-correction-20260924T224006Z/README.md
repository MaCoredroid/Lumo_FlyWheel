# GDN Tree-Scan v2: verifier design for coding-agent workloads

Current revision: 24 September 2026, patch-producing workload comparison. Among recorded runs producing nonempty patches on both shared SWE-bench Verified Astropy tasks (12907 and 13033), the best tree run reaches **29.09 pooled tokens/s** versus **26.89 for SGLang EAGLE**, an **8.18% higher rate**.

The performance subset is selected retrospectively. Every task in a compared run-pair must finish without agent timeout/task-budget exhaustion and produce a nonempty patch. Test failures remain eligible. The same criterion retains all three recovered tree pairs and one of the two recovered SGLang pairs. The excluded SGLang pair produced an empty patch on task 13033 after a thinking-only response at its output cap. The entire pair is excluded from the performance table; its failed outcome and original evidence remain in the artifact. This is a conditional comparison, not an estimate over all attempted tasks.

Eligible patch sizes for tasks 12907/13033: Sr12 504/1017 bytes; Cqc16 504/1092; Cqc15 504/1450; SGLang 504/912. The excluded pair has 504/0 bytes. Every included pair has one resolved task and one failed-test task.

All rates use `(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`. The included tree observations are 29.09, 28.28 and 27.27; the separate ten-task Cqc10 result remains 25.63 and full four-task Sr12 remains 26.21. Both context cohorts also produce nonempty patches for every completed task. Best recorded is distinct from latest configuration and full-campaign performance.

The complete decoding methods compete: SGLang EAGLE uses steps/top-k/draft-tokens 3/1/4; Hydra27 uses 27 valid draft nodes, maximum depth 11 and 32 physical verification rows. Method-specific optimizations are part of that comparison. No common tuning sweep is established. All 41 headline tree and 45 eligible SGLang responses are at most 20,000 tokens, below both configured caps.

Reproduce the selection and rates without inference:

```sh
python3 results/agent-workload/patch_producing_rate_reduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output must match `results/agent-workload/patch-producing-rate-audit.json`. The selection overlay replays the unfiltered audit byte-for-byte, verifies original metadata hashes, records every inclusion/exclusion, and retains 94 input bindings plus five safe projections. No original record, raw rate or earlier sealed reducer was edited.

The separate ten-task deployment retains six resolved and four failed attempts, all terminating with nonempty patches and no degeneration detected by the recorded checks. Numerical qualification remains separate from NVFP4 agent serving; local-document timing and superseded proxies remain excluded from performance claims.

- `main.pdf`, `main.tex`, `abstract.tex`: current paper.
- `notes/patch-producing-comparison-redteam-2026-09-24.md`: independent metadata, selection and current-source audit.
- `results/agent-workload/patch-producing-rate-audit.json`: conditional comparison and full eligibility ledger.
- `results/agent-workload/competitive-rate-audit.json`: unchanged unfiltered audit.
- `FINAL-REVIEW.md`, `artifacts/FINAL-DELIVERY.json`: verification and final artifact identities.

No inference, publication or push was performed. Future confirmation should predeclare eligibility/tuning and report all attempted outcomes alongside conditional speed.
