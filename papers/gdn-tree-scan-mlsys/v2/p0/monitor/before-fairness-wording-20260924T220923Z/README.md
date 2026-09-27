# GDN Tree-Scan v2: verifier design for coding-agent workloads

Current revision: 24 September 2026. The abstract leads with the **best recorded shared-task pooled decode result: 29.09 tokens/s for GDN Tree-Scan versus 26.89 for SGLang EAGLE, an 8.18% higher observed rate**. Both use Qwen3.8-27B NVFP4 on GB10 and the same two SWE-bench Verified Astropy task IDs, 12907 and 13033. The result is Sr12 on August 19; it is explicitly the best recovered shared-task observation, not the latest configuration or a full-campaign average.

Every decode-rate claim uses `(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`. Counts and durations are pooled before division. Sr12 has 25027 output tokens, 41 completed requests and 858.818953 seconds after first output; SGLang has 47854 tokens, 45 requests and 1777.685661 seconds. Later tree runs on the pair remain visible: Cqc16 28.28 and Cqc15 27.27. The whole four-task Sr12 cohort pools 26.21. Different Python revisions, serving stacks and response caps (tree 32768 versus SGLang 24000) remain explicit. This is an observed deployment comparison, not a controlled task-completion speedup.

The separate August 24 Cqc10 deployment covers ten Astropy tasks and 265 requests at 25.63 pooled tokens/s. All ten attempts terminated, used tools and produced nonempty patches, with no degeneration or malformed tool-call arguments detected by the recorded checks. Six resolved and four failed tests; agent durations total 181.15 minutes. Detector blind spots and earlier adverse/incomplete attempts remain in the results and audit. These selected subsets are not full-benchmark scores.

The manuscript focuses on tree verification, GPU scan/replay, state publication and route-specific optimizations. The NVFP4 Hydra27 graph/cache-enabled stochastic deployment is distinct from the FP8 Cat10 eager/cache-off numerical qualification. Neither compact candidate passes every frozen criterion. **Only named coding-agent tasks supply performance results.** Local-document E1/E8 rates, mixed-support historical proxies and cross-workload percentage gains remain excluded. Dated raw records and earlier calculations are preserved without relabeling their estimators.

Reproduce all current rates without inference from this directory:

```sh
python3 results/agent-workload/shared_task_reduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The reducer output must match `results/agent-workload/shared-task-rate-audit.json`. It reads 80 original task files plus the new raw manifest, validates the 32 recovered original files and three explicitly safe audit projections, and uses the unchanged population/clock checks in `shared_rate_reduce.py`. The previous `shared-rate-audit.json` and calculation manifests remain unchanged as dated evidence.

- `main.pdf`, `main.tex`, `abstract.tex`: current paper.
- `results/agent-workload/raw/shared-tasks-20260924/`: original shared-task records, provenance manifest and safe configuration/behavior projections.
- `notes/nvfp4-b1-stronger-history-audit-2026-09-24.md`: source/trace/configuration audit, including adverse records.
- `notes/claim-evidence-ledger.md`: allowed claims and limits.
- `FINAL-REVIEW.md`, `p0/monitor/`: current review and build evidence.
- `artifacts/FINAL-DELIVERY.json`: final source/PDF/private-package identities.

A paired current task campaign remains conditional on a causal mechanism or task-completion speedup claim. No new inference, public submission, external message or push was performed for this revision.
