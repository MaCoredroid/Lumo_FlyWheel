# GDN Tree-Scan v2: verifier design for coding-agent workloads

Current revision: 24 September 2026, competitive-comparison correction. The best recorded tree run reaches **29.09 pooled tokens/s** on SWE-bench Verified Astropy tasks 12907 and 13033. The two recovered SGLang EAGLE runs reach **26.89 and 30.70** on those same tasks. The tree is 8.18% above the later run and 5.23% below the earlier run; there is no best-versus-best speed win in these records.

The comparison evaluates complete decoding systems, including proposal topology, draft length, candidate budget and serving optimizations. Different decoder implementations are the intended comparison. SGLang uses EAGLE steps/top-k/draft-tokens 3/1/4; tree Hydra27 has 27 valid draft nodes, maximum draft depth 11 and 32 physical verification rows. The recovered records do not establish optimal configurations from a common tuning sweep.

All decode rates use `(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`. The earlier SGLang record has 47,850 tokens and 17 completed requests; its failed task 13033 produced an empty patch, so the test harness was not invoked. Its final response contains 32,768 output tokens of thinking only, at its configured cap. That unsuccessful outcome remains in the comparison. The later SGLang record has 47,854 tokens and 45 requests. All 41 Sr12 responses and all 45 later SGLang responses are at most 20,000 tokens, below both configured caps. The earlier SGLang and Sr12 limits both equal 32,768.

Later tree runs remain visible: Cqc16 28.28 and Cqc15 27.27. Sr12's complete four-task cohort pools 26.21. The separate latest complete ten-task Cqc10 deployment pools 25.63, with six resolved and four failed attempts. All ten terminated, used tools and produced nonempty patches, with no degeneration or malformed tool-call arguments detected by the recorded checks. Adverse and incomplete campaign records are preserved. These selected subsets do not establish a full SWE-bench score or broad task-quality equivalence.

The methods focus on tree verification, GPU scan/replay, state publication and route-specific optimizations. NVFP4 Hydra27 graph/cache-enabled serving remains distinct from FP8 Cat10 eager/cache-off qualification. Only named coding-agent tasks supply performance results; local-document E1/E8 rates and superseded proxies remain excluded.

Reproduce all current rates without inference:

```sh
python3 results/agent-workload/competitive_rate_reduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output must match `results/agent-workload/competitive-rate-audit.json`. The aggregate replays both child reducers byte-for-byte, uses the unchanged common population/clock checks, and binds 94 input files plus five configuration/behavior projections. Its inputs include 88 original task files, two raw manifests and four original supporting launch/note files. Earlier reducers and sealed outputs are retained as dated evidence.

- `main.pdf`, `main.tex`, `abstract.tex`: current paper.
- `results/agent-workload/raw/sglang-step2-20260924/`: recovered original earlier SGLang inputs.
- `notes/serving-comparison-fairness-2026-09-24.md`: draft settings, tuning evidence and stronger comparator discovery.
- `notes/competitive-comparison-redteam-2026-09-24.md`: independent calculation/source review.
- `notes/headline-response-cap-check-2026-09-24.json`: original response-length histogram checks.
- `FINAL-REVIEW.md`, `artifacts/FINAL-DELIVERY.json`: current verification and delivery identities.

A broader comparison would predeclare a comparable tuning budget and repeated task runs. Equal-draft-budget component ablations answer a separate mechanism question. No inference, publication or push was performed for this correction.
