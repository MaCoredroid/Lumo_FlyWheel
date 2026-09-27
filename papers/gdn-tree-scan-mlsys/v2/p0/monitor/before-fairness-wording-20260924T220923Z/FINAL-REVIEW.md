# Best shared-task revision — 24 September 2026

The 13-page paper now leads with **29.09 pooled tokens/s for GDN Tree-Scan versus 26.89 for SGLang EAGLE, an 8.18% higher observed rate**, on the same two SWE-bench Verified Astropy tasks (12907 and 13033). Both recorded deployments use Qwen3.8-27B NVFP4 on GB10 at batch one. The headline is explicitly the best recorded shared-task observation, Sr12 on August 19, rather than the latest configuration or a full-campaign mean.

All decode rates use `(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`. Later tree runs remain in the table: Cqc16 28.28 and Cqc15 27.27. Sr12's full four-task cohort pools 26.21. The separate ten-task Cqc10 deployment remains at 25.63. Different serving stacks, Python revisions and response caps (tree 32768 versus SGLang 24000) are explicit. The comparison describes the recorded deployments; it does not attribute a causal mechanism gain or task-completion speedup. No B4 aggregate throughput or inverse-mean-TPOT rate has been substituted for pooled request decode rate.

The new reducer reproduces `results/agent-workload/shared-task-rate-audit.json` byte-for-byte from 80 original task files and one raw manifest. It verifies all 32 newly recovered originals, three safe configuration/behavior projections, idle boundaries, and matching token/request/latency populations. It reuses the unchanged checks in the common reducer. The earlier 48-input result and sealed manifests remain unchanged as dated evidence.

All six selected tree attempts on the shared pair used tools, produced nonempty patches and passed the recorded degeneration/malformed-call checks. Every shared-pair row has one resolved and one failed task. The separate ten-task deployment has six resolved and four failed attempts; all terminated and produced nonempty patches, with no detected degeneration. The abstract keeps that behavior statement attached to the ten-task deployment. Earlier adverse and incomplete attempts remain visible, including Cqc16 task 13236 and Cqc15 task 13398. These are selected development subsets, not full-benchmark scores or a universal quality guarantee.

The methods remain focused on tree verification, GPU scan/replay, state publication and route-specific optimizations. NVFP4 Hydra27 graph/cache-enabled stochastic serving remains distinct from FP8 Cat10 eager/cache-off qualification. Neither compact candidate passes every frozen criterion. Local-document E1/E8 rates, superseded proxies and cross-workload optimization gains remain excluded from current performance claims.

Verification records:

- `p0/monitor/paper-best-shared-task-redteam-2026-09-24.md`: independent manuscript review, including original selected-tree counters and the full Sr12 cohort; PASS.
- `notes/best-shared-task-artifact-review-2026-09-24.md`: exact isolated replay with 81 input bindings, three projections and two scripts; all three negative controls rejected; PASS.
- `p0/monitor/2026-09-24-best-shared-task-speed-audit.json`: current loaded-source claim inventory, exact reducer replay and stale-rate checks.
- `p0/monitor/2026-09-24-best-shared-task-build.json`: clean 13-page build, citation checks and rendered-page inspection.
- `artifacts/FINAL-DELIVERY.json`: final source/PDF/private-package identities and extracted replay/rebuild outcome.

No new inference, public submission, external message or push was performed. The paired current task campaign remains conditional on a stronger causal or task-completion claim. Earlier notes recommending the latest run remain dated evidence; the user's September 24 instruction selects the best verified observation for this revision.
