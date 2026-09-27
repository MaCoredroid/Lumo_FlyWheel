# Pooled decode revision — 23 September 2026

The current 13-page paper uses one completed-request pooled decode estimator in the abstract, results, comparison table and conclusion:

`D_pool = (sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)`.

GDN Tree-Scan achieves **25.63 tokens/s** on ten SWE-bench Verified Astropy tasks (265 completed requests); SGLang EAGLE achieves **26.89 tokens/s** on two different completed Astropy tasks (45 requests). Both deployments serve Qwen3.8-27B NVFP4 on GB10. The tree rate is **4.69% lower** in these records. The manuscript identifies different task subsets and serving configurations and makes a descriptive comparison, without attributing a causal mechanism gain or task-completion speedup.

The common reducer reproduces `results/agent-workload/shared-rate-audit.json` byte-for-byte from 48 hash-bound inputs: 40 tree files and eight SGLang files. It pools counts and durations before division, covers all completed engine requests including internal agent traffic, and verifies idle boundaries and matching token/request/latency populations. Both selected SGLang brackets have zero non-streaming requests. Accumulated request time excludes TTFT and between-request tool work; it is not machine-wall throughput. The settings projection records the source-bound nonsecret comparator configuration without copying unfiltered environments.

The prior request-weighted inverse-TPOT headline is absent from the current paper and current claim documents. All numerical speed claims use the pooled definition. E1/E8 internal-document rates, superseded mixed-support proxies and cross-workload optimization gains remain excluded. Dated raw evidence, earlier calculations and their manifests retain their original values for audit. Task durations and numerical error measurements remain explicitly distinct from decode rates.

All ten tree attempts terminated, used tools and produced nonempty patches; the recorded trace checks detected no degeneration or malformed tool-call arguments. All outcomes remain visible: six resolved and four failed tests. Detector blind spots and earlier adverse campaign observations remain in the results. The two completed SGLang task outcomes are one resolved and one failed. These selected development subsets are not full SWE-bench scores.

The NVFP4 Hydra27 graph/cache-enabled stochastic deployment remains distinct from the FP8 Cat10 eager/cache-off numerical qualification. Neither compact candidate passes all frozen criteria; bounded mechanism checks do not establish full-model equivalence or broad quality preservation.

Current verification records:

- `p0/monitor/2026-09-23-pooled-speed-claims-audit.{md,json}`: full loaded-source speed-claim inventory, current-document consistency, rendered-PDF stale-rate search and exact rate replay.
- `p0/monitor/2026-09-23-pooled-decode-build.json`: canonical build and visual inspection of all pages, with detailed checks of the abstract, equation/table, conclusion and final layout.
- `p0/monitor/paper-pooled-speed-redteam.md`: independent manuscript and current-claim review.
- `notes/pooled-decode-paper-artifact-redteam-2026-09-23.md`: independent estimator, configuration and package-dependency review.
- `notes/agent-behavior-abstract-evidence-2026-09-23.md`: source-bound behavior checks; original scope unchanged.

No new inference was launched. A paired task campaign remains conditional on a stronger causal/task-completion claim, not a prerequisite for reporting this descriptive comparison. The private package, extracted replay/rebuild and final source/PDF identities are recorded in `artifacts/FINAL-DELIVERY.json`. Earlier receipts remain in dated snapshots. No publication, submission or push was performed.
