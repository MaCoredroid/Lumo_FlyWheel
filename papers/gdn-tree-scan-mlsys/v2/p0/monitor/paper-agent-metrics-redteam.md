# Final Cqc10 manuscript/metric review — 2026-09-23

**PASS for the reviewed source scope.** The manuscript now reports only the latest complete Cqc10 single-configuration agent-workload segment as its performance case. It does not claim a native-versus-tree task-speed/quality advantage, use the superseded 9/16 outcome, or promote the older censored split-K pair's gain. The full investigation is in `notes/nvfp4-workload-evidence-review-2026-09-23.md`.

Independently reconstructed the original 10 per-task pre/post metric brackets, all 10 runner metadata records and evaluator outcomes. Selected metric origins are contiguous; raw deltas match the stored reduction. Six tasks resolved, four failed; agent elapsed sum 10868.910 s and range 100.411–2474.673 s agree with the table. The 265 completed engine TPOT observations reconcile with 256 normal requests and 9 successful internal compaction requests, 0 failed compactions, 265 stop completions, 0 length completions. Their TPOT sum 9.39803353270284 gives request-mean 35.46427748 ms and inverse 28.1973882172 tok/s. This is request-weighted service evidence, not aggregate task throughput, task speedup, or the mean of per-request rates.

Checked actual Cqc10 source revision e7af6b595a8b3a09b7e88d25a3653d381432b99c, served model path/modelopt_mixed/head route, engine seed 0, graph/prefix-cache controls, visible 24,000 / compaction 20,000 token budgets and 9,000 s task timeout, proxy sampling/auto-continuation, Qwen Code 0.19.4 and per-task network-policy attestation. Only the authorized case-study.tex was amended for explicit auto-continuation/network/support details; main/abstract were not edited by this reviewer.

The Hydra27 method paragraph correctly distinguishes 27 active draft nodes from 31 physical draft slots plus root, depth 11,15 MTP head nodes plus 6 principal + 4/2 rescue suffix nodes, and 4 post-root MTP forwards. It does not import Cat10's stock-attention flat-map qualification into the stochastic NVFP4 task deployment.

The latest split-K paragraph was checked against `results/fr14_nvfp4_port_20260816/splitk_fa2.md:20–54` and actual Cqc10 engagement: paired query heads share staged KV; four context partitions produce partial softmax quantities which the combine rescales/merges; floating accumulation order changes. Loaded candidate is split-K4, all 16 full-attention layers, binary 28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857, no fallback. The paragraph claims mechanism/engagement and no isolated attention gain. Fixed32 single-launch GDN is explicitly inactive in the task case. No material method/result mismatch remains in these reviewed paragraphs.

The CPU reducer was executed against the durable raw bundle at `results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10`. It reproduced `cqc10-metric-audit.json`; four lightweight in-memory parser negatives independently refused missing metric, duplicate metric, nonfinite value and unexpected engine label. No GPU or inference was run. Raw bundle assembly/manifest and the final rendered PDF are separate owner checks.

## Reviewed identities

| File | SHA256 |
|---|---|
|`main.tex`|`bf8f1f47a95e95e0a4732691dbcf56d86f2150b4a9da70d6b762005baf10e90c`|
|`abstract.tex`|`018538d3d3938a177950976e49f99662e38027cc12a34a2693d5b136abeb9c80`|
|`results/agent-workload/case-study.tex`|`4acf538efb64f6d27ce0abc6069fce3371a4d67454343f4db0f5b904195d86c4`|
|`results/agent-workload/cqc10_reduce.py`|`59f6d35aad7a409e9a00b551697da3205bb5fa8706ef4a54977c62089fb4e8a3`|
|`results/agent-workload/cqc10-metric-audit.json`|`d46418013862609dd494f1b2bcbe0dba82dada578cfeb2ce01816e539fabc761`|
|`notes/nvfp4-workload-evidence-review-2026-09-23.md`|`d6a31f364d412f8da4aa5ad5e65bcfcf849537db497884ab4117626ee5f88d94`|
