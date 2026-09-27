# Coding-agent workload manuscript review — 2026-09-23

**PASS for the stated descriptive Cqc10 case. No unresolved material finding.** Read-only manuscript/evidence review and CPU reduction; no model execution. PDF layout is the parent's separate check.

## Reviewed source

| File | SHA-256 |
|---|---|
| `main.tex` | `44c07c9e018bbbc9cc7dbc93b63941089273fa0f2154c1968e265fcead0e7300` |
| `abstract.tex` | `018538d3d3938a177950976e49f99662e38027cc12a34a2693d5b136abeb9c80` |
| `results/agent-workload/case-study.tex` | `ad053cd02521a04e99fd29f743e99a71e102815845864168ac4561a1ab05a1b3` |
| `figures/pipeline.tikz` | `8c9dbd43516be8703d4ba841a06d1f2d17caad2c434a61ec29ede40035f0762b` |
| `figures/state-contract.tikz` | `50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898` |
| `figures/scan-replay.tikz` | `affc55cd6972d929de38dc8ebdd86c6dcea77fb0beb73c96551010bfaffadcfe` |

## Evidence and disposition

- **Task outcomes and times:** case-study lines 9–31 and abstract line 2 agree with all ten original `eval/eval_report.json`, `runner_metadata.json` and `eval/normalized_eval.json` records under `results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/swe_out/verified/per_task/`. These are six resolved and four failed evaluations, not a composite sixteen-task score. The agent elapsed sum is 10,868.91 seconds (181.15 minutes); the minimum/maximum are 100.411/2,474.673 seconds (1.67/41.24 minutes). All ten displayed task values round correctly. Evaluation and startup are not included in that agent field.
- **Service statistic:** case-study lines 33–39 correctly compute inverse request-mean TPOT: `265 / 9.39803353270284 = 28.19738821721218` tokens/s, or 35.46427748189751 ms mean TPOT. The original ten pre/post brackets are contiguous and have no selected-counter reset. Support reconciles to 256 normal requests plus nine successful internal compaction requests; all 265 completion reasons are stop and none length. This is not total tokens divided by agent wall, not an average of individual rates, and not task-speed evidence. Re-executing `cqc10_reduce.py` produced byte-identical `cqc10-metric-audit.json`.
- **Configuration:** case-study lines 1–6 match the recorded August 24 run: Qwen3.8-27B NVFP4 port, ModelOpt mixed/bfloat16, Hydra27/32 physical rows, patched FA2 split-K4, graph/prefix-cache deployment, one active sequence, context 131,072, scheduler budget 4,096, engine seed zero, declared proxy sampling, 24,000/20,000 normal/compaction caps, 9,000-second agent limit, and disabled proxy auto-continuation. All ten task metadata records bind Qwen Code 0.19.4, repository-image execution, the same network-policy fingerprint, and untimed-out attempts. The source commit and runtime receipts remain attached; this does not independently establish immutable model-weight identity or a globally latest model release.
- **Selection and parity:** case-study lines 41–42 disclose the interrupted development lineage, changed earlier source/caps, one-repository selection, single completed boot, and lack of a matched native ten-task arm. The earlier repeated/incomplete/capped records remain auditable and are not silently counted as a homogeneous benchmark. There is no 9/16, 8/16, or 8/13 headline and no native/tree speed or quality advantage claim.
- **Route and freshness boundaries:** main lines 239–256, 274, 298–314, 319–332 and 358–384 keep the August NVFP4 deployment distinct from September Qwen3.6 FP8 Cat10 numerical/continuation qualification. E7a adverse findings and finite E2/E7b limits remain explicit. The paper does not claim that the eager/cache-off/temperature-zero route qualifies graph/cache-enabled stochastic NVFP4 serving.
- **Performance scope:** searched main, abstract, included case study and all three loaded design figures. No E1/E8 throughput, gain, interval, multiplier or performance plot remains. E8's untimed same-input head/candidate counts remain correctness evidence only. Loaded references and citations have no orphan/missing keys. No isolated kernel or document-prefix gain is multiplied into the workload result.

## Closed correction

The first reviewed case-study wording said the earlier segments' “original failures, retries, and missing evaluations remain in the artifact.” The compact bundle contains source-bound audit extracts for that earlier lineage, not all complete original raw records. The current source explicitly says audit records are in the artifact and originals remain at recorded source paths. This closes the only material wording finding in this pass.

## Durable evidence

`results/agent-workload/raw/cqc10/MANIFEST.json` binds 139 payloads (3,482,997 bytes), SHA `01c3922b06377c85d86b213bded399af377809f22d0df63829c5358ef7a86bcf`. All local payload sizes and hashes were verified. Original evaluator, metadata, normalized output, submitted patch/prediction, metric and runtime/configuration receipts are included. Two explicitly named `SANITIZED` configuration projections exclude credential values; original hashes are retained and the unsafe originals are not packaged. The evidence does not include complete model weights, images, agent bundle, tool workspaces or evaluator implementation bytes.

`results/agent-workload/EVIDENCE_MANIFEST.json` binds those raw files plus the raw manifest, task audit, metric audit/reducer, README and two earlier-lineage audit extracts: 146 payloads; SHA `75200c9ba4538ae8d1c9889ac798e1604dc6cbb8e98646568e2b9a741b00980c`. Manuscript text is deliberately outside this evidence manifest.

| Derived file | SHA-256 |
|---|---|
| `cqc10-task-audit.json` | `7e2e11b682006687dc46ca1afd19bd5ce907ed29fcd41ca4c68936af11f17132` |
| `cqc10-metric-audit.json` | `d46418013862609dd494f1b2bcbe0dba82dada578cfeb2ce01816e539fabc761` |
| `cqc10_reduce.py` | `59f6d35aad7a409e9a00b551697da3205bb5fa8706ef4a54977c62089fb4e8a3` |

CPU replay from `results/agent-workload/`: `python3 cqc10_reduce.py > /tmp/cqc10-replay.json && cmp /tmp/cqc10-replay.json cqc10-metric-audit.json`.

No new experiment is indispensable to report this dated, selected task ledger and its declared request statistic. A current repeated native/tree campaign with the same checkpoint, agent, tasks, settings, tool/network policy and budgets is still necessary for an application-speedup or matched quality claim. The manuscript correctly leaves that stronger claim unestablished.
