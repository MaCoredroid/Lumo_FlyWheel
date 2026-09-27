# Patch-producing comparison red-team — 2026-09-24

## Raw eligibility check

**PASS for the declared retrospective eligibility rule.** Apply the same rule to every recovered run-pair: both shared task attempts must complete and each must produce a nonempty patch. Failed tests remain eligible. The rule is based on patch production, not test success, rate, engine identity, or a separate response-cap filter. It is conditional on these two task IDs and must not be described as a prospectively specified or unconditional performance result.

I independently read the original-byte runner metadata and saved evaluation reports for all ten attempts, without any new inference or recovery. Every attempt has exit code zero, no task timeout or campaign-budget capping, and a terminal `ended_at`. Every metadata `patch_bytes` field is an integer. Recorded patch sizes are:

| Pair | 12907 patch bytes | 13033 patch bytes | Attempt completion | Pair disposition |
|---|---:|---:|---|---|
| Sr12 | 504 | 1017 | PASS | Eligible |
| Cqc16 | 504 | 1092 | PASS | Eligible |
| Cqc15 | 504 | 1450 | PASS | Eligible |
| SGLang_later | 504 | 912 | PASS | Eligible |
| SGLang_earlier | 504 | 0 | PASS | Excluded: empty patch |

All five task12907 records are resolved. Sr12/Cqc16/Cqc15 and later SGLang task13033 record `tests_failed`; those failures remain eligible. Earlier SGLang task13033 records `synthetic_no_patch=true`, `harness_invoked=false`, `error=empty_patch`, and zero patch bytes. Its task attempt did terminate; it failed to produce a patch. Thus exactly four pairs pass and the earlier SGLang pair fails under the same criterion.

The earlier SGLang raw pooled rate remains 30.70049467291224 tokens/s. The raw artifacts and unchanged competitive receipt retain it. Excluding it from a patch-producing comparison changes the population, not the recorded rate or its arithmetic validity. The explicit retrospective qualification is necessary because patch production is observed after the run. The later eligible SGLang pair pools 26.893956024189762; the best eligible tree pair remains Sr12 at 29.093442700158857, or +8.178367935125518%. These values describe the patch-producing subset, not a quality-adjusted rate or an unconditional system advantage.

## Raw metadata bindings

Paths below are repository-relative. Each record's sibling `eval/eval_report.json` was inspected for the outcome and synthetic-no-patch distinction. The existing audit manifests bind those evaluation files and metric brackets.

| Pair / task | Metadata path | SHA-256 |
|---|---|---|
| Sr12 / 12907 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Sr12/swe_out/verified/per_task/astropy__astropy-12907/runner_metadata.json` | `fe7e5a21b8016b3336717a8c2d73118b1f63d5fe5f21d037ddcd0c506f664e99` |
| Sr12 / 13033 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Sr12/swe_out/verified/per_task/astropy__astropy-13033/runner_metadata.json` | `7640209c38cb66497f19a7bc84fe5c99833a56bf789ddf6bbb6077517984ac0c` |
| Cqc16 / 12907 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Cqc16/swe_out/verified/per_task/astropy__astropy-12907/runner_metadata.json` | `bbb23507db18bd5bd70fa7cdd8f4dc544efc1075c1f3f5161db22e80f4419554` |
| Cqc16 / 13033 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Cqc16/swe_out/verified/per_task/astropy__astropy-13033/runner_metadata.json` | `a2cbc8e1d1245139918005f1f275eaa6378270535c3a31f6e02bb56c4ebe511d` |
| Cqc15 / 12907 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Cqc15/swe_out/verified/per_task/astropy__astropy-12907/runner_metadata.json` | `193c989c10fd94143d5b0765dc9b690ba0a594ea1f068eccd4f6b50ec312fc8b` |
| Cqc15 / 13033 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/shared-tasks-20260924/Cqc15/swe_out/verified/per_task/astropy__astropy-13033/runner_metadata.json` | `333b2328235d562351607e833b8b44094d30baffb8d2b1cf81cff900f2102c38` |
| SGLang_later / 12907 | `results/fr14_nvfp4_port_20260816/sglang16_evidence/run1/swe_out/verified/per_task/astropy__astropy-12907/runner_metadata.json` | `78679842646c92cc927c0cc102836580b965ced7a7ab8d5d4a1f0d0a77d97f71` |
| SGLang_later / 13033 | `results/fr14_nvfp4_port_20260816/sglang16_evidence/r1/swe_out/verified/per_task/astropy__astropy-13033/runner_metadata.json` | `a8197a47470987fc87076b2163a02dcfac7f30a3652a56f1a4031a3ee9f89ff1` |
| SGLang_earlier / 12907 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/sglang-step2-20260924/astropy__astropy-12907/runner_metadata.json` | `5a1d8c0257a20769b74295a89f497a8cf5e51da48fb3d05f94ec75bbdb3040cf` |
| SGLang_earlier / 13033 | `papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/sglang-step2-20260924/astropy__astropy-13033/runner_metadata.json` | `ffdd5969ccbab6993f9ec88297b2c8b4eb968e265eb9caab39fd460e2dc2ba02` |

## Final overlay and manuscript review

**PASS for the final overlay and reviewed manuscript source.** No remaining material discrepancy was found for this declared conditional comparison. Final PDF rendering and final delivery-archive extraction are the parent's separate packaging checks.

`patch_producing_rate_reduce.py:29–46` first requires the unchanged competitive reducer to replay byte-for-byte, locates each arm/task metadata file uniquely, checks its bound hash and task identity, and then reads patch production directly from that metadata. `:47–61` rejects invalid patch-size types, computes attempt completion from exit status/timeout/task-budget/terminal fields, and applies the same positive-patch rule to every task in every arm. `:62–69` excludes whole pairs and refuses to estimate when either system has no eligible pair. It does not inspect test success or rate to decide eligibility. Existing Cqc10 and full-four-task Sr12 context arms also pass this check; their populations remain separate.

The output preserves a complete eligibility ledger for all recovered arms, including earlier SGLang's failed empty-patch task. It selects all three tree pairs and later SGLang, then takes the maximum over eligible tree observations. The excluded rate remains in the unchanged, hash-linked unfiltered audit. The overlay does not reinterpret that rate as erroneous.

`abstract.tex:2` conditions 29.09 versus 26.89 on nonempty patches for both shared tasks. `results/agent-workload/case-study.tex:50` explicitly dates the decision after inspection, applies it symmetrically, retains failed tests, identifies one excluded SGLang pair and its empty-patch/capped-thinking outcome, and excludes the whole pair. `:55–75` and the table keep every eligible pair and disclose the retrospective subset. `main.tex:256,330` applies the same conditional scope. The excluded early run is no longer a performance-table row, while its unsuccessful outcome and evidence remain disclosed. No unconditional superiority or task-success benefit follows from this filter.

## Independent isolated replay and symmetry controls

I copied the exact transitive dependency set into an otherwise empty temporary repository: 94 raw/input files, five projections, five reducer scripts, two child audit receipts, and the unfiltered competitive audit, totaling **107 files**. Every copied file matched its declared hash. Command:

```sh
python3 /var/folders/xc/sy7ktq0n42d1n78zg10b8p_r0000gn/T/patch-producing-review-20260924-r0duweit/papers/gdn-tree-scan-mlsys/v2/results/agent-workload/patch_producing_rate_reduce.py
```

Clean output was byte-identical to `patch-producing-rate-audit.json`. To test the actual eligibility logic beyond the upstream byte-integrity gates, I made three explicit synthetic changes only in disposable copies, refreshing their temporary manifests and child receipts:

1. Set Sr12 task12907 patch bytes to zero: the entire Sr12 pair was excluded and Cqc16 became the selected tree observation.
2. Set Sr12 task12907 agent exit code to one: the entire pair was excluded with `incomplete_attempt` even though its patch remained nonempty.
3. Set later SGLang task12907 patch bytes to zero: neither SGLang pair was eligible and the overlay raised `no eligible shared-task comparison`, rather than emitting a tree-only estimate.

Restoring the copied inputs restored exact clean replay. Real test-failure tasks remained eligible throughout the original result, verifying that test success is not the selection criterion. No original run, raw evidence, source file, or prior receipt was edited.

## Final source identities

All paths in the following table are relative to paper v2.

| File | SHA-256 |
|---|---|

| `main.tex` | `baee91032667889733aed5307b0d3a40fc0cc6a880df8df8aa17d8782b70066d` |
| `abstract.tex` | `f160cd35a25f331d625bed688d7875372ca27234178727fb77a6bf23aeeb4b3b` |
| `results/agent-workload/case-study.tex` | `73b37d657faa261fba11eaed22bb914ec1fc61229a0b22280b204a5d07ac4b64` |
| `results/agent-workload/patch_producing_rate_reduce.py` | `d57459bc5706c1e2df849dfa5a04f1f7974e7731ebcd1e04945f7695d0c24e52` |
| `results/agent-workload/patch-producing-rate-audit.json` | `279475d182ecd259790e2c57663d65c37205f2ee67c80aa46b0bd2e9b4e460c4` |
| `results/agent-workload/competitive_rate_reduce.py` | `a2d2a669cf63061d5fabddcd39778dadd0dff9c031fec3b9d99ccd9d713c310c` |
| `results/agent-workload/competitive-rate-audit.json` | `e30060ec1c4ade6f4b3765e4a0c55c25cff3ba121715e4ad26f2f2a44abecac7` |
| `results/agent-workload/shared_task_reduce.py` | `d007110971ed8d3d0b2829a4f1f8e617ff12faaf88ca6231263997cfff4881ca` |
| `results/agent-workload/shared_rate_reduce.py` | `c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f` |
