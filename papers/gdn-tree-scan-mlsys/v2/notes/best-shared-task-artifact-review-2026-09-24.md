# Shared-task artifact review — 2026-09-24

**PASS for the source-bound CPU reduction and dependency set.** This review does not assert that the final delivery archive has already been built or extracted; the parent owns that final packaging check. No inference was run and no attempted source/run was modified.

## Selection and estimator

`results/agent-workload/shared_task_reduce.py:15,44–61` fixes the shared IDs to `astropy__astropy-12907` and `astropy__astropy-13033`, verifies their ordering against the completed SGLang comparator tasks, and takes the maximum only over `Sr12`, `Cqc16`, and `Cqc15`. It pools counters before applying `(N-R)/(E-F)`. It does not maximize individual task rates or mix the separate four-task and ten-task populations into the selected comparison.

All three recovered shared-task runs remain in the output. The maximum is the older Sr12 run, not the latest Cqc15 run or an assertion of the best possible configuration. `:64–71` explicitly records different builds, output caps, and trajectories; tree cap 32,768 versus SGLang 24,000. These are descriptive recorded request-interval rates, not an isolated engine speedup, task-time speedup, or machine throughput estimate.

| Population | Tasks | Completed output tokens N | Completed requests R | Pooled tokens/s | Resolved |
|---|---:|---:|---:|---:|---:|
| Sr12 | 2 | 25027.0 | 41.0 | 29.093442700159 | 1 |
| Cqc16 | 2 | 26239.0 | 35.0 | 28.283046868773 | 1 |
| Cqc15 | 2 | 70424.0 | 53.0 | 27.268008150270 | 1 |
| SGLang_EAGLE | 2 | 47854.0 | 45.0 | 26.893956024190 | 1 |
| Sr12_full_four_tasks | 4 | 101923.0 | 124.0 | 26.205364927916 | 2 |
| Cqc10 | 10 | 248342.0 | 265.0 | 25.633622465853 | 6 |

The shared-task relative differences versus SGLang are Sr12 **+8.178367935126%**, Cqc16 **+5.165066988785%**, and Cqc15 **+1.390840848197%**. The two-task selection follows completed comparator availability and is retrospective. The report preserves Sr12's full four-task value and Cqc10's later ten-task value so the selected comparison is not presented as either full population.

## Raw recovery and guards

The new `raw/shared-tasks-20260924/MANIFEST.json` binds eight task records / 32 original files: pre/post metrics, runner metadata, and evaluation report for Sr12's four completed tasks and the shared two tasks each for Cqc16 and Cqc15. All 32 were retrieved from the DGX originals, matched the earlier extraction SHA-256 and size, and copied byte-for-byte after a credential-pattern check. No raw file required projection or redaction. No raw environment or full agent trace was copied. Three separately labeled safe configuration/engagement/cap projections are included with their original source bindings. The manifest is itself an input to the reducer.

The imported, unchanged `shared_rate_reduce.Audit.task` checks matched completion populations and metric counts, nonnegative finite deltas, pre/post idle gauges, task and evaluation identity, and completion/budget status. `shared_task_reduce.py:27–33` independently checks each recovered payload/projection size, SHA-256, and containment within its raw root before reduction. Existing scope limits and adverse/incomplete attempts remain in the dated history audit; they are not converted into completed task observations.

## Isolated replay and negative controls

I copied exactly the declared dependency set to a temporary repository layout: **81 input files** (80 original raw files plus the new manifest), **3 projection files**, and **2 reducer scripts**, for **86 files**. The 80 originals comprise the new 32 files, 40 Cqc10 files, and 8 SGLang files. The latter retain their repository-relative paths under `results/fr14_nvfp4_port_20260816/sglang16_evidence/`; this mapping is required when packaging outside the paper directory.

Command, with only the declared files available:

```sh
python3 /var/folders/xc/sy7ktq0n42d1n78zg10b8p_r0000gn/T/best-shared-task-review-20260924-12j_2lte/papers/gdn-tree-scan-mlsys/v2/results/agent-workload/shared_task_reduce.py
```

The output was byte-identical to `shared-task-rate-audit.json`. Three negatives in the disposable copy were rejected:

1. Append a newline to Sr12/12907's post metrics without updating the manifest: size/hash rejection.
2. Append a newline to the cap projection without updating the manifest: size/hash rejection.
3. Increment Sr12/12907's E2E completed-request count and update only the disposable manifest's size/hash to admit the altered bytes: rejection at the `E2E request count` population check.

After restoration, clean replay was again byte-identical. These tests changed only disposable copies. Source/artifact review found no remaining material reduction or dependency issue. Final bundle extraction should preserve the repository-relative layout and run this command again; the parent is updating the builder to explicitly collect both old and new audit input maps.

## Reviewed identities

| File, relative to paper v2 | SHA-256 |
|---|---|
| `results/agent-workload/shared_task_reduce.py` | `d007110971ed8d3d0b2829a4f1f8e617ff12faaf88ca6231263997cfff4881ca` |
| `results/agent-workload/shared_rate_reduce.py` | `c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f` |
| `results/agent-workload/shared-task-rate-audit.json` | `dcc6f3fb69bdd01f7ff8bed0c920ea797c18c18418c06044bfa039a88a7b286a` |
| `results/agent-workload/raw/shared-tasks-20260924/MANIFEST.json` | `2e20393e2fa238d489215a11d3b3315a08e9516609f341713ecd1ae62ed4dbd2` |

Recovery chronology, model/source/route distinctions, cap and behavior checks: `notes/nvfp4-b1-stronger-history-audit-2026-09-24.md` (SHA-256 `657d190e3179768b4fe5a4f8b000a019bd8d01358d55de2b62a18905b40dcbfb`).
