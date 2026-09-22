# E1 engine-seed discrepancy: bounded paper-scope adjudication

2026-09-22 10:49 UTC. Read-only source/configuration audit; no GPU, test suite, inference, campaign modification, or replacement data. Paper reviewed: `main.tex` SHA-256 `f52a828f4cb13aca925be7d081da0c4c438072762b673de9a2933b7396032212`; abstract `0eddd1990dfe88aa1b0251526448d24983c7d0f67821d2f2921bb63df18755ce`.

**Decision: explicit disclosure is sufficient for the proposed bounded comparison of the as-executed instrumented configurations. No additional experiment is necessary for that claim.** The seed mismatch is a real deviation from the frozen settings, not proof of a measurement error and not evidence of seed invariance. Preserve the original freeze, every initial cell, and the dated deviation. The first-tree raw-support PASS and the runner's VALID status must not be represented as complete frozen-protocol compliance.

## Exact basis

Let `R` be `experiments/out-20260922T100028Z-e1-18cells/` under the remote paper directory, `N = R/cell_01_b1_native-5_B1_a1/`, and `T = R/cell_05_b1_tree_B1_a1/`.

1. `R/campaign_snapshot/E1_FREEZE.md:17–18` explicitly places seed 20260921 under settings described as identical across arms. `N/docker_inspect.json[0].Config.Cmd` contains `--seed '20260921'`, and `N/docker_logs.txt:32` initializes `seed=20260921`. `T`'s command and environment contain no seed argument/variable; its log at line 31 initializes `seed=0`. Independently read all 18 native and 10 tree `cohort/*/capture_request.json` files: every `request.seed` is 20260921 and every `request.temperature` is zero. These 28 observations establish this pair's actual scopes; they do not substitute for final per-cell configuration reporting.
2. `T/loaded_backend/gpu_model_runner.py:1152–1159` creates a request generator only for `SamplingType.RANDOM_SEED`; otherwise it assigns `None`. The greedy sampler branch at `T/loaded_backend/rejection_sampler.py:3762–3775` passes the generator map and `all_greedy=True`; its device call at 2751–2761 forwards them. The dependency path is selected at 2725–2738. The actual dependency hash matches `R/campaign_snapshot/campaign_identity.txt:6`.
3. In that dependency, repository `scripts/fr13_device_multidraft_kernel.py:1034–1058` excludes TAW/depthsync for `all_greedy`; 1087–1089 chooses point-mass argmax rows; 1098–1108 creates a fresh explicit device generator, manually seeding it only when a request generator exists; 1149–1150 passes this explicit generator to child selection. The engine-global seed is not an input at this inspected selector. This supports neither “temperature zero performs no RNG operations” nor “the request seed overrides the engine seed,” and does not establish invariance elsewhere in the engine. Previously qualified duplicate-sibling path selection is not newly invalidated by the configuration discrepancy.

## Consequence for the claim

E1's present estimand is API-bound emitted tokens divided by wall time on the declared retained intervals, comparing complete instrumented configurations on one model/hardware/workload. Those observed numerators, denominators, support exclusions, and paired configuration-rate contrasts remain interpretable despite the mismatch. No current numerical result is being attributed to an isolated engine-seed effect, and full-model equivalence is already unestablished. The existing claim does not depend on proving that changing the engine seed would leave outputs or timing unchanged.

The seed assignment is confounded with arm, so final results cannot isolate a tree-versus-chain algorithm effect under otherwise identical engine settings. A paired-block interval characterizes the observed configuration contrast; it does not bound or remove any systematic engine-seed effect. Final methods/results should state both engine seeds alongside the common API seed and identify this as a protocol deviation, not bury it only in artifacts. Avoid unqualified “all settings matched,” “seed-controlled algorithm speedup,” “harmless mismatch,” or “whole-engine seed invariance.” “Matched workload” remains accurate when its specific matching dimensions and this deviation are explicit.

Suggested minimal wording: “All requests used temperature zero and request seed 20260921. Contrary to the frozen configuration statement, native engines used seed 20260921 and tree engines used the default seed 0. We retain the original cells and compare these as-executed instrumented configurations; source inspection found no engine-global seed input at the active tree child selector, but does not establish whole-engine seed invariance.”

No corrective boot is required or recommended for this bounded claim. An added matched-engine-seed control would become necessary only if the paper instead required a conclusion that depends on holding that factor fixed or on ruling out its effects; that stronger claim is unnecessary here. Any such future evidence must be supplementary and cannot retroactively make the original campaign comply with the freeze. All unrelated qualification, support, uncertainty, and pending E1-result checks remain in force.

## Reviewed identities

All hashes are SHA-256. `notes/e1-protocol-deviations.md` and the first-tree review were read locally; the raw configuration and loaded sources above were independently read on DGX. The note was not yet present remotely at this audit, so no synchronized remote-note identity is claimed.

| Artifact | SHA-256 |
|---|---|
| `notes/e1-protocol-deviations.md` | `f04e6d62844ffe6962cac57b624934f7bbe5f9ae7f701b0f7e817a231ce2322b` |
| `p0/monitor/e1-first-tree-timing-redteam.md` | `9995f60a61a45a9db69a2fa1fec013d10efc5b4f98bc26c811fed5908d81774e` |
| `R/campaign_snapshot/E1_FREEZE.md` | `11f5dd00b8d8b7073575f10cd9a00a4282760746be05f16bb9c7d188fa027a4e` |
| `R/campaign_snapshot/e1_cells.json` | `d5aa5b6eb3da19bec9955f57162b2508c85b4d6413eedf93cfdd1f3157a75363` |
| `R/campaign_snapshot/campaign_identity.txt` | `28c91e906fc0f6592b623ba442d21a575c2544de36763f13a2f792654e94d164` |
| `N/docker_inspect.json` | `5d0f809c42c5311b421c44d4400def8413d8fcb856ed51ab87786be3eff4a6e1` |
| `N/docker_logs.txt` | `4987b695345c606940bc552a21bea902786467fd4ff136032bbf28cb5212822c` |
| `N/cohort/req_t0_p072/capture_request.json` | `4d7ed4eae05be9347cb0edab8990a30fc8891130f1d356873cc8543b3af47e5e` |
| `T/docker_inspect.json` | `98fbcdb5b367dbe8b171d8b9b98daab74bd94c55a2ef26852417d9f4898e842c` |
| `T/docker_logs.txt` | `8b7678970e0af0bd51a83bce59c88be82532c8e7efe837e3bd3ab536f2ad49bf` |
| `T/cohort/req_t0_p072/capture_request.json` | `cfa190c9d18413b28af3dfafa98a71626a047e31f636c49939107c855b9594fe` |
| `T/loaded_backend/gpu_model_runner.py` | `b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40` |
| `T/loaded_backend/rejection_sampler.py` | `5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f` |
| repository `scripts/fr13_device_multidraft_kernel.py` | `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9` |
