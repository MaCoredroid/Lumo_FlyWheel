# Auto-Research Status Report — 2026-04-23

## Scope

This report covers the current status of the serving auto-research round started from:

- Spec: `docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md`
- Model: `qwen3.5-27b`
- Family: `proposal-ranking-manager-judgment`
- Sprint: `sprint-0`
- Round ID: `qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z`
- Round branch: `autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/20260423T224422Z`
- Live backend: vLLM on `127.0.0.1:8100`, inference proxy on `127.0.0.1:8101`

## Current Round State

The round is not actively running right now. Monitoring from 23:25–23:30 UTC showed:

- No active `lumoserve auto-research measure` process.
- No active `codex exec` process.
- vLLM stayed healthy and idle.
- The round ledger did not change during monitoring.

The actual round ledger is on the round branch, not on `main`. Reading `auto-research status` from `main` reports a misleading `bootstrapped` state because the tracked `output/auto_research/.../results.tsv` from the round branch is not present on `main`.

## Measurement Ledger

Current committed rows on the round branch:

| iteration | status | feasible | TTFT p95 ms | TPOT p95 ms | Turn latency p95 ms | rollout throughput | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_a` | `baseline` | false | 50905.318 | 86.275 | 145443.767 | 6.667 | default-config baseline replay a |
| `baseline_b` | `baseline` | false | 50793.29 | 86.226 | 145123.686 | 6.667 | default-config baseline replay b |
| `001` | `discard` | false | 50777.689 | 86.291 | 145079.112 | 6.667 | failed TTFT/TPOT/turn-latency SLOs and rollout floor |

Candidate `001` used the prior live winner shape:

```yaml
max_num_seqs: 1
max_num_batched_tokens: 6144
enable_chunked_prefill: true
enable_prefix_caching: false
gpu_memory_utilization: 0.3
max_model_len: 32768
kv_cache_dtype: fp8_e5m2
```

It did not improve the live measured SLOs in this harness run. All three measured rows failed:

- TTFT SLO: target `35000 ms`, observed about `50.8 s`.
- TPOT SLO: target `80 ms`, observed about `86.2 ms`.
- Turn latency SLO: target `35000 ms`, observed about `145 s`.
- Rollout floor: target `10.0`, observed `6.667`.

## Relevant Commits

Implementation and round commits are split across `main` and the round branch.

On `main`:

- `1f23977` — Phase A auto-research substrate.
- `dc4ca31` — Phase A preflight/invariant fixes.
- `3638329` — round ledger invariants.
- `4b0fef5`, `e8cf1c5`, `ec74212`, `9cb0344` — follow-up fixes for measurement, staging/index hygiene, and seed-capture contract.

On the round branch:

- `0816c62` — measurement/finalize contract fixes on round branch.
- `b543d8b` — allow ledger commits for ignored `output/` artifacts.
- `8b92b61` — committed `baseline_a`.
- `3bc1bb4` — committed `baseline_b`.
- `4a0929e` — ledger staging/brief fixes on round branch.
- `0c37626` — committed candidate `001` as discarded.

## Issues Encountered

1. `commit-candidate` initially could not commit round artifacts because `output/` is ignored by git. This was fixed by force-adding the explicit ledger paths selected by the CLI.

2. The first `codex exec` launch used per-round `CODEX_HOME`, which isolated Codex auth and failed with OpenAI websocket `401 Unauthorized`. Retrying with normal Codex home fixed auth.

3. The Codex iteration violated the Phase B contract by switching the repo back to `main` and making source-code commits instead of only writing `candidates/001/candidate.yaml`. I stopped that process and completed measurement/commit manually from its candidate file.

4. Branch drift makes monitoring unsafe unless commands explicitly read the round branch. When checked out on `main`, round status can read as `bootstrapped` even though the round branch has three committed rows.

5. Current measured performance is below every live acceptance rail tested so far. Even the previous tuned candidate shape failed the real harness in this run.

## Recommended Next Step

Before starting iteration `002`, stabilize the loop mechanics:

1. Run the round from the round branch only.
2. Prevent per-iteration Codex from running git branch/source-edit commands.
3. Use explicit allow-listed commands or a wrapper that only permits writing `candidate.yaml`, calling `measure`, and calling `commit-candidate`.
4. Reconcile whether the latest `main` fixes should be merged/cherry-picked into the round branch before continuing.
5. Only then launch `002`; otherwise the branch drift and out-of-contract Codex edits are likely to recur.

