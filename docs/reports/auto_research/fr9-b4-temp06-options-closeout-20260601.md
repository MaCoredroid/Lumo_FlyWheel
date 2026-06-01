# FR9 B4 Temp 0.6 Options Closeout - 2026-06-01

Status: **BLOCKED_NEEDS_USER_HELP**.

I ran the requested FR9 independent-row option sweep sequentially until arm 3
could not pass the vLLM prelaunch gate under the required settings. Arms 1 and 2
have accepted artifacts. Arm 3 failed three fresh-tag relaunch attempts before
the SWE driver launched, with the same vLLM free-memory threshold failure. Arms
4 and 5 were not run because the matrix was required to be sequential and arm 3
never produced a valid campaign.

## Fixed Settings

- Subset: `docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`
- Subset size: 16 exact `instance_ids`
- Suite: `swe`
- Concurrency: 4
- Temperature: 0.6
- Agent wall: 1800 s
- Eval timeout: 1800 s
- Nsight: off
- Config: `Fb`
- Row mode: `independent`
- No `--limit`
- No `--no-commit`
- `LUMO_PROXY_FORCE_TOP_P` was unset before launch attempts
- `LUMO_SUDO_PASSWORD` was sourced from `.lumo.local.env`

Canonical command shape used:

```bash
source .lumo.local.env 2>/dev/null
export LUMO_SUDO_PASSWORD
unset LUMO_PROXY_FORCE_TOP_P
SUB=docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json
COMMON="--suite swe --subset $SUB --concurrency 4 --temp 0.6 --agent-wall-s 1800 --eval-timeout-s 1800 --nsight off"
.venv/bin/python scripts/run_codex_experiment.py \
  --exp-tag <tag> --config Fb --row-mode independent --mtp <M> --spines <S> --apply-config $COMMON
```

## Campaign Results

| Arm | Decision | Exp tag | instances_total | Verdict counts | resolved_rate | Summary commit |
|---|---:|---|---:|---|---:|---|
| mtp=5, spines=1 | clean | `fr9_b4temp06_mtp5_s1_20260601T230213Z` | 16 | failed=15, resolved=1 | 0.0625 | `a31ae394` |
| mtp=5, spines=2 | invalid prelaunch, then clean rerun | `fr9_b4temp06_mtp5_s2_20260601T233000Z` | 16 | failed=16 | 0.0 | `daadbe38` |
| mtp=3, spines=2 | blocked | none accepted | missing | missing | missing | none |
| mtp=2, spines=2 | not run | blocked by sequential arm 3 gate | missing | missing | missing | none |
| mtp=2, spines=3 | not run | blocked by sequential arm 3 gate | missing | missing | missing | none |

The invalid arm-2 prelaunch tag was `fr9_b4temp06_mtp5_s2_20260601T232538Z`.
It produced no SWE driver output and no accepted metrics.

The blocked arm-3 attempts were:

- `fr9_b4temp06_mtp3_s2_20260601T234537Z`
- `fr9_b4temp06_mtp3_s2_20260601T234919Z`
- `fr9_b4temp06_mtp3_s2_20260601T235241Z`

All three produced no local or remote SWE output directories.

## Artifact Paths

Arm 1:

- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/driver.log`
- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/campaign_summary.json`
- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/agentic_summary.json`
- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/per_req_spec_trace.jsonl`
- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/dgx_steptrace.jsonl`
- `output/fr9_b4temp06_mtp5_s1_20260601T230213Z/fr9_b4temp06_mtp5_s1_20260601T230213Z/per_task/`

Arm 2:

- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/driver.log`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/campaign_summary.json`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/agentic_summary.json`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/per_req_spec_trace.jsonl`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/dgx_steptrace.jsonl`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/independent_winner_trace.jsonl`
- `output/fr9_b4temp06_mtp5_s2_20260601T233000Z/fr9_b4temp06_mtp5_s2_20260601T233000Z/per_task/`

## Decode And Spec Summaries

Summarizer command used, omitting `--nodes` because node-count semantics were
not known and should not be guessed:

```bash
.venv/bin/python scripts/summarize_round_f_agentic_arm.py \
  --exp-dir output/<tag> --label <label> --out output/<tag>/agentic_summary.json
```

| Arm | spec_events | accepted_tokens | draft_tokens | accept/event | accept/draft | generation_tokens | decode_tps | mean_gpu_util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mtp5_s1 | 6990 | 18430 | 34950 | 2.6366 | 0.5273 | 25397 | 34.3035 | 89.9916 |
| mtp5_s2 | 467 | 1305 | 2335 | 2.7944 | 0.5589 | 891 | 6.5691 | 68.8409 |

These summaries are sliced by the driver start/end timestamps recorded in
`driver.log`. The arm-2 decode window is much shorter because all 16 tasks failed
quickly. It is not a valid speed comparison against arm 1.

## Independent Winner Trace Invariants

| Arm | Trace status | winner_events | winner_spine_counts | superset_violations | missing_state_copy_sum | winner_acc_sum | spine0_acc_sum |
|---|---|---:|---|---:|---:|---:|---:|
| mtp5_s1 | absent as expected for `spines=1` | n/a | n/a | n/a | n/a | n/a | n/a |
| mtp5_s2 | present | 253 | spine0=243, spine1=10 | 0 | 0 | 908 | 889 |

For `mtp5_s2`, the winner trace accepted 19 more tokens than spine 0 over the
driver-window events. Superset violations were computed by checking
`winner_acc >= max(counts.values())` for each winner event. Missing state-copy
sum was computed from `copy.missing`.

## Contamination Audit

| Arm / attempt | Decision | Evidence |
|---|---|---|
| `fr9_b4temp06_mtp5_s1_20260601T230213Z` | clean | Preflight: clean `main`, no local/remote runner process, output dir absent locally/remotely, subset count 16, no limit, top-p unset. Driver: fresh start at `2026-06-01T23:12:05Z`, `n=16`, `concurrency=4`, no skip/resume lines. Postcheck: 16 expected task IDs, top-level summary copied from generated nested summary, `instances_total=16`. |
| `fr9_b4temp06_mtp5_s2_20260601T232538Z` | invalid | vLLM failed before SWE launch. Root cause: free memory `104.99/117.51 GiB` was below requested `gpu_memory_utilization=0.9` threshold `105.76 GiB`. No local or remote SWE output accepted. |
| `fr9_b4temp06_mtp5_s2_20260601T233000Z` | clean rerun | Fresh tag after memory recovery. Preflight clean, output absent locally/remotely, top-p unset. Driver: fresh start at `2026-06-01T23:39:14Z`, `n=16`, `concurrency=4`, no skip/resume lines. Postcheck: 16 expected task IDs, `instances_total=16`, winner trace present with violations=0 and missing=0. |
| `fr9_b4temp06_mtp3_s2_20260601T234537Z` | invalid prelaunch | vLLM failed before SWE launch at required launch shape: `--mtp 3 --spines 2`, `max_num_seqs=8`, `gpu_memory_utilization=0.9`. Free memory was `105.25/117.51 GiB`, below requested `105.76 GiB`. No local or remote SWE output. |
| `fr9_b4temp06_mtp3_s2_20260601T234919Z` | invalid prelaunch | Fresh tag after cleanup. Same vLLM engine-start failure before SWE launch. No local or remote SWE output. |
| `fr9_b4temp06_mtp3_s2_20260601T235241Z` | blocked | Fresh tag after drop-caches, swap cycle, and memory compaction. Same vLLM engine-start failure before SWE launch. Current durable log excerpt: `ValueError: Free memory on device cuda:0 (105.05/117.51 GiB) on startup is less than desired GPU memory utilization (0.9, 105.76 GiB).` No local or remote SWE output. |
| mtp=2, spines=2 | not run | Sequential sweep stopped because arm 3 could not produce a valid campaign under required launch settings. |
| mtp=2, spines=3 | not run | Sequential sweep stopped because arm 3 could not produce a valid campaign under required launch settings. |

## Commit Hashes Pushed

Arm 1 task commits:

`5a1aefef`, `2ed6e55d`, `88c29d8f`, `842d5e63`, `a9141c00`,
`6cdde850`, `c044467e`, `05698e2b`, `6109c923`, `24f02a56`,
`330e3ee2`, `e7ee6e26`, `faa78061`, `6884c349`, `b4872832`,
`9b85e270`.

Arm 1 summary commit: `a31ae394`.

Arm 2 task commits:

`f21918ca`, `b2ce569b`, `72a2c23e`, `39e4e52b`, `0f24eac7`,
`3bdf49cf`, `ca02ecb0`, `171a3ad9`, `82f86a2e`, `1b7cbc6e`,
`3a1ebcfa`, `977522a6`, `7c9aff2c`, `73e32536`, `8f9be747`,
`769b6e62`.

Arm 2 summary commit: `daadbe38`.

This report commit is recorded in git after this file is committed.

## Red-Team Notes

- The two accepted arms have sharply divergent workloads. Arm 1 resolved 1/16
  and generated 25,397 tokens in the driver window. Arm 2 resolved 0/16 and
  generated only 891 tokens. Speed/throughput comparisons are weak and should
  not be presented as a clean mtp5_s1 vs mtp5_s2 performance result.
- Temp 0.6 is still stochastic. Even with the same subset and settings, the
  agents sampled different trajectories, many tasks produced empty-patch
  retries, and low-resolved arms can collapse into short, low-information
  workloads.
- Arm 2 has a valid winner-trace invariant result but not a meaningful
  throughput win/loss result. The accepted trace says the mechanism did not
  violate the winner superset or state-copy invariants over the recorded events.
- The blocked arm is an infrastructure capacity problem under the required
  launch settings, not a SWE task contamination. Lowering
  `gpu_memory_utilization`, killing unrelated user processes, or changing the
  command shape might get past it, but that would no longer be the requested
  run without operator approval.
- The first arm-2 tag and all arm-3 tags are not carried forward as valid
  metrics.

## What Was Not Done

- No enhanced MTP plus suffix-tree work was run.
- No direct greedy Phase A/B sweep was run.
- No unsupported performance claim is made from these agentic traces.
- Arms `mtp=2, spines=2` and `mtp=2, spines=3` were not run because the
  sequential arm-3 gate was blocked.

## Verification Performed

- Confirmed every accepted tag has `driver.log`.
- Copied generated nested `campaign_summary.json` to the requested top-level
  `output/<tag>/campaign_summary.json` and verified `instances_total=16`.
- Verified accepted task IDs match the 16 expected `instance_ids`.
- Verified runner logs record `config=Fb`, `row_mode=independent`, intended
  `mtp`/`spines`, `temp=0.6`, and `concurrency=4`.
- Verified `independent_winner_trace.jsonl` is present for accepted `spines=2`.
- Ran `scripts/summarize_round_f_agentic_arm.py` for accepted arms, with no
  guessed `--nodes`.
- Inspected git status, git log, pushed task commits, and pushed summary commits.
