# Round-5 B=4 Sweep — Runbook & Methodology (2026-05-25)

How the config-D vs config-E (MTP depth) sweep was actually run, end to end, so it
is reproducible. This is a **how-we-ran-it** record, not a results analysis (the
depth-scaling analysis from the per-agent traces is deferred).

## 1. Goal

Compare, on a fixed task set at a fixed batch size, the shipping suffix stack
against Qwen3.6's native MTP head at increasing speculative depth:

| Round | exp tag | config | spec method | num_speculative_tokens |
|------:|---------|--------|-------------|:-:|
| 1 | `q36a_D_b4`  | D (full T1+T2+T3+T4 suffix) | `suffix` | 12 (bundle default) |
| 2 | `q36a_E1_b4` | E (Qwen3.6 native MTP head) | `qwen3_5_mtp` | 1 |
| 3 | `q36a_E2_b4` | E | `qwen3_5_mtp` | 2 |
| 4 | `q36a_E3_b4` | E | `qwen3_5_mtp` | 3 |
| 5 | `q36a_E6_b4` | E | `qwen3_5_mtp` | 6 |

Fixed across all rounds: **B=4 (concurrency 4), temperature 0.6, top_p 0.95**, the
same **16 SWE-Bench Verified astropy instances**
(`docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json`),
1800 s codex wall + 1800 s eval budget per task.

> Note on `qwen3_5_mtp`: the served checkpoint `/models/qwen3.6-27b-fp8` is dense,
> `text_config.model_type=qwen3_5_text`, with an in-checkpoint MTP head
> (`mtp_num_hidden_layers=1`, 22 `mtp.*` tensors). `qwen3_5_mtp` is the only vLLM
> MTP path that reads `mtp_num_hidden_layers` (every other reads
> `num_nextn_predict_layers`, which this checkpoint lacks); vLLM normalizes the
> method to `mtp`. "3.6" is branding; vLLM's family is `qwen3_5`.

## 2. Topology

- **DGX Spark (GB10, this host):** vLLM container `lumo-vllm-track-b-suffix` on
  :9950, the codex-bench proxy on :8022, Nsight (host-side). vLLM + proxy + the
  per-agent trace all live here.
- **alienware (x86):** runs `codex-runner:v1` (codex CLI 0.128.0) + the SWE-Bench
  orchestrator, over a reverse SSH tunnel (DGX→alienware: 8022 proxy, 9950 vLLM
  metrics). Keeps the DGX inference-only. See
  [[project-swe-bench-concurrency-probe]] for the tunnel/streamer/steptrace infra
  (`swe_infra` tmux).

## 3. The three scripts (all repo-resident under `scripts/`)

1. **`scripts/swe_x86_helpers/relaunch_qwen36_round.py`** — parameterized vLLM
   relaunch. `--config D` (full T1-T4 suffix stack + the suffix bundle) or
   `--config E --mtp N` (KEEP-prefix-only prelaunch, no suffix patches; generates an
   MTP bundle with `spec_decode: {method: qwen3_5_mtp, num_speculative_tokens: N}`).
   **Both** variants source-edit `Scheduler.make_spec_decoding_stats` to emit the
   per-agent step trace (see §4). Runs ModelServer (sudo, needs `LUMO_SUDO_PASSWORD`).

2. **`scripts/run_codex_experiment.py`** — one experiment (one config, one task set):
   `--config {D,E} [--mtp N] --apply-config` relaunches vLLM into the config;
   `--temp {1.0,0.6}` restarts the proxy with that forced temperature;
   `--concurrency 4`; `--suite swe --subset <on-alienware> [--limit N]`. It launches
   the orchestrator on alienware, then **per finished task** rsyncs artifacts back,
   (optionally) joins Nsight GPU metrics, and **commits + pushes** that task's
   bundle incrementally. `--skip-existing` lets a run resume. `--nsight
   {off,first-task,<secs>}` captures one representative GPU-metrics window.

3. **`scripts/swe_x86_helpers/run_round5_b4_sweep.sh`** — the sweep driver. Runs the
   5 rounds sequentially via `run_codex_experiment.py`, COMMON args
   `--suite swe --subset <16> --concurrency 4 --temp 0.6 --agent-wall-s 1800
   --eval-timeout-s 1800 --nsight off`. Round 1 (D) skips re-applying config unless
   `--r1-apply` is passed (it was passed in the real run, so D relaunched fresh).
   Prints `ROUND k/5` lines and `SWEEP COMPLETE`.

## 4. Per-agent spec-step trace (the load-bearing instrumentation)

At B>1 the global `/metrics` deltas (decode_sum, spec_decode, iteration_tokens)
blend all concurrent streams, so per-stream decode analysis is impossible from
them. Fix: a prelaunch **source-edit** of `vllm/v1/core/sched/scheduler.py`
`Scheduler.make_spec_decoding_stats` (called per-request, per-step, with
`request_id`) appends `{ts, rid, draft, acc}` rows to a bind-mounted host path
`/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl` (container `/logs`).
`rid` is the proxy's session-prefixed request id → maps to each of the 4 agents, so
acceptance + per-step timing stay **clean at any batch size**. Verified per round:
`draft` per row = 1/2/3/6 matching n_spec; `draft_tokens_total ≈ n_spec × rows`.

Gotchas learned the hard way:
- The patch must be a **source-edit** (file is imported fresh by the engine), not a
  runtime monkeypatch.
- Build the injected source with `chr(10)` for **every** newline — a literal `"\n"`
  collapses through the raw-string→heredoc→inner-python→written-source layers and
  yields `SyntaxError: unterminated string literal`, crashing vLLM on start. A
  `py_compile` guard in the patch fails fast at prelaunch if it ever breaks again.
- Reset the trace file **before** each relaunch (`apply_config` rm's it), never
  after the container opened its handle — deleting a live-handle file sends writes
  to an unlinked inode (lost round-1 data once; fixed).
- The proxy also now emits per-request `engine_iterations` /`iteration_tokens`
  (vllm:iteration_tokens_total_{count,sum} deltas) so per-task per-step latency is
  self-contained in `vllm_request_metrics.jsonl`.

## 5. Exact commands used

```bash
# one-time: confirm config D/E relaunch + per-agent trace work, then:
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env && export LUMO_SUDO_PASSWORD          # for sudo vLLM relaunch
nohup bash scripts/swe_x86_helpers/run_round5_b4_sweep.sh --r1-apply \
  > /tmp/round5_b4_sweep.out 2>&1 &
```

The driver internally calls, per round (example for round 3):
```bash
.venv/bin/python scripts/run_codex_experiment.py \
  --exp-tag q36a_E2_b4 --config E --mtp 2 --apply-config \
  --suite swe --subset docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json \
  --concurrency 4 --temp 0.6 --agent-wall-s 1800 --eval-timeout-s 1800 --nsight off
```

## 6. Monitoring during the run

- **10-min cron** (round-agnostic integrity check): verifies vLLM serving, the
  *active round's* config (suffix+T1 for D; `mtp`+no-T1 for E), drafting rising,
  per-agent trace growing, sweep/runner liveness, no fast-fail (<300 s). Restarts
  matching the active round only.
- **5-min `experiment_supervisor.sh`** (tmux `swe_infra:supervisor`): auto-fixes
  tunnel/tmux/proxy; ALERTS (does not auto-restart) on vLLM/runner death.
- A persistent Monitor on `/tmp/round5_b4_sweep.log` for round transitions /
  verdicts / `SWEEP COMPLETE`.

## 7. Pacing & operational notes

- ~B=4 → 16 tasks run in ~4 waves of 4; each task uses the full 1800 s codex wall
  (the agent is wall-bound at this decode speed, not decode-bound). Empty-patch
  `agent_gave_up` triggers one codex retry (~doubles that task's slot). Round
  wall-clock ≈ 2.5–3.2 h; whole sweep ≈ 14.5 h (06:25→20:47 UTC) incl. ~10-min
  vLLM relaunches between rounds.
- `git push` to GitHub:22 from the DGX flaps intermittently; commits are always
  safe locally and a retry loop catches a good window. No ssh-config change made.

## 8. Results (verdicts only; analysis deferred)

Pass rate, 16 instances each, B=4 / temp 0.6 / top_p 0.95:

| config | resolved/16 |
|---|---|
| D (suffix) | 6 |
| E-mtp1 | 7 |
| E-mtp2 | 7 |
| E-mtp3 | 7 |
| E-mtp6 | 6 |

Flat across the suffix baseline and all MTP depths — expected: speculative
decoding is **lossless** (changes decode *speed*, not the sampled distribution).
Per-instance verdicts vary run-to-run (stochastic at temp 0.6); only the aggregate
is comparable. The depth-scaling **speed** story (per-position acceptance +
tokens/step at n_spec 1→2→3→6) is in the committed traces and is the next analysis.

## 9. Committed artifacts (per round dir `output/q36a_{D,E1,E2,E3,E6}_b4/`)

- `q36a_*_b4/per_task/<instance>/` — `patch.diff`, `eval/eval_report.json`
  (verdict), `codex_trace.jsonl` (+`_retry` on empty-patch retries),
  `vllm_request_metrics.jsonl` (per-request decode/accept + engine-iteration
  counters), `runner_metadata.json`.
- `per_req_spec_trace.jsonl` — the per-agent step trace (the §4 instrumentation).
- `dgx_steptrace.jsonl` — 10 Hz global vLLM step/GPU sampler (shared file; per-dir
  snapshot).

Related: [[nsight-gb10]], [[project-swe-bench-concurrency-probe]],
[[project-swe-bench-campaign]], [[reference-lumo-local-env]].
