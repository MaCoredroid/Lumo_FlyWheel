# SWE-Bench Q36-A telemetry — authoritative field definitions (2026-05-23)

Source of truth: `src/lumo_flywheel_serving/inference_proxy.py`
(`TrackBRequestMetricsCapture`, `_build_request_metrics_row`,
`compute_deltas`). This resolves the ambiguity that produced the >100%
`decode_sum_s / wallclock_s` ratio at "B=1".

## Per-request JSONL rows (vllm_request_metrics.jsonl / dgx_proxy_capture_full.jsonl)

One row is emitted by the **proxy** when a `/v1/responses` request completes.

| field | exact meaning | clean per-request? |
|---|---|---|
| `ts_request_received` | proxy wall-clock when it received the client request (ISO, ms) | yes |
| `ts_first_byte` | proxy wall-clock at first SSE byte from upstream | yes |
| `ts_completed` | proxy wall-clock when the response finished | yes |
| `wallclock_s` | `ts_completed - ts_request_received` (proxy side) — includes queue + prefill + decode + network/SSE overhead | yes (but proxy-side, not GPU time) |
| `first_byte_s` | `ts_first_byte - ts_request_received` — TTFT incl. queue+prefill | yes |
| `prompt_tokens` / `completion_tokens` | from the response `usage` block. **completion_tokens is the final emitted/accepted token count** (net of speculative rejects). | yes |
| `prefill_sum_s` | **DELTA of the global counter `vllm:request_prefill_time_seconds_sum`** between a `/metrics` snapshot taken just before the request was forwarded and one taken just after it completed | **NO — global, see below** |
| `decode_sum_s` | **DELTA of the global counter `vllm:request_decode_time_seconds_sum`** over the same before/after window | **NO — global, see below** |
| `spec_decode_num_accepted_tokens` / `_draft_tokens` / `_drafts` | DELTAS of the corresponding global `vllm:spec_decode_*_total` counters over the same window | **NO — global** |

### NEW (2026-05-23): network-deducted per-request vLLM compute time
To get clean per-agent decode timing **without** turning off NONSTREAM_BYPASS
(which would disable the proxy AUTO_CONTINUE solve-rate feature), the proxy now
records the upstream (proxy→vLLM) call boundaries:
| field | meaning |
|---|---|
| `ts_upstream_sent` | proxy wall-clock just before it POSTed the request to vLLM |
| `ts_upstream_recv` | proxy wall-clock when the full upstream (non-streamed) response was received |
| `upstream_compute_s` | `ts_upstream_recv - ts_upstream_sent` |

Because proxy and vLLM are **co-located on the DGX (127.0.0.1)**, `upstream_compute_s`
≈ vLLM's actual generation wall (prefill+decode) for that request, with the
**codex↔proxy tunnel network latency deducted**. Validated: a 64-token request
showed `upstream_compute_s`=6.44s vs `wallclock_s`=6.73s (Δ≈0.29s = tunnel RTT).
**Use `completion_tokens / upstream_compute_s` as the clean per-request /
per-agent throughput** (attributable via `oracle_session_id`). Caveat: still
prefill+decode combined (bypass = non-streamed upstream, so no first-token split),
and the AUTO_CONTINUE retry issues extra upstream calls whose time is not in this
single delta (it captures the initial generation).

### The critical caveat (why decode_sum_s/wall can exceed 1)
`prefill_sum_s`, `decode_sum_s`, and the `spec_decode_*` fields are **deltas of
vLLM-global Prometheus histogram/counter sums**, captured by snapshotting
`/metrics` before forwarding and after completion (`compute_deltas`). They are
**NOT clean per-request values**:
- Under **any concurrency** the window aggregates the prefill/decode time of
  *every other request that completed in that window*. At B=2 a row's
  `decode_sum_s` roughly double-counts.
- `request_decode_time_seconds` is **vLLM-internal decode-phase wall time** (the
  request's decode phase as the engine measures it), whereas `wallclock_s` is the
  **proxy-side** elapsed time. The histogram observes on request completion, so
  if another request completed inside the snapshot window its full decode time is
  added — producing `decode_sum_s > wallclock_s` even at nominal B=1 (e.g. an
  overlapping auto-continue/retry, or a tail of a prior request).

**Implication for analysis:** per-request decode tps (`completion_tokens /
decode_sum_s`) is only trustworthy under **strict single-stream isolation with no
overlapping traffic**. For arm-level (B=1 vs B=2) per-stream throughput, prefer
the **server-side iteration step-trace** (below), sliced by arm-window, which is
unambiguous.

## Server-side step-trace (dgx_steptrace.jsonl) — the load-bearing measurement

High-frequency (~1.5s) `/metrics` scrape; cumulative counters, delta per window:
| field | meaning |
|---|---|
| `iter_cnt` | `vllm:iteration_tokens_total_count` — **cumulative engine steps (forward passes)**. Δcount over a window = #forward steps → **per-step latency = window_seconds / Δcount**. |
| `iter_sum` | `vllm:iteration_tokens_total_sum` — cumulative tokens across steps. Δsum/Δcount = **tokens per forward step**. |
| `running` | `vllm:num_requests_running` — instantaneous batch size B (sample to get the B-occupancy distribution per arm). |
| `waiting` | `vllm:num_requests_waiting` — queued requests (should be 0 for B≤max_num_seqs=4). |
| `gen` / `prompt` | `vllm:generation_tokens_total` / `prompt_tokens_total`. Δgen/window = aggregate gen tps. |
| `acc`/`draft`/`drafts` | spec-decode totals; Δacc/Δdraft = acceptance rate; Δacc/Δdrafts = accepted-per-step. |
| `dec_sum`/`pre_sum` | same global histogram sums as above (server-wide). |
| `gpu_util` | nvidia-smi GPU util %. |
| `mem_util` | nvidia-smi memory-controller util %. **Reads 0 on GB10 (unified LPDDR5x exposes no real bandwidth counter)** — do NOT use as a bandwidth signal; use per-step latency instead. |
| `power_w` / `temp_c` | board power / temp. |

**Per-step latency** from `iter_cnt` deltas is the direct test:
~155 ms/step observed at B=1 in a spot check (vs the 121 ms bandwidth-only
prediction) → consistent with per-pass overhead above pure weight+KV streaming.
Comparing B=1 vs B=2 per-step latency decides bandwidth-bound batching
(step time rises ~143/121≈1.18× at B=2) vs overhead-amortization (B=1 step time
inflated by fixed per-pass overhead that B=2 amortizes).

## Concurrent-stream timeseries (derived, committed as concurrency_overlap.json)
Computed post-hoc from the proxy capture rows' `[ts_request_received,
ts_completed]` intervals: for each arm, the fraction of wall-time with 0/1/2
requests open simultaneously, and per-instance the fraction of its wall paired
with another instance. Explains how much of a c=2 arm was truly B=2 vs solo.
