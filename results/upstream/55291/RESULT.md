# RESULT — vllm-project/vllm #55291 reproduction attempt

**Verdict: DOES NOT REPRODUCE** on stock vLLM 0.28.0, one GB10, TP=1.

Protocol pre-registered in `CARD.md` (written before the server was launched), with
`CARD.md`'s addendum and `DEVIATIONS.md` recording everything that changed and why.
Nothing was posted to GitHub.

## What was run

One server launch, 3 h 16 min of traffic, the pre-registered phases P0 → P4.

| | |
|---|---|
| Server up | 2026-09-17 23:09:34 → 2026-09-18 02:31 UTC (ready after 350 s) |
| Soak | 23:14:24 → 02:30:58 UTC (3 h 16 min), ended on the pre-set budget, not on a failure |
| Requests | **298** — 26 CANARY, 120 SHORT, 120 LONG, 32 GROW |
| Generated tokens | **249,544** |
| Peak context | **36,496** tokens (prompt + generated), `max_model_len` 128,000 |
| Client-side errors | **0** |
| Server `request_success_total` | 261 `length` + 33 `stop` |
| Preemptions | **0** |
| Prefix cache | 1,435,101 queries / 406,112 hits (28.3 %) — caching was genuinely exercised |
| Latency | p50 166 s, p95 363 s (20-way concurrency, long generations) |
| Launch attempts | 1 — `max_model_len 128000` worked first try, no ladder fallback |

Phases actually reached: P0 baseline (serial) ✓, P1 serial soak 6/6 rounds ✓,
P2 20-way concurrent 8/8 batches ✓, P3 long-context growth 7 rounds ✓, P4 began at
02:30:58 and was cut by the budget. Roughly 75 minutes of the run was at the reporter's
concurrency (`--max-num-seqs 20`).

## The measurement

Detection criterion, fixed in advance (`CARD.md` §7), on the **generated token ids**:
a run of ≥ 16 consecutive `!` characters, or ≥ 90 % `!` tokens over ≥ 32 tokens.

| | |
|---|---|
| COLLAPSES | **0 / 298** |
| Longest `!` run anywhere | **1 character** (threshold: 16) |
| Longest run of *any* repeated token | **4** (token 22, in `P1_r0_long_1`) |
| Canaries collapsed | **0 / 26** — the issue's verbatim probe, sent solo, throughout |
| Server log lines matching `traceback\|error\|nan\|assert` | **0** |

The detector is not silently dead: it registered a naturally occurring single `!`
during the run, and before the run it was unit-tested on synthetic sequences **and**
validated against a real saved response with synthetic collapses injected (a 3-token
`!!!!!!!!` run, a 20-token `!` run, and a 50-token non-`!` repeat as a negative).

## Why it likely did not reproduce — the substantive finding

`STATIC_FINDINGS.md` has the full read-only analysis of the pinned wheel. The load-bearing
point, which also **corrects** a claim made in `CARD.md` §2 before the run:

**In 0.28.0, `VLLM_GDN_DECODE_KERNEL` defaults to `cuda`.** This model satisfies every
condition in `_fused_gdn_decode_unsupported_reason()`, so GDN decode runs the fused CUDA
kernel `torch.ops._C.fused_gdn_decode_post_conv_mtp`, **not** the Triton
`fused_sigmoid_gating_delta_rule_update` readout that PR #54146 patches. The server
confirms it:

    qwen_gdn_linear_attn.py:505  GDN decode kernel: cuda
    qwen_gdn_linear_attn.py:158  Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)

with no `Falling back to the Triton GDN decode path:` line anywhere in the log.

That unfixed Triton line (`fused_sigmoid_gating.py:225`, `o = q.new_empty(NK, *v.shape)`)
is still present in 0.28.0 and still on main — PR #54146 is open, unmerged, and the file's
last commit is the July `626c90b2d` refactor. It is simply no longer on this model's
decode path.

**This is not a Blackwell artefact.** The only GPU-dependent condition is compute
capability ≥ 8.0. The reporter's L20-D is Ada (8.9) and clears it, so 0.28.0 on their
hardware should also take the fused CUDA kernel. The version, not the GPU, is what moves
the Triton readout off the decode path here.

Secondary answers for 0.28.0, from source plus this run:

- **Q2** — `mamba_ssm_cache_dtype=auto` still resolves to **float32**
  (`models/config.py:781-806` copies HF `text_config.mamba_ssm_dtype`).
- **Q3** — the model is **bfloat16**, whose exponent range matches fp32, so #54146's
  fp16 ~65504 overflow → `inf` → `NaN` cannot fire on it; downcasting loses precision,
  not magnitude.
- **Q4/Q5** — `KVBlockZeroer` does still skip Mamba (`v1/worker/utils.py:100+`, verbatim:
  *"Only AttentionSpec layers are processed; Mamba layers are skipped."*), and
  `needs_kv_cache_zeroing` is switched on *by* Mamba layers while the zeroing applies
  only to attention blocks. But GDN state is addressed per request, and the prefill path
  zeroes it for any sequence without an initial state
  (`initial_state[~prefill_has_initial_state, ...] = 0`), so a fresh request does not
  inherit a stale one. The remaining way in is a request that never takes the prefill
  branch — the bug class of #51483 (merged 2026-09-16) and its open sibling #51562.

## What this does and does not establish

**Does.** On one GB10, stock 0.28.0, the pinned FP8 snapshot, TP=1, bf16 activations,
prefix caching and chunked prefill on, `--max-num-seqs 20`, 298 requests / 249,544
generated tokens / 3 h 16 min / contexts to 36 k: the repeated-`!` collapse did not
occur once, and the issue's own canary stayed coherent from the first request to the
last. The service also showed none of the reporter's secondary signs — no errors, no
preemptions, no NaN/Inf in the log.

**Does not.** It does not show the reporter is wrong. Untested: vLLM 0.21.0 (their
version), TP=2, 2 × L20-D, `--gpu-memory-utilization 0.85`, 128 k contexts actually
filled, multimodal requests, tool-calling traffic, and production prompt diversity. The
trigger is stated to be non-deterministic, so a null over one 3-hour window bounds the
rate; it does not prove absence. And both structural findings the thread raised remain
true of 0.28.0 — the unfixed readout line exists, and the zeroer skips Mamba — they are
just not reachable on this configuration's decode path.

**Most useful next step for the issue:** ask the reporter to check whether their startup
log says `GDN decode kernel: cuda` or `Falling back to the Triton GDN decode path:
<reason>`. That one line decides whether the #54146 mechanism is even live for them.

## Artifacts

All under `/home/mark/shared/exp54928/exp55291/`:

- `CARD.md` — pre-registered protocol + post-hoc addendum
- `STATIC_FINDINGS.md` — read-only source analysis (F1-F7), including the F2 correction
- `DEVIATIONS.md` — everything that departed from the card
- `RESULT.md` — this file
- `DRAFT_COMMENT.md` — ≤ 200-word issue comment, **not posted**
- `soak.py`, `run_soak.sh`, `analyze_soak.py`, `download.sh`, `wait_markers.sh`
- `runs/soak/` — `records.jsonl` (298 per-request records with token-id analysis),
  `meta.json` (resolved argv, `/v1/models`, `/version`), `server_attempt1_mml128000.log`,
  26 `metrics_*.txt`, `full_*.json` + `req_*.json` for kept requests
- `runs/aborted_detector_bug_soak/` — the 8-minute first launch, excluded from the result
- `logs/soak.log`, `logs/download.log`
