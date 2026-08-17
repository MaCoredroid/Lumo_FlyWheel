# FR14 Ablation Arm A — Step 2: their stack runs our SWE agent

**Lane: DIAGNOSTIC. Calibration-grade, NON-CITABLE.**

2026-08-16 23:42Z → 2026-08-17 01:10Z · GB10 · agent + proxy on alienware, GB10 engine-only
(identical topology to the stock B1 arm, so decode TPS is uncontended).

**Status: PARTIAL.** 3 of 4 tasks completed with verdicts; task 4 (`astropy-13398`) was stopped
in flight at ~40 min when the 2.5 h GPU budget expired. Decode telemetry covers the whole window.

## The completed triangle

| leg | workload | engine | sustained decode TPS | accept | step |
|-----|----------|--------|----------------------|--------|------|
| 1 (pass 11) | random 1024/1024 | sglang EAGLE 3/1/4 | 26.10 | 2.90 | ~111 ms |
| 2 (pass 10) | 4x SWE-Verified, qwen-code | our fixed32 tail6 tree | 25.26 | 4.526 | 218.8 ms |
| 3 (leg 3)   | random 1024/1024 | our fixed32 tail6 tree | 19.27 | 2.358 | 171.6 ms |
| **4 (this)** | **4x SWE-Verified, qwen-code** | **sglang EAGLE 3/1/4** | **29.22** | **3.363** | **115.1 ms** |

Restricting leg 4 to the 3 tasks that completed: **31.71 TPS, accept 3.450, 108.8 ms/step**.

## The finding

**On our own workload, their chain is faster than ours: ~29.2 tok/s vs 25.3.**

The mechanism is now fully separated:

- our tree **does** accept more per verify — 4.526 vs 3.363 = **1.31x**;
- their step costs **115 ms** against our **219 ms** — a **1.90x** step-cost advantage;
- 1.31 < 1.90, so they win. Our acceptance edge does not pay for our machinery.

### Correction to pass 11

Pass 11 recorded *"Tree accepts 1.9x more per verify"* by putting our 4.53 (SWE traffic) next to
their 2.90 (random-1024 traffic). That was a **cross-workload** comparison and it overstated the
premium. Measured on the same workload:

- on **our** traffic: ours 4.526 vs theirs 3.363 → tree premium **1.31x**
- on **their** traffic: ours 2.358 vs theirs 2.90 → tree premium **0.81x** (we accept *less*)

The tree's acceptance advantage is real but it is roughly **1.3x on agentic code traffic and
negative on unpredictable traffic** — not 1.9x anywhere.

### What the step-cost gap is made of

Their 115 ms step is **not** a pure machinery result. Two things differ at once:

1. **machinery** — a 4-token EAGLE chain (3 steps, topk 1) with no tree attention, no merged
   drafter, no committer, against our 31-draft fixed32 tree with drafter + tree-attention verify
   + committer;
2. **bytes** — they serve RadixArk **as-shipped** (aggressive quant, auto-fp8 KV); we serve the
   conservative NVFP4 with bf16 KV. Our own weight floor for that checkpoint is 102.5 ms, i.e.
   *our floor alone is ~89% of their entire measured step*.

That second point is the actionable one: their whole step fits inside our mandatory weight read.
No amount of machinery work closes this while the byte budget differs. **Bytes first.**

## Quality (bonus, non-citable)

| task | ours (stock B1) | theirs (sglang) |
|------|-----------------|-----------------|
| astropy-12907 | resolved | **resolved** |
| astropy-13033 | failed   | **failed**   |
| astropy-13236 | resolved | **resolved** |
| astropy-13398 | failed   | *stopped in flight* |

Verdict pattern is identical on all three completed tasks. Their chain is **not** trading
correctness for speed on this subset, and the RadixArk as-shipped checkpoint is not degenerate.

## Wall and tokens per task

| task | ours wall | ours out tok | ours e2e tps | theirs wall | theirs out tok | theirs e2e tps | theirs reqs |
|------|-----------|--------------|--------------|-------------|----------------|----------------|-------------|
| 12907 | 192.1 s | 3,572 | 18.59 | 238.7 s | 5,895 | 24.70 | 11 |
| 13033 | 326.9 s | 7,812 | 23.89 | 1,363.5 s | 41,955 | 30.77 | 6 |
| 13236 | 515.7 s | 10,907 | 21.15 | 1,195.5 s | 38,737 | 32.40 | 39 |
| 13398 | 2,313.6 s | 49,165 | 21.25 | *2,403 s (partial)* | *58,711* | *24.43* | *70* |
| **3-task total** | 1,034.7 s | 22,291 | 21.55 | **2,797.7 s** | **86,587** | **30.95** | 56 |

**Wall-per-task is NOT a like-for-like speed comparison here.** Their trajectories are far
longer: 86,587 output tokens against our 22,291 over the same three tasks (3.9x). Their wall is
longer *because they generate almost four times as many tokens*, at a higher rate. `13033` is the
extreme case: a single ~32k-token capped runaway turn (6 model requests, 41,955 tokens, 0-byte
patch) — the "flavor-2 endless-reasoning" failure the proxy's 32,768-token output cap exists to
bound. The **decode TPS and accept numbers are the comparable ones**; wall is confounded by
trajectory length.

## Engagement proof

- sglang's own decode lines carry `accept len` continuously (1,017 samples), so EAGLE spec
  decoding was live for the entire run — `ablation_a_step2_sglang_decode_tail.txt`.
- `--reasoning-parser qwen3` demonstrably active: responses carry a separate `reasoning_content`
  field (`ablation_a_step2_models.json` / smoke).
- `--tool-call-parser qwen3_coder` demonstrably active: the agent executed 39 tool calls on
  13236 and produced a real 1,273-byte patch that the official grader **resolved**.
- proxy pins captured live from `/proc/<pid>/environ` on alienware:
  `ablation_a_step2_proxy_env.txt` (temp 0.6 / top_p 0.95 / top_k 20 / presence 1.0 / min_p 0,
  upstream `http://100.103.10.122:9950`).
- 5,069,585 prompt tokens and 145,298 output tokens across 126 model requests went through their
  engine.

## Compromises

1. **PARTIAL.** Task 4 stopped in flight at the budget wall; no verdict for `astropy-13398`.
   Its partial numbers are marked *italic* above and flagged in the JSON. The 3-task totals and
   all decode telemetry are complete.
2. **Trajectory length not matched** — see the table note. Only decode TPS / accept are
   like-for-like.
3. **Checkpoint not matched** — theirs is RadixArk as-shipped (aggressive bytes + fp8 KV), ours
   conservative NVFP4 + bf16 KV. Deliberate ("their chain as they ship it"), but it means the
   1.90x step gap is machinery **and** bytes.
4. **The proxy is ours.** The agent talks to the lumo offload proxy (sampling pins, 32k output
   cap) which talks to sglang. That is exactly the stock arm's topology — controlled — but it is
   their **engine + checkpoint + parsers**, not their chain end to end.
5. **Proxy start gate false-negative.** `offload_codex_proxy.sh start` exits 5 on a class-9 pin
   check that greps for a literal `~` path (`REMOTE_PAIR_DUMPS`) that the remote shell expands.
   The proxy was up and correctly pinned; the gate is a cosmetic bug on the non-fixed32 branch.
   The run deliberately continued past it. Worth fixing.
6. `running_req_max=3` appears in the telemetry despite harness concurrency 1 — qwen-code issues
   occasional parallel sub-requests. The overwhelming majority of samples are `#running-req: 1`.
7. sglang was served on **:9950** with `--host 0.0.0.0` and `--enable-metrics` (the calibration
   used 127.0.0.1:30000, no metrics), so the offload proxy could reach it and the harness's
   `DEFAULT_METRICS_URL` bracketed it per task. Every model/recipe/parser flag is theirs verbatim.

## Artifacts

- `ablation_a_step2.json` — the numbers
- `ablation_a_step2_sglang_cmdline.txt` — their engine's PID-1 argv as served
- `ablation_a_step2_sglang_decode_tail.txt` — decode-batch telemetry sample
- `ablation_a_step2_proxy_env.txt` — live proxy pins on alienware
- `ablation_a_step2_models.json` — served model identity
- `ablation_a_step2_sglang_boot.sh`, `ablation_a_step2_run_swe.sh`, `ablation_a_step2_reduce.py`
  — the vehicle
