# FR14 Ablation Arm A — Leg 3: our tree on their bench shape

**Lane: DIAGNOSTIC. Calibration-grade, NON-CITABLE.** Screen for the ceiling work; not an
acceptance instrument.

Head 61e6b98a1 · 2026-08-16 · GB10, uncontended (engine only, no harness/proxy/agent).

## The triangle

|                        | workload                              | engine                   | sustained TPS | accept |
|------------------------|---------------------------------------|--------------------------|---------------|--------|
| leg 1 (pass 11)        | random 1024/1024, bs=1                | sglang EAGLE 3/1/4       | **26.10**     | 2.90   |
| leg 2 (pass 10)        | 4x SWE-Verified, qwen-code, 35k ctx    | our fixed32 tail6 tree   | **25.26**     | 4.526  |
| **leg 3 (this pass)**  | random 1024/1024, bs=1                | our fixed32 tail6 tree   | **19.27**     | 2.358  |
| leg 4 (step 2)         | 4x SWE-Verified, qwen-code             | sglang EAGLE 3/1/4       | see ablation_a_step2 |  |

## What leg 3 says

On **their** workload our tree loses by ~26% (19.27 vs 26.10). The mechanism is **acceptance,
not step cost**:

- accept/event: **4.526** on real SWE agent traffic → **2.358** on 1024-token random-token prompts.
- step wall: ~172 ms here vs 219 ms on the 35k-context SWE arm — the step actually got *cheaper*
  (short contexts), so the whole loss is the acceptance collapse.
- committed/event 3.358 / 0.1716 s = 19.6 tok/s, which reconciles with the client's 19.27.
- their EAGLE chain barely notices the workload change (2.90 here vs whatever it does on SWE
  traffic — see step 2).

**Reading:** the 1.9x acceptance premium the tree bought in pass 11 is a property of
**predictable agentic code traffic**, not of the tree per se. On unpredictable random-token
traffic the premium evaporates and the fixed 31-draft verify cost is dead weight. Any claim of
the form "our tree beats their chain" must name the workload.

## bs=8 — structurally not comparable, and only partial

Our stock arm's admission width is `--max-num-seqs 1`. Offering 8-way concurrency to it
**serializes**: `/metrics` showed `num_requests_running=1`, `num_requests_waiting=7`
(reason=capacity), `num_preemptions_total=0`. The measured decode-only TPS over the 4 requests
that completed was **18.30** at accept 2.15 — i.e. *zero* batch amortization, as expected. This
is a queueing measurement, **not** a bs=8 throughput measurement, and it must **not** be put
beside sglang's 147.78. A real bs=8 leg would need a B8 boot of the fixed32 arm, which the
launcher's fixed32 memory gates are not provisioned for (they pin expectations at B1 and B4).

The sweep was also cut short (16 prompts offered, client stopped at ~5 min) to protect the
step-2 GPU budget.

## Engagement proof

- `spec_decode_num_draft_tokens / spec_decode_num_drafts = 31.00` exactly, over both phases —
  the fixed32 tail6 tree ran on **every** decode step (31 physical drafts + implicit root).
- engine PID-1 argv is byte-identical to the reference stock arm
  (`output/fr14_b1_stock_20260816T204931Z/tail6_fixed32_b1stock`) except the dropped middleware
  flag — see `ablation_a_leg3_engine_cmdline.txt`.
- the 548-variable container env was diffed against the reference: every decode-path variable
  matches (`ablation_a_leg3_container_env.txt`).
- the merged drafter logged `[FR13_MERGED ENGAGED] TAIL[...] HYDRA[...]` during the run.

## Compromises (all forced, all documented)

1. **Middleware dropped.** The stock arm serves behind
   `Fixed32EngineIngressMiddleware`, an ASGI guard that authenticates a per-task HMAC header
   only the SWE harness mints. `bench_serving` cannot mint it. Leg 3 ran a **one-line-patched
   copy** of the launcher — `scripts/fr14_leg3_launch_nomiddleware.sh`, diff vs HEAD is exactly
   `FR13_FIXED32_MIDDLEWARE_FLAGS=""` plus a comment. The guard is an ingress audit device, not
   a decode-path component.
2. **Sampling asymmetry.** `bench_serving` defaults to `temperature 0.0`. Our fixed32 committer
   hard-refuses greedy — *"FR13 fixed32 requires sampled temp>0, no draft_probs, and
   max_spec_len=31"* — and the engine **died on the first request**, twice
   (`ablation_a_leg3_temp0_refusal.txt`; the flag must go inside `--extra-request-body`, the
   `--temperature` CLI flag is not wired on the vllm-completions path). Leg 3 therefore ran at
   our **deployment** sampling profile (temp 0.6 / top_p 0.95 / top_k 20) — the same profile the
   25.26 number was measured at. Their 26.10 was greedy. The two stacks cannot be run at the
   same temperature; this asymmetry is unavoidable and favours *them* by an unknown small amount
   (greedy accepts more).
3. **Sequence file HEAD-pinned.** A concurrent agent is live-editing `scripts/` on this branch
   (a floor re-derivation moving `FR13_MANDATORY_WEIGHT_BYTES` 27977022848 → 25210209416). Leg 3
   sources a HEAD copy of `scripts/fr13_fixed32_floor_timers_seq.sh` so it reproduces the
   committed b1-stock arm. `src/` was verified unmodified, so the engine's runtime code is
   HEAD-clean.
4. **No SWE runner.** Engine-only, so the fixed32 ready-ack / work-census / boundary-flush
   handshakes never ran and the `fr13_decode_step_wall_*` / `fr13_decode_forward_gpu_*` counters
   stayed 0 (they publish only through the explicit flush transaction). The vLLM-native
   `spec_decode_*` / `request_decode_time_*` counters used above are unaffected.

## Artifacts

- `ablation_a_leg3.json` — the numbers
- `ablation_a_leg3_bench_bs1.txt` — sglang bench_serving summary
- `ablation_a_leg3_engine_cmdline.txt`, `ablation_a_leg3_container_env.txt` — engine identity
- `ablation_a_leg3_boot.sh`, `ablation_a_leg3_bench.sh` — the vehicle
- `ablation_a_leg3_temp0_refusal.txt` — the greedy refusal
- `scripts/fr14_leg3_launch_nomiddleware.sh` — the one-line-patched launcher copy
