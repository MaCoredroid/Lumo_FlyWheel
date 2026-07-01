# FR13 — Spec-decode / prefix-cache degeneration literature: banked + applicability to OUR config

**Date:** 2026-07-01
**Purpose:** bank the vLLM/GDN/Mamba spec-decode + prefix-cache degeneration reports surfaced for the 13033
garble, and record a **grounded** per-issue applicability verdict against our EXACT config, plus a fix ladder
**gated on the live matrix arm results** (do NOT apply blind; the arm is the discriminator).

Companion: `FR13_DEGENERATION_INVESTIGATION.md` (the 13033 verdict). This file is the literature + fix plan.

---

## 0. OUR EXACT CONFIG (grounded from boot_log_snapshot.txt + fr13_launch_forked_fa2_tree_server.sh)

| Knob | Value | Source |
|---|---|---|
| model | qwen3.6-27b-**fp8** (Qwen3-Next hybrid **GDN** = gated-delta-net recurrent) | serve exec |
| **kv_cache_dtype** | **auto (UNQUANTIZED)** — fp8 only if `FR13_FULL_ATTN_KV_FP8=1` (default 0) | boot log + script |
| **mamba_ssm_cache_dtype** | **float32 (UNQUANTIZED)** | script default |
| spec method | `qwen3_5_mtp` (MTP), **num_speculative_tokens=5**, tree | SPEC_CONFIG |
| cudagraph_mode | **FULL_AND_PIECEWISE** | CG_FLAGS |
| enable_prefix_caching | **True** (+ chunked prefill), EXACT_SEED=1 | boot log |
| tensor_parallel_size | **1** | boot log |
| batch | B=1, temp 0.6 / top_p 0.95 / top_k 20 (Qwen anti-degeneration preset) | proxy |

**THE load-bearing config fact:** both the KV cache **and** the SSM cache are **UNQUANTIZED** for us. The most
frequently-cited trigger in this literature — **KV quantization (TurboQuant/fp8) × spec decode** — is therefore
**ABSENT in our runs.** This *removes* the top reported cause and narrows our hypothesis to the recurrent-state
/ MTP-reject-rollback / CUDA-graph / APC-restore surface.

---

## 1. Banked reports + per-issue applicability

| # | Report (verified live) | What it says | Applies to us? | Grounding |
|---|---|---|---|---|
| [#40831](https://github.com/vllm-project/vllm/issues/40831) | TurboQuant KV × any spec (MTP/ngram) → degenerate token loops | **NO** | trigger is **KV quant**; we run `kv_cache_dtype=auto` + f32 SSM → trigger absent |
| [#40880](https://github.com/vllm-project/vllm/issues/40880) | MTP × TurboQuant × **CUDA-graph capture** → degenerate on Qwen3-Next hybrid | **PARTIAL** | KV-quant part absent, but **MTP × FULL_AND_PIECEWISE cudagraph** part is ours; `cudagraph_mode=NONE` workaround is testable |
| [#39809](https://github.com/vllm-project/vllm/issues/39809) | Mamba prefix-cache + MTP: **Triton kernel doesn't roll back SSM state on spec-reject** | **YES (mechanism)** | hybrid GDN + MTP-5 + APC is exactly this surface; reject-rollback of recurrent state is quant-independent |
| [#36872](https://github.com/vllm-project/vllm/issues/36872) | **Gibberish output** + collapsing TPS, Qwen3.5-FP8 + spec decode | **YES (symptom, sibling)** | sibling FP8 model; gibberish-with-spec even (implied) without KV quant → spec itself a carrier |
| [#40756](https://github.com/vllm-project/vllm/issues/40756) | MTP crash (illegal mem) long seq **Qwen3.6-27B-FP8** | **PARTIAL** | our EXACT model, but symptom = **crash**, not garble; we have not crashed |
| [#41190](https://github.com/vllm-project/vllm/issues/41190) | **TP=2** qwen3_next_mtp GDN crash at num_accepted_tokens_event.sync | **NO** | we run **TP=1** |
| [#34650](https://github.com/vllm-project/vllm/issues/34650) | MTP breaks `</think>` detection in structured-output + reasoning | **MAYBE** | we use reasoning + tool calls, but our thinking-cap handles `</think>` proxy-side; 13033 garble is pre-tool-call |
| [#40875](https://github.com/vllm-project/vllm/issues/40875) | ngram `prompt_lookup_min=2` tool-call corruption | **NO** | we use **MTP**, not ngram |
| [#43587](https://github.com/vllm-project/vllm/issues/43587) / [#40696](https://github.com/vllm-project/vllm/issues/40696) | APC ineffective / incremental-fail on Qwen3.5 mamba-hybrid | **CONTEXT** | APC-on-hybrid correctness/efficiency; informs block-granularity of our EXACT_SEED |
| [#26201](https://github.com/vllm-project/vllm/issues/26201) | Tracking: prefix caching for hybrid models | **CONTEXT (umbrella)** | our EXACT_SEED is a custom impl on this surface |
| [#46187](https://github.com/vllm-project/vllm/issues/46187) | ReplaySSM ring-buffer faster hybrid decode | **CONTEXT (perf)** | relevant to our fixed-buffer decode work |
| [AI21 blog](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) | "One token to corrupt them all" Mamba state-corruption debug tale | **METHODOLOGY** | same isolation pattern (single-token state corruption) |

### Net applicability (grounded)
Because KV+SSM caches are **unquantized**, the KV-quant reports (#40831; the quant half of #40880) are **off the
table**. Three live hypotheses remain, all on the recurrent-state path — none require quant:

- **H1 — MTP spec-reject rollback gap** (#39809-class): on draft-reject the GDN/SSM recurrent state isn't rolled
  back → "reused one block too many" → wrong recurrent seed → merged/doubled tokens + early EOS.
- **H2 — MTP × FULL_AND_PIECEWISE cudagraph race** (#40880 minus quant): capture/replay timing on the draft path.
- **H3 — our recent commits** reopened it: `.cpu()`-drop (`faecc88d`, ~15%) and/or FIX-A LRU cap (`1e1df386`).

---

## 2. EMPIRICAL UPDATE from the live run (2026-07-01, run_20260701T072605Z)

**The garble is INTERMITTENT on cache-ON, not deterministic.**

| Replica | config | 13033 outcome | garble signature |
|---|---|---|---|
| old run 063919Z m_e5_ON | cache-ON + .cpu()-drop | garble → empty patch → patch_apply_failed | **YES** (spaceless + doubled subword, seq-34) |
| live run 072605Z m_e5_ON | cache-ON + .cpu()-drop (**same code**) | `tests_failed` (**real patch**, 3 attempts) | **NONE** across all 3 traces |

**Reading:** a *deterministic* APC-restore off-by-block would corrupt **every** cache-ON replica of the same
prompt. Non-reproduction on the 2nd cache-ON replica (1 garble / 2) means the mechanism is **intermittent** — a
**low-probability race** (H1/H2) or a genuine **sampling flake amplified by the recurrent path**, *not* a
deterministic lossiness. It also **partially exonerates** the `.cpu()`-drop (a deterministic break would have
reproduced here). Still not settled: N is tiny, and the **cache-OFF control has not run yet**.

---

## 3. FIX LADDER — GATED ON ARM RESULTS (do NOT apply blind; arm is the discriminator)

**Gate 0 (data, in progress):** finish the matrix so 13033 (and all 16) run on **cache-OFF (`EXACT_SEED=0`)**
vs **cache-ON**. Discriminator: garble/early-stop appears **ON-only, repeatably (≥2/N)** → lossiness; appears on
OFF too, or 0 on all ON replicas → flake. *Given §2, "0 on all ON replicas" is currently trending.*

Execute the cheapest applicable fix **only after** Gate 0 says lossiness, **after** the matrix (never mid-arm):

1. **Revert `.cpu()`-drop (`faecc88d`)** — one-line, restores the defensive host-copy + capture-time sync;
   re-run 13033 cache-ON ×N. If garble vanishes → `.cpu()`-drop (H3) was the carrier → keep reverted. *[tests H3]*
2. **`cudagraph_mode=PIECEWISE` (or NONE) on the cache-ON path** — the documented #40880 workaround; re-run.
   If garble vanishes → MTP×cudagraph race (H2). Costs TPS; accept only if it's the carrier. *[tests H2]*
3. **num_speculative_tokens 5→0 on cache-ON** (spec OFF, cache ON) ×N. If clean without spec → the MTP
   reject-rollback gap (H1, #39809) → needs the SSM-state-rollback fix (deeper; see #39809 / ReplaySSM). *[tests H1]*
4. **Raise `FR13_ES_CKPT_CAP` above 13033's live-block count** — rules out FIX-A evicting a live checkpoint. *[tests FIX-A]*

**If Gate 0 says flake** (clean on both / garbles on OFF too / 13033 resolves on ON): **no state-path fix** —
the win is durable and the garble is sampling noise on the recurrent path; document and move on.

**Constraint:** every step above needs the GPU → **BLOCKED while the live arm runs.** Prepared, not executed.
Never temp 0. cache-ON = EXACT_SEED=1.

---

## Files
- Verdict doc: `FR13_DEGENERATION_INVESTIGATION.md`
- Old garble trace: `run_20260701T063919Z/m_e5_ON/proxy_pair_dumps/pair_..._000034_initial.json`
- Live clean 13033: `run_20260701T072605Z/m_e5_ON/swe_out/verified/per_task/astropy__astropy-13033/codex_trace*.jsonl`
- Config: `boot_log_snapshot.txt`; `scripts/fr13_launch_forked_fa2_tree_server.sh`
- Suspect commits: `faecc88d` (.cpu()-drop), `1e1df386` (FIX-A LRU cap)
