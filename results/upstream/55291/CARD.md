# EXPERIMENT CARD — vllm-project/vllm issue #55291

**Written before any GPU work.** Run dir: `/home/mark/shared/exp54928/exp55291/`
Author: Claude (agent), for Mark Ma. Read-only on GitHub; nothing posted.
Card written: 2026-09-17 ~22:30 UTC. GPU not yet touched (waiting on A/B markers).

---

## 1. Target

vllm-project/vllm **#55291** — "[Bug]: Qwen3.6-27B-FP8 eventually collapses into
repeated `!` tokens, affecting all subsequent requests" (reporter `dikongfeixing8`,
opened 2026-09-04, open, 5 comments, label `bug`).

**Claim under test.** A vLLM server running `Qwen/Qwen3.6-27B-FP8` works normally,
then at some non-deterministic point one request emits only `!` tokens; from then
on *every* subsequent, independent, short, unrelated request also returns only `!`.
The process stays alive, HTTP stays responsive, throughput looks normal, and the
state never self-heals without a restart.

**What the reporter asks reproducers to do.** The issue's "Reproduction request"
section gives one probe — a plain short chat completion:

```bash
curl -s http://127.0.0.1:PORT/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen3.6-27B",
  "messages": [{"role":"user","content":"Hello, please introduce yourself in one sentence."}],
  "max_tokens": 100, "stream": false }'
```

The reporter's own framing: the *first* corrupted request is not deterministic, but
once triggered the bad behaviour reproduces without restart. So the experiment is a
**soak that tries to trigger**, with this exact probe used as a repeated canary.

**What the maintainers asked for (this is the decisive comment).** The whole thread:

| # | Author | Substance |
|---|--------|-----------|
| 1 | dikongfeixing8 | pings maintainers |
| 2 | github-actions[bot] | auto-CC for a ROCm label (mis-routed; reporter is on NVIDIA L20-D) |
| 3 | sizzlecar | At v0.21.0 `auto` resolves to **float32** (Qwen3.6 `text_config.mamba_ssm_dtype=float32`, copied into `mamba_ssm_cache_dtype` by `Qwen3_5ForConditionalGenerationConfig`); that tag's `KVBlockZeroer` **only accepts `FullAttentionSpec` and skips Mamba/GDN groups**; PR #54146 still open, so disabling prefix caching is an isolation step, not a fix |
| 4 | ZJY0516 | "vLLM 0.21.0 is too old" |
| 5 | sizzlecar | "The useful next step is **reproducing on 0.28.0 or current main**; if it disappears there, debugging the old cache path is not actionable." |

**This experiment is exactly that requested next step: an attempt to reproduce on
stock 0.28.0.**

## 2. Prior art / has anyone reproduced or fixed it?

Read-only `gh` search over issues and PRs (`repeated exclamation`, `Qwen3.6 collapse`,
`Qwen3.6 FP8`, `GDN readout fp32`, `gated delta net state leak`, plus the #55291
timeline). Findings as of 2026-09-17:

- **No one has reproduced #55291.** No cross-reference on its timeline, no linked PR,
  no comment claiming a repro or a fix. It is open and unassigned.
- **#54308** (open, 2026-08-29, JackDanger) — "GDN / Qwen3-Next decode degenerates to a
  single repeated token on non-Blackwell GPUs with an fp32 SSM cache". Same symptom
  class, different model. Note the title's scope: **non-Blackwell**.
- **PR #54146** (open, **NOT merged**, JackDanger) — "[Bugfix][Kernel] GDN readout:
  honor fp32 SSM state precision to avoid fp16 overflow → NaN". 2 files, +64/-1. It
  changes `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`
  from `o = q.new_empty(NK, *v.shape)` to allocating `o` in fp32 when
  `initial_state.dtype == torch.float32`. One outside comment (JartX, 2026-09-08)
  says it helped but is incomplete for Qwen3.5-arch models.
- **#51483** (merged/closed 2026-09-16) and **#51562** (open) — GatedDeltaNet metadata
  builder classifying a stateless first chunk as a decode. Adjacent GDN-state-correctness
  family, different mechanism.
- **#42426 / #36763** — Kimi-K2.x "only !!!!!!" in the reasoning field. Same *surface*
  symptom, unrelated (parser/reasoning-field, not GDN state).
- **#45238** — hybrid-model prefix caching vs Mamba checkpoint alignment (open).

### Static reading of the pinned 0.28.0 wheel (read-only; no source patched)

Done before running, so the run has a hypothesis to confirm or refute.

1. **The PR #54146 fix is absent from 0.28.0.**
   `.venv/lib/python3.12/site-packages/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:225`
   is still `o = q.new_empty(NK, *v.shape)` — output materialized in `q.dtype`.
2. **The GDN decode path on this model is that Triton kernel, on Blackwell too.**
   `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` calls
   `fused_sigmoid_gating_delta_rule_update` at lines 1445 / 1472 / 1530 for the
   spec-decode, peeled-decode and decode-only branches; only the *prefill chunk*
   branch goes to `chunk_gated_delta_rule`. So GB10 (sm_121) does exercise the
   unfixed readout.
3. **`mamba_ssm_cache_dtype=auto` still resolves to float32 here on 0.28.0.**
   `model_executor/models/config.py:781-806` (`Qwen3_5ForConditionalGenerationConfig`)
   copies HF `text_config.mamba_ssm_dtype` into `cache_config.mamba_ssm_cache_dtype`
   when it is `auto`; the pinned HF config sets `mamba_ssm_dtype: "float32"`.
   Answers the reporter's Question 2 for 0.28.0.
4. **`KVBlockZeroer` still skips Mamba.** `v1/worker/utils.py:100+`, docstring verbatim:
   *"Only AttentionSpec layers are processed; Mamba layers are skipped."* and the loop
   does `if not isinstance(spec, AttentionSpec): continue`. The reporter's suspicion #2
   is still structurally true on 0.28.0. (That alone is not a bug — GDN state is
   addressed by per-request state index, not by the prefix-cache block pool — but it
   means nothing zeroes a recycled GDN page.)
5. **A caveat that weakens the PR-#54146 mechanism for *this* model.** PR #54146's
   overflow story is fp16-specific (~65504 max). The pinned Qwen3.6-27B-FP8 config has
   `text_config.dtype: "bfloat16"`, so `q.dtype` is **bfloat16**, whose exponent range
   matches fp32 (~3.4e38). A large fp32 state downcast to bf16 loses *precision*, it
   does not overflow to `inf`. Unless the server is launched with `--dtype float16`,
   the exact PR-#54146 inf→NaN path should not fire here. **Prediction recorded up
   front: this soak is more likely to come out negative than positive.** Recording that
   before the run is the point of the card.

## 3. Model pin

The issue names `qwen/Qwen3.6-27B-FP8`. HF canonical id: **`Qwen/Qwen3.6-27B-FP8`**
(lowercase org 404s). Repo head sha = **`e89b16ebf1988b3d6befa7de50abc2d76f26eb09`**
(`lastModified` 2026-04-24T02:39:18Z), 80 files, 66 safetensors.

**The campaign copy at `/home/mark/shared/models/qwen3.6-27b-fp8` is NOT usable and
is NOT the HF snapshot.** Verified:

- Its `.cache/huggingface/download/*.metadata` all record revision
  `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`, and all 66 safetensors + tokenizer files
  match the HF blob sizes exactly.
- But `config.json` was **replaced in place on 2026-05-28** (a sibling
  `config.json.lumo_pre_fp8_fix.bak` is left next to it). Local `config.json` is
  21,854 bytes vs HF's 51,346. Diffs against the pinned revision:
  - `quantization_config.modules_to_not_convert`: HF has **882** entries, campaign copy
    has **371** (539 HF-only, 28 campaign-only — e.g. HF also excludes
    `model.embed_tokens`, every `input_layernorm`, `linear_attn.A_log`,
    `linear_attn.dt_bias`, `linear_attn.in_proj_ba`, `linear_attn.norm`).
  - `quantization_config` keys differ: HF has `fmt: e4m3`; the campaign copy drops
    `fmt` and adds `weight_per_tensor: false`, `act_per_tensor: false`.
  - The campaign copy adds a top-level `torch_dtype` that HF does not have.
  - The `.bak` is *also* not the HF file (3,662 bytes, no `quantization_config` at all).

  A different `modules_to_not_convert` means **different layers are quantized** — using
  it would test the campaign's model, not the issue's. **Decision: download the pinned
  revision fresh into the rig's `HF_HOME`** (`/home/mark/shared/exp54928/hf`), started
  2026-09-17T22:22:30Z, no GPU involved. Serve by repo id + `--revision`, so the pin
  is in the resolved argv and cannot silently drift.

## 4. Environment pins

| Pin | Value |
|-----|-------|
| Rig | `/home/mark/shared/exp54928` (stock, isolated `.venv`, **no source patches**) |
| Wheel | `vllm-0.28.0-cp38-abi3-manylinux_2_28_aarch64.whl`, sha256 `817b8181f7f61b4a62dc1d5d9ab39f2bfb60a6cb86c29879a78a147b85756787` |
| vLLM / torch | 0.28.0 / 2.13.0+cu130, CUDA 13.0, capability (12, 1) |
| Deps | `lock.txt` (195 pkgs) in the rig |
| GPU | 1 × NVIDIA GB10 (sm_121, Blackwell, unified memory), driver 590.48.01, CUDA 13.1 |
| Model | `Qwen/Qwen3.6-27B-FP8` @ `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| `HF_HOME` | `/home/mark/shared/exp54928/hf`, `HF_HUB_OFFLINE=1` at serve time |
| Conventions | venv `bin` on `PATH` (FlashInfer JIT needs `ninja`); isolated `FLASHINFER_WORKSPACE_BASE`; `sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before each launch (unified memory counts page cache as used); teardown via `pkill -f '[v]llm serve'`; every GPU command under `flock /home/mark/shared/exp54928/gpu.lock` |

## 5. Serve flags — mirroring the issue as far as one GB10 allows

Reporter's command vs ours, with every deviation named:

| Issue flag | Ours | Why |
|---|---|---|
| `--tensor-parallel-size 2` (2 × L20-D) | `--tensor-parallel-size 1` | **Forced.** One GB10. Biggest deviation; the reporter's GPUs are also non-Blackwell and ours is Blackwell. |
| `--gpu-memory-utilization 0.85` | `0.50` | **Rig convention.** GB10 unified memory counts page cache as used; 0.85 fails the startup check. 0.50 of 128 GB ≈ 64 GB, ~31 GB weights, ~30 GB caches. |
| `--max-model-len 128000` | `128000`, fallback ladder `65536` → `32768` | Long context is the hypothesised trigger, so try full first. Fallback only on a startup memory failure; the value actually used is recorded. |
| `--model qwen/Qwen3.6-27B-FP8/` (local dir) | `Qwen/Qwen3.6-27B-FP8 --revision e89b16eb…` | Pins the snapshot in the resolved argv. |
| `--served-model-name Qwen3.6-27B` | same | Canary body is copied verbatim from the issue. |
| `--trust-remote-code` | same | |
| `--max-num-seqs 20` | same | Sets the concurrency ceiling the soak drives to. |
| `--max-num-batched-tokens 16384` | same | |
| `--enable-chunked-prefill` | same | |
| `--enable-prefix-caching` | same | Reporter had it on; comment #3 calls disabling it an isolation step, so the primary arm keeps it **on**. |
| `--max-cudagraph-capture-size 64` | same | |
| `--limit-mm-per-prompt '{"image":16,"video":1}'` | same | |
| `--mm-processor-cache-gb 0` | same | |
| `--mm-encoder-tp-mode data` | same | No-op at TP=1; kept for fidelity. |
| `--enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_xml` | same | All three names verified present in the 0.28.0 wheel. |
| `--mamba-ssm-cache-dtype` (not set) | **not set** | Must stay `auto` — that is the configuration under test. Expected to resolve to `float32`; the resolved value is read back from the server log and recorded. |
| `--host 0.0.0.0 --port 30015` | `127.0.0.1` port `8000` | Local only. |

Resolved `/proc/<pid>/cmdline`, `/v1/models` and `/version` are captured, so the
actual argv is evidence, not a claim.

## 6. Prompt set and request cadence

One server launch. `seed` fixed per request template; `temperature=0` for canaries
(so a collapse cannot be blamed on sampling) and `temperature=0.7` for stress traffic
(the reporter's production traffic was not greedy, and greedy-only traffic would
under-explore). All requests ask for token ids and logprobs.

- **CANARY** — the issue's Reproduction request, byte-for-byte: `"Hello, please
  introduce yourself in one sentence."`, `max_tokens: 100`, `stream: false`, greedy.
  Sent serially, alone, with no other request in flight.
- **SHORT** — 8 short unrelated prompts (arithmetic, a capital city, a one-line shell
  command, a haiku, …), `max_tokens` 128. These are the "short and unrelated" requests
  the reporter says also collapse.
- **LONG** — 4 long-context prompts: a ~6k-token synthetic document plus a question,
  `max_tokens` 2048. Purpose: drive long decode over low-decay context, which is the
  regime in which the GDN recurrent state grows.
- **GROW** — a conversation replayed with its own output appended, round after round,
  pushing context toward `max-model-len`. Purpose: maximise recurrent-state magnitude,
  the stated precondition of the #54146 mechanism.

**Cadence — phases, in order, on one server:**

- **P0 baseline (serial).** 5 CANARY + 8 SHORT, serial. Establishes "works normally".
  If anything collapses here the result is immediate and the run stops.
- **P1 serial soak.** Repeat SHORT + LONG serially, CANARY every 10 requests.
- **P2 concurrent soak.** 20 concurrent workers (= `--max-num-seqs`) mixing SHORT/LONG,
  with a solo CANARY between every batch. This is the regime the reporter was in.
- **P3 long-context growth.** GROW rounds to near `max-model-len`, interleaved with
  CANARY, concurrency 4.
- **P4 repeat P2** until the time budget is spent.

`/metrics` snapshot after every canary; server log captured continuously.

## 7. Detection criterion

Primary, mechanical, on the **token ids** (not the rendered string, so whitespace and
parser behaviour cannot hide or fake it):

> **COLLAPSE** ⇔ the completion's generated token id sequence contains a run of
> **≥ 16 consecutive** occurrences of the `!` token id, **or** ≥ 90 % of all generated
> tokens are the `!` token id (with ≥ 32 tokens generated).

The `!` token id is resolved from the served tokenizer at run start and recorded; both
a bare `!` token and any `!`-only multi-character token are counted. For every response
we log: token ids, decoded text, `finish_reason`, generated-token count, the **first
index of the `!` run** (or `null`), and the top-5 logprobs at that index when present
(the "logprob signature" — a collapse driven by NaN logits typically shows a
degenerate/uniform or `-inf`-saturated distribution, which distinguishes it from the
model merely *choosing* to say `!`).

Secondary signals recorded but not sufficient alone: any other single token repeated
≥ 64 times; `finish_reason` anomalies; non-finite values reported in the server log;
throughput from `/metrics` staying flat while output turns to garbage (the reporter's
observation 5).

## 8. Soak length, budget, stop rules

Budget: ≤ 5 h exclusive GPU, ≤ 2 h setup (setup — read, card, model download — is done
off-GPU while waiting for the markers).

- T+0 to ~T+0:30 — drop caches, launch, wait for `/health`, record resolved argv and
  the resolved `mamba_ssm_cache_dtype`.
- ~T+0:30 to T+4:15 — P0…P4.
- T+4:15 — **hard stop** regardless of state; drain, snapshot, tear down.

**Stop rules:**

1. **First COLLAPSE → stop the soak immediately** and switch to the poisoning protocol
   (§9). Do not keep loading the server.
2. Server exits / crashes / OOMs → record exit code and the last 500 log lines, stop.
3. Startup memory failure → step the `max-model-len` ladder (128000 → 65536 → 32768),
   at most 3 launch attempts total; if all 3 fail, result is **inconclusive
   (could not launch)**.
4. GPU markers absent after 4 h of polling → stop, report "GPU not released", no run.
5. Any single request exceeding a 900 s timeout twice in a row → record and stop.

## 9. If it collapses — poisoning protocol

The cross-request poisoning claim is the substantive part of the issue, so it gets its
own measurement rather than a remark:

1. Stop all stress traffic; let the server drain.
2. Send **10 solo CANARY** requests serially, ≥ 5 s apart. Record how many collapse.
   *Poisoning confirmed* iff a clear majority collapse with no other traffic present.
3. Send 8 SHORT requests with prompts never used before in this server's lifetime
   (fresh prefixes, so prefix-cache reuse cannot be the explanation).
4. Wait 120 s idle, re-send 5 CANARY — does it self-heal?
5. Snapshot `/metrics` and grep the server log for `nan`, `inf`, `assert`, `error`,
   `Traceback` around the first collapsed request's timestamp.
6. Restart the server (same flags) and send 5 CANARY — confirms a restart clears it,
   which is what the reporter states.

## 10. Outcomes — defined before the run

- **REPRODUCES** — at least one COLLAPSE occurred under §7, *and* step 2 of §9 shows
  the poisoning (a majority of solo canaries collapse afterwards).
- **PARTIALLY REPRODUCES** — a COLLAPSE occurred but subsequent solo canaries are
  clean: the numerical failure exists on 0.28.0 but the cross-request persistence does
  not, which would be a materially different (milder) bug than reported.
- **DOES NOT REPRODUCE** — the soak completed its planned phases within budget, the
  recorded request count / generated-token count / peak context length are all in the
  log, and **no** response met the §7 criterion. Explicitly this means: *on this one
  GB10, on stock 0.28.0, with this pinned FP8 snapshot, at TP=1, bf16 activations, with
  prefix caching and chunked prefill on, for this much traffic* — the collapse did not
  appear. It is **not** evidence the reporter is wrong on 0.21.0 / 2 × L20-D / TP=2, and
  it does not clear the two structural findings in §2 (the unfixed readout line and the
  Mamba-skipping zeroer).
- **INCONCLUSIVE** — could not launch within the ladder, ran out of budget before
  finishing P0–P2, or the server died for an unrelated reason.

## 11. Scope limits (fixed in advance)

One device; one GPU (TP=1 vs the reporter's TP=2); Blackwell sm_121 vs the reporter's
non-Blackwell L20-D — and #54308's title explicitly scopes that symptom to
non-Blackwell; vLLM **0.28.0**, not the reporter's 0.21.0; bf16 activations; single
soak, so a non-deterministic trigger can simply be missed — absence of a collapse in
N hours bounds the rate, it does not prove absence.

## 12. Deliverables

`CARD.md` (this file, pre-run) · `RESULT.md` (verdict + numbers) · `runs/` (per-request
JSON, server log, `/metrics`, `meta.json` with resolved argv) · `DRAFT_COMMENT.md`
(≤ 200 words, **not posted**) · report copied to
`/home/mark/shared/tmp-scratch/C_55291_repro.md`.

---

## Addendum (appended 2026-09-17 ~23:15 UTC, after an aborted first launch)

The card body above is unedited. Two implementation defects were found ~1 minute into
the first launch, during P0, and fixed before any measurement was kept. Recording them
because they would both have produced a **guaranteed false negative**:

1. **`token_ids` were read from the wrong place.** In vLLM 0.28.0 the generated ids are
   returned on `choices[0].token_ids`, not `choices[0].message.token_ids`. The detector
   was reading the message field, got `[]` for every response, and therefore scored
   zero `!` on everything.
2. **The text was invisible.** With `--reasoning-parser qwen3`, a 100-token canary is
   entirely "thinking", so `message.content` is `None` and the text is in
   `message.reasoning`. Detection is on token ids either way, but the logs were blank.

Two improvements made at the same time:

3. **`!` runs are now scored in `!` characters, not tokens.** The vocab holds **10**
   bang-only tokens — `!`(0), `Ġ!`(729), `!!`(2834), `Ġ!!`(10693), `!!!`(11726),
   `!!!!`(16582), `Ġ!!!`(31780), `!!!!!!!!`(48962), `!!!!!`(67437), `Ġ!!!!`(230802) —
   so eight `!` characters can be a single token. The §7 threshold is unchanged in
   meaning (≥ 16 consecutive `!`) but is now measured in characters, which makes it
   strictly more sensitive. The id set is enumerated from `tokenizer.json` rather than
   probed, and cross-checked against the served `/tokenize`.
4. The detector was unit-tested (synthetic sequences) **and** validated against the real
   saved response from the aborted run plus injected synthetic collapses, before restart.

The 4 canaries from the aborted launch are kept in
`runs/aborted_detector_bug_soak/` and are **excluded** from the result.
