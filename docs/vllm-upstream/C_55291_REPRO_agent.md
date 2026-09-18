> **CORRECTION 2026-09-18 (independent Codex check, see F8 in STATIC_FINDINGS.md):** the kernel-path conclusion in F2/F7 (and the 'likely why' in RESULT.md / DRAFT_COMMENT.md) is WRONG. The startup line 'GDN decode kernel: cuda' is configuration only; the fused CUDA MTP path additionally requires speculative metadata and 8:1 V/K heads, and this run had neither (speculative_config=None; 48:16 = 3:1). Default decode-only batches use the packed Triton path; mixed decode/prefill batches can call #54146's readout. No per-batch kernel trace was captured. The L20-D bypass prediction is withdrawn. The negative result itself stands. The evidence tarball is the pristine agent output and still contains the uncorrected text.

# C — vllm-project/vllm #55291 reproduction attempt (GB10, stock 0.28.0)

**Verdict: DOES NOT REPRODUCE.** Read-only on GitHub throughout; nothing posted.
Work dir: `/home/mark/shared/exp54928/exp55291/`
Evidence: `/home/mark/shared/exp54928/exp55291/evidence_55291_20260918T023237Z.tar.gz`

## The issue and what the thread asked for

#55291 (dikongfeixing8, open): `Qwen3.6-27B-FP8` on vLLM **0.21.0**, 2 × L20-D, TP=2,
prefix caching + chunked prefill, 128 k context. After some time one request emits only
`!`, and from then on every independent short request does too; the process stays alive
and never recovers without a restart. The reporter supplies a verbatim "Reproduction
request" curl and says the first corrupted request is non-deterministic but the state,
once entered, persists.

Five comments. The decisive one is sizzlecar's last: *"The useful next step is
reproducing on 0.28.0 or current main; if it disappears there, debugging the old cache
path is not actionable."* **That is exactly what this run did.** Nobody had reproduced
it: no cross-reference on the timeline, no linked PR, no repro claim.

## Result

One server launch, 3 h 16 min, pre-registered phases P0-P4 (`CARD.md`, written before
launch).

- **298 requests, 249,544 generated tokens, 0 errors**, peak context 36,496 tokens
- **0 / 298 collapses.** Longest `!` run anywhere: **1 character** (threshold: 16).
  Longest run of any repeated token: 4.
- **0 / 26 canaries** collapsed — the issue's verbatim probe, sent solo, throughout
- 0 preemptions; prefix caching genuinely exercised (406 k hits / 1.44 M queries)
- **0** lines matching `traceback|error|nan|assert` in the whole server log
- ~75 min of the run at the reporter's `--max-num-seqs 20` concurrency

Detector was unit-tested and validated against a real response with synthetic collapses
injected; it also registered a naturally occurring `!` during the run, so it was live.

## The finding worth more than the null

**In 0.28.0 `VLLM_GDN_DECODE_KERNEL` defaults to `cuda`.** This model meets every
condition, so GDN decode runs the fused CUDA kernel `fused_gdn_decode_post_conv_mtp` —
**not** the Triton readout that PR #54146 patches. Server log:
`qwen_gdn_linear_attn.py:505 GDN decode kernel: cuda`, with no fallback line.

The unfixed line (`fused_sigmoid_gating.py:225`, `o = q.new_empty(NK, *v.shape)`) is
still in 0.28.0 and still on main — #54146 is open and unmerged — it is just off this
model's decode path. **Not a Blackwell artefact:** the only GPU condition is capability
≥ 8.0, which the reporter's Ada L20-D (8.9) clears, so 0.28.0 on their hardware should
take the fused kernel too. The vLLM version, not the GPU, is what moved it.

Also, for 0.28.0: `auto` still resolves to **float32** (their Q2); the model is **bf16**,
so #54146's fp16 65504 overflow cannot fire (Q3); `KVBlockZeroer` still skips Mamba, but
GDN prefill zeroes state for any sequence without an initial state, so a fresh request
does not inherit one (Q4/Q5) — the remaining way in is a request that never takes the
prefill branch, which is the #51483 / #51562 bug class.

## Model pin — the campaign copy was unusable

`/home/mark/shared/models/qwen3.6-27b-fp8` was downloaded from the right revision
(`e89b16eb…`) and all 66 safetensors match HF byte-for-byte, **but its `config.json` was
replaced by campaign work on 2026-05-28** (a `config.json.lumo_pre_fp8_fix.bak` sits
next to it): 21,854 bytes vs HF's 51,346, and `quantization_config.modules_to_not_convert`
has **371** entries vs HF's **882**. Different layers quantized = a different model. So
the pinned revision was downloaded fresh into the rig's `HF_HOME`; all 80 files verified
against the HF manifest and `config.json` md5-identical to the pristine blob.

## Pins

vLLM wheel `817b8181…f309d` (0.28.0 aarch64) · torch 2.13.0+cu130 · 1 × GB10 sm_121,
driver 590.48.01 · `Qwen/Qwen3.6-27B-FP8` @ `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` ·
TP=1 · `--gpu-memory-utilization 0.50` (rig convention, not the issue's 0.85) ·
`--max-model-len 128000` · `--max-num-seqs 20` · prefix caching + chunked prefill on ·
`mamba-ssm-cache-dtype` left at `auto`.

## Scope limits

One GPU, TP=1 (issue: TP=2), Blackwell (issue: L20-D), **0.28.0 (issue: 0.21.0)**, bf16
activations, one 3-hour window against a self-described non-deterministic trigger — this
bounds the rate, it does not prove absence. Untested: 0.21.0, TP=2, 128 k contexts
actually filled, multimodal and tool-calling traffic.

## Budget and discipline

GPU 23:00:52 → 02:31:14 UTC ≈ **3 h 30 min** of the 5 h allowance (includes an 8-minute
aborted first launch, below). Setup (reading, card, 29 GB download) ran off-GPU while
waiting for the markers. Every GPU command under `flock .../gpu.lock`; GPU work started
only after both `A_done.marker` and `B_done.marker` appeared (22:32 UTC); `drop_caches`
before launch; bracket-pattern `pkill`; no source patches. GPU confirmed free at exit.

**One aborted launch, disclosed:** ~1 minute into the first attempt the detector was
found to be reading `token_ids` from `message` instead of `choices[0]` (vLLM 0.28.0 puts
them on the choice) and to be blind to text in `message.reasoning` under
`--reasoning-parser qwen3`. Both would have forced a false negative. Fixed, re-validated,
and restarted; those 4 canaries are archived in `runs/aborted_detector_bug_soak/` and
excluded. See the `CARD.md` addendum and `DEVIATIONS.md` (which also records a mid-run
P3 defect that left the growth conversations' assistant turns repetitive — context still
grew as intended via the per-round filler).

## Deliverables

`CARD.md` (pre-run) · `RESULT.md` · `STATIC_FINDINGS.md` (F1-F7, incl. the F2 correction
to the card) · `DEVIATIONS.md` · `DRAFT_COMMENT.md` (181 words, **not posted**, artifact
link left as `{{ARTIFACT_LINK}}`) · `runs/soak/` raw evidence · evidence tarball above.
