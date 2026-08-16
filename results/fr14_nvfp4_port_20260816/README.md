# FR14 — Qwen3.8-27B NVFP4 port: campaign kickoff (2026-08-16)

Mark's directive: *"port our work to qwen 3.8 27b nvfp4 and see how fast we could go on b1 and b4, also run the swe verified 16 tasks."* Successor campaign to FR13 (closed 2026-08-15 at `1d4258b36`, B1 honest 1.833×, B4 honest 2.574×).

## Day-0 verdicts (recon evidence in this dir)

1. **Architecture: ZERO diff.** `Qwen/Qwen3.8-27B` config is byte-identical to Qwen3.6-27B on every architectural field (sole delta: `transformers_version` 4.57.1→5.8.0.dev0). Same 64 layers (48 GDN + 16 full-attn, interval 4), hidden 5120, 24 heads / 4 KV / head_dim 256, GDN 16/48/128/128 conv4, vocab 248 320, same 15-tensor MTP block, byte-identical generation_config. The entire FR13 stack carries over architecturally; the campaign's whole risk surface is the quantization. (`recon_model_intel.json`)
2. **NVFP4 kernels: SMOKE PASS on this box.** `scaled_fp4_quant` + `cutlass_scaled_fp4_mm` ran clean inside the pinned image (`vllm 0.19.2rc1.dev134+gfe9c3d6c5`, torch 2.11.0+cu130) on GB10 sm_121; `cutlass_scaled_mm_supports_fp4=true`; mean rel err 0.134 on W4A4 random-gaussian toy GEMM (quantization noise, not garbage). **The Sprint-0.5 "ARM64 NVFP4 CUDA illegal instruction" record (docs/LLD-01 §4.2) does NOT reproduce on the pinned image.** (`nvfp4_smoke_result.json`, script `nvfp4_smoke.py`)
3. **Loader plumbing present:** `compressed-tensors` registry entry with `compressed_tensors_w4a4_nvfp4` scheme (unsloth path); `modelopt_fp4`/`modelopt_mixed` with `MIXED_PRECISION` (RadixArk fallback); `qwen3_5_mtp` speculative method registered — 3.8 keeps `model_type qwen3_5`, so the MTP spec-decode wiring is untouched. Residual: unsloth's top-level `format=mixed-precision` needs one live boot to confirm the group-wise scheme selection path.
4. **No official NVFP4 exists** (Qwen org ships only BF16 + FP8 for 27B; nvidia/RedHatAI ship none). Field of ~12 community repacks audited tensor-by-tensor over HTTP-range safetensors headers; **every serious one keeps the 15 MTP tensors in BF16** (blacklist: lokeshe09, YCWTG, mlx-community — they drop MTP; r0b0tlab uploaded ONLY the MTP shard).

## Checkpoints (downloads in flight, revision-pinned)

| arm | repo @ revision | GB | why |
|---|---|---|---|
| baseline | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7af6b5` | 30.9 | **Structurally identical to the served 3.6 dir** (same 66-file layers-N/mtp/outside layout, 1606 tensors, same 882-entry exclusion list) → drop-in for the existing pipeline; separates "model 3.6→3.8" from "FP8→NVFP4" in every B1/B4/QC readout |
| primary NVFP4 | `unsloth/Qwen3.8-27B-NVFP4` @ `16b6615af354` | 23.4 | Most conservative recipe in the field: NVFP4 W4A4 g16 on MLPs only (last 8 layers held FP8), FP8 on attn+GDN projections + lm_head, BF16 embed/GDN in_proj_a/b/conv, calibrated FP8 KV scales, MTP BF16 in own shard; mirror of the nm-testing/Neural-Magic reference quant; complete tokenizer set |
| fallback | `RadixArk/Qwen3.8-27B-NVFP4` @ (pin at download) | 22.0 | ModelOpt path; only repo with published verification (tensor audit, GSM8K 97.27%, **measured SGLang MTP acceptance 2.775 on Blackwell**); liability: NVFP4 lm_head (K64 drafter-logits quality + loader risk) |

Local dirs: `/home/mark/shared/models/qwen3.8-27b-fp8`, `/models/qwen3.8-27b-nvfp4` (pinned revision recorded in `.lumo_pinned_revision`).

## Floor math directive (do NOT scale the old constant)

The floor does not halve. Of the 32.667 GB mandatory bytes, ~5.9 GB (BF16 verifier head 2.543 + 5×K64 draft-head 3.355) never shrinks; GDN convs/in_proj_a/b are BF16 today and stay BF16; converted tensors go ×0.5625 (4-bit + e4m3 per-16 scales) — and unsloth converts only most MLPs to FP4 (attn/GDN stay FP8). **Re-derive `TARGET_MODEL_BYTES` and `MTP_FORWARD_BYTES_PER_PASS` by summing the actual NVFP4 tensor ledger** (weights + block scales + global scales), then re-emit `fr13_hardware_floor_ledger` arithmetic to a new results dir. Expected landing: high-70s to mid-80s ms for the K64/root1 arm (vs 119.658). The honest-floor per-request term (+7.117 ms at C=18k) is geometry-only and carries over unchanged.

## Correctness bar — PROPOSED, AWAITING MARK

FR13's Tier-A byte gates are *weight-relative* (candidate vs reference at the same weights) — they all still work after the swap and remain the kernel-legality instrument. What dies is any losslessness claim *across* the swap: NVFP4 is lossy vs FP8 by construction (in-repo doctrine `FR13_KERNEL_SPEEDUP_HYPOTHESES.md:81` said "do not propose" for exactly this reason — this campaign is the deliberate scope change). Proposed bar:

- **Kernel legality**: unchanged — Tier-A byte gates re-run against the NVFP4 model's own activations.
- **Model quality**: exact16 SWE-Verified QC re-banked on the new model (first run structurally cannot pass — the c2 comparator needs ≥4 new-model reference arms; the 9/16 band is a Qwen3.6 number). FP8-3.8 baseline arm run first so the QC delta decomposes into model-refresh vs quantization.
- **Speed**: B1 step-wall + B4 width-4 windowed instruments unchanged; all floors/MDEs/caps re-derived; no FR13 ratio quoted against FR14 numbers.

## Execution ladder

0. ✅ Recon (3-agent sweep, this dir) · smoke PASS · downloads started · SGLang MTP-under-NVFP4 reference recon in flight.
1. **FP8-3.8 drop-in boot** (registry smoke-test path, credential-free) → confirms model refresh alone breaks nothing.
2. **NVFP4 stock boot** via `LUMO_MODEL_LOCAL_PATH_OVERRIDE` escape hatch (`model_server.py:938`) — the format=mixed-precision loader answer.
3. **Model-bound constant train** (ONE commit): launcher serve line + ~40 name literals, contract 81-file manifest regen (`layers-{0..63}` count survives — FP8 twin — but NVFP4 dir has different file set), floor table (`fr13_fixed32_floor_timers_seq.sh:105-127` + ~25 mirrored literals), DVK block-map check (tokenizer identical ⇒ likely carries; verify by hash), KV sizing. Verify offline with `external-manifest` before any GPU.
4. **Stock B1 timing** (exact4, `fr13_measure.py deploy-speed` idiom) + **stock B4 width-4** (pool16 + windowed reduction + batch-conditioned strata) on NVFP4 → the "how fast" headline numbers vs the new floor.
5. **exact16**: N≥4 stock arms to re-bank the band (FP8-3.8 + NVFP4), then the QC gate proper.
6. **Lever re-earn** (only after 1–5 green): b34 dual gate at new HEAD (also retires FR13 capstone item #1 in passing), B1 gqa_pair gate, then timing pairs; single_launch/TAW stay parked.

## Dead on arrival under NVFP4 (from port-surface recon)

The fp8 GEMM lever portfolio is inert with a non-fp8 checkpoint: OPT-A/`FR13_GB10_FP8_GEMV_CFG` (guard requires `weight_block_size==[128,128]`), `FR13_FIXED32_B1_FP8_QUANT_REGCACHE`, all ~18 `FR13_FIXED32_CUTLASS_WAVE` variants. They don't crash — they silently measure nothing. Step-3 train must add a fail-loud guard refusing fp8-only levers under a non-fp8 checkpoint.

## Evidence files

- `recon_model_intel.json` — full HF field audit (12+ repacks, per-tensor dtype/exclusion maps, MTP verdicts, blacklist)
- `recon_port_surface.json` — every model pin/floor constant/shape literal with file:line @ origin/main; port checklist; hard blockers
- `recon_harness_map.json` — B1/B4/exact16 instrument idioms, model-bound vs HEAD-bound credential split, stock-serve runbook
- `nvfp4_smoke.py` + `nvfp4_smoke_result.json` — the GB10 sm_121a kernel go/no-go, PASS 2026-08-16
