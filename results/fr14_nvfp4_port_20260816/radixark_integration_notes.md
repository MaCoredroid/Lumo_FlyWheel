# RadixArk aggressive-4bit integration design (staged 2026-08-16 ~21:45Z, commit at run drain)

Checkpoint: RadixArk/Qwen3.8-27B-NVFP4 @ 554ebba9, total 21,921,428,072 B, 2194 tensors.
lm_head is ModelOpt NVFP4 4-tensor set (packed weight U8 [248320,2560] + weight_scale
[248320,320] F8 + weight_scale_2 + input_scale scalars) ≈ 0.715 GB vs BF16 2.543.
MTP 15 tensors BF16 (0.849/pass, same as unsloth). Decomposition from total: target
(non-head/embed/visual/mtp) ≈ 16.89 GB.

## Floor bands on our stack (root_64k formula, 273 GB/s)
- Aggressive + BF16-dequant DVK slice (draft head stays BF16 GEMV): 16.89 + 0.715
  + 5×0.849 + 5×0.671 = 25.21 GB → **92.3 ms** (ceiling ~60 TPS @ accept 5.5)
- Aggressive + FP4 draft-head reads (needs FP4 GEMV unit or resurrect FP8 slice):
  ≈ 22.80 GB → **83.5 ms** (ceiling ~66 TPS)
- + MTP-FP8 requant lever: → high-70s ms (ceiling ~70 TPS)
Compare: conservative unsloth arm currently pinned at 102.480 ms (ceiling ~54).

## Integration steps (in order)
1. Boot-smoke RadixArk via credential-free path when GPU free — answers the flagged
   loader risk: does vLLM 0.19.2's modelopt path route ParallelLMHead NVFP4? (ModelOpt
   config class exists; lm_head dispatch unverified — the compressed-tensors analogue
   was present but the model file bypassed it; expect possibly the same
   "no parameter lm_head.weight_scale" class → then options: model-file patch via
   fr10 patcher (route lm_head through quant method) or lm_head-only dequant surgery
   (loses the 10.8 ms floor win — defeats the purpose; patch preferred).
2. Expect kv surgery #2 analogue: hf_quant_config kv_cache_quant_algo FP8 (uncalibrated,
   scales=1.0 per their own qualification) must be stripped or TREE_ATTN legality may
   refuse again. Their KV FP8 is uncalibrated anyway — BF16 KV is both our doctrine and
   strictly safer here.
3. DVK slice on NVFP4 lm_head: row-aligned by construction (scales are [out, in/16] —
   index_select rows works for weight AND scale). Phase 1 = dequant-at-slice to BF16
   (boot-time, keeps sealed BF16 GEMV units + 128-block map untouched). Phase 2 (floor
   lever) = FP4 or FP8 draft-head GEMV, own byte gate.
4. Contract/floor train for the RadixArk arm mirrors the unsloth one (generator script
   reusable as-is); floor re-derivation from post-surgery ledger, never from this estimate.
5. sglang container native bench = calibration reference on identical hardware (their
   recipe: EAGLE 3/1/4 chain, mem-fraction 0.70 per the only real GB10 datapoint).
