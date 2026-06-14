# FR13 — vLLM SOURCE OF TRUTH + grounding rule (read 2026-06-14)

## GROUNDING RULE (user 2026-06-14, EMPHATIC)
**Read vLLM source DIRECTLY from the pinned running image — NEVER from a stale `/tmp` cache.** Cached
extractions DRIFT and silently corrupt analysis. Use `scripts/vllm_src.sh <relpath>` (cats one file fresh
from the pinned image) or `scripts/vllm_src.sh` (re-extracts the full tree to `/tmp/vllm_cu130_src`).

## What we run (CONFIRMED)
- The locked launchers (`fr13_launch_locked.sh` → `fr13_launch_forked_fa2_tree_server.sh`,
  `fr10_launch_speed_server.sh`, `fr10_phase4_launch_tree_capture_probe.sh`) ALL pin
  **`vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`**.
- That digest = tag **`cu130-nightly`** = **vLLM `0.19.2rc1.dev134+gfe9c3d6c5`** (verified: digest match +
  in-container `_version.py` + the live boot engine log). It is the **newest WORKING** vLLM image on
  GB10/cu130 (CUDA13, flashinfer, cuda-graphs, B4 defaults).
- The running container = this image + the FA2-fork `.so` swap (`_vllm_fa2_C.abi3.so`) + the patcher
  (`fr10_phase4_patch_vllm_tree_gdn.py`) applied at boot. The fla/mamba kernel ops are the base image's.

## The confound this fixes (user caught it)
Agents were reading `/tmp/vllm_live_019` = a **STALE vLLM 0.19.0** extraction. vs the running **0.19.2** it
diverged: `fused_recurrent.py` 15 lines, `fused_sigmoid_gating.py` 40, `causal_conv1d.py` **123**. Line
citations in the FR13 binds that reference `/tmp/vllm_live_019` are off the real source; conclusions that
re-verify on the image still hold (see below), but **re-ground any `/tmp/vllm_live_019:LINE` citation** via
`scripts/vllm_src.sh` before trusting it. Stale caches DELETED 2026-06-14: `/tmp/vllm_live_019`,
`/tmp/vllm_img_0192`, `/tmp/vllm_pristine_019`, `/tmp/vllm-0.22-src`, `/tmp/vllm-0.22-probe`, `/tmp/fr10_vllm_src`.
Canonical = `/tmp/vllm_cu130_src` (stamped `SOURCE_OF_TRUTH.txt`, re-extractable via the helper).

## Scan-math finding RE-VERIFIED on the real image (survives the correction)
`fused_recurrent_gated_delta_rule_packed_decode_kernel` IS recurrent rank-1 — `tl.program_id` one-token,
the 5 ops (`b_h*=exp(g); b_v-=tl.sum(b_h*b_k,1); b_v*=beta; b_h+=b_v*b_k; b_o=tl.sum(b_h*b_q,1)`), **ZERO**
chunk-loops / `tl.dot`; `num_warps=1, num_stages=3`. So the "carrier is codegen-alignable (geometry + l2norm
opcode + beta cast), NOT a chunk-vs-recurrent irreducible gap" conclusion HOLDS on the running image.

## Why NOT vLLM 0.22 (the answer)
Two separate reasons, both evidenced (FR9 study `/tmp/fr9_vllm_upgrade_feasibility.md` + `/tmp/fr9_vllm_cu130_dgxspark.md`, 2026-06-03):
1. **An upgrade buys NOTHING for the lossless objective.** Batch-invariant GDN does NOT exist in ANY released
   vLLM incl. 0.22.0 and `main`: issue **#42960 is OPEN, zero merged PRs**; GDN/mamba backends inherit
   `supports_batch_invariance()==False`; the hard abort `"VLLM batch_invariant mode is not supported for
   GDN_ATTN"` is unchanged. The 0.22 "batch-invariant" features are **GEMM/linear only** (Cutlass FP8/NVFP4),
   NOT GDN recurrent-state. No isolated `num_reqs=1` recurrent-forward primitive in any release either. Per
   speed-is-the-goal / cost-gate: do not migrate for a capability that isn't there.
2. **The local `lumo-vllm-audit:v0.22.0-cu129-min` image is BROKEN on GB10** (cu12 `_C` vs CUDA13.1, no
   flashinfer). A WORKING `vllm/vllm-openai:v0.22.0-cu130` (multi-arch incl arm64) DOES exist and is cheap to
   pull — but see (1): it doesn't help. All FR13 investment (patcher, FA2-fork, locked pipeline) is on the
   cu130-nightly 0.19.2 build; re-porting to 0.22 is churn for zero lossless benefit.

## Docker image inventory (2026-06-14)
- **KEEP — `vllm/vllm-openai:cu130-nightly`** (`ffa30d66`) = the pinned running image. THE one.
- KEEP (referenced by 6 old FR9/L0c scripts) — `lumo-flywheel-vllm:26.01-py3-v0.19.0` + `:0.19.0-fr9iso`
  (old 0.19.0 custom build). NOT the FR13 path; retire only if those scripts are retired.
- DELETED (unused) — `vllm/vllm-openai:latest` (`c17fbdfa`, superseded by cu130-nightly) +
  `lumo-vllm-audit:v0.22.0-cu129-min` (`af07ec6b`, BROKEN on GB10).
