# FR13 — Version-Staleness Audit (2026-06-14, CPU read-only)

**Why this exists:** we got burned. Agents read `/tmp/vllm_live_019` = a STALE vLLM **0.19.0** extraction
while the running image is **0.19.2rc1.dev134** (digest `3dbe092e`, tag `cu130-nightly`). The two diverged
15 / 40 / **123** lines (`fused_recurrent.py` / `fused_sigmoid_gating.py` / `causal_conv1d.py`), silently
off-citing every kernel analysis. This audit re-grounds the load-bearing citations on the REAL image (via
`scripts/vllm_src.sh`, which `docker run --entrypoint cat`s the pinned image — image confirmed present
locally `ffa30d66`, `_version.py` = `0.19.2rc1.dev134+gfe9c3d6c5`) and re-checks the upstream "why not
upgrade" decision online.

**Bug-class playbook rows in scope (FR13_BUG_CLASS_PLAYBOOK):**
- **#11 batch-composition / version-skew** — the core failure here: line citations keyed to the wrong source
  version. The cache "looked like vLLM" but was a different release; near-identical files masked the skew.
- **#10 codegen-identity (shared-source ≠ shared-SASS)** — relevant to the kernel re-cites: the bar is the
  RUNNING image's bytes, not a same-named cache; re-confirm structure on the deployed source, not by name.
- **#9 silent/vacuous instrument** — the skipif tests that depend on `/tmp/vllm_pristine_019` now silently
  SKIP (the path is deleted), so they pass while testing nothing. Flagged below.

---

## PART 1 — CLEANUP LIST (dangling refs to DELETED paths/images)

Deleted 2026-06-14 (per FR13_VLLM_SOURCE_OF_TRUTH): `/tmp/vllm_live_019`, `/tmp/vllm_img_0192`,
`/tmp/vllm_pristine_019`, `/tmp/vllm-0.22-src`, `/tmp/vllm-0.22-probe`, `/tmp/fr10_vllm_src`;
images `vllm/vllm-openai:latest`, `lumo-vllm-audit:v0.22.0-cu129-min`. Canonical replacement =
`scripts/vllm_src.sh <relpath>` (reads fresh from the pinned image) or `/tmp/vllm_cu130_src` (re-extract).

### 1A. LIVE-SCRIPT BREAKS (would raise / silently no-op at runtime — FIX these)

| file:line | dangling ref | runtime effect | fix |
|---|---|---|---|
| `scripts/fr10_gdn_tree_algebra_reference.py:16` | `/tmp/vllm-0.22-src/.../cpu/recurrent_gated_delta_rule.py` | `spec_from_file_location` → `exec_module` **RAISES** `RuntimeError("cannot import vLLM CPU rule")` if invoked | repoint to `scripts/vllm_src.sh model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py` (extract once) OR `/tmp/vllm_cu130_src/vllm/...`. NOTE: 0.22 CPU-rule API ≠ 0.19.2 — re-validate the import target exists in 0.19.2 first (see caveat below). |
| `scripts/fr10_phase2_triton_tree_gdn_microbench.py:34` | `/tmp/vllm-0.22-src/.../cpu/recurrent_gated_delta_rule.py` | `RuntimeError` on `load_cpu_rule()` | same repoint |
| `scripts/fr10_phase2_triton_tree_gdn_microbench.py:38` | `/tmp/vllm-0.22-src/.../fla/ops` | source-load path dead | same repoint to `.../fla/ops` |
| `scripts/fr13_lossless_fast_derivation_validate.py:66` | `/tmp/vllm-0.22-src/.../cpu/recurrent_gated_delta_rule.py` | **SILENT no-op**: `load_vllm_oracle()` returns `None` when path absent (`if not VLLM_CPU_RULE.exists(): return None`) → oracle column quietly disappears = **class #9 vacuous**. Worse than a raise. | repoint + make the None-path FAIL LOUD (the oracle is the whole point of the validator). |

> **CAVEAT (do NOT blindly repoint to a 0.19.2 CPU rule):** these 3 scripts wanted the vLLM **0.22** CPU
> serial recurrence (`mamba/ops/cpu/recurrent_gated_delta_rule.py`) as a clean ℝ-reference oracle. That file
> is a **0.22 path** — confirm it EXISTS in the running 0.19.2 image before repointing (`scripts/vllm_src.sh
> model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py`). If 0.19.2 lacks the CPU rule, the
> correct oracle for the deployed image is `fused_recurrent_gated_delta_rule_packed_decode` (the kernel the
> decode path actually dispatches — see 7441fc43 / `fr13_native_packed_decode_ref.py`, which already imports
> the LIVE kernel). These 3 are FR10/early-FR13 microbench/derivation scripts, NOT on the locked pipeline —
> repoint or retire; they are not load-bearing for the current campaign.

### 1B. LIVE-SCRIPT, SELF-HEALING (no fix strictly required, but tidy)

| file:line | ref | why it's OK | recommended |
|---|---|---|---|
| `scripts/fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py:88,90` | `/tmp/vllm_live_019/...`, `/tmp/vllm_pristine_019/...` | candidate LIST tries `/usr/local/lib/.../dist-packages/vllm/...` (line 86) **FIRST**; in-container that path exists so the deleted /tmp entries are never reached. Has `--src` override + raises if none found. | drop the 2 dead /tmp candidates (lines 88-91) to avoid future confusion; functionally inert today. |

### 1C. TESTS THAT NOW SILENTLY SKIP (class #9 — they pass while testing nothing)

All gate on `PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")` with
`@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")`. Path is DELETED →
these now **SKIP, not fail** → green CI hides that they verify nothing:
- `tests/test_fr13_chase_diag_wiring.py:41,329,339`
- `tests/test_fr13_nondet_chase_fixes.py:353,356,362,370,376,402…`
- `tests/test_fr13_s1_bonus_row.py:230,233,243`
- `tests/test_fr13_commit_argmax_gate_wiring.py:40,228,234`
- `tests/test_fr13_conv_committed_path.py:584,587,598`

**Fix:** repoint `PRISTINE` to a re-extractable canonical (`/tmp/vllm_cu130_src/vllm`, produced by
`scripts/vllm_src.sh` with no args). The current skipif means these wiring/diff tests are dormant. (They diff
patched-vs-pristine `rejection_sampler.py` / `gpu_model_runner.py` — repointing to the 0.19.2 tree also
re-validates those diffs against the version actually deployed, which is the right thing.)

### 1D. HISTORICAL NARRATIVE (markdown — annotate, do NOT chase line-by-line; they are dated records)

These reference deleted /tmp caches or deleted images but are FROZEN historical docs (FR10/FR11/FR12 era,
auto-research reports). They do not run. The single-line fix is an annotation banner, not a per-citation
rewrite. Group by file (representative line in parens):

- **Stale `/tmp/vllm_live_019` citations in load-bearing FR13 binds** (these are the ones whose CONCLUSIONS
  are re-verified in Part 2 — keep the conclusion, annotate the line citation as 0.19.0-keyed):
  `FR13_L4_CONV_VERDICT.md:9`, `FR13_CONV_FIX_DESIGN.md:138,274`, `FR13_CONV_CROSSEVENT_INVESTIGATE.md:28`,
  `FR13_REPLAY_KERNEL_ALIGNMENT_PLAN.md:41,246`, `FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND.md:9`,
  `FR13_SCAN_ALIGNMENT_MATH.md:5`, `FR13_PLUS2_SPINE_ALIGN_RESEARCH.md:94`, `FR13_ORACLE_FRAME_DECISION.md:44,
  291,293,296,299`, `FR13_RESIDUAL13_HYPOTHESES.md:4`, `FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md:25`,
  `FR13_COST_GATE_BI_TREE_BACKEND_BLOCKED.md:15,36,44`, `FR13_LADDER_LOG.md:1301`, `FR13_SUBOP_MAB_REBUILD.md:
  111,118`, `FR13_BF16_FP32_SEAM_SCAN_BIND.md:4`, `FR13_PREFILL_DRIFT.md:14`,
  `docs/archive/wy/FR13_WY_*.md` (LAYER0_SUBOP_VERDICT:6, STATE_FIX:57, MULTISTEP_SEAMS:9, SEAM_FIXES:29,
  CASCADE_MAP:3, RESIDUAL_CLOSURE:3).
- **Stale `/tmp/vllm-0.22-src` citations** (FR10/FR11/FR12 narrative): `FR10_KICKOFF.md:24,28`,
  `FR11_KICKOFF.md:30`, `FR11_RESULTS.md:102`, `FR10_STATUS.md:274,309`, `FR12_SCAN_ROOT_TASK.md:22,36`,
  `FR12_WY_KICKOFF.md:14`, `FR13_LOSSLESS_FAST_DERIVATION.md:165`, `FR13_FULLATTN_OP_LOCALIZE_TASK.md:28`,
  `docs/reports/auto_research/fr10-gdn-tree-*.md`.
- **Deleted-image references** (`lumo-vllm-audit:v0.22.0-cu129-min`, BROKEN; `vllm-openai:latest`, deleted):
  `FR10_STATUS.md:260,269,334`, `docs/reports/auto_research/fr7-current-vllm-token-tree-capture-audit-
  20260531.md:14`, `docs/reports/auto_research/fr10-gdn-tree-kernel-microbench-20260603.md:15,58,110,245`,
  `docs/reports/auto_research/fr10-gdn-tree-algebra-proof-20260603.md:191`. All historical; the broken-image
  decision is RECORDED (FR10_STATUS:334 already says "should not be used as production reference").
- **`FR13_VLLM_SOURCE_OF_TRUTH.md:19-24,41,50-51`** — DELIBERATELY names the deleted caches/images as the
  source-of-truth correction. Do NOT "fix" these; they are the canonical record of what was deleted and why.

**Recommended single-line annotation for the narrative `.md` group** (not a per-line edit): the grounding
banner already exists in `FR13_VLLM_SOURCE_OF_TRUTH.md` and `scripts/vllm_src.sh`. No further edit is needed
on frozen FR10/11/12 docs — they are dated. The actionable cleanup is 1A (3-4 scripts) + 1C (5 test files).

---

## PART 2 — RE-VERIFY LIST (load-bearing conclusions re-confirmed on the REAL 0.19.2 source)

All re-reads via `scripts/vllm_src.sh` against `3dbe092e` (`0.19.2rc1.dev134`). Line numbers below are the
REAL 0.19.2 lines (the binds' numbers are 0.19.0-keyed and OFF).

### 2A. CONV conclusion (PRIME SUSPECT — `causal_conv1d.py` diverged 123 lines) → **RE-CONFIRMED, HOLDS**

Real 0.19.2 `model_executor/layers/mamba/ops/causal_conv1d.py` = **1241 lines** (binds cite 0.19.0 lines
~565/749/831/845-863/156). The FR13_CONV_CROSSEVENT_INVESTIGATE / FR13_L4_CONV_VERDICT claims, re-checked:

- **"3-tap FIFO, width=4 → width-1=3"** — CONFIRMED. Non-spec: `state_len = width - 1` (0.19.2 **L1186**,
  also L567); spec: `state_len = width - 1 + (seqlen - 1)` (0.19.2 **L1184**). The cross-event conv state is
  a fixed width-1 sliding window; it does NOT grow with event count. ✓
- **"native conv_state_token_offset = num_accepted_tokens - 1 (spec path)"** — CONFIRMED verbatim. 0.19.2
  spec-decode update kernel: `conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1`
  (**L852-853**); else (non-spec) `= 0` (**L856**); `prior_tokens = conv_states_base +
  conv_state_token_offset*stride` (**L866**). The rolling comment block (L838-851) is the exact
  `[history2,...,historyM, draft1, draft2]` accept-2 semantics the bind quotes. ✓
- **"with num_accepted_tokens=None → state_len=width-1, offset=0"** (the A/B-design geometry that PREVENTS
  the device assert) — CONFIRMED. 0.19.2 wrapper L1183-1186 (`if num_accepted_tokens is not None: state_len
  = width-1+(seqlen-1) else: width-1`); `IS_SPEC_DECODING = num_accepted_tokens is not None` (L1234). ✓

- **ONE stale citation to fix (not a conclusion change):** `FR13_L4_CONV_VERDICT.md:9` cites native reading
  `prior_tokens = conv_states_base + (state_len-1)*stride` at "~:156". In 0.19.2 that `(state_len-1)*stride`
  read at **L157** is the **prefill** `_causal_conv1d_fwd_kernel` init-state branch (`chunk_offset==0`,
  `load_init_state`), NOT the spec-decode update read. The spec-decode read is the FIFO-tail
  `conv_state_token_offset*stride` at L866. The two land near the same line number by coincidence of drift,
  but are different code regions. The CONCLUSION (FIFO tail at offset=num_accepted-1) is correct; the
  `(state_len-1)` characterization conflates the prefill init-state read with the spec read. Annotate.

**VERDICT: "conv FIXED+CLOSED / 3-tap FIFO / native offset=num_accepted-1" HOLDS on the real 0.19.2
`causal_conv1d.py` despite the 123-line diff. The spec-path state_len/offset logic the binds claim matches
the deployed kernel byte-for-byte (logic, not line numbers).** No conclusion shifted.

### 2B. `fused_recurrent` packed-decode (recurrent rank-1, num_warps=1) → **RE-CONFIRMED, HOLDS**

Real 0.19.2 `fla/ops/fused_recurrent.py` (619 lines):
`fused_recurrent_gated_delta_rule_packed_decode_kernel` def **L256**, `tl.program_id` 2-axis one-token
(L282), the 5 recurrent ops **L327-331** (`b_h *= exp(g_val); b_v -= tl.sum(b_h*b_k); b_v *= beta_val;
b_h += b_v*b_k; b_o = tl.sum(b_h*b_q)`), ZERO `tl.dot`/chunk-loop, `num_stages=3` (L438) **`num_warps=1`
(L439)**. Wrapper `fused_recurrent_gated_delta_rule_packed_decode` def L339. Matches the source-of-truth doc
exactly. **The "carrier is codegen-alignable (geometry/l2norm/beta-cast), NOT a chunk-vs-recurrent
irreducible gap" conclusion HOLDS.**
- Stale-cite to annotate: `FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND.md:9` cites `fused_recurrent.py:438`
  for `num_warps=1` — in 0.19.2 L438 is `num_stages=3`, L439 is `num_warps=1`. Off-by-one from drift.

### 2C. `fused_sigmoid_gating` (sequential verify-dispatch) → **RE-CONFIRMED, HOLDS**

Real 0.19.2 `fla/ops/fused_sigmoid_gating.py` (279 lines):
`fused_sigmoid_gating_delta_rule_update_kernel` def **L24**, same 5 rank-1 recurrent ops **L144-153**, in-
kernel `b_beta = sigmoid(b)` (L136) + `g = -exp(A_log)*softplus` (L130-133), `num_stages=3` (L211)
`num_warps=4` (L212). This is the sequential rank-1 verify kernel underpinning
`reference_gdn_verify_sequential_dispatch` and `FR13_REPLAY_KERNEL_ALIGNMENT_PLAN`. **Structure intact; only
line numbers drifted.** No basis shift.

### 2D. Commit 7441fc43 (scan A/B gate + `fr13_native_packed_decode_ref.py`) → **CORRECT for 0.19.2**

The reference was BUILT against the stale 0.19.0 source, so I checked its native-kernel reference call:
- `scripts/fr13_native_packed_decode_ref.py:94-96` does `from vllm.model_executor.layers.fla.ops import
  fused_recurrent_gated_delta_rule_packed_decode` — it imports the **LIVE installed kernel at runtime (inside
  the container)**, NOT a /tmp cache. So the actual reference is bound to whatever 0.19.2 ships. ✓
- The kwargs it passes (L126-136: `mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out,
  ssm_state_indices, use_qk_l2norm_in_kernel`) **EXACTLY match** the real 0.19.2 wrapper signature
  (`fused_recurrent.py` L339-350). No signature drift. ✓
- The dispatch-path the commit message asserts (gdn_linear_attn → packed_decode, gated by
  `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` default True + spec_masks None + num_prefills==0 + num_decodes>0)
  is correct in 0.19.2: flag at `gdn_linear_attn.py` L385; gate in `_forward_core` L786 with the conditions
  at L809-812; `_forward_core_decode_non_spec` L1045 calls packed_decode L1085. ✓ **But** the commit
  doc-comment's specific line numbers ("L845-855 / L1261 / L1295") are the 0.19.0 lines — OFF by ~40-200 vs
  0.19.2. The LOGIC and the runtime import are correct; only the comment's line citations are stale.

**VERDICT: 7441fc43's native-kernel reference is CORRECT for 0.19.2** (it imports live + kwargs match). Fix
only the stale line numbers in its doc-comment. The empirical re-confirmation it cites
(`fr13_recurrent_decode_oracle` counter, 1104 calls / 24 toks) is a runtime measurement, version-independent.

### Re-verify summary
| conclusion | basis on real 0.19.2 | verdict |
|---|---|---|
| conv FIXED+CLOSED, 3-tap FIFO, offset=num_accepted-1 | causal_conv1d.py L852-866, L1184-1186, L1234 | **HOLDS** |
| packed-decode = recurrent rank-1, num_warps=1 | fused_recurrent.py L256,282,327-331,439 | **HOLDS** |
| fused_sigmoid_gating sequential verify dispatch | fused_sigmoid_gating.py L24,144-153,212 | **HOLDS** |
| 7441fc43 native reference call correct | imports live kernel; kwargs match L339-350 | **CORRECT** |

No load-bearing conclusion's BASIS shifted on the real source. The 123/40/15-line diffs moved line numbers,
not logic. The remaining work is citation hygiene (annotate 0.19.0 line numbers), not re-deriving conclusions.

---

## PART 3 — UPSTREAM verdict (is our work redundant vs vLLM LATEST? does why-not-upgrade hold?)

Latest released vLLM = **0.22.1 (2026-06-05)** / 0.22.0 (2026-05-29). Checked online 2026-06-14.

### (a) #42960 GDN batch-invariance — **STILL OPEN, zero merged PR**
GitHub API: `state=open`, `closed_at=null`, `pull_request=null`, `updated_at=2026-06-12`. Title "[Feature]:
Batch-invariant support for GDN_ATTN (Qwen3-Next / Qwen3.6 hybrid Mamba+GDN MoE models)". The hard abort
`"VLLM batch_invariant mode is not supported for GDN_ATTN"` is unchanged. 0.22's batch-invariant additions
are **Cutlass FP8 / GEMM-linear only** (release notes), NOT GDN recurrent-state. The FR9 finding (2026-06-03)
HOLDS unchanged.

### (b) Did upstream change the GDN kernels we align to, or the spec-decode tree machinery? — **NO basis shift**
- The decode/verify kernels we align to — `fused_recurrent_gated_delta_rule_packed_decode` and
  `fused_sigmoid_gating_delta_rule_update` (PR #30860, already in our 0.19.2 image) — are stable into
  0.21/0.22. The v0.21/0.22 GDN ADDITIONS are **prefill** kernels (v0.21 "GDN for Qwen3.5/3.6 on CPU"; v0.22
  "FlashInfer Blackwell GDN prefill" + "GDN prefill kernel for SM100"). GB10 is **sm_121**, which routes to
  Triton/FLA (per `reference_gb10_gdn_backend_fla`: flashinfer GDN is Hopper/Blackwell-only) — the new
  SM100/Blackwell prefill kernels do not even run on GB10, and they are prefill not decode/verify. Our
  alignment target (packed-decode + sigmoid-gating) is unaffected. **No conclusion would differ on 0.22.**
- Spec-decode TREE machinery: #18327 "Tree-Attention Support for Speculative Decoding" is **CLOSED
  not_planned** (stale-bot, 2025-11-06). #3960 also a dormant request. Upstream has NOT shipped tree-attn
  spec-decode in any release. So there is no upstream tree machinery our 0.19.2 work would have to track.

### (c) Does upstream latest NATIVELY provide anything we BUILT? — **NO. Nothing for us.**
| our deliverable | upstream-latest native equivalent? | evidence |
|---|---|---|
| FA2 tree-bias fork (additive -inf bias, no-copy tree decode) | **No** | tree-attn spec #18327 CLOSED not_planned; no FA2 tree-bias in releases |
| tree-conv-fused | **No** | no tree-spec → no tree conv path upstream |
| GDN tree-scan (rank-1 over co-resident chain) | **No** | no GDN tree-spec verify kernel upstream |
| committer / GDN SSM state-rollback on rejection | **No — it's an OPEN BUG upstream** | **#39273 OPEN/unfixed** (2026-04-08, assignee tdoublep, 9 comments, no resolution): ngram spec-decode "advances GDN SSM state by N steps" with "no mechanism to revert state when tokens are rejected"; `causal_conv1d_update_v2` lacks the snapshot path. sglang #25587 mirrors it (GDN-MTP not lossless). |
| no-copy tree verify / batch-invariant GDN verify | **No** | #42960 OPEN (no GDN BI primitive); no isolated num_reqs=1 recurrent-forward primitive in any release |

**The GDN-state-rollback primitive our committer implements is not even a FIXED upstream feature — it's an
OPEN upstream bug (#39273).** Upgrading would inherit the corruption, not a solution. This STRENGTHENS the
why-not-upgrade decision beyond the FR9 study.

### Upstream verdict
**Why-not-upgrade HOLDS, on two independent legs (both re-confirmed online 2026-06-14):**
1. An upgrade buys NOTHING for lossless GDN — #42960 OPEN (no GDN batch-invariance), no isolated
   recurrent-forward, tree-spec not_planned (#18327). 0.22 GDN work is prefill/Blackwell, off-path for GB10
   (sm_121 → Triton/FLA) and off-target for our decode/verify alignment.
2. Everything we built (FA2 tree-fork, tree-conv, GDN tree-scan, committer, no-copy verify) has NO native
   upstream equivalent; the closest — GDN state-rollback — is an OPEN upstream bug (#39273), not a feature.
   Re-porting FR13 to 0.22 is churn for zero lossless benefit.

**Not redundant. Do not upgrade for the lossless objective.** (Speed-tuning is a separate question; nothing
in 0.22 changes the GB10 GDN decode/verify path that governs FR13 speed.)

---

## Sources (online, AXIS 2)
- vLLM #42960 (GDN batch-invariance, OPEN): https://github.com/vllm-project/vllm/issues/42960 — API
  state=open, closed_at=null, updated 2026-06-12
- vLLM #18327 (Tree-Attention spec-decode, CLOSED not_planned): https://github.com/vllm-project/vllm/issues/18327
- vLLM #39273 (GDN ngram spec-decode state-rollback corruption, OPEN/unfixed): https://github.com/vllm-project/vllm/issues/39273
- vLLM PR #30860 (fused_sigmoid_gating_delta_rule_update — the verify kernel we align to): https://github.com/vllm-project/vllm/pull/30860
- vLLM v0.22.0 release (batch-invariant = Cutlass FP8; GDN prefill SM100/Blackwell): https://github.com/vllm-project/vllm/releases/tag/v0.22.0
- vLLM v0.21.0 release (GDN for Qwen3.5/3.6 CPU): https://github.com/vllm-project/vllm/releases/tag/v0.21.0
- sglang #25587 (GDN-MTP not lossless, mirror): https://github.com/sgl-project/sglang/issues/25587

## Provenance (AXIS 1, REAL 0.19.2 source)
All re-reads via `scripts/vllm_src.sh <relpath>` against the pinned image
`vllm/vllm-openai@sha256:3dbe092e…` (`0.19.2rc1.dev134+gfe9c3d6c5`, local `ffa30d66`), 2026-06-14.
Files: `model_executor/layers/mamba/ops/causal_conv1d.py` (1241 L),
`model_executor/layers/fla/ops/fused_recurrent.py` (619 L),
`model_executor/layers/fla/ops/fused_sigmoid_gating.py` (279 L),
`model_executor/layers/mamba/gdn_linear_attn.py` (1211 L).
