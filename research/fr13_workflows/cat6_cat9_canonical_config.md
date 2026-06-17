# FR13 Canonical Config — cat6 + cat9 SHIPPED Deploy (Flag Consolidation)

**Status:** authoritative as of HEAD `5b74042d` (verified against source 2026-06-17).
**Supersedes for flag-classification:** the stale `FR13_FLAGS.md` (drift-gate framing) and the cat9-only `FR13_PIPELINE_LOCK.md` (predates the LUMO_FB pad bake-in and cat6).
**Scope:** cat6 and cat9 are the two SHIPPED Qwen3-Next-27B fp8 GDN-hybrid TREE speculative-decode configs on DGX Spark GB10 (vLLM 0.19.x). Both beat native E5 MTP-5 and are lossless within the E5 floor (merged to main 2b29e599). This doc is a catalog only — it does **not** change runtime behavior.

---

## 1. Where we are

cat6 and cat9 share **one** container launcher: `scripts/fr13_launch_forked_fa2_tree_server.sh`, which emits ~120 `-e` env flags. cat9 boots it via the thin wrapper `scripts/fr13_launch_locked.sh` (pins every pipeline flag explicitly, forces diagnostics OFF unless `--arm`). cat6 (`KIND=cat6root` in `scripts/fr13_bigdenom_swe_serve_variant.sh`) boots the same forked launcher **directly** with the TREE overridden to the 6-node depth-5 root-sibling shape and `LUMO_FB_KERNEL_ROWS=1 / LUMO_FB_PROJ_PAD_ROWS=16` pinned.

**The only real per-arm differences are:**
1. The speculative tree SHAPE — cat9 = 9-node 5-spine+top-2-leaf (launcher default); cat6 = 6-node depth-5 root-sibling override; `EXPECT_RATIO` 9 vs 6.
2. `GPU_UTIL=0.82` / `MAX_NUM_SEQS=1` set by the variant wrapper for both arms.

Everything else is identical.

**Why the ~120 flags collapse to ~14:** over FR10→FR13, 9 former pipeline-FIX flags were **BAKED** into `scripts/fr10_phase4_patch_vllm_tree_gdn.py` as literal `True`/`"1"` (commits 219d41de / a09ef5b5 / 45dc05a2). Verified at HEAD: those 9 flags have **0 `os.environ.get` reads** remaining in the patch, so the launcher's `-e …=1` for them is **inert**. The remaining sprawl is (a) ~80 diagnostic default-OFF capture/MAB/gate/profiling/trace flags used only for A/B byte-identity proofs, and (b) ~12 dead/superseded flags (FR12 native-spine oracle splice, `FR13_SCAN_ALIGN`, `FR13_NPAD_INVARIANT`, `FR13_TREE_REMAP_SEQ`-as-env, the OPT-A/OPT-1/device-multidraft speed candidates).

**Two load-bearing flags whose CODE default is OFF (must be set explicitly):**
- `FR13_FA2_TREE_BIAS` and `FR13_FA2_PREFILL_NATIVE` — the injected FA2 code reads them with default `"0"` (verified `scripts/fr13_patch_fa2_tree_bias.py` L499/L570/L783/L784). Left env-gated **on purpose** to preserve the patcher's re-patch idempotency anchor. Unset = the lossy `unified_attention` qq_bias fallback.
- `LUMO_FB_KERNEL_ROWS=1` (+ `LUMO_FB_PROJ_PAD_ROWS=16`) — the patch early-returns unless `== "1"` (L5670) **and** the forked launcher's default is empty/OFF (L131). It is pinned ON only by the locked wrapper (L34) and the variant (L139). It is the authorized #42960 targeted in_proj_ba batch-invariance (removes ~8 of the +17 leaf co-residency flips). Must be explicit.

---

## 2. Minimal load-bearing config

### Shared core (identical for cat6 and cat9)

| Flag | Value | Why load-bearing |
|---|---|---|
| `FR10_ENABLE_TREE_GDN` | `1` | Master engage; patch reads `=="1"` with NO fallback (env-read ×5) — unset = tree path inert |
| `FR10_DECODE_MODE_DEFAULT` | `tree_mtp` | Selects tree-verify mode; committer/replay keyed off `=="tree_mtp"` (env-read ×9) |
| `FR13_FA2_TREE_BIAS` | `1` | Forked-FA2 additive -inf ancestry bias decode path; **code default `"0"`**, unset = lossy. NOT baked |
| `FR13_FA2_PREFILL_NATIVE` | `1` | Native FA2 prefill (bias only on decode); **code default `"0"`**, unset = wrong prefill. NOT baked |
| `LUMO_FB_KERNEL_ROWS` | `1` | in_proj_ba pad-to-fixed-M batch-invariance; **patch + launcher default OFF**, must pin. Drops ~8 leaf flips |
| `LUMO_FB_PROJ_PAD_ROWS` | `16` | Fixed pad-row count ≥ max tree_n (16 ≥ cat9's 9, cat6's 6); pin explicitly (config-dependent) |
| `BATCH_INVARIANT` | `0` | Global BI OFF; `=1` = GB10 REDUCED branch, perturbs fp8/scan (cat9+BI=34 flips) |
| `LUMO_BATCH_INVARIANT_VLLM` | `0` | lumo-side BI mirror; variant needle hard-asserts both `=0` |
| `FR11_TREE_CONV_NATIVE_BF16_TAPS` | `1` | Native bf16 conv taps (conv bit-exact); patcher **RAISES if 0** (L841); launcher hardcodes (L330) |
| `FR12_TREE_CONV_NATIVE_BF16_TAPS` | `1` | FR12 alias (falls back to FR11 default `"1"`); launcher hardcodes (L331) |
| `--attention-backend` (CLI) | `TREE_ATTN` | Patched tree-bias FA2 decode backend |
| `--gdn-prefill-backend` (CLI) | `triton` | GB10 `fla_chunk` GDN prefill backend |
| `GPU_UTIL` / `MAX_NUM_SEQS` | `0.82` / `1` | Operational co-residency knobs (B=1 deploy), set by the variant wrapper |

### Per-arm: the SPEC_CONFIG tree (the ONLY differentiator)

**cat9** — 9-node, 5-spine + top-2-leaf, `num_speculative_tokens=9`, `EXPECT_RATIO=9`:
```
SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":9,
  "speculative_token_tree":"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"}
```

**cat6** (cat6root) — 6-node, depth-5, root-sibling, `num_speculative_tokens=6`, `EXPECT_RATIO=6`:
```
SPEC_CONFIG={"method":"qwen3_5_mtp","num_speculative_tokens":6,
  "speculative_token_tree":"[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"}
```

> `SPEC_CONFIG` is auto-constructed by the forked launcher from `TREE` + `NUM_SPECULATIVE_TOKENS` (L175/L181). You can pass `TREE` and let the launcher build `SPEC_CONFIG`, or pass `SPEC_CONFIG` directly.

### Also load-bearing as an *absence* (not a positive flag)

| Flag | State | Why |
|---|---|---|
| `FR10_ALLOW_LINEAR_FALLBACK` | **UNSET** (launcher actively `unset`s it, L424) | Fail-loud: a stray `=1` would silently fall back to linear and mask a vacuous tree run. Patch checks `!= "1"` in 19 places. Keep the `unset`. |

---

## 3. Full classified flag catalog

| Flag / family | Value | Classification | Load-bearing | Notes |
|---|---|---|---|---|
| `FR10_ENABLE_TREE_GDN` | 1 | needed_default | yes | Master engage; `=="1"` no fallback |
| `FR10_DECODE_MODE_DEFAULT` | tree_mtp | needed_default | yes | Tree-verify mode select |
| `FR13_FA2_TREE_BIAS` | 1 | needed_default | yes | **Code default `"0"`** — must set; NOT baked |
| `FR13_FA2_PREFILL_NATIVE` | 1 | needed_default | yes | **Code default `"0"`** — must set; NOT baked |
| `LUMO_FB_KERNEL_ROWS` | 1 | needed_default | yes | **Patch + launcher default OFF** — must pin |
| `LUMO_FB_PROJ_PAD_ROWS` | 16 | needed_default | yes | Pad rows ≥ max tree_n |
| `BATCH_INVARIANT` | 0 | needed_default | yes | Must stay OFF |
| `LUMO_BATCH_INVARIANT_VLLM` | 0 | needed_default | yes | Tracks BATCH_INVARIANT |
| `FR11_TREE_CONV_NATIVE_BF16_TAPS` | 1 | needed_default | yes | Patcher RAISES if 0; launcher hardcodes |
| `FR12_TREE_CONV_NATIVE_BF16_TAPS` | 1 | needed_default | yes | FR12 alias of bf16-tap fix |
| `SPEC_CONFIG` / `TREE` / `NUM_SPECULATIVE_TOKENS` | per-arm | needed_default | yes | **The cat6 vs cat9 differentiator** |
| `--attention-backend` (CLI) | TREE_ATTN | needed_default | yes | Tree-bias FA2 decode backend |
| `--gdn-prefill-backend` (CLI) | triton | needed_default | yes | GB10 fla_chunk prefill |
| `GPU_UTIL` / `MAX_NUM_SEQS` | 0.82 / 1 | needed_default | yes (op) | Variant wrapper; B=1 deploy |
| `VLLM_SERVER_DEV_MODE` | 1 | needed_default | no (op) | `/reset_prefix_cache` dev endpoint the harness calls between runs; not model-behavior |
| `FR10_ALLOW_LINEAR_FALLBACK` | UNSET | needed_default | yes (as absence) | Fail-loud guard; keep `unset` |
| `FR13_TREE_BONUS_SELF` | 1 | needed_default | yes (default) | Live env-read default `"1"`; default IS locked path → belt-and-suspenders |
| `FR13_REPLAY_ROUTE` | 1 | baked_permanent* | yes | Mostly baked (`if True`) + 1 surviving injected gate (L10372 default `"1"`); dep anchor for conv-fusion |
| `FR13_DRAFTER_SINGLE_LOGITS` | 1 | baked_permanent | no | Baked True L11429 (0 env reads). Speed FIX-1 |
| `FR13_EAGER_PACK` | 1 | baked_permanent | no | Baked True L788. Speed FIX-2 transport-only |
| `FR13_TREE_CONV_FUSED` | 1 | baked_permanent | no | Baked True L797. Speed FIX-3, bit-exact |
| `FR13_TREE_SAMPLE_ROW` | 1 | baked_permanent | no | Baked True L11147. FIX-A leaf-row (accept-superset crossing) |
| `FR13_CONV_COMMITTED_PATH` | 1 | baked_permanent | no | Baked True L2123. Branch-winner conv-prior seam |
| `FR13_TREE_PER_REQ_GEN` | 1 | baked_permanent | no | Baked True L8257. Non-det fix |
| `FR13_TREE_REQKEY` | 1 | baked_permanent | no | Baked (dep-guard `if False` L11154). FIX-A prereq |
| `FR13_TREE_ATTN_EXP2_SOFTMAX` | 1 | baked_permanent | no | Baked (alt-patch `if False` L13086). exp2 = locked path |
| `FR13_TREE_REMAP_SEQ` | 1 | dead_superseded | no | **DEAD as env** (0 reads); baked into kernel |
| `FR13_NPAD_INVARIANT` | 0 | dead_superseded | no | **DEAD** (0 serving reads); bench-workflow only |
| `FR13_SCAN_ALIGN` / `_MODE` | 0 | dead_superseded | no | **DEAD/forbidden** (0 serving reads); variant exits 3 if `=1`; WY parked |
| FR12 native-spine ORACLE family (`FR12_NATIVE_SPINE_ORACLE`, `FR12_TREE_CONV_NATIVE_SPINE`, `FR12_TREE_SCAN_NATIVE_SPINE`, `FR12_TREE_CONV_NATIVE_PRIOR_READ`, `FR12_TREE_CONV_STATE_FULL_CAPTURE`) | 0 | dead_superseded | no | Oracle-only splice (reward-hack if baked); superseded by bit-exact GDN kernel |
| `FR13_GB10_FP8_GEMV_CFG` | 0 | diagnostic_off | no | OPT-A speed candidate; only cat9-opta arm sets `=1` |
| `FR13_GPU_COMMITTER` / `FR13_COMMITTER_SYNCKILL` | 0 | diagnostic_off | no | OPT-1 (greedy-only, dead at t0.6); only cat9-opt1 arm |
| `FR13_DEVICE_MULTIDRAFT` (+`_KERNEL`) | 0 | diagnostic_off | no | Device multidraft candidate; temp06 TV gate pending; not shipped |
| `FR13_COMMIT_ARGMAX_GATE` (+`_DUMP`) | 0 | diagnostic_off | no | Binding per-token argmax lossless instrument, EAGER-only. **NEVER bind =1 into serving** |
| `FR13_HIDDEN_SUBSTITUTE` | empty | diagnostic_off | no | Layer-splice oracle for localization, OFF=byte-identical |
| `FR13_FORCE_SPINE_COMMIT` | 0 | diagnostic_off | no | A/B footgun; **NEVER bind =1** (breaks branch accept/superset) |
| `FR13_FIX1_SELFCHECK` (+`_DUMP`) | 0 | diagnostic_off | no | A/B byte-identity proof for baked FIX-1 |
| `FR13_FORK_MARGIN_DUMP` (+`_PATH`) | 0 | diagnostic_off | no | Read-only committer-fork margin classifier, EAGER-only |
| `FR13_CHASE_DIAG` family (`_DIR/_TOPK/_KV_WINDOW/_H3/_H3_LAYER/_KV_ALLOW_EMPTY`) | 0 | diagnostic_off | no | Superset-chase taps; family inert unless `CHASE_DIAG=1` |
| `FR13_GDN_SUBOP_MAB` family (+`_DUMP/_LAYER/_SKIP/_LIMIT/_EXPECT_TREE_N/_THRESHOLD`) + `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY/_PREFIXES` | 0 | diagnostic_off | no | M-invariance taps; ray-copy vars measured INSUFFICIENT (curated worker env) = dead belt-and-suspenders |
| `FR13_FA2_MAB` family (+`_DUMP/_LAYER/_SKIP/_LIMIT`) | 0 | diagnostic_off | no | FA2 M-invariance taps |
| `FR13_BI_TREE_ATTN` | 0 | diagnostic_off | no | BI allowlist; doubly-inert (needs `BATCH_INVARIANT=1`) |
| `FR10_METRICS` | 0 | diagnostic_off | no | Metrics emission; ON slows GEMMs |
| `FR13_REPLAY_BOUNDARY_LOG` / `FR13_REPLAY_DURABLE_AB` families | 0 | diagnostic_off | no | Replay-boundary state-parity taps |
| `FR13_TCF_SELFCHECK` / `FR13_TCF_DIAG_OVERRIDE` | 0 | diagnostic_off | no | Conv-prior byte self-check |
| `FR13_SFWD_GPU_TIMER` (+`_JSON/_MAXPENDING`) / `FR13_TORCH_DET_WARN` (+`_LOG`) | 0 | diagnostic_off | no | Profiling/debug hooks |
| All `*_CAPTURE` families (~40 sub-flags) | empty/0 | diagnostic_off | no | Per-layer/per-op tensor capture (top-down lossless ladder); inert when unset. **Bulk of the sprawl** |
| `LUMO_NSYS_*` family (`WRAP_VLLM/BIN/DELAY_S/DURATION_S/FLUSH_MS/CONFIG_DIRECTIVES/TRACE/OUTPUT`) | WRAP_VLLM=0 | diagnostic_off | no | nsys profiling; family inert unless `WRAP_VLLM=1` |
| `LUMO_MTP_DRAFT_TRACE_FILE` / `LUMO_TREE_SAMPLER_DEBUG_LOG` / `LUMO_TREE_PATH_LCP_LOG` | empty | diagnostic_off | no | Trace logs; inflate t0.6 committer tax if set |
| `CUDA_LAUNCH_BLOCKING` / `TORCH_USE_CUDA_DSA` / `ENFORCE_EAGER` | 0 | diagnostic_off | no | Debug/eager; shipped config is CUDA-graph-captured |

\* `FR13_REPLAY_ROUTE` is the one nuance: the main route is baked `if True` (L313/733/2484/4149/10964) but **one** injected gate still reads the env with default `"1"` (L10372). Because the default equals the locked value, an explicit `=1` is redundant — but it is the dependency anchor for the baked conv-fusion/eager-pack, so it is conservatively retained. See open questions.

*The `*_CAPTURE` family enumerated:* `FR10_ROOT/LAYER_HIDDEN_CAPTURE*`, `FR10_TREE_GDN_CAPTURE_PAYLOAD*/COMMIT_HANDOFF*/SRC_NATIVE_PAYLOAD/ROOT_H0_LOG/COUNTER_DUMP/DEPTH_POSITION_LOG`, `FR10_ROOT/SPINE_LOGIT_CAPTURE*`, `FR12_FULL_ATTN_CAPTURE*`, `FR12_SUBKERNEL_CAPTURE*`, `FR13_TREE_ATTN_OP_CAPTURE*`, `FR13_FLASH_ATTN_OP_CAPTURE*`, `FR13_PREPROCESS_INPUT_CAPTURE*`, `FR13_PREFILL_GDN_CAPTURE*`, `FR13_FINAL_LOGIT_CAPTURE*`. All default empty/OFF → inert.

---

## 4. Diagnostic / dead list to drop from a clean launcher

**Baked-permanent (env now inert — drop the `-e`):** `FR13_DRAFTER_SINGLE_LOGITS`, `FR13_EAGER_PACK`, `FR13_TREE_CONV_FUSED`, `FR13_TREE_SAMPLE_ROW`, `FR13_CONV_COMMITTED_PATH`, `FR13_TREE_PER_REQ_GEN`, `FR13_TREE_REQKEY`, `FR13_TREE_ATTN_EXP2_SOFTMAX`. (Behavior is permanently the locked ON path; toggling the env does nothing.)

**Dead / superseded (drop):** `FR13_TREE_REMAP_SEQ` (env dead, baked in kernel), `FR13_NPAD_INVARIANT` (bench-only), `FR13_SCAN_ALIGN`/`_MODE` (forbidden — variant exits 3 if `=1`), the FR12 native-spine oracle splice family.

**Speed candidates not in shipped default (drop):** `FR13_GB10_FP8_GEMV_CFG` (cat9-opta), `FR13_GPU_COMMITTER`/`FR13_COMMITTER_SYNCKILL` (cat9-opt1), `FR13_DEVICE_MULTIDRAFT`.

**Diagnostic default-OFF (drop from serving launch; arm via `--arm` only):** `FR13_COMMIT_ARGMAX_GATE`, `FR13_HIDDEN_SUBSTITUTE`, `FR13_FORCE_SPINE_COMMIT`, `FR13_FIX1_SELFCHECK`, `FR13_FORK_MARGIN_DUMP`, the `FR13_CHASE_DIAG` family, the `FR13_GDN_SUBOP_MAB`/`FR13_FA2_MAB` families + `VLLM_RAY_EXTRA_ENV_VARS_*`, `FR13_BI_TREE_ATTN`, `FR10_METRICS`, the replay-boundary/durable-AB families, `FR13_TCF_*`, `FR13_SFWD_GPU_TIMER`/`FR13_TORCH_DET_WARN`, **all `*_CAPTURE` families (~40)**, the `LUMO_NSYS_*` family, the `LUMO_*_LOG`/`_TRACE_FILE` taps, and `CUDA_LAUNCH_BLOCKING`/`TORCH_USE_CUDA_DSA`/`ENFORCE_EAGER`.

> **Footgun reminder:** `FR13_FORCE_SPINE_COMMIT=1` and `FR13_COMMIT_ARGMAX_GATE=1` (and the FR12 oracle splices) must NEVER be bound into a serving/gate/speed number. They are A/B and localization instruments only.

---

## 5. Clean canonical launch invocation

```bash
# CANONICAL cat6 / cat9 LAUNCH — only the load-bearing flags.
# Both arms identical except the SPEC_CONFIG tree (TREE + num_speculative_tokens).
# Boots via scripts/fr13_launch_forked_fa2_tree_server.sh; the launcher builds
# SPEC_CONFIG from TREE + NUM_SPECULATIVE_TOKENS and emits the vllm serve CLI with
# --attention-backend TREE_ATTN --gdn-prefill-backend triton.

# ---- shared load-bearing env (identical for cat6 and cat9) ----
export FR10_ENABLE_TREE_GDN=1            # master engage; patch reads =="1" no fallback
export FR10_DECODE_MODE_DEFAULT=tree_mtp # tree-verify decode mode
export FR13_FA2_TREE_BIAS=1              # forked-FA2 -inf tree bias (code default "0" -> MUST set)
export FR13_FA2_PREFILL_NATIVE=1         # native FA2 prefill (code default "0" -> MUST set)
export LUMO_FB_KERNEL_ROWS=1             # in_proj_ba pad batch-invariance (patch+launcher default OFF -> MUST pin)
export LUMO_FB_PROJ_PAD_ROWS=16          # pad rows >= max tree_n
export BATCH_INVARIANT=0                 # global BI must stay OFF
export LUMO_BATCH_INVARIANT_VLLM=0       # BI mirror, tracks BATCH_INVARIANT
export FR11_TREE_CONV_NATIVE_BF16_TAPS=1 # native bf16 conv taps (patcher RAISES if 0)
export FR12_TREE_CONV_NATIVE_BF16_TAPS=1 # FR12 alias of bf16-tap fix
export GPU_UTIL=0.82                     # B=1 deploy co-residency
export MAX_NUM_SEQS=1                    # B=1 deploy
unset FR10_ALLOW_LINEAR_FALLBACK         # fail-loud: no silent linear fallback
# CLI emitted by the launcher: --attention-backend TREE_ATTN --gdn-prefill-backend triton

# ---- cat9 (9-node, 5-spine + top-2-leaf, EXPECT_RATIO=9) ----
TREE="[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]" \
NUM_SPECULATIVE_TOKENS=9 \
  scripts/fr13_launch_forked_fa2_tree_server.sh
# (canonical cat9 path: scripts/fr13_launch_locked.sh pins exactly this set.)

# ---- cat6 / cat6root (6-node, depth-5, root-sibling, EXPECT_RATIO=6) ----
TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]" \
NUM_SPECULATIVE_TOKENS=6 \
  scripts/fr13_launch_forked_fa2_tree_server.sh
# (in practice cat6 boots via:
#  scripts/fr13_bigdenom_swe_serve_variant.sh <ARM> cat6root
#  which sets the TREE override + GPU_UTIL=0.82 MAX_NUM_SEQS=1 + the LUMO_FB pins.)
```

---

## 6. Flag-count reduction

- **cat6:** ~120 `-e` flags emitted by the forked launcher → **~14 load-bearing** (10 env + 2 CLI + SPEC_CONFIG + the GPU_UTIL/MAX_NUM_SEQS op pair).
- **cat9:** same ~120 `-e` (via the locked wrapper, which already curates) → **~14 load-bearing**.
- **~106 droppable:** ~9 baked-permanent (inert env) + ~80 diagnostic default-OFF + ~12 dead/superseded. Plus `FR10_ALLOW_LINEAR_FALLBACK` stays UNSET (a guard, not a positive flag).

---

## 7. Source-of-truth references (cite, don't restate)

- **Env defaults:** `scripts/fr13_launch_forked_fa2_tree_server.sh` (the ~120 `-e` lines + their `${:-}` defaults).
- **Baked literals / live env reads:** `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (grep `baked ON`; the `os.environ.get` reads at the line numbers cited above).
- **FA2 flags:** `scripts/fr13_patch_fa2_tree_bias.py` (the `os.environ.get("FR13_FA2_*","0")` reads).
- **cat9 executable lock:** `scripts/fr13_launch_locked.sh`.
- **Arm wiring (cat6/cat10/opt arms) + container-env needle:** `scripts/fr13_bigdenom_swe_serve_variant.sh`.
- **Lock rationale + dependency invariants:** `FR13_PIPELINE_LOCK.md`.
