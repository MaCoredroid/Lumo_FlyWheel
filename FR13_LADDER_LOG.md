# FR-13 ladder log — per-commit strict drift gates (verify-path + regular-decode)

Each entry binds a commit to its STRICT top-down ladder results. Two gates per commit:
1. **VERIFY-PATH** — tree verify (tree-bias) byte-exact to native-FA2-on-path oracle: input + every full_attn layer + final logits = 0.0, spine AND branch.
2. **REGULAR-DECODE (verifier-only)** — plain decode (no tree/spec/bias) with the forked FA2 == ORIGINAL pristine stock FA2: 0.0 at every layer.

A `FR10_ALLOW_LINEAR_FALLBACK` run is DIAGNOSTIC ONLY (GDN may go linear) → NEVER bound here. Both gates must be STRICT + 0.0 to record a PASS. (codex_fr14 appends; this first entry recorded by the monitor from verified artifacts.)

---

## Commit `2900203b` (fr13 handle suffix tree bias offsets) — forked FA2 tree-bias kernel

### Kernel smoke (direct CUDA, isolated)
- `suffix_offset_tree_bias_smoke`: **equal=True, max_abs=0.0** vs native-FA2-on-path oracle (whole tree, one FA2 call). PASS.

### Gate 2 — REGULAR-DECODE (verifier-only) — **PASS, byte-exact**
- Artifacts: `output/fr13_regular_decode_gate_latest.json` (`fr13.ladder_table.v1`), `output/fr13_regular_decode_full_attn_gate_latest.json` (`fr13.regular_full_attn_pair_table.v1`).
- Config: strict — plain decode, no tree/spec, no bias; forked `_vllm_fa2_C` vs **pristine stock** `_vllm_fa2_C`; B=1 eager.
- Result: input `torch_equal=true, max_abs=0.0`; **all full_attn layers passed=True**; final logits `torch_equal=true, max_abs=0.0`. **`passed=True`.**
- ⟹ the fork does NOT touch regular decode; the E5 lossless baseline (which runs the same forked FA2 at no-bias) is provably unmoved; the `apply_tree_bias` insertion does not perturb FA2 codegen/MMA.

### Gate 1 — VERIFY-PATH — **PENDING (rerun in progress)**
- The strict verify run captured full_attn stage outputs + spine logits, but the FA2 route bypassed the TREE_ATTN op-capture hook, so the low-level tree-op (q/k/v) for **branch**-path replay was not emitted. codex is patching the FA2 route to call `_fr13_tree_attn_op_capture(...)` before the forked FA2, then rerunning the strict verify gate (no `FR10_ALLOW_LINEAR_FALLBACK`). Spine full-attn byte-exact pending confirmation; branch-path oracle pending the rerun. NOT yet a PASS.

---

## Commit `fe21cb73` (FR13 WIP: route TREE_ATTN through forked FA2 + path oracle) — Gate 1 rerun result

### Gate 1 — VERIFY-PATH — **NOT byte-exact 0.0 (single-ULP fragment-grouping floor); strict config, NOT a PASS-at-0.0**
- **Strict config (verified):** `--attention-backend TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` **UNSET**, GDN tree branch active, all 16 `tree_attn_op.call*.pt` captured. Run dir `output/fr13_verify_strict_tree_20260607T091935Z` (forked FA2 computing, splice OFF — no reward-hack).
- **Bias is true `-inf`** (capture `tree_attn_bias` `uniq_nonzero=[-inf]`, `has_neginf=True`) ⟹ masked (non-ancestor) keys contribute `exp2(-inf)=0` **exactly**; `x+0.0` is exact in IEEE fp. So the residual is **not** a masking/bias bug.
- **Residual = a single-ULP MMA fragment reduction-grouping floor on scattered no-copy keys.** In the shared no-copy tree KV the ancestor keys sit at scattered slots (`TREE_PARENT=[-1,0,1,1,2,2,4,4,6,6]`; branch nodes 3,5,7,9 interleaved between spine nodes), so the path keys land in different accumulation lanes than the **packed** oracle's contiguous slots → fp32 non-associative partial sums differ by one ULP. Measured (call15 corrected oracle `fa2_path_refs_v2`):
  - **stacked-spine packed oracle (`tree_vs_fa2_spine`): `max_abs 0.00390625`, `nonzero: 1`** (one element, one bf16 ULP).
  - branch row 6 (path `[0,1,2,4,6]`): 1 nonzero, 0.0039. call10 branch row 9 (path `[0,1,2,4,6,9]`): 0.00098 (1 ULP).
  - per-row oracle row-0 `0.125/512-nonzero` is a **single-query-harness artifact** (row 0 is **exact** inside the stacked-spine oracle, which has only 1 total nonzero on row 6) — to confirm via the new-direction workflow.
- **Interpretation (red-team, monitor):** this is the **same** reduction-grouping irreducibility `FR13_FA2_TREE_BIAS_FORK_RESEARCH.md` pinned on Triton-vs-CUTLASS — and it is now shown **intrinsic to no-copy shared-KV itself**, independent of kernel. Literal **byte-exact 0.0 on scattered rows is unreachable without per-row contiguous repacking** (which is a design change / possible copy). It IS within the E5 self-noise floor (~0.059) ⟹ **distributionally/argmax lossless** (the theorem-backed branch gate, `[[reference_gdn_tree_branch_oracle_losslessness]]`).
- **Verdict:** **NOT a 0.0 PASS** under the literal byte-exact-0.0 gate. The single-ULP floor is the expected no-copy regime, not a bug. **Open gate-definition decision (user):** accept the 1-ULP grouping floor and proceed to the e2e deliverable gate (lossless within E5 floor + superset accept/event ≥ E5) vs insist on literal 0.0 via per-row gather. New-direction workflow launched to verify the floor is irreducible + confirm the row-0 artifact + produce the e2e measurement plan. NOT bound as a PASS.
- Gate 2 (regular-decode) unchanged from `2900203b`/`d2f1ba18`: byte-exact 0.0, still valid (no regular-decode code changed).

> **CORRECTION (workflow `w86uygp1x`, `FR13_FLOOR_WORKFLOW_VERDICT.md`):** the v1 numbers above ("row-0 0.125/512", phrasing implying broad drift) are superseded by the v2 oracle on raw `.pt`: **14/16 calls byte-exact 0.0 on the whole tree; spine byte-exact 15/16; exactly 2 single-bf16-ULP elements total across 16 events** (call15 spine row6 head20 dim158 = 0.0039; call10 branch row9 = 0.00098). row-0 = **0.0078/1037**, confirmed single-query-vs-stacked FA2 harness artifact (tree[0] == stacked oracle exactly). The residual is a ~2e-6 probabilistic single-ULP rounding event (no depth growth = rounding signature, not bug signature), NOT a deterministic floor; no impossibility theorem. Still NOT a literal-0.0 PASS; the real verdict is the e2e vs-E5 measurement (gate decision with the user pending).

---

## Commit `ed0390df` (FR13: use ex2.approx SiLU for tree conv) — offline clean-input GDN conv gate

### Gate A sub-op — causal-conv clean-input replay — **OFFLINE 0.0, pre-live**

- Config: boot-free same-input replay; clean tree/native captures only; L0 from `output/fr13_gdn_conv_multilayer_capture_20260607T193044Z`, fresh L45 from `output/fr13_gdn_l45_fullstate_20260607T175434Z`.
- Strictness: no `FR10_ALLOW_LINEAR_FALLBACK`; no native `causal_conv1d_update` reroute. The tree path uses our Triton helper `triton_ex2_silu_bf16`, whose `tl.exp` lowers to NVIDIA `ex2.approx.f32`, matching the native activation instruction sequence.
- Artifact: `output/fr13_clean_input_l0_l45_ex2_replay.json` (`fr13.ex2_silu_replay.v1`).

| layer | clean pre_conv | captured tree conv vs native | ex2 helper conv vs native |
| ---: | --- | ---: | ---: |
| 0 | yes | 0.0 / 0 nz | 0.0 / 0 nz |
| 45 | yes | 0.0009765625 / 1 nz | 0.0 / 0 nz |

Clean aggregate: **max_abs=0.0, nonzero=0** across layers `[0,45]`.

This is not a full verify-path pass. The required live follow-up remains: strict top-down ladder for spine+branches+final logits, plus Gate 2 regular-decode/no-bias no-regression.

### Live follow-up on server commit `42d49580` / ex2 code commit `ed0390df` — **VERIFY-PATH FAILED**

- Run dir: `output/fr13_ex2_live_ladder_20260608T021853Z`
- Strict tree config: `TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset by `scripts/fr13_launch_forked_fa2_tree_server.sh`, `FR10_ENABLE_TREE_GDN=1`, B=1 eager, `GPU_UTIL=0.4`, `MAX_MODEL_LEN=65536` for diagnostics.
- Matched native config: `FLASH_ATTN`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`, 5-token spine, B=1 eager.
- Correct row map for the current scheduler: tree spine rows `[0,1,2,4,6,8]` -> native rows `[0,1,2,3,4,5]`; branch rows `[3,5,7,9]`.

Top-down spine ladder artifact: `output/fr13_ex2_live_ladder_20260608T021853Z/gateA_spine_ladder.json`.

| stage | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| layer 3 full_attention hidden | 0.0 |
| layer 7 full_attention hidden | 0.0 |
| **layer 8 linear_attention hidden** | **0.00390625** |
| final_norm_hidden | 1.65625 |
| final logits | 1.90625 |

First nonzero: **layer 8 `linear_attention`**, hidden `0.00390625`; the drift compounds through later GDN/full-attn layers. This is **not a pass**.

Full-attn/branch oracle artifact: `output/fr13_ex2_live_ladder_20260608T021853Z/gateA_full_attn_tree_path_table.json`.

- The full-attn path is not the primary source: layer 3 full-attn stages are 0.0, and the large later full-attn drift enters as `input_hidden`.
- Branch FA2-on-path oracle from tree op captures: `tree_vs_fa2_branch=0.0` for 15/16 full-attn layers, with one small residual at layer 55 (`0.00048828125`), within the accepted FA2 floor. The verify failure is the compounding GDN spine drift, not the forked FA2 tree-bias call.

Gate 2 kernel-level no-bias check artifact: `output/fr13_ex2_live_ladder_20260608T021853Z/gate2/no_bias_compare.json`.

| case | stock-vs-fork no-bias max_abs | nonzero |
| --- | ---: | ---: |
| fp16 | 0.0 | 0 |
| bf16 | 0.0 | 0 |

Gate 2 full regular-decode model ladder was **not rerun** in this failed verify-path turn; do not treat this entry as a full two-gate pass.

### Follow-up spread replay after L8 live failure — **VERIFY-PATH STILL FAILED / ROOT CAUSE REDIRECTED**

- Bound diagnostic commit: this commit (`FR13: capture spine conv spread replay`; final hash in git log/push output).
- Run dir: `output/fr13_conv_spread_20260608T025907Z`.
- Strict config: spine-only `TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset, B=1 eager, `GPU_UTIL=0.4`; matched native `FLASH_ATTN`/`naive_mtp`.
- Capture layers: GDN `0,4,8,12,24,36,44`.
- Replay artifact: `output/fr13_conv_spread_20260608T025907Z/conv_spread_ex2_replay.json`.

Offline PTX-style conv replay:

| GDN layer | clean `pre_conv` | captured tree conv vs native | PTX replay vs native |
| ---: | --- | ---: | ---: |
| 0 | yes | 0.0 | 0.0 |
| 4 | yes | 0.0 | 0.0 |
| 8 | yes | 0.0 | 0.0 |
| 12 | no | 0.012451171875 | 0.012451171875 |
| 24 | no | 0.046875 | 0.046875 |
| 36 | no | 0.06640625 | 0.06640625 |
| 44 | no | 0.125 | 0.125 |

Sub-op localization on the same run:

| layer | first diverging stage | key result |
| ---: | --- | --- |
| 4 | none | all captured stages 0.0 |
| 8 | `h0_state_in` | `input_hidden=0.0`, `pre_conv=0.0`, `conv1d_out=0.0`, `h0_state_in=0.0007215589284896851`, `o_proj_out=0.003906` |
| 12 | `input_hidden` | already inherited drift: `input_hidden=0.09375` |

Conclusion for this gate entry: the L8 live ladder failure is **not** explained by a conv/SILU mismatch on this capture; the clean L8 conv output is 0.0 vs native. Deeper spread layers are contaminated and cannot be used for conv tuning. The next target is the recurrent state content/write path feeding L8 (`h0_state_in` bank row 1). Gate A remains **not passed**; Gate 2 full regular-decode model ladder was not rerun in this failed verify-path turn.
