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

---

## Commit `c776581b` (executed from `FR13_SPINE_ARGMAX_LOSSLESS.md` @ `16660de9`) - B=4 CUDA-graph e2e binding

### Scope
- User direction: stop the prefill/GDN literal-0 grind; accepted spine gate is **argmax-lossless** and already met 6/6 at `16660de9`; execute Next-action step 2 e2e.
- Run dir: `output/fr13_argmax_e2e_20260608T055851Z`.
- One GPU; no Docker `--rm`; tree and native arms run sequentially. Host recovery/sync/drop-caches was run between arms; final `recover_host_memory()` succeeded with swap back to 0.

### Launch config - pinned and aligned
- Saved E5 reference: `output/fr10_native_mtp5_same8_20260604T210257Z`; 8 prompts x 4 samples, `batch_size=4`, `max_tokens=64`, temperature `0.6`, top_p `0.95`, `naive_mtp`, 5 draft tokens/event, `FLASH_ATTN`, Triton GDN prefill, CUDA graph (`enforce_eager=False`). Saved summary: accept/event `3.076171875`, accept/token `0.615234375`, decode TPS `17.987313578432634`.
- Fresh native arm: `output/fr13_argmax_e2e_20260608T055851Z/native_mtp5`; same shape/config as E5 (`FLASH_ATTN`, `naive_mtp`, 5 speculative tokens, B=4, CUDA graph). Result: accept/event `3.2132796780684103`, accept/token `0.6426559356136821`, decode TPS `15.647809832189031`, returned tokens `2048`.
- Tree arm: forked FA2 `.so` sha256 `97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1`, `TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR13_TREE_ATTN_EXP2_SOFTMAX=1`, 9-node tree, `MAX_NUM_SEQS=4`, `GPU_UTIL=0.86` (0.88 failed vLLM free-memory guard), CUDA graph (`enforce_eager=False`). Result: accept/event `1.1133786848072562`, accept/token `0.1237087427563618`, decode TPS `2.6654573600104485`, returned tokens `1855`.

### CUDA graph and hook state
- Tree FULL capture confirmed in `docker_full_gpu086.log`: `Profiling CUDA graph memory: PIECEWISE=8 (largest=80), FULL=4 (largest=40)` and `Capturing CUDA graphs (decode, FULL)` completed before startup.
- Native FULL capture confirmed in `native_mtp5/logs/server_pre_probe.log`: `Profiling CUDA graph memory: PIECEWISE=7 (largest=48), FULL=4 (largest=24)` and `Capturing CUDA graphs (decode, FULL)` completed before startup.
- Capture-path hooks were unset in the runtime env, but both launches emitted FR12 capture-warning spam during CUDA graph capture (`scan`, `pre-conv`, `conv`, `native h0`). Treat this as a recorded hooks-off caveat, not as a hidden diagnostic capture run.

### E2E gate - **FAIL**
- Artifact: `output/fr13_argmax_e2e_20260608T055851Z/e2e_compare_tree_vs_native_fresh.json`.
- Accept/event gate: tree `1.1133786848072562` < saved E5 `3.076171875` and < fresh native `3.2132796780684103`.
- Bag-TV gate: saved E5 artifact has no token records, so bag-TV was computed against the fresh aligned native arm. Full-token bag-TV = `0.5017672885781671` (threshold `0.059`); first-token TV = `0.0`.
- Paired records: 32/32 paired, exact sequence matches `0`, prefix LCP mean `4.375`, median `2.5`, max `12`.
- Tree engagement was non-vacuous: `tree_engagement.engaged=true`, metadata rows all OK, draft-count rows include 9-node tree events.

### Gate 2 - no-bias fork pristine check - **PASS**
- Artifact: `output/fr13_argmax_e2e_20260608T055851Z/gate2_no_bias/no_bias_compare.json`.
- Stock vs fork, no tree bias: fp16 `torch_equal=true`, `max_abs=0.0`, `nonzero=0`; bf16 `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.

### Verdict
- The accepted spine argmax-lossless gate from `16660de9` remains the gate definition; this run did not pursue literal GDN 0.0.
- The requested B=4 CUDA-graph e2e **does not pass**: the forked-FA2 tree arm loses the accept/event superset gate and the distributional bag-TV gate versus aligned native/E5.

---

## Takeover run `codex_fr15` - prefill-native root grind and L12 wall (2026-06-08T07:35:45Z)

### Scope
- User direction after failed e2e: apply `patches/fr13_fa2_prefill_native.patch`, run `FR13_FA2_PREFILL_NATIVE=1`, stop grinding GDN to literal 0 only after top-down Gate A reaches drift 0, and defer clean B=4 e2e until Gate A is clean.
- Run dir: `output/fr13_prefill_grind_20260608T063838Z`.
- One GPU; no Docker `--rm`; host `recover_host_memory()`, `sync`, and `drop_caches` used between server arms.

### Code under test
- `scripts/fr13_patch_fa2_tree_bias.py`: native FA2 prefill path is installed behind `FR13_FA2_PREFILL_NATIVE=1`.
- `scripts/fr13_launch_forked_fa2_tree_server.sh`: passes `FR13_FA2_PREFILL_NATIVE` into Docker and fails loud if patched container `tree_attn.py` lacks the prefill-native anchor.
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`: FR12 diagnostic clones / CPU `.tolist()` paths are env-gated default-off, so clean e2e can run with `FR10_METRICS=0` without unconditional FR12 clone overhead.
- Added default-off scan-payload filters: `FR10_TREE_GDN_CAPTURE_PAYLOAD_LAYER_PREFIX` and `FR10_TREE_GDN_CAPTURE_PAYLOAD_NUM_TOKENS`.

Container patch anchors:
- `anchor_ok ... tree_attn.py contains FR13_FA2_PREFILL_NATIVE`.
- `capture_filter_anchor_ok` for the installed `gdn_linear_attn.py`.

### Gate A top-down ladder - **FAIL**
- Tree arm: `TREE_ATTN`, forked FA2, `FR13_FA2_TREE_BIAS=1`, `FR13_FA2_PREFILL_NATIVE=1`, B=1 eager, 9-node tree.
- Native arm: `FLASH_ATTN`, `naive_mtp`, 5 speculative tokens, B=1 eager.
- Row map: tree spine rows `[0,1,2,4,6,8]` -> native rows `[0,1,2,3,4,5]`.
- Artifact: `output/fr13_prefill_grind_20260608T063838Z/gateA_spine_ladder.json`.

| stage | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| layers 0-11 | 0.0 |
| **layer 12 linear_attention hidden** | **0.00634765625** |
| final_norm_hidden | 0.59375 |
| final logits | 0.5234375 |

Prefill-native fixed the previous root/prefill front: model input and layers through full-attn layer 11 are byte-exact. Gate A still fails, now first at GDN layer 12.

### L12 GDN sub-op localization
- Decode-filtered captures:
  - Tree: `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_decode_subop/tree/logs/gdn_l12_decode_subop.call0.pt`, `num_tokens=10`.
  - Native: `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_decode_subop/native/logs/gdn_l12_decode_subop.call0.pt`, `num_tokens=6`.
- Mapped diff artifact: `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_decode_subop/gdn_l12_decode_subop_mapped_diff.json`.

| stage | mapped max_abs | note |
| --- | ---: | --- |
| input_hidden | 0.00390625 | row `1->1` only |
| pre_conv | 0.0 | clean |
| conv1d_out | 0.0 | clean |
| h0_state_in | 0.0 | same row/col `[1]/[0]` |
| gdn_scan_out | 0.00000095367431640625 | downstream tiny delta |
| gate_z | 0.0 | clean |
| gate_out | 0.000244140625 | downstream |
| o_proj_out | 0.0009765625 | downstream |

Fresh same-input scan replay:
- Payload: `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_scan_payload/logs/fr10_tree_gdn_scan_l12.pt`, layer `language_model.model.layers.12.linear_attn`, `n_actual=10`.
- Artifacts:
  - `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_scan_payload/scan_probe_default.json`
  - `output/fr13_prefill_grind_20260608T063838Z/gdn_l12_scan_payload/scan_probe_bf16.json`
- Result: same-input tree-kernel vs native FLA **scan output is 0.0**; state differs only `1.49e-08`. The `fla_bf16_boundaries` flag remains a no-op in current kernel code.

### Current wall
This is not a clean Gate A pass and no clean B=4 CUDA-graph e2e was run. The next root is no longer the prefill full-attn path and not an isolated same-input scan arithmetic mismatch. The remaining failure enters the L12 GDN block as a tree-vs-native row-1 `input_hidden` delta while conv, h0, gate_z, and same-input scan output are clean. The next step is to pin why that L12 per-row input differs despite the layer-hidden ladder showing layers 0-11 as 0.0 under the earlier capture.

## Takeover run `codex_fr15` - L12 offline replay and native-path state wall (2026-06-08T08:19:00Z)

### Scope
- User direction: stop live FR12 sub-op capture; localize L12 via offline replay, max 2-3 boots, do not run e2e before Gate A drift-0.
- Run dir: `output/fr13_l12_offline_replay_20260608T075019Z`.
- Boots used: one tree capture boot for L12 scan + source-native handoff; one patched tree ladder boot; one native ladder boot. No B=4 e2e was run.

### Offline replay result
- Fresh payloads:
  - Tree scan: `output/fr13_l12_offline_replay_20260608T075019Z/tree/logs/fr10_tree_gdn_scan_l12.pt`.
  - Source-native handoff: `output/fr13_l12_offline_replay_20260608T075019Z/tree/logs/fr10_src_native_handoff_l12.pt`.
- Added `scripts/fr13_l12_handoff_replay.py`.
- Baseline replay confirmed a real native-path state mismatch without live sub-op hooks:
  - serving tree replay vs serving state: `0.0`.
  - serving accepted state vs next-read SSM state: `0.0`.
  - native FLA on accepted GDN path `[1,2,4,6,8]` vs serving/next-read state: `0.024476230144500732`.
  - conv handoff remained exact: `0.0`.
- Added a scoped handoff-capture layer-prefix filter in `scripts/fr10_phase4_patch_vllm_tree_gdn.py`; previous `FR10_TREE_GDN_COMMIT_HANDOFF_LAYER_PREFIX` launcher env was not enforced for source-native payloads.

### Patch tested
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` now stores non-root recurrent states on the accepted/native GDN path (`1..i` under the visible mask) while leaving verify outputs on the full visible ancestry.
- Offline patched replay:
  - native accepted-path final state vs patched tree replay state: `1.862645149230957e-09` (down from `0.024476230144500732`).
  - verify output remained mismatched for that native-path comparison (`0.0003662109375`) because the patch only changed stored state, not current verify output.

### Gate A top-down ladder after patch - **FAIL**
- Artifact: `output/fr13_l12_offline_replay_20260608T075019Z/gateA_spine_ladder_after_patch.json`.
- Tree arm: forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, patched state-store kernel, B=1 eager.
- Native arm: `FLASH_ATTN`, `naive_mtp`, B=1 eager.

| stage | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| layers 0-11 | 0.0 |
| **layer 12 linear_attention hidden** | **0.00634765625** |
| final_norm_hidden | 0.59375 |
| final logits | 0.5234375 |

Per-row L12 hidden max_abs: row `0->0` = `0.0`; row `1->1` = `0.00634765625`; row `2->2` = `0.001953125`; row `4->3` = `0.0010986328125`; row `6->4` = `0.001953125`; row `8->5` = `0.002685546875`.

### Current wall
The offline replay localized and fixed a real accepted-path recurrent-state store mismatch, but Gate A did not move: the remaining first nonzero is still the current L12 verify output, not only the stored state read by the next event. L0-L11 did not regress. With the requested boot budget exhausted, e2e remains blocked until the current-output L12 native-path/op-order gap is fixed and the strict ladder reaches drift-0.

## Takeover run `codex_fr15` - WY one-pass tree-GDN build start (2026-06-08T08:43:00Z)

### Scope
- User direction: stop replay-kernel L12 grind and build the no-copy WY one-pass GDN tree kernel from `FR13_WY_KERNEL_BUILD.md`.
- Code path: added flag-gated `FR10_TREE_GDN_WY=1`; default remains replay fallback.

### Offline WY sub-gate - **PASS**
- Artifact: `output/fr13_l12_offline_replay_20260608T075019Z/wy_l12_offline_probe_ieee.json`.
- Payload: `output/fr13_l12_offline_replay_20260608T075019Z/tree/logs/fr10_tree_gdn_scan_l12.pt`.
- Kernel: revived native-basis WY/UT form from `_tree_gdn_gqa_kernel`, with raw-gating and h0-bank serving support. Important alignment: `tl.dot(..., input_precision="ieee")`; the initial TF32 dot missed (`out=1.52587890625e-05`, `state=0.00011709332466125488`).

| comparison | max_abs |
| --- | ---: |
| WY L12 spine output vs native FLA | 0.000000003725290298461914 |
| WY L12 spine state vs native FLA | 0.000000059604644775390625 |

This clears the boot-free L12 scan arithmetic/op-order sub-gate and is ready for live `FR10_TREE_GDN_WY=1` ladder validation. No e2e has been run.

## Commit `f614979e` (FR13 WY handoff) - live WY ladder attempt (codex_fr16, 2026-06-08T16:29Z)

### Scope
- User direction: execute `FR13_WY_KERNEL_BUILD.md` from the WY scaffold/handoff state, with `FR10_TREE_GDN_WY=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset, one GPU, host recovery between arms, and no e2e until strict Gate A clears.
- Code under test: no code changes after `c0448bd7`/`f614979e`; this entry binds the first live WY validation attempt.

### WY boot + CUDA graph capture - **CONFIRMED**
- Run dir: `output/fr13_wy_live_ladder_20260608T162920Z`.
- Config: `TREE_ATTN`, forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_TREE_GDN_WY=1`, `FR10_METRICS=1`, `MAX_NUM_SEQS=1`, `GPU_UTIL=0.86`, fallback unset by the launch entrypoint.
- FULL capture evidence from container logs:
  - `Profiling CUDA graph memory: PIECEWISE=1 (largest=10), FULL=1 (largest=10)`.
  - `Capturing CUDA graphs (decode, FULL)` completed.
  - `Graph capturing finished in 6 secs, took 0.33 GiB`.
- Non-vacuous tree request evidence: `tree_path_lcp.jsonl`, `independent_winner_trace.jsonl`, and `per_req_spec_trace.jsonl` emitted rows for a 9-draft / 10-node tree.
- WY-compute evidence: container env had `FR10_TREE_GDN_WY=1`; the GDN verifier branch was active; signal-dumped counters after a tree request included three `n10_pad16`, `has_sibling=true` rows with `count=192` each and path0 nodes `[0,1,2,4,6,8]`. Artifact: `output/fr13_wy_gateA_20260608T163915Z/tree/logs/fr10_tree_gdn_counters.after_request.json`.

### Gate A top-down spine ladder - **FAIL**
- Run dir: `output/fr13_wy_gateA_20260608T163915Z`.
- Tree arm: `TREE_ATTN`, forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_TREE_GDN_WY=1`, `FR10_METRICS=1`, B=1 eager diagnostic capture, rows `0..9`.
- Native arm: `FLASH_ATTN`, `naive_mtp`, 5-token linear MTP, B=1 eager diagnostic capture, rows `0..5`.
- Row map: tree spine `[0,1,2,4,6,8]` -> native `[0,1,2,3,4,5]`.
- Artifact: `output/fr13_wy_gateA_20260608T163915Z/gateA_spine_ladder.json`.

| stage | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| layer 0 linear_attention hidden | 0.0 |
| **layer 1 linear_attention hidden** | **0.0001220703125** |
| layer 2 linear_attention hidden | 0.015625 |
| layer 3 full_attention hidden | 0.005859375 |
| final_norm_hidden | 5.294921875 |
| final logits | 3.3203125 |

This is not a strict Gate-A pass. The first nonzero moved earlier than the previous replay L12 wall: live WY now fails at layer 1 under the full serving ladder. Branch rows were captured in the tree arm, but the gate was stopped at the first strict spine failure; no branch pass is claimed.

### Micro-check
- Short CUDA-image micro-check compared `use_wy=True` against the existing replay kernel on a random GQA `n10_pad16` tree with raw gating and in-kernel q/k l2norm.
- Result: WY output matches replay to `9.313225746154785e-10` max_abs (`2` nonzero elements), but materialized per-node state differs (`0.4837808310985565`) because the kernels store different state surfaces. This supports that the live failure is not a silent fallback, but it does not clear Gate A.

### Gate 2 / e2e
- Gate 2 was not rerun in this failed Gate-A turn; no verifier-only no-bias code changed from the previously bound PASS entries.
- Clean B=4 e2e was intentionally not run because Gate A is blocked at layer 1.

## Commit `5d90a70d` follow-up - L1 WY offline localization (codex_fr16, 2026-06-08T17:12Z)

### Scope
- User direction: stop double-booting servers for WY alignment; reuse the fixed native ladder reference and captured L1 scan payload; iterate `_tree_gdn_wy_kernel` offline before any further live ladder.
- Fixed references:
  - Native top-down reference: `output/fr13_wy_gateA_20260608T163915Z/native/logs/native_layer_hidden.pt` and `native_final_logits.pt`.
  - L1 WY scan payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.

### Offline L1 WY scan replay - **WY SCAN IS NOT THE 1.2e-4 L1 WALL**
- Artifact: `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_offline_probe.json`.
- Runner: lightweight CUDA image only (`--entrypoint bash`), no model/server boot.
- Kernel under test: `_tree_gdn_wy_kernel` via `launch_tree_gdn_prepared(use_wy=True)`.
- Native-on-path oracle: vLLM `fused_sigmoid_gating_delta_rule_update`, per-node path, raw `a/b`, in-kernel q/k l2norm, captured h0, captured L1 tree topology.

| comparison | max_abs |
| --- | ---: |
| WY replay vs captured serving out | 0.0 |
| WY replay vs captured serving state | 0.0 |
| WY replay vs native-on-path serial out | 0.0000000009313225746154785 |
| WY replay vs native-on-path serial state | 0.00000008940696716308594 |
| captured serving out vs native-on-path serial out | 0.0000000009313225746154785 |
| captured serving state vs native-on-path serial state | 0.00000008940696716308594 |

Per-node WY-vs-serial output max_abs: `[0.0, 0.0, 1.1641532182693481e-10, 0.0, 0.0, 0.0, 9.313225746154785e-10, 0.0, 5.820766091346741e-11, 4.656612873077393e-10]`.

### Interpretation / wall
- The live full-model ladder wall remains: L1 `linear_attention` hidden `0.0001220703125` -> final logits `3.3203125` (artifact `output/fr13_wy_gateA_20260608T163915Z/gateA_spine_ladder.json`).
- The captured L1 WY scan itself is already at the fp32 floor against native-on-path serial and is bit-exact to the serving capture. Changing WY `kk` dot precision, basis, solve order, raw-g/softplus, l2norm eps, or decay exp has no justified target from this payload: the observed `1.2e-4` full-block hidden drift is not present at the scan output/state boundary.
- Therefore no in-kernel WY patch was made in this turn. A further live ladder would violate the user's offline-first condition without a kernel change, and a kernel change would be blind/random relative to the captured L1 WY evidence.
- Next required evidence, if continuing, is a fixed-target L1 GDN sub-op capture (tree and native) for `input_hidden -> pre_conv -> conv1d_out -> h0_state_in -> gdn_scan_out -> gate_z -> gate_out -> o_proj_out`; the currently available L1 scan payload cannot localize gate/o_proj or layer-output drift.

## Commit `1f0c7237` (FR13 align WY FLA bf16 boundaries) - offline L1 WY vs live-FLA seam check (codex_fr16, 2026-06-08)

### Scope
- User supplied `FR13_WY_SEAM_FIXES.md`: the prior `9.3e-10` L1 WY scan result was against the fp32 oracle, while the live ladder compares against native FLA's bf16 boundary behavior.
- Code change: `_tree_gdn_wy_kernel` now accepts the existing `FLA_BF16_BOUNDARIES` constexpr and, when enabled, bf16-rounds normalized q/k after in-kernel l2norm and bf16-rounds each triangular solve coefficient. The fp32 oracle path remains available with the flag off.
- Probe-only script change: `scripts/fr12_spine_scan_rounding_probe.py` and `scripts/fr12_scan_batch_invariance_probe.py` gained `--use-wy` so the boot-free probes exercise `_tree_gdn_wy_kernel`, not the replay fallback.

### Offline L1 scan vs live FLA - **within one bf16 ULP; not literal 0.0**
- Runner: CUDA entrypoint container only, no vLLM server boot and no native reboot.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_spine_scan_live_fla_bf16_final.json`.
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_batch_scan_live_fla_bf16_final.json`.
- Config: `--use-wy --fla-bf16-boundaries --max-depth 6`; native reference is `vllm.fused_sigmoid_gating_delta_rule_update`.

| comparison | max_abs |
| --- | ---: |
| isolated spine WY scan output vs live FLA | 0.0001220703125 |
| isolated spine WY scan state vs live FLA | 0.0008958578109741211 |
| original-full spine WY scan output vs live FLA | 0.0001220703125 |
| original-full spine WY scan state vs live FLA | 0.0008958280086517334 |
| spine-only output vs original-full spine output | 0.000000003725290298461914 |
| sibling-reordered full output vs original-full spine output | 0.0 |

Interpretation: the patched WY scan is now measured against the live FLA target and sits at the expected one-bf16-ULP output floor, while remaining row-order/context invariant. Extra experimental boundaries (`KKt` bf16-input fold and `tv_i/k_j` state-update rounding) did not reduce output drift; `tv_i/k_j` worsened state drift, so they were not kept.

### Next gate
- Run one live WY Gate-A ladder with `FR10_TREE_GDN_WY=1`, fallback unset, and `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES=1` to see whether the one-ULP L1 scan floor no longer amplifies into final-logit drift.
- Gate 2 and clean B=4 e2e remain blocked until Gate A passes.

## Commit `43434f01` (FR13 apply WY cascade-map bf16 taps) - six-tap implementation and readout wall (codex_fr16, 2026-06-08)

### Scope
- User supplied `FR13_WY_CASCADE_MAP.md`, which supersedes the earlier two-tap seam fix.
- Code change: `_tree_gdn_wy_kernel` now gates the complete documented bf16 boundary set behind `FLA_BF16_BOUNDARIES`:
  - #1 normalized q/k bf16 store after l2norm: kept.
  - #2 solve-T rounding relocated from per-iteration `coeff_j` to final `solved_v` / `solved_k` stores.
  - #3 KKt gram uses bf16 dot inputs with beta pre-folded into k, dropping the duplicate post-dot beta multiply.
  - #4 initial WY operands `beta*v` and `beta*k*exp(cum_g)` round to bf16 before substitution.
  - #5 transformed value delta `tv_i` rounds to bf16.
  - #6 readout tap rounds `q_i` and the per-ancestor intra outer-product contribution to bf16. No readout reduction-order rewrite was made.

### Boot-free L1 smoke vs live FLA - **USER-DECISION WALL, not a pass/fail**
- Runner: CUDA entrypoint container only; no vLLM server boot and no native reboot.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_spine_scan_live_fla_6tap.json`.
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_batch_scan_live_fla_6tap.json`.
- Config: `--use-wy --fla-bf16-boundaries --max-depth 6`; native reference is `vllm.fused_sigmoid_gating_delta_rule_update`.

| comparison | max_abs |
| --- | ---: |
| isolated spine WY scan output vs live FLA | 0.000244140625 |
| isolated spine WY scan state vs live FLA | 0.0018433183431625366 |
| original-full spine WY scan output vs live FLA | 0.000244140625 |
| original-full spine WY scan state vs live FLA | 0.0018433183431625366 |
| spine-only output vs original-full spine output | 0.000000010593794286251068 |
| sibling-reordered full output vs original-full spine output | 0.0 |

Interpretation: the six taps compile and remain row-order/context invariant, but the L1 scan output is above the cascade-map target floor (~6e-5). The next plausible lever is the #6 readout reduction-order match described in `FR13_WY_CASCADE_MAP.md` as a user-decision item, so no live ladder pass/fail is claimed and no readout restructure was attempted.

### Blocked next action
- Ask user whether to authorize the #6 readout reduction-order rewrite or to run the requested live ladder anyway with the known above-floor L1 scan result.
- Gate 2 and clean B=4 e2e remain blocked until Gate A is resolved.

## Commit `1f11b9d7` (FR13 remove WY misapplied readout over-rounds) - tap red-team fix (codex_fr16, 2026-06-08)

### Scope
- User supplied `FR13_WY_TAP_REDTEAM.md`: the six-tap `2.44e-4` result was a mis-applied #6 tap, not proof of a reduction-order rewrite need.
- Code change: removed the two native-mismatched over-rounds while keeping #1-#5:
  - Deleted post-scale `q_i.to(bf16).to(f32)`; q is already bf16-rounded by #1 before `OUTPUT_SCALE`.
  - Deleted per-j `state_update_ij.to(bf16).to(f32)`; native accumulates outer products in fp32 after rounding `tv_i`.
- No #6 reduction-order rewrite was attempted; no live ladder was run.

### Offline L1 smoke vs live FLA - **restored pre-#6 floor, still not Gate A**
- Runner: CUDA entrypoint container only; no vLLM server boot and no native reboot.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_spine_scan_live_fla_redteam_fix.json`.
  - `output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_batch_scan_live_fla_redteam_fix.json`.
- Config: `--use-wy --fla-bf16-boundaries --max-depth 6`; native reference is `vllm.fused_sigmoid_gating_delta_rule_update`.

| comparison | max_abs |
| --- | ---: |
| isolated spine WY scan output vs live FLA | 0.0001220703125 |
| isolated spine WY scan state vs live FLA | 0.001657634973526001 |
| original-full spine WY scan output vs live FLA | 0.0001220703125 |
| original-full spine WY scan state vs live FLA | 0.001657634973526001 |
| spine-only output vs original-full spine output | 0.000000014901161193847656 |
| sibling-reordered full output vs original-full spine output | 0.0 |

Interpretation: deleting the two over-rounds restores the expected `1.221e-4` L1 spine output smoke and preserves row-order/context invariance. This still exceeds the gate floor; per user direction, hold for the remaining-seam workflow before any live ladder, #6 rewrite, Gate 2, or e2e.

## Commit `b709d488` (FR13 WY B=4 e2e preconditions) - Gate-2 hooks-off and clean-launch fix (codex_fr16, 2026-06-08)

### Scope
- User supplied `FR13_WY_RESIDUAL_CLOSURE.md` verdict B: stop tapping; measure WY B=4 e2e with `TREE_ATTN`, forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_TREE_GDN_WY=1`, fallback unset, splice off, `FR10_METRICS=0`, and CUDA graph FULL.
- Launcher fix before boot: `scripts/fr13_launch_forked_fa2_tree_server.sh` now passes `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES` into the container. Without this pass-through, the accepted live-FLA bf16 boundary configuration would be dropped at server launch.

### Gate-2 no-bias hooks-off compare
- Artifact dir: `output/fr13_wy_b4_e2e_20260608T183138Z/gate2`.
- Stock and forked FA2 were run in separate entrypoint containers; no model server boot.
- Compare artifact: `output/fr13_wy_b4_e2e_20260608T183138Z/gate2/no_bias_compare.json`.

| dtype | torch_equal | max_abs | nonzero |
| --- | --- | ---: | ---: |
| fp16 | true | 0.0 | 0 |
| bf16 | true | 0.0 | 0 |

Interpretation: Gate-2 hooks-off/no-bias regression remains byte-exact for the current forked FA2 `.so`; proceed to the clean WY B=4 server boot and timed e2e.

## Commits `ced25bd3` + `74dfb33d` (FR13 gate diagnostic captures for clean B=4 Gate-A) - prerequisite (codex_fr16, 2026-06-08)

### Scope
- User red-team caught a gate gap: the prior Gate-A ladder was B=1/eager and pre-final-B=4 conditions; the B=4 WY e2e symptom (`accept/event 1.1989`) must not be treated as a verdict until the final bf16-tapped WY build is ladder-validated at B=4.
- Code-gated diagnostic capture paths in `scripts/fr10_phase4_patch_vllm_tree_gdn.py`:
  - FR12 GDN h0/scan subkernel capture now constructs CPU metadata only when `FR12_SUBKERNEL_CAPTURE` is set.
  - A follow-up removed a Dynamo-hostile `torch.cuda.is_current_stream_capturing()` check from compiled model code; layer/final capture startup avoidance must be controlled by explicit skip/limit settings.
- Purpose: run the B=4 Gate-A ladder with CUDA graph FULL intact and without FR12 diagnostic clone/copy contamination during graph capture.

### Status
- The measured WY e2e artifact (`output/fr13_wy_b4_e2e_20260608T183138Z/tree/quick_tree_wy_b4_spp16_seed1313.json`, accept/event `1.1989480198019802`) is retained as a symptom only.
- Next action: clean B=4 Gate-A ladder on the final bf16-tapped WY build, including spine rows and branch-path argmax oracles, before assigning structural-vs-numerical cause.

## Commit `codex_fr17-offline-state-fix` - WY scan STATE replay to native floor (codex_fr17, 2026-06-08)

### Scope
- User direction: execute `FR13_FR17_HANDOFF.md`; grind the WY scan STATE write path offline before any live ladder.
- Code change: `_tree_gdn_wy_kernel` now splits the WY readout surface from the stored recurrent-state surface when `FLA_BF16_BOUNDARIES` is enabled. Output keeps the bf16-normalized/tapped WY path that was already at the one-bf16-ULP readout floor. The stored state uses a separate raw-normalized WY basis and raw fp32 triangular solve, matching native `fused_sigmoid_gating_delta_rule_update` state storage instead of carrying the readout bf16 taps into the recurrent state.
- No copy/splice/reroute/dense path was used; the WY kernel still computes the state.

### Offline fixed-payload replay - **STATE floor reached**
- Runner: named CUDA entrypoint containers, no vLLM server boot, no `--rm`.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_patch2_spine_state.json`
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_patch2_batch_state.json`

| comparison | out max_abs | state max_abs |
| --- | ---: | ---: |
| spine WY vs native FLA update | `0.0001220703125` | `0.00000008940696716308594` |
| original-full spine vs native FLA update | `0.0001220703125` | `0.00000008940696716308594` |
| reverse-sibling spine vs original-full spine | `0.0` | `0.0` |
| spine-only spine vs original-full spine | `0.000000014901161193847656` | `0.000000059604644775390625` |

Baseline before this patch on the same payload was state `0.001657634973526001`; the fixed state is now below the requested bf16 floor and at fp32 replay noise. Output remains at the known one-bf16-ULP readout floor. This clears the offline-first state replay gate and is ready for one live ladder with `FR10_TREE_GDN_WY=1` and `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES=1`.

## Commit `codex_fr17-bf16-bank-s1-probe` - post-round state-bank seam check (codex_fr17, 2026-06-08)

### Scope
- User direction: apply `FR13_WY_MULTISTEP_SEAMS.md`; do **not** touch `b_h0`, do **not** BF16-round `state_store_i` in-kernel, and proactively test S1 offline before another live ladder.
- Code change: the offline spine and batch replay probes now report per-element `state_bf16_bank` equality for `tree_state.to(bfloat16)` vs native `ht.to(bfloat16)`, not just fp32 `state_max_abs`.
- Code change: `_tree_gdn_wy_kernel` now computes the stored recurrent-state surface with native's direct per-token recurrent update order using the raw normalized state basis. The WY output/readout path is unchanged, and the stored state is still written fp32 for the caller's single cache-bank round.

### Offline BF16-bank replay - **S1 confirmed, reduced but not cleared**
- Runner: named CUDA entrypoint containers, no vLLM server boot, no `--rm`.
- Payload: `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Pre-change artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_bf16_bank_spine_state.json`
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_bf16_bank_batch_state.json`
- Best-current artifacts:
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_bf16_bank_spine_softplus.json`
  - `output/fr13_wy_l1_payload_20260608T170530Z/codex_fr17_bf16_bank_batch_best.json`

| check | fp32 state max_abs | BF16-bank mismatches |
| --- | ---: | ---: |
| pre-change spine WY vs native FLA | `0.00000008940696716308594` | `365 / 4718592` |
| native-order direct-load spine WY vs native FLA | `0.000000007450580596923828` | `8 / 4718592` |
| native-order original-full spine vs native FLA | `0.000000029802322387695312` | `58 / 4718592` |
| reverse-sibling full spine vs original-full spine | `0.0` | `0 / 4718592` |

Interpretation: S1 is real. The previous fp32 floor could round into different BF16 cache-bank buckets. Matching native's recurrent update order and direct token-local loads reduces the post-round bank mismatch sharply, but the offline BF16-bank gate is **not yet literal bit-exact**. Remaining first mismatch is at pre-round scale `1.33e-12`, enough to straddle a BF16 midpoint.

### S2/S3 wiring audit - no static off-by-one found
- `launch_tree_state_linear_remap` copies column `k` from `accepted_paths[b,k]` for `k < accepted_len`, and the next WY h0 read uses `accepted_len - 1`; this matches native's tail-column read contract.
- The committer writes accepted node paths and accepted lens into the same global device buffers that the remap and h0-read paths consume.
- Captured `tree_path_lcp.jsonl` rows show `superset_violation=false`; this is not a substitute for a future live remap-column assertion, but no static accepted-column lag was found in the code path.

Status: no live ladder/e2e verdict is claimed from this entry. The remaining S1 BF16-bank mismatch is the current wall before a meaningful full-model live validator.

## Commit `codex_fr17-output-tap-live-ladder` - prompt-pinned paired ladder + output split (codex_fr17, 2026-06-08)

### Scope
- User direction: execute `FR13_WY_LADDER_PROMPT_FIX.md`; stop reusing the `163915Z` native capture because its prompt is unrecoverable, do one fresh paired run with a saved deterministic request, then apply `FR13_WY_OUTPUT_TAP_PATCH.md` and re-run only the tree arm.
- Fixed request saved at `output/fr13_wy_paired_ladder_20260608T211749Z/request.json`:
  - endpoint body: `{"model":"qwen3.6-27b","prompt":"Explain hash tables.","max_tokens":16,"temperature":0,"vllm_xargs":{"fr10_decode_mode":"naive_mtp"}}`
  - tree copy changes only `fr10_decode_mode` to `tree_mtp`.
- Native arm: FLASH_ATTN, MTP-5, B=1 eager, layer/final captures pinned under `output/fr13_wy_paired_ladder_20260608T211749Z/native/`.
- Tree baseline arm: TREE_ATTN WY, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_TREE_GDN_WY=1`, `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES=1`, output split OFF, captures under `.../tree/`.
- Patch applied: `_tree_gdn_wy_kernel` now has gated `FLA_BF16_OUTPUT_SPLIT`; default is OFF. When ON, the readout separates carried-state/inter contribution from per-node intra contribution and bf16-rounds the scalar intra score before accumulating `out_intra_i`. Host flag is wired through `fla_bf16_output_split` and launcher env `FR13_WY_FLA_BF16_OUTPUT_SPLIT`.

### OFF-path smoke
- Attempted boot-free replay against `output/fr13_wy_l1_payload_20260608T170530Z/tree/logs/fr10_tree_gdn_scan_l1.pt`.
- Artifact: `output/fr13_wy_paired_ladder_20260608T211749Z/off_path_byte_identical_smoke.json`.
- Result was **not byte-identical**: output max_abs `0.000244140625`, state max_abs `0.00000008940696716308594`.
- Limitation: this is not a clean pre-output-tap reference because the payload predates the later state-store/order changes; no newer pre-tap WY serving payload exists in the paired ladder run. Compile and `git diff --check` passed.

### Fresh paired baseline, output split OFF
- Artifacts:
  - `output/fr13_wy_paired_ladder_20260608T211749Z/gateA_spine_ladder_baseline.json`
  - `output/fr13_wy_paired_ladder_20260608T211749Z/gateA_summary_baseline.json`

| check | max_abs | mean_abs | nonzero |
| --- | ---: | ---: | ---: |
| input spine | `0.0` | `0.0` | `0` |
| layer0 hidden spine | not summarized | not summarized | not summarized |
| final norm spine | `2.625` | `0.12346107512712479` | `30085` |
| final logits spine | `1.25` | `0.13377708196640015` | `1449401` |

Baseline reducer verdict: failed; first above-threshold hidden drift was layer-0 residual max_abs `0.0625`. Per-depth spine argmax still matched native for rows `[0,1,2,4,6,8]` vs native rows `[0,1,2,3,4,5]`.

### Output split ON live tree-only ladder
- Tree arm: same pinned native request/reference; `FR13_WY_FLA_BF16_OUTPUT_SPLIT=1`; no native reboot.
- Response artifact: `output/fr13_wy_paired_ladder_20260608T211749Z/tree_tap/response.json`; HTTP 200, prompt_tokens `5`, completion_tokens `16`.
- Capture artifacts:
  - `output/fr13_wy_paired_ladder_20260608T211749Z/tree_tap/logs/tree_layer_hidden.pt`
  - `output/fr13_wy_paired_ladder_20260608T211749Z/tree_tap/logs/tree_final_logits.pt`
  - `output/fr13_wy_paired_ladder_20260608T211749Z/gateA_spine_ladder_tap.json`
  - `output/fr13_wy_paired_ladder_20260608T211749Z/gateA_summary_tap.json`

| check | max_abs | mean_abs | nonzero |
| --- | ---: | ---: | ---: |
| input spine | `0.0` | `0.0` | `0` |
| layer0 hidden spine | `0.015625` | `0.00017748985555954278` | `27310` |
| layer0 residual spine | `0.0625` | `0.00009679019422037527` | `21944` |
| final norm spine | `2.1875` | `0.09377077966928482` | `29953` |
| final logits spine | `1.02734375` | `0.11143794655799866` | `1444953` |

Per-depth final-logit max_abs with output split ON:

| depth | tree row | native row | max_abs | argmax match | native margin |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 0 | 0 | `0.484375` | true | `1.125` |
| 1 | 1 | 1 | `0.78125` | true | `2.5` |
| 2 | 2 | 2 | `0.7392578125` | true | `2.25` |
| 3 | 4 | 3 | `0.5625` | true | `1.625` |
| 4 | 6 | 4 | `1.02734375` | true | `0.75` |
| 5 | 8 | 5 | `0.66015625` | true | `0.375` |

Branch proxy rows `[3,5,7,9]` matched logged self-target argmaxes `[332,332,198,32]`, but this remains only a tree self-target proxy. A true native-on-branch-path oracle was not captured in this paired run because the pinned native MTP-5 capture has spine rows only.

### Verdict
- Prompt pin is fixed: paired input spine max_abs is `0.0` with saved `request.json`.
- Output split improves aggregate final-logit drift (`1.25` -> `1.02734375`) but **does not close Gate A**. The first live drift remains layer-0 residual `0.0625`, and final logits remain far above the E5 floor.
- No e2e run was performed because the live ladder did not pass.

## Commit `fr13-seq-pivot-bind` - FR13 sequential rank-1 tree-scan pivot bound (codex_fr13, 2026-06-08T22:34:03Z)

### Scope
- User direction: read `FR13_SEQ_TREE_SCAN_TASK.md` and `FR13_WY_VS_SEQUENTIAL_VERDICT.md` at `HEAD 93ad85e4`, stand down the WY deliverable path, and execute the sequential rank-1 GDN tree-scan task.
- Verified `HEAD`: `93ad85e46604fe7ae2e152d7b2b6980ead13f088`.
- Decision bound from the task docs: WY is oracle-only because native verify dispatches to the sequential rank-1 recurrence; the deliverable is the default `use_wy=False` path in `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.
- Discipline bound: one GPU, recover host memory between live arms, no native splice/reroute/dense fallback, and commit + push this log at each implementation or gate step.

### Execution Plan
1. Audit live native `fused_sigmoid_gating.py` and the current sequential ancestor-replay kernel side by side.
2. Patch only the sequential tree-scan path to mirror native gate, l2norm, beta rounding, state decay, delta read, rank-1 write, and readout order; keep WY as an oracle path.
3. Preserve tree ancestry semantics without HBM per-node state replay: parent-derived working state stays in registers/SRAM and only the accepted path state is committed.
4. Run cheap local checks first, then the pinned paired ladder with native-on-path oracle: B=1 eager spine and true branch, Gate-2 pristine decode, B=4 captured ladder, e2e vs E5, and B=4 TPS vs native MTP-5 and WY.

### Status
- No code changes or gate claims in this entry.

## Commit `fr13-seq-register-checkpoint-kernel` - default sequential tree-scan rewritten around parent checkpoints (codex_fr13, 2026-06-08)

### Scope
- Code change: rewrote the default `use_wy=False` `_tree_gdn_kernel` in `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.
- The sequential path now launches one Triton program per `(value_head, value_channel)` and keeps an `N_PAD x DIM_K` fp32 checkpoint table in the program working set.
- Each node selects the latest strict ancestor checkpoint from `strict_mask`, applies exactly one native-style rank-1 recurrence update for that node, stores the same post-update state used for readout, and writes the node output/state surface.
- WY remains behind `FR10_TREE_GDN_WY=1` and unchanged; this entry does not splice or call native FLA as the deliverable kernel.

### Native Op-Order Alignment
- Per node: load `q`, `k`, `v`, `b`, derive raw gate from `a + dt_bias`, compute `sigmoid(b)`, l2-normalize `q/k` in-kernel with `+1e-6`, scale `q`, decay state, delta-read, beta-scale, rank-1 write, and read out from the updated state.
- Removed the prior second state replay that skipped token `0` for `i > 0`; stored state now matches the actual post-token state used for `out_i`.

### Local Checks
- `git diff --check`: passed.
- `python3 -m py_compile src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py scripts/fr10_phase4_patch_vllm_tree_gdn.py`: passed.
- `python3 -m pytest tests/test_fr10_gdn_tree_algebra.py -q`: `15 passed`.
- `python3 -m pytest tests/test_fr10_lossless_equivalence.py tests/test_fr10_tree_conv.py -q`: `15 passed, 1 skipped`.

### Status
- Host Python reports `torch.cuda.is_available() == False`, so no direct host Triton smoke is claimed here.
- Next gate is GPU/container validation of the sequential default path with the pinned paired ladder and native-on-path oracle.

## Commit `fr13-seq-b1-eager-gatea-fail-l2` - B=1 eager serving ladder on sequential default kernel (codex_fr13, 2026-06-08)

### Scope
- Tree arm run dir: `output/fr13_seq_paired_ladder_20260608T223903Z/tree/`.
- Request: pinned prompt copied from `output/fr13_wy_paired_ladder_20260608T211749Z/tree/request.json` (`Explain hash tables.`, `max_tokens=16`, `temperature=0`, `tree_mtp`).
- Config: `TREE_ATTN`, B=1 eager (`--enforce-eager`), `FR10_ENABLE_TREE_GDN=1`, `FR10_TREE_GDN_WY=0`, `FR13_FA2_PREFILL_NATIVE=1`, fallback unset by launcher, layer/final captures enabled.
- Runtime evidence: request returned HTTP `200`; response usage was prompt `5`, completion `16`; container env check showed `FR10_TREE_GDN_WY=0` and no `FR10_ALLOW_LINEAR_FALLBACK`.

### Spine Ladder vs Pinned Native
- Reducer: `scripts/fr13_ladder_table.py`.
- Artifact: `output/fr13_seq_paired_ladder_20260608T223903Z/gateA_spine_ladder_seq.json`.
- Rows: tree `[0,1,2,4,6,8]` vs native `[0,1,2,3,4,5]`, pinned native from `output/fr13_wy_paired_ladder_20260608T211749Z/native/`.

| check | max_abs | nonzero |
| --- | ---: | ---: |
| input hidden | `0.0` | `0` |
| layer 0 hidden / residual | `0.0 / 0.0` | `0 / 0` |
| layer 1 hidden / residual | `0.0 / 0.0` | `0 / 0` |
| first nonzero: layer 2 hidden | `0.015625` | `3345` |
| layer 2 residual | `0.0009765625` | not summarized |
| final norm hidden | `1.25` | `29704` |
| final logits | `0.7265625` | `1434157` |

### Verdict
- Gate A **does not pass**. The new default sequential kernel compiles/runs and clears layers 0-1 on the spine, but the first visible drift is now layer 2.
- No branch/native-on-branch or B=4 claim is made from this entry.
- Next action: recover host memory, then capture layer-2 GDN subops for tree and native to localize the first nonzero.
