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
