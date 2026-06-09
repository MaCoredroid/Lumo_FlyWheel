# FR-13 ladder log — per-commit strict drift gates (verify-path + regular-decode)

Each entry binds a commit to its STRICT top-down ladder results. Two gates per commit:
1. **VERIFY-PATH** — tree verify (tree-bias) byte-exact to native-FA2-on-path oracle: input + every full_attn layer + final logits = 0.0, spine AND branch.
2. **REGULAR-DECODE (verifier-only)** — plain decode (no tree/spec/bias) with the forked FA2 == ORIGINAL pristine stock FA2: 0.0 at every layer.

A `FR10_ALLOW_LINEAR_FALLBACK` run is DIAGNOSTIC ONLY (GDN may go linear) → NEVER bound here. Both gates must be STRICT + 0.0 to record a PASS. (codex_fr14 appends; this first entry recorded by the monitor from verified artifacts.)

---

## Commit `e9d2a701` + measurement-tool worktree - guarded prompt-0 argmax localization (codex_fr13, 2026-06-09T05:26Z)

- Pairing red-team: `scripts/fr10_quick_decode_tps_probe.py` now records `prompt_token_ids`; `scripts/fr13_argmax_lcp_localize.py` fails closed unless tree/native prompt tokens are byte-identical.
- Guarded run: `output/fr13_argmax_lcp_prompt0_20260609T052640Z`, B=1 eager, prompt `0`, temp `0`, no warmup, tree `TREE_ATTN`/`tree_mtp` vs native `FLASH_ATTN`/`naive_mtp`.
- Prompt identity: byte-identical, `15` prompt tokens on both arms. The earlier `lcp=0` token-0 divergence is invalidated as a pairing artifact.
- Served-token alignment: exact through `8` tokens; first mismatch at position `8` (`tree=727`, `native=1005`), matching the prior e2e miss shape rather than token-0 divergence.
- Authoritative target-argmax flip: stream position `7` / completion position `8`, tree call `2` row `0` token `727` vs native call `2` row `0` token `1005`.
- Flip-row layer localization: `input_hidden max_abs=0.0`; first nonzero layer `0` `linear_attention`, `max_abs=0.0625`; final norm max_abs `9.03125`.
- Interpretation: first guarded flip is not full-attention. It is already in the layer-0 GDN/linear-attention path on verify event 2 after an accepted path. Next root is that event's GDN recurrent-state/current-event handling, not prompt pairing and not the rejected lcp=0 artifact.
- `num_warps=8` red-team gate: `output/fr13_numwarps8_gdn_scan_gate_20260609T043954Z.json` gives output max_abs `0.0` at `N_PAD=1` and deployed `N_PAD=16`; no revert indicated from this gate.

---

## Commit `a586ac84` - sequential e2e prefill-native binding (codex_fr13, 2026-06-09T04:15Z)

- Code changes: `FR13_FA2_PREFILL_NATIVE` defaults on for the forked FA2 TREE_ATTN launcher, `FR10_METRICS` defaults off, and the GDN tree kernel launch pins `num_warps=8`.
- Static checks: `python3 -m py_compile src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py scripts/fr13_patch_fa2_tree_bias.py scripts/fr12_deliverable_swe4_probe.py scripts/fr13_compare_deliverable.py`; `bash -n scripts/fr13_launch_forked_fa2_tree_server.sh`; `pytest -q tests/test_fr10_phase4_sampled_committer_wiring.py`.
- Live run: `output/fr13_seq_e2e_prefill_native_20260609T041654Z`, sequential 9-node TREE_ATTN, forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_METRICS=0`, B=4 CUDA graph. FULL decode capture completed (`PIECEWISE=8`, `FULL=4`).
- Tree engagement: `engaged=true`, `gpu_tree_metadata_ok_rows=248/248`, `tree_accept_rows=795`.
- E2E result: tree accept/event `1.6583442838370566` versus saved E5 bar `3.076171875`; tree warm decode TPS `5.746200172868099` versus saved E5 `17.987313578432634`.
- Bag-TV: saved E5 same8 JSON has no token records, so bag-TV was computed against the fresh paired native artifact with records. Result `0.42584828811470293`, above the `0.0593` floor; first paired mismatch at prompt `0`, sample `0`, position `8`.
- Binding caveat: this e2e path emitted canonical stochastic committer rows (`tree_sample_accept`), not authoritative `tree_path_lcp_max` argmax-ladder rows. It binds the distribution/acceptance miss, not the exact per-depth argmax-flip layer. Do not pivot back to GDN/literal-zero from this artifact; next localization remains the full-attention/tree-verify argmax front.

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
- User direction: stop replay-kernel L12 grind and build the no-copy WY one-pass GDN tree kernel from `docs/archive/wy/FR13_WY_KERNEL_BUILD.md`.
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
- User direction: execute `docs/archive/wy/FR13_WY_KERNEL_BUILD.md` from the WY scaffold/handoff state, with `FR10_TREE_GDN_WY=1`, `FR10_ALLOW_LINEAR_FALLBACK` unset, one GPU, host recovery between arms, and no e2e until strict Gate A clears.
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
- User supplied `docs/archive/wy/FR13_WY_SEAM_FIXES.md`: the prior `9.3e-10` L1 WY scan result was against the fp32 oracle, while the live ladder compares against native FLA's bf16 boundary behavior.
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
- User supplied `docs/archive/wy/FR13_WY_CASCADE_MAP.md`, which supersedes the earlier two-tap seam fix.
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

Interpretation: the six taps compile and remain row-order/context invariant, but the L1 scan output is above the cascade-map target floor (~6e-5). The next plausible lever is the #6 readout reduction-order match described in `docs/archive/wy/FR13_WY_CASCADE_MAP.md` as a user-decision item, so no live ladder pass/fail is claimed and no readout restructure was attempted.

### Blocked next action
- Ask user whether to authorize the #6 readout reduction-order rewrite or to run the requested live ladder anyway with the known above-floor L1 scan result.
- Gate 2 and clean B=4 e2e remain blocked until Gate A is resolved.

## Commit `1f11b9d7` (FR13 remove WY misapplied readout over-rounds) - tap red-team fix (codex_fr16, 2026-06-08)

### Scope
- User supplied `docs/archive/wy/FR13_WY_TAP_REDTEAM.md`: the six-tap `2.44e-4` result was a mis-applied #6 tap, not proof of a reduction-order rewrite need.
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
- User supplied `docs/archive/wy/FR13_WY_RESIDUAL_CLOSURE.md` verdict B: stop tapping; measure WY B=4 e2e with `TREE_ATTN`, forked FA2, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_TREE_GDN_WY=1`, fallback unset, splice off, `FR10_METRICS=0`, and CUDA graph FULL.
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
- User direction: apply `docs/archive/wy/FR13_WY_MULTISTEP_SEAMS.md`; do **not** touch `b_h0`, do **not** BF16-round `state_store_i` in-kernel, and proactively test S1 offline before another live ladder.
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
- User direction: execute `docs/archive/wy/FR13_WY_LADDER_PROMPT_FIX.md`; stop reusing the `163915Z` native capture because its prompt is unrecoverable, do one fresh paired run with a saved deterministic request, then apply `docs/archive/wy/FR13_WY_OUTPUT_TAP_PATCH.md` and re-run only the tree arm.
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
- User direction: read `FR13_SEQ_TREE_SCAN_TASK.md` and `docs/archive/wy/FR13_WY_VS_SEQUENTIAL_VERDICT.md` at `HEAD 93ad85e4`, stand down the WY deliverable path, and execute the sequential rank-1 GDN tree-scan task.
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

## Commit `fr13-seq-native-l2-capture-miss` - native layer-2 localization arm did not write captures (codex_fr13, 2026-06-08)

### Scope
- Native arm run dir: `output/fr13_seq_paired_ladder_20260608T223903Z/native_l2/`.
- Config: `FLASH_ATTN`, B=1 eager, `FR10_DECODE_MODE_DEFAULT=naive_mtp`, `FR10_ENABLE_TREE_GDN=0`, layer-2 subkernel capture requested with `FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX=language_model.model.layers.2.linear_attn`.
- Request: pinned native request copied from `output/fr13_wy_paired_ladder_20260608T211749Z/request.json`.

### Result
- Server reached `/health`; request returned HTTP `200`.
- Response usage: prompt `5`, completion `16`, total `21`.
- Request wall time: `50.67s`.
- Expected capture files under `native_l2/logs/` were **not written**: no `native_l2_gdn_subop.pt`, `native_l2_layer_hidden.pt`, or `native_l2_final_logits.pt`.
- Container source audit confirmed the capture hooks were installed and env vars were present; this appears to be a capture-filter/matching miss, not an unpatched container.
- Host recovery after teardown succeeded: swap `0B`, GPU compute process clear.

### Status
- No layer-2 native subop oracle is available from this arm, so no layer-2 subop verdict is claimed.
- Last valid binding gate remains the B=1 spine ladder failure at layer 2 from `gateA_spine_ladder_seq.json`.

## Commit `fr13-subkernel-capture-debug` - FR12 subkernel capture miss tooling (codex_fr13, 2026-06-08)

### Scope
- User direction: execute `FR13_SEQ_LAYER2_REDTEAM.md` at `HEAD 48c261b6`; do **not** add bf16 rounding to beta or any sequential-scan beta path.
- Code change: added optional `FR12_SUBKERNEL_CAPTURE_DEBUG_LOG` support to the FR12 GDN subkernel capture helper in `scripts/fr10_phase4_patch_vllm_tree_gdn.py`.
- Code change: passed `FR12_SUBKERNEL_CAPTURE_DEBUG_LOG` through both `scripts/fr10_launch_speed_server.sh` and `scripts/fr13_launch_forked_fa2_tree_server.sh`.
- This is observability only: it records `skip_prefix_mismatch`, `skip_num_tokens_mismatch`, `skip_limit`, CUDA-capture skip, and `capture_created` events when the debug env is set.

### Why
- The prior native L2 arm returned HTTP `200` but wrote no capture files. Evidence now points to a silent capture filter miss: the native relaunch used the default 9-node tree speculative config while the capture filters requested `NUM_TOKENS=6`.
- The next native arm will use the MTP-5 speculative config and the debug log so a no-file result has a concrete reason.

### Local Checks
- `python3 -m py_compile scripts/fr10_phase4_patch_vllm_tree_gdn.py`: passed.
- `bash -n scripts/fr10_launch_speed_server.sh scripts/fr13_launch_forked_fa2_tree_server.sh`: passed.
- `git diff --check`: passed.

## Commit `fr13-seq-l2-subop-ladder` - live layer-2 subop ladder localizes first drift (codex_fr13, 2026-06-08)

### Scope
- User direction: run the `FR13_SEQ_LAYER2_REDTEAM.md` pivot, keep beta pure fp32, and fix the failed native capture tooling before running the live L2 ladder.
- Run dir: `output/fr13_seq_l2_subop_20260608T231135Z/`.
- Pinned prompt/request: copied from `output/fr13_wy_paired_ladder_20260608T211749Z/`.
- Discipline: one GPU; native arm torn down and `recover_host_memory()` run before tree arm; tree arm torn down and recovery rerun after capture. Final host state: swap `0B`, GPU compute clear.
- No beta or sequential-scan rounding changes were made.

### Native Capture Tooling Fix Validation
- Native arm used `FLASH_ATTN`, `FR10_ENABLE_TREE_GDN=0`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`, and corrected MTP-5 speculative config (`num_speculative_tokens=5`).
- Capture filters: layer prefix `language_model.model.layers.2.linear_attn`, `NUM_TOKENS=6`, rows `0..5`.
- Request returned HTTP `200`, usage prompt `5`, completion `16`.
- Files written:
  - `native/logs/native_l2_gdn_subop.pt`
  - `native/logs/native_l2_layer_hidden.pt`
  - `native/logs/native_l2_final_logits.pt`
  - `native/logs/native_l2_gdn_subop.debug.jsonl`
- The debug log records the prior failure mode concretely: layer-2 calls can be skipped by `skip_num_tokens_mismatch` when the requested token count does not match the live speculative config. With MTP-5 and `NUM_TOKENS=6`, `capture_created` fired.

### Tree Capture
- Tree arm used `TREE_ATTN`, `FR10_DECODE_MODE_DEFAULT=tree_mtp`, `FR10_TREE_GDN_WY=0`, `FR13_FA2_PREFILL_NATIVE=1`, and the default sequential rank-1 GDN tree-scan.
- Capture filters: layer prefix `language_model.model.layers.2.linear_attn`, `NUM_TOKENS=10`, rows `0..9`.
- Request returned HTTP `200`, usage prompt `5`, completion `16`.
- Files written:
  - `tree/logs/tree_l2_gdn_subop.pt`
  - `tree/logs/tree_l2_layer_hidden.pt`
  - `tree/logs/tree_l2_final_logits.pt`
  - `tree/logs/tree_l2_spine_logits.pt`
  - `tree/logs/tree_l2_gdn_subop.debug.jsonl`

### Row-2 Subop Verdict
- Reducer: `scripts/fr13_layer0_subop_localize.py`.
- Row-2 artifact: `output/fr13_seq_l2_subop_20260608T231135Z/l2_subop_row2_seq_vs_native.json`.
- Compact marker summary: `output/fr13_seq_l2_subop_20260608T231135Z/l2_row2_marker_summary.json`.
- Spine cross-check artifact: `output/fr13_seq_l2_subop_20260608T231135Z/l2_subop_spine_seq_vs_native.json`.

| stage | row-2 max_abs | nonzero | verdict |
| --- | ---: | ---: | --- |
| `input_hidden` | `0.0` | `0` | identical |
| `pre_conv` | `0.0` | `0` | identical |
| `conv1d_out` | `0.0` | `0` | identical |
| `h0_state_in` | `0.0` | `0` | identical full tensor |
| `gdn_scan_out` | `1.1920928955078125e-07` | `2` | **first divergence** |
| `gate_z` | `0.0` | `0` | identical |
| `gate_out` | `6.103515625e-05` | `2` | propagated from scan |
| `o_proj_out` | `0.00048828125` | `1393` | amplified downstream |

The requested visible marker is reproduced in the layer-hidden capture: layer `2`, row `2`, flat `3994` is tree `-2.1875` vs native `-2.203125`, diff `0.015625` exactly one bf16 ULP. At the raw subop level, flat `3994` / `h31,d26` is still identical through `gdn_scan_out`, `gate_z`, `gate_out`, and `o_proj_out`; the first raw scan mismatch is at flat `5535`.

### Interpretation
- This rules out beta and input/convolution wiring for the first L2 row-2 drift.
- Since `h0_state_in` is identical and the first mismatch appears inside `gdn_scan_out`, the remaining suspect is the sequential scan recurrence itself, specifically the parent-resume/register-carry path around `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:277`.
- The spine-row cross-check (`tree rows 0,1,2,4,6,8` vs native rows `0..5`) reports the same first diverging stage: `gdn_scan_out`.

## Commit `fr13-archive-wy-path` - WY path archived off main before L2 recurrence fix (codex_fr13, 2026-06-08)

### Scope
- User direction: preserve all WY work on branch `fr13-wy-archive`, then remove the WY kernel path from `main` before touching the sequential recurrence.
- Archive branch updated and pushed: `fr13-wy-archive -> 680e0848`.
- Moved the 16 `FR13_WY_*.md` notes to `docs/archive/wy/` and updated in-repo markdown references to the archived paths.
- Removed `_tree_gdn_wy_kernel` from `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.
- Removed host/API plumbing for `use_wy`, `fla_bf16_boundaries`, `fla_bf16_output_split`, `FLA_BF16_BOUNDARIES`, `FLA_BF16_OUTPUT_SPLIT`, `FR10_TREE_GDN_WY`, `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES`, and `FR13_WY_FLA_BF16_OUTPUT_SPLIT`.
- Removed the vestigial `FLA_BF16_BOUNDARIES` constexpr from the sequential `_tree_gdn_kernel` signature.
- Updated the FR12 scan probe scripts to exercise only the default sequential `launch_tree_gdn_prepared` path.
- Fixed stale FR10 launcher-default test expectation for `FR11_TREE_CONV_NATIVE_BF16_TAPS` to match the documented/script default `1`.

### Checks
- `python3 -m py_compile src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py scripts/fr10_phase4_patch_vllm_tree_gdn.py scripts/fr12_spine_scan_rounding_probe.py scripts/fr12_scan_batch_invariance_probe.py`: passed.
- `bash -n scripts/fr10_launch_speed_server.sh scripts/fr13_launch_forked_fa2_tree_server.sh`: passed.
- `python3 -m pytest tests/test_fr10_* -q`: `83 passed, 1 skipped`.
- Source/script grep for WY runtime symbols returned empty for `src`, `scripts`, and `tests`.

### Gate-2 No-Bias Check
- Run dir: `output/fr13_wy_archive_validation_20260608T234116Z/gate2/`.
- Reducer: `scripts/fr13_fa2_no_bias_pristine_compare.py`.
- Result artifact: `no_bias_compare.json`.
- Stock vs fork, no tree bias:
  - `float16`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.
  - `bfloat16`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.

### Sequential Ladder No-Regression Check
- Run dir: `output/fr13_wy_archive_validation_20260608T234116Z/tree_ladder/`.
- Config: `TREE_ATTN`, B=1 eager, `FR10_METRICS=0`, `FR13_FA2_PREFILL_NATIVE=1`, default sequential rank-1 GDN tree-scan only.
- Pinned request: `Explain hash tables.`, `max_tokens=16`, `temperature=0`.
- Request returned HTTP `200`, usage prompt `5`, completion `16`.
- Reducer artifact: `tree_ladder/gateA_spine_ladder_seq_after_wy_archive.json`.
- Native reference: `output/fr13_seq_l2_subop_20260608T231135Z/native/`.

| check | max_abs | nonzero / status |
| --- | ---: | --- |
| input hidden | `0.0` | bit-exact |
| layer 0 hidden / residual | `0.0 / 0.0` | bit-exact |
| layer 1 hidden / residual | `0.0 / 0.0` | bit-exact |
| first nonzero | `0.015625` | layer 2 hidden |
| layer 2 residual | `0.0009765625` | unchanged known L2 seam |
| final norm hidden | `1.25` | unchanged known downstream drift |
| final logits | `0.7265625` | unchanged known downstream drift |

### Status
- WY is now archived off `main`; the only served GDN tree path in `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` is the sequential rank-1 kernel.
- Gate A is not claimed here; the known L2 recurrence seam is intentionally left for the next ordered task.
- Host recovery after the live ladder succeeded: swap `0B`, GPU compute clear.

## Commit `fr13-seq-scanout-bv16` - sequential scan-out reduction order aligned (codex_fr13, 2026-06-09)

### Scope
- User direction: fix the L2 scan-out seam from `FR13_SEQ_SCANOUT_FIX.md`; do not change beta or inject bf16 rounding.
- Code change: restored the sequential tree-scan value lane tile to the native/FR12 2-D reduction form.
- First attempt kept `BLOCK_V=1` with `state_i` shaped as `[1, DIM_K]` and axis-1 reductions. It preserved the small h-cache but Triton still produced the old row-2 `gdn_scan_out` delta.
- Second attempt escalated to existing `BV=16`: grid is now `(num_vh, ceil(dim_v / BV))`, `h_cache` is `[N_PAD, BV, DIM_K]`, and the scan reductions use `tl.sum(..., axis=1)`.

### BV=1 Measurement
- Run dir: `output/fr13_seq_scanout_fix_20260608T235731Z/`.
- Request returned HTTP `200`, wall `68.62s`.
- Row-2 artifact: `l2_subop_row2_after_scanout_fix.json`.
- Spine artifact: `l2_subop_spine_after_scanout_fix.json`.
- Verdict: failed to clear the injector.
  - Row 2 `gdn_scan_out`: `max_abs=1.1920928955078125e-07`, `nonzero=2`.
  - Spine `gdn_scan_out`: `max_abs=1.1920928955078125e-07`, `nonzero=7`.

### BV=16 Measurement
- Run dir: `output/fr13_seq_scanout_fix_bv16_20260609T000522Z/`.
- Request returned HTTP `200`, wall `69.68s`.
- Row-2 artifact: `l2_subop_row2_after_scanout_fix_bv16.json`.
- Spine artifact: `l2_subop_spine_after_scanout_fix_bv16.json`.
- Verdict: cleared the L2 injector.

| stage | row-2 max_abs | spine max_abs | verdict |
| --- | ---: | ---: | --- |
| `input_hidden` | `0.0` | `0.0` | identical |
| `pre_conv` | `0.0` | `0.0` | identical |
| `conv1d_out` | `0.0` | `0.0` | identical |
| `gdn_scan_out` | `0.0` | `0.0` | fixed |
| `gate_z` | `0.0` | `0.0` | identical |
| `gate_out` | `0.0` | `0.0` | identical |
| `o_proj_out` | `0.0` | `0.0` | identical |

### Ladder After Fix
- Reducer artifact: `output/fr13_seq_scanout_fix_bv16_20260609T000522Z/gateA_spine_ladder_after_scanout_fix_bv16.json`.
- L2 is now clean on the captured spine path; first hidden mismatch moved to layer `4`.
- Summary: input hidden `0.0`; hidden first nonzero layer `4`, `max_abs=0.0125732421875`; final norm `2.125`; logits `1.00390625`.

### Spill Guard
- Standalone compile artifact: `output/fr13_seq_scanout_fix_bv16_20260609T000522Z/spill_check/`.
- Triton metadata for `_tree_gdn_kernel`: `shared=2048`, `global_scratch_size=0`, `profile_scratch_size=0`.
- Generated PTX/LLIR scan: `.local=0`, `spill=0`, `alloca=0`.

## Commit `fr13-l4-subop-ladder` - layer-4 subop localization after BV16 scan fix (codex_fr13, 2026-06-09)

### Scope
- User correction accepted: `BLOCK_V=1` was a failed diagnostic, not the fix. Its `235731Z` subop artifact still has `gdn_scan_out=1.1920928955078125e-07`; Triton did collapse the degenerate `[1,128]` form.
- Working scan fix is BV16 and is pushed on `main` as `e4a6a2f2`.
- Spill/Triton metadata rabbit-hole is deferred. The practical check is the later TPS gate: an HBM spill should show up as a decode TPS drop versus native. Optional future tuning is smallest non-collapsing BV, likely BV2/BV4, after correctness front is localized.

### L4 Capture
- Run dir: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/`.
- Native arm: `FLASH_ATTN`, `FR10_ENABLE_TREE_GDN=0`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`, explicit `num_speculative_tokens=5`, layer prefix `language_model.model.layers.4.linear_attn`.
- Tree arm: `TREE_ATTN`, pushed BV16 sequential GDN tree-scan, default 9-node tree, layer prefix `language_model.model.layers.4.linear_attn`.
- One GPU discipline: native arm was stopped and host memory recovered before tree arm; tree arm was stopped and host memory recovered after capture. Final host state: swap `0B`, GPU compute clear.
- Request status: native HTTP `200` wall `48.51s`; tree HTTP `200` wall `69.22s`.

### L4 Subop Verdict
- Row-2 artifact: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_subop_row2_bv16_vs_native.json`.
- Spine artifact: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_subop_spine_bv16_vs_native.json`.
- Layer/logit ladder artifact: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/gateA_spine_ladder_l4_capture_bv16.json`.

| stage | row-2 max_abs | spine max_abs | verdict |
| --- | ---: | ---: | --- |
| `input_hidden` | `0.0` | `0.0` | identical |
| `pre_conv` | `0.0` | `0.0` | identical |
| `conv1d_out` | `0.00390625` | `0.0556640625` | **first divergence** |
| `gdn_scan_out` | `0.0018157958984375` | `0.0018157958984375` | downstream of conv |
| `gate_z` | `0.0` | `0.0` | identical |
| `gate_out` | `0.1328125` | `0.1328125` | downstream |
| `o_proj_out` | `0.0703125` | `0.25` | downstream |

The state-store/h-cache compounding caveat is not the first L4 injector on this live capture. `input_hidden` and `pre_conv` are bit-exact; the mismatch starts at the causal-conv output before `h0_state_in` / scan recurrence can be blamed. By-depth spine `conv1d_out` max_abs: depth0 `0.0556640625`, depth1 `0.017578125`, depth2 `0.00390625`, depths3-5 `0.0`.

### Ladder After L4 Capture
- L2 remains fixed under BV16. Layer/logit ladder first hidden mismatch is layer `4`, `max_abs=0.0125732421875`.
- Final norm max_abs `2.125`; logits max_abs `1.00390625`.

## Commit `fr13-l4-conv-detail-root` - layer-4 conv-state/source-index drill (codex_fr13, 2026-06-09)

### Scope
- User direction: do not fix yet; localize the L4 conv root after the L4 subop ladder showed `conv1d_out` is the first broad divergence.
- Artifacts reused from the live L4 capture because the subop hooks already wrote the conv-detail payload:
  - Summary: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_conv_detail_summary.json`.
  - Row-2 subop compare: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_subop_row2_bv16_vs_native.json`.
  - Spine subop compare: `output/fr13_seq_l4_subop_bv16_20260609T003555Z/l4_subop_spine_bv16_vs_native.json`.
- No beta or scan recurrence changes were made.

### Conv State / Window
- Native conv detail: `conv_state_shape=[1196,10240,8]`, `prior_window_source=post_update_fallback`, `metadata_num_accepted_tokens=[1]`, `prior_bank_row=1`, `prior_cols=[0,1,2]`.
- Tree conv detail: `conv_state_shape=[1173,10240,12]`, `prior_read_mode=legacy_remapped_head`, `metadata_num_accepted_tokens=[1]`, `accepted_lens=[0]`, `read_cols=[[0]]`, `prior_bank_rows=[[1]]`, `prior_cols=[0,1,2]`.
- `pre_conv_path0` versus native `pre_conv_rows` is bit-exact: `max_abs=0.0`, `nonzero=0`.
- Conv prior window/state is already divergent before the convolution tap multiply:
  - `prior_window`: shape `[10240,3]`, `max_abs=6.0546875`, `mean_abs=0.39506155252456665`, `nonzero=30671`.
  - Top observed element: index `[2658,1]`, tree `-1.8046875`, native `4.25`.
- Path0 assembled conv window versus native:
  - shape `[6,4,10240]`, `max_abs=6.0546875`, `mean_abs=0.09588509052991867`, `nonzero=61340`.
  - bf16 tap products: `max_abs=0.8046875`, `mean_abs=0.0008135128300637007`, `nonzero=61323`.

### Source Indices
- Native chain source indices:
  - `[[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8]]`
- Tree path0 source indices:
  - `[[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,7],[4,5,7,9],[5,7,9,11]]`
- Equal-by-depth verdict: `[true,true,true,false,false,false]`.
- Branch row memory flag is real for the selected branch node: selected node `3` uses source indices `[2,3,4,6]`.
- But the spine source-index divergence is not the first L4 injector on this capture: the live `conv1d_out` mismatch occurs at depths `0`, `1`, and `2`, where source indices still match native.

| depth | source indices match | window max_abs | window nonzero | bf16 tap max_abs | bf16 tap nonzero |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `true` | `6.0546875` | `30671` | `0.8046875` | `30664` |
| 1 | `true` | `6.0546875` | `20447` | `0.1962890625` | `20441` |
| 2 | `true` | `2.75390625` | `10222` | `0.150390625` | `10218` |
| 3 | `false` | `0.0` | `0` | `0.0` | `0` |
| 4 | `false` | `0.0` | `0` | `0.0` | `0` |
| 5 | `false` | `0.0` | `0` | `0.0` | `0` |

### Conv Root Verdict
- The L4 conv input itself is clean (`pre_conv=0.0`), so the broad `conv1d_out` divergence is not from the incoming hidden stream.
- The conv state/window used for the causal-conv history is divergent, and that divergence is present on the first three spine depths before any path0 source-index divergence can explain the output.
- Current root is therefore the tree conv-state/prior-window read path (`legacy_remapped_head` / bank-resume handling), not beta, not the sequential scan, and not the path0 source-index mismatch as first injector.
- Why L4 and not L0-L3 is now narrowed but not yet fixed: L4 is the first hidden mismatch after full-attn L3; the L4 capture shows the conv-state/prior-window read is live-divergent there. Earlier layers had bit-exact emitted conv outputs after the BV16 scan fix, so the next fix should inspect the layer-indexed conv state bank/update/read transition rather than changing the GDN recurrence.

## Commit `fr13-conv-prior-slot-fix` - conv prior slot root fix, substate/e2e/profile bind (codex_fr13, 2026-06-09)

### Patch
- Code commit: `3a9039cc` (`Fix FR13 tree conv prior slot read`).
- Binding commit: `beca6897` (`Bind FR13 conv prior slot fix`).
- Changed the live tree GDN conv prior read in `scripts/fr10_phase4_patch_vllm_tree_gdn.py:804-811` from `accepted_len` to `accepted_len - 1`.
- This aligns the conv read with the already-clean h0 read convention and fixes the compact-bank row/slot selection root. It is not a tail-column band-aid.

### Substate Gate
- Run dir: `output/fr13_conv_slot_fix_prompt0_20260609T063933Z`.
- Substate artifact: `output/fr13_conv_slot_fix_prompt0_20260609T063933Z/fr13_conv_substate_compare.json`.
- Prior failing metric: call2 row0 `conv1d_out=18.375036`.
- Fixed metric: call2 row0 `conv1d_out=0.0`.

| call | input_hidden | pre_conv | conv1d_out | h0_state_in | gdn_scan_out | gate_out | o_proj_out |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| 1 | `0.0` | `0.0` | `0.0` | `3.725e-09` | `0.0` | `0.0` | `0.0` |
| 2 | `0.0` | `0.0` | `0.0` | `9.537e-07` | `0.0` | `0.0` | `0.0` |

Call2 conv-detail alignment after the fix:
- Tree prior bank rows: `[[5]]`.
- Tree read cols: `[[4]]`.
- Tree compact prior cols: `[0,1,2]`.
- Native prior bank row: `1`.
- Native rolled-tail prior cols: `[5,6,7]`.
- `prior_window_max_abs=0.0`; `window_row0_max_abs=0.0`.

### Prompt0 E2E Gate
- Orchestrator artifact: `output/fr13_conv_slot_fix_prompt0_20260609T063933Z/fr13_e2e_measure.json`.
- `valid=true`; prompt pairing byte-identical.
- `first_mismatch=None`.
- `bag_tv=0.0`.
- Native accept/event: `2.6`.
- Tree accept/event: `2.8`.
- Native TPS: `10.320005524789147`.
- Tree TPS: `6.149089367549517`.
- `first_flip_layer=None`.
- Note: the argmax localizer still reports a tail event after native stream exhaustion, but the deliverable comparison is the token-output gate and reports no mismatch plus zero bag-TV.

### Clean Per-Forward Profile
- Profile artifact: `output/fr13_conv_slot_fix_prompt0_20260609T063933Z/profile/fr13_profile_summary.json`.
- Surface: `scripts/fr10_quick_decode_tps_probe.py` + vLLM `/metrics`, B=4, CUDA graphs enabled, `FR10_METRICS=0`, prompt0, `max_tokens=64`, `temperature=0.6`, `top_p=0.95`.
- `GPU_UTIL=0.86`; `0.88` tripped vLLM startup free-memory guard (`102.37 GiB` free vs `103.41 GiB` requested).

| arm | nodes | decode iters | decode seconds sum | ms/request-forward | accept/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree | 9 | 20 | `21.937230` | `274.215` | `2.173` |
| native | 6 | 15 | `11.654068` | `194.234` | `3.714` |

- Measured tree/native per-forward ratio: `1.412x`.
- Node-count ratio: `9/6 = 1.50x`.
- Residual versus native scaled by node count: `-17.136 ms`; this `/metrics` surface does not show a positive extra penalty beyond node-count scaling.
- GDN tree-scan state traffic / num_warps spill and forked-FA2 whole-tree cost are not separately isolatable from this metrics-only profile; separating them still requires component ablation or a valid server-side kernel trace.
- Do not use returned-token TPS alone for this profile: stochastic sampling returned 163 tree tokens vs 256 native tokens.

### Speed Correction / Caveat
- User red-team workflow `w7m85fq0e` adversarially verified that the `1.412x` ratio above is a **raw, internally inconsistent metric artifact**, not a speed verdict.
- The broken surface is `vllm:request_decode_time_seconds_sum`: it is directionally inverted versus per-request `decode_sum_s` and wall clock on the checked run (`tree 8.60s` vs `native 37.20s` wall-consistent aggregate), while returned-token counts also differ.
- The regime remains weight-bandwidth-bound for this prompt length (`~99ms` weight-stream floor; attention about `0.14%`), but per-forward speed is **UNDETERMINED** pending a clean wall-consistent measurement.
- Future speed measurement must use per-request decode TPS / wall-consistent timing, `VLLM_BATCH_INVARIANT=0`, seeded fixed token totals, many prompts, and E5 `FLASH_ATTN`; do **not** use `decode_seconds_sum` as the denominator.
- Correctness note: do **not** disable or gate off `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:349-353`; the per-node state store is the committed recurrent-state handoff data source, not a discardable HBM tax.

## Commit `3273c411` + measurement worktree - greedy same8 token-level lossless gate (codex_fr13, 2026-06-09)

### Tooling
- Added `scripts/fr13_branch_token_oracle.py` and `tests/test_fr13_branch_token_oracle.py`.
- Local checks passed: `python3 -m py_compile scripts/fr13_branch_token_oracle.py scripts/fr13_e2e_measure.py scripts/fr13_argmax_lcp_localize.py`; `pytest -q tests/test_fr13_branch_token_oracle.py tests/test_fr13_e2e_measure.py`.

### Greedy Same8 Served-Token Gate
- Run dir: `output/fr13_conv_fix_same8_greedy_token_20260609T074753Z`.
- Shape: 8 default prompts, `samples_per_prompt=1`, `batch_size=1`, `max_tokens=64`, `temperature=0`, `top_p=1`.
- Tree arm: forked FA2 tree server, `tree_mtp`, 9-node tree, `FR10_METRICS=0`, CUDA graph, engagement checked (`tree_path_lcp.jsonl` rows present).
- Native arm: `FLASH_ATTN`, `naive_mtp`, 5 speculative tokens, same prompt slice and request shape.
- Pairing guard: **PASS**, `prompt_token_ids` byte-identical for all `8/8` request rows.
- Served output token identity: **FAIL**.
  - Exact token sequence matches: `0/8`.
  - Per-position token matches: `26/512` (`0.05078125`).
  - First mismatch: prompt `0`, sample `0`, position `16`, native token `82546`, tree token `264`.
  - First mismatches for prompts `1`, `2`, `3`, `5`, `6`, `7` occur at position `1`; prompt `4` at position `2`.
  - Bag-TV: `0.5672704646017699`; first-token TV `0.0`; token-count TV `0.125`.
  - Native accept/event: `3.192`; tree accept/event: `1.896774193548387`.
- Summary artifact: `output/fr13_conv_fix_same8_greedy_token_20260609T074753Z/fr13_greedy_token_lossless_summary.json`.
- Token compare artifact: `output/fr13_conv_fix_same8_greedy_token_20260609T074753Z/tree_vs_native_token_compare.json`.

### Branch Oracle Status
- The real branch token oracle was **not bound** because the guard correctly failed before issuing native branch queries.
- Reason: `tree_path_lcp_max` diagnostic rows do not reconstruct the served tree output for this capture. First failure: record `(prompt_id=0, sample=0)`, position `0`, LCP emitted `[248068]`, served tree target slice `[271]`.
- Therefore a branch oracle built from `tree_path_lcp_max` rows would bind a different policy from the actual served output. This is a measurement/policy-alignment blocker, not a branch pass or fail.

### Regular-Decode Status
- Full regular-decode stock-vs-fork layer capture was **not rerun** in this turn. The primary served-token lossless gate already failed, so no lossless verdict is claimed.
- Existing no-bias pristine FA2 comparator remains available, but this entry does not bind a fresh full-layer regular-decode pass for HEAD.

## Commit `6f11a7ca` + measurement worktree - prompt0 next-divergence class (codex_fr13, 2026-06-09)

- Binding note: `FR13_POS16_DIVERGENCE_BIND.md`.
- Temp0.6 same8 after the conv fix: bag-TV `0.24993977147577093` versus prior `0.4258`; improved but still above the `0.059` floor. Exact sequence matches `1/32`; first mismatch prompt `1`, sample `0`, position `1`.
- Targeted prompt0 greedy eager substate run: `output/fr13_pos16_substate_20260609T081638Z`.
- Capture note: the original CUDA-graph same8 first mismatch was prompt0 position `16`; the eager substate run shifted the first prompt0 mismatch to position `18`, so this entry binds the **class** of the next divergence, not a literal CUDA-graph position-16 row.
- Row0 layer-0 GDN substate:
  - calls `0`, `1`, `2`, and `4` are clean through `o_proj_out`;
  - calls `3`, `5`, and `6` first diverge at `input_hidden`, before layer-0 GDN.
- At the served flip event, spine rows remain clean through layer-0 GDN, while the tree selects off-spine branch path `[0,1,3,5,7]` and emits `10278` where native greedy emits `52589`.
- Verdict: this frontier is **not another systematic row0 conv/bank num-accepted-class issue**. It is a **branch selector / native-on-branch-path oracle alignment front**. Do not tune row0 conv writeback/readback from this artifact.

### Branch Oracle Follow-Up
- Code/test update: `scripts/fr13_branch_token_oracle.py` now aligns `tree_path_lcp_max` rows as an ordered subsequence of served output, allowing the leading unlogged token `[271]` and a truncated final tree event; `tests/test_fr13_branch_token_oracle.py` covers both cases.
- Native-on-path artifact: `output/fr13_branch_oracle_20260609T084849Z/branch_event4_path_0_1_3_5_7_oracle.json`.
- Prompt-token sanity: `prompt: [prompt_token_ids]` reproduced the native prompt0 text-prompt baseline for the first 24 tokens; integer-token branch prompts are valid for this oracle.
- Event `4`, branch path `[0,1,3,5,7]`: tree verify targets match real native-on-branch-path `/v1/completions` for `6/6` checks (`364`, `264`, `82546`, `10278`, `1103`, leaf bonus `2357`).
- Classification: **alignment/gate mismatch**, not branch-kernel argmax drift. The paired native greedy spine emits `52589`, but native-on-that-branch-path emits `10278`, matching the tree. The committer selects the branch because branch LCP `5` > path0 LCP `1`; this is the `greedy_tree_lcp_max` policy, not a GDN/FA2 sub-op failure.

## Commit (this commit) - FR13 branch-node self-noise red-team / current same8 minus native-self-noise gate (codex_fr13, 2026-06-09)

- Binding note: `FR13_BRANCH_NODE_SELF_NOISE_BIND.md`.
- Run root: `output/fr13_branch_node_redteam_20260609T090932Z`.
- Reason for rerun: stale artifacts disagreed on native's own prompt0 branch-point token (`10278` vs `52589`), so no losslessness verdict can be bound from those pairs.
- Fresh current arms:
  - tree: `TREE_ATTN`, forked FA2, `tree_mtp`, 9-node tree, `MAX_NUM_SEQS=1`, batch probe `1`;
  - native1: `FLASH_ATTN`, `naive_mtp`, 5 MTP tokens, `MAX_NUM_SEQS=1`, batch probe `1`;
  - native2 self-noise: `FLASH_ATTN`, `naive_mtp`, 5 MTP tokens, `MAX_NUM_SEQS=4`, batch probe `4`.
- Prompt identity: byte-exact for all `8/8` current records.
- Branch-node red-team result: prompt0 position `18` is native self-noise (`native1=10278`, `native2=52589`, `tree=52589`), so that specific branch flip is not a real tree loss.
- A/B/C gate:
  - (a) tree-vs-native1 mismatching positions: `368`;
  - (b) native self-noise mismatching positions: `85`;
  - (c) tree mismatches outside native self-noise: `287`.
- Sequence level: tree/native exact `0/8`; prefix `0/8`; ordered-subsequence `0/8`; native1/native2 exact `4/8`.
- First outside-self-noise divergences: prompt0 pos `46`; prompt1 pos `28`; prompt2 pos `35`; prompt3 pos `10`; prompt4 pos `1`; prompt5 pos `1`; prompt6 pos `1`; prompt7 pos `15`.
- Verdict: native self-noise explains the stale branch-node contradiction, but not the current same8 tree divergence. The accepted minus-self-noise gate still fails; localize from the first current outside-self-noise divergence, not the stale pos18 artifact.

## Commit (this commit) - FR13 Seam 1 uniform conv write-back validation (codex_fr13, 2026-06-09)

- Binding note: `FR13_SEAM1_UNIFORM_CONV_BIND.md`.
- Run root: `output/fr13_decisive_seam1_20260609T173330Z`.
- Code status: the layer-conditioned rolled-tail band-aid is already absent from HEAD (`_fr10_use_rolled_tail_prior`, `layer_idx >= 4`, and `rolled_tail_remapped` all absent). The live write-back is the uniform accepted-path source `cat(prior_window, node_path_x)` with per-node `node_path_len + arange(conv_state_len)` store rows.
- Code delta in this stage: normalized the accepted-length clamp cast ordering in `scripts/fr10_phase4_patch_vllm_tree_gdn.py`; this is a guard/contract normalization, not a new conv behavior change.
- CPU gates: `pytest -q tests/test_fr10_phase4_sampled_committer_wiring.py tests/test_fr10_tree_commit_gates.py tests/test_fr10_tree_conv.py` -> `22 passed, 1 skipped`; `py_compile` passed for the patch and sub-op reducer scripts.
- Live capture: eager prompt0 tree/native, all 48 GDN layers, three verify calls per layer. Tree captures use `num_tokens=10`; native captures use `num_tokens=6`; prefill/profile captures excluded.
- Capture counts: `144` tree + `144` native = `48` GDN layers x `3` calls.
- Reducer artifact: `output/fr13_decisive_seam1_20260609T173330Z/fr13_seam1_uniform_conv_validation.json`.
- Gate result: row0 clean-input cases `99`; row0 clean-input `conv1d_out` nonzero cases `0`; spine clean-input `conv1d_out` nonzero cases `0`.
- Verdict: Seam 1 validated. No new clean-input conv divergence was exposed; remaining non-clean divergences are upstream input/branch-state issues, not a conv read/write-back seam. Advance to Seam 2.

## Commit (this commit) - FR13 Seam 2 prefill-native validation (codex_fr13, 2026-06-09)

- Binding note: `FR13_SEAM2_PREFILL_NATIVE_BIND.md`.
- Code status: the `FR13_FA2_PREFILL_NATIVE` implementation was already present in `scripts/fr13_patch_fa2_tree_bias.py` and launch-defaulted on by `scripts/fr13_launch_forked_fa2_tree_server.sh`. This stage validates it; it does not re-derive or replace it.
- Harness updates: `scripts/fr13_fa2_no_bias_pristine_compare.py` now includes explicit prefill-shaped cases (`q_lens == k_lens`) in addition to decode-shaped cases; `scripts/fr10_launch_speed_server.sh` passes the existing `FR13_PREFILL_GDN_CAPTURE*` diagnostic env vars so native FLASH_ATTN can be paired for L8 h0/state.
- CPU gates: `pytest -q tests/test_fr13_fa2_no_bias_pristine_compare.py tests/test_fr10_phase4_sampled_committer_wiring.py` -> `15 passed`; `bash -n` for both launchers and `py_compile` for Seam 2 scripts passed.
- Run root: `output/fr13_decisive_seam2_20260609T175913Z`.
- Extended gate-2 stock-vs-fork no-bias compare: `gate2/no_bias_compare.json`.
  - `float16_decode`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.
  - `float16_prefill`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.
  - `bfloat16_decode`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.
  - `bfloat16_prefill`: `torch_equal=true`, `max_abs=0.0`, `nonzero=0`.
- Live L7 prefill full-attn replay: `prefill_full_attn_l7_replay.json`.
  - Tree capture: `tree_prefill_l7_live/logs/full_attn_tree_l7.call0.pt`.
  - Native capture: `native_prefill_live/logs/full_attn_native_l7.call0.pt`.
  - Same text prompt; both arms reported `prompt_tokens=14`.
  - Result: no first diverging layer/stage; `attn_out_raw=0.0`, `o_proj_out=0.0`, and all captured stages `0.0`.
- Live L8 prefill-GDN replay: `prefill_gdn_l8_state_replay.json`.
  - Tree capture source: `tree_prefill_live/logs/prefill_gdn_tree.call*.pt`.
  - Native capture: `native_prefill_live/logs/prefill_gdn_native.call0.pt`.
  - Result: no first diverging layer/stage; `pre_conv=0.0`, `conv_out=0.0`, `initial_state/h0=0.0`, `core_out=0.0`, `final_state=0.0`.
- Verdict: Seam 2 validated for HEAD. Forked FA2 no-bias regular decode and prefill are byte-identical to pristine stock FA2; live TREE_ATTN prefill L7 is byte-identical to native FLASH_ATTN; downstream GDN L8 prefill recurrent seed is byte-identical. Advance directly to the all-8 branch-oracle plus self-noise-corrected B=4 superset e2e comparator.

## Commit (this commit) - FR13 all-8 branch oracle targeted winner-path gate (codex_fr13, 2026-06-09)

- Binding note: `FR13_ALL8_BRANCH_ORACLE_BIND.md`.
- Run root: `output/fr13_decisive_final_20260609T182455Z`.
- Reducer update: `scripts/fr13_branch_token_oracle.py` gained `--targets` and `--winner-only` so the oracle can check the eight requested first outside-self-noise flip events without running every branch path. It also records skipped leading overrun rows from max-token truncation while preserving the fail-closed mismatch guard.
- CPU gates: `pytest -q tests/test_fr13_branch_token_oracle.py` -> `3 passed`; `py_compile` passed.
- Tree B1 greedy branch capture: `tree_b1_greedy_branch/tree_greedy_probe.json`; engagement asserted by the probe; tree accept/event `1.965986394557823`.
- Native B1 greedy reference: `native_b1/native_greedy_probe.json`; native accept/event `3.8545454545454545`.
- Native B1 temp0.6 self-noise baseline captured: `native_b1/native_temp06_probe.json`; native accept/event `3.8878504672897196`.
- Targeted oracle artifact: `all8_branch_oracle.json`.
  - events aligned: `143`; skipped overrun rows: `5`.
  - target positions: `0:46,1:28,2:35,3:10,4:1,5:1,6:1,7:15`.
  - winner-only checks: `17`; tree/native-on-branch matches: `13/17`.
  - first mismatch: prompt0 event13 path `[0,1,3,5,8]` depth4 parent target, tree `198` vs native-on-branch `1358`.
  - other mismatches: prompt0 leaf self target `262` vs `71093`; prompt3 leaf self target `265` vs `1302`; prompt7 depth0 parent target `417` vs `7620`.
- Verdict: the all-8 branch oracle does **not** confirm lossless-by-branch for HEAD. Several targeted committed winner-path checks differ from native-on-their-branch-path. The B4 temp0.6 superset e2e deliverable was not started in this stage.

## Commit (this commit) - FR13 SWE B=4 CUDA-graph diagnostic (codex_fr13, 2026-06-09)

- Binding note: `FR13_SWE_B4_DIAGNOSTIC_BIND.md`.
- Run root: `output/fr13_swe_verified_b4_diag_20260609T190931Z`.
- Scope: two sequential one-GPU arms using `scripts/fr12_deliverable_swe4_probe.py` on the SWE-Verified four-prompt subset, B=4, `temperature=0.6`, `top_p=0.95`, `max_tokens=128`, `samples_per_prompt=4`, `seed=1313`; `recover_host_memory()` before each arm.
- TREE arm: `TREE_ATTN`, forked FA2 tree verify, `tree_mtp`, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR13_FA2_PREFILL_NATIVE=1`, `FR10_METRICS=0`.
- Native E5 arm: `FLASH_ATTN`, `naive_mtp`, MTP-5, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR10_METRICS=0`.
- CUDA graph validation:
  - TREE: `enforce_eager=False`, `cudagraph_mode=FULL_AND_PIECEWISE`, `Capturing CUDA graphs (decode, FULL): 4/4`.
  - Native: `enforce_eager=False`, `cudagraph_mode=FULL_AND_PIECEWISE`, `Capturing CUDA graphs (decode, FULL): 4/4`.
  - No FR13 subkernel/prefill capture hook artifacts were produced; TREE standard engagement traces were present (`tree_sampler_debug.jsonl`, `tree_path_lcp.jsonl`, `fr10_mtp_draft_trace.jsonl`).
- Artifacts:
  - TREE probe: `tree_b4_swe4/tree_b4_swe4_probe.json`.
  - Native probe: `native_b4_swe4/native_b4_swe4_probe.json`.
  - Raw compare: `native_vs_tree_swe4_compare.json`.
- Wall-consistent speed:
  - TREE returned `1856` tokens in `102.950s`: `18.028` returned tokens/s; request TPS mean/median `4.902/4.966`.
  - Native returned `2048` tokens in `49.629s`: `41.266` returned tokens/s; request TPS mean/median `11.875/11.417`.
- Deployed-regime speed verdict for this diagnostic: using wall-consistent per-request TPS mean, TREE is `2.42x` slower than native E5 (`4.902` vs `11.875`). By returned-token wall TPS, TREE is `2.29x` slower (`18.028` vs `41.266`); by measured wall window, TREE took `2.07x` longer (`102.950s` vs `49.629s`) while returning fewer tokens.
- Acceptance:
  - TREE accept/event `2.024` (`1255` accepted, `5580` draft tokens, `620` drafts).
  - Native accept/event `3.794` (`1643` accepted, `3897` draft tokens, `433` drafts).
  - Native E5 has `1.87x` higher accept/event than TREE (`3.794 / 2.024`), equivalently TREE has `1.87x` fewer accepts per draft event.
- Raw output-level token comparison:
  - records compared `16`;
  - exact token sequences `0/16`;
  - emitted-token bag TV `0.2194066541`;
  - first-token TV `0.0`;
  - token-count TV `0.25`;
  - first mismatch prompt `0`, sample `0`, position `15`, native `1970`, tree `5759`;
  - prefix match min/median/max `1/16/68`;
  - TREE returned fewer than `128` tokens for `4/16` records; native returned `128/128` for all records.
- Limitations: `fr12_deliverable_swe4_probe.py` is an output-level SWE prompt decode probe, not the full Codex SWE agent/evaluator run, so task grader verdicts were not produced. Self-noise correction was not established because this stage ran the user-requested two arms only; a same-regime native-vs-native repeat would be required to subtract native self-noise from raw TREE-vs-native token differences.
- Diagnostic bind: record the numbers only; no pass/fail self-declaration. The real-workload B=4 CUDA-captured diagnostic is clearly negative for TREE on speed/acceptance versus native E5.

## Commit (this commit) - FR13 branch seam CPU prep / no-boot inventory (codex_fr13, 2026-06-09)

- Binding note: `FR13_BRANCH_SEAM_CPU_PREP_BIND.md`.
- GPU status: held; no server boot, no GPU work.
- Target seam from `all8_branch_oracle.json`: prompt `0`, sample `0`, event index `13`, served prefix len `45`, path `[0,1,3,5,8]`, node `8`, depth `4`, parent-target check, prefix token IDs `[638,4381,283,1727]`, tree `198` vs native-on-branch `1358`.
- Exact tree trace row exists: `output/fr13_decisive_final_20260609T182455Z/tree_b1_greedy_branch/logs/tree_path_lcp.jsonl` line `14`; it has `accepted_node_ids=[0,1,3,5,8]`, `winner_path=[0,1,3,5,8]`, `emitted_tokens=[638,4381,283,1727,198,262]`.
- Existing capture inventory result: no matching `.pt` substate/layer/full-attn/final-logit capture exists for this all-8 seam event. The all-8 run root has only JSONL/probe/log artifacts.
- Checked nearby B1 captures:
  - `fr13_gdn_substate_prompt0_20260609T061732Z`: L0 subkernel/layer-hidden calls exist, but not the `[638,4381,283,1727]` seam event.
  - `fr13_pos16_substate_20260609T081638Z`: L0 subkernel calls `0..6` exist, but not the seam event.
  - `fr13_conv_fix_same8_greedy_token_20260609T074753Z`: nearby prefix exists in LCP JSONL, but no `.pt` substate and not the all-8 native-on-branch mismatch.
  - `fr13_branch_node_redteam_20260609T090932Z`: LCP/sampler traces only; no `.pt` substate.
- Prepared future capture if CPU workflow calls for a GPU seam boot: tree `FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=10`, `FR12_SUBKERNEL_CAPTURE_SKIP=13`, `FR12_SUBKERNEL_CAPTURE_LIMIT=1`, first at `language_model.model.layers.0.linear_attn`; native-on-branch uses the existing branch-oracle path construction with prefix len `45` plus `[638,4381,283,1727]`, `NUM_TOKENS=6`, `SKIP=0`, `LIMIT=1`.
- Conclusion: CPU prep cannot localize where `198` vs `1358` is born from existing captures. The next localizing evidence requires one targeted paired capture if and only if the parallel CPU root workflow says the seam fix is worth a GPU boot.

## Commit (this commit) - FR13 B4 superset CPU gate (codex_fr13, 2026-06-09)

- Binding note: `FR13_B4_SUPERSET_CPU_GATE_BIND.md`.
- Reducer: `scripts/fr13_b4_superset_cpu_gate.py`.
- Artifact: `output/fr13_swe_verified_b4_diag_20260609T190931Z/b4_superset_cpu_gate.json`.
- CPU-only; no server boot, no GPU work.
- Gate input: existing SWE B4 diagnostic artifacts under `output/fr13_swe_verified_b4_diag_20260609T190931Z`, using tree `tree_path_lcp.jsonl`, `tree_sampler_debug.jsonl`, `fr10_mtp_draft_trace.jsonl`, tree probe, and native probe.
- Topology result: pass. Runtime `speculative_token_tree` equals the expected top-1 spine plus top-2 branch tree; path0/top-1 spine nodes are `[0,1,3,5,7]`; sibling branch nodes are `[2,4,6,8]`; `193/193` `gpu_tree_metadata` rows have `mode=tree_mtp`, `tree_len=9`, `reason=ok`, `has_tree_parent_indices=true`, and `has_draft_token_indices=true`.
- Measured-tail tree verifier/committer classification from `tree_summary.spec_drafts=620`:
  - spine path accepted `250/620`;
  - branch path accepted `132/620`;
  - root reject `238/620`;
  - accepted-length counts `{0:238, 1:60, 2:85, 3:74, 4:37, 5:126}`.
- Committer/source-index evidence:
  - accepted source `0`: `1098`;
  - accepted source `1`: `132`;
  - in `116/132` source-1 branch accepts, the verifier assigned zero probability to the top-1/spine child;
  - reject steps total `367`, root-step rejects `227`, reject steps with both candidates at zero probability `332`.
- Native limitation: the existing native B4 artifact has aggregate spec counters and served token IDs but no per-event MTP draft/accept trace, so exact native per-event accepted depth cannot be reconstructed CPU-only from this run.
- Served-token surface remains bad: exact sequences `0/16`; first mismatch examples include prompt0/sample0 pos `15` tree `5759` vs native `1970`.
- Classification: `topology_break=false`, `committer_break=false`, `verify_or_target_row_break=true`.
- Conclusion: the tree contains the superset topology, and the committer follows verifier scores. The B4 accept drop is a verify/target-row contamination or alignment bug, not a drafter-quality structural floor and not a missing-topology bug. Defer speed/forward-cost analysis.

## Commit (this commit) - FR13 corruption gate script bind (codex_fr13, 2026-06-09)

- Binding note: `FR13_CORRUPTION_GATE_BIND.md`.
- File committed: `scripts/fr13_corruption_gate.py`.
- Validation: `python3 -m py_compile scripts/fr13_corruption_gate.py` passed.
- Purpose: CPU reduction of the three-arm tree/native/native-noise corruption gate: served-token argmax match corrected by native self-noise, emitted-token bag-TV floor, accept/event superset against native, prompt identity fail-closed, and optional per-event superset checks when traces are present.
- Note for the next run: the script expects `fr13_e2e_measure.py`-style filenames (`tree_greedy_probe.json`, `native_greedy_probe.json`); the earlier SWE B4 direct probe used `fr12_deliverable_swe4_probe.py` names, so the fresh three-arm run must write or normalize the expected arm filenames before invoking the gate.

## Commit (this commit) - FR13 corruption B4 three-arm gate (codex_fr13, 2026-06-09)

- Binding note: `FR13_CORRUPTION_B4_GATE_BIND.md`.
- Run root: `output/fr13_corruption_b4_gate_20260609T194841Z`.
- Arms: TREE forked-FA2 `TREE_ATTN/tree_mtp` seed `1313`; native E5 `FLASH_ATTN/naive_mtp` seed `1313`; native-noise `FLASH_ATTN/naive_mtp` seed `2313`.
- All arms were B=4, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR10_METRICS=0`, sequential one-GPU, with `recover_host_memory()` between arms.
- CUDA graph validation: all three logs show `enforce_eager=False`, `Profiling CUDA graph memory: ... FULL=4`, and `Capturing CUDA graphs (decode, FULL): 4/4`.
- Prompt guard: tree/native prompt token IDs byte-identical; prompt token counts `681`, `1080`, `829`, `1614`; record sets paired.
- Gate output: `fr13_corruption_gate.json`, exit code `2`, `valid=true`, `passed=false`, `verdict=FAIL`.
- Self-noise: native self-noise mask positions `277`; native self bag-TV `0.10986328125`.
- Self-noise-corrected token surface: compared positions `467`; eligible positions `221`; outside-self-noise losses `105/221 = 0.4751`; first real loss prompt `0` position `16`, tree `369` vs native `3051`; longest real-loss run `44`; depth-collapse detector fired.
- Bag-TV: tree/native emitted-token bag-TV `0.2335064444` > budget `0.10986328125`.
- Accept/event: tree `2.1016042781` vs native `3.6191536748`, delta `-1.5175493968`; native-noise `3.8139534884`.
- Depth-0 root signal from `fr13_depth0_root_gate.json`: measured tail root accept `348/549 = 0.6339`; root reject `201/549 = 0.3661`; `target_argmax != draft0` in `201` rows.
- Native per-event limitation: no `per_req_spec_trace.jsonl` was emitted by the native arms, so per-event native target-argmax comparison is unavailable in this bind. The paired served-token/self-noise gate remains valid.
- Conclusion: not a drafter-quality stop. The three-arm evidence shows a real tree-verify contamination / target-row bug surface; speed/forward-cost analysis remains deferred.

## Commit (this commit) - FR13 eager B4 bisection (codex_fr13, 2026-06-09)

- Binding note: `FR13_EAGER_B4_BISECT_BIND.md`.
- Run root: `output/fr13_b4_eager_bisect_20260609T203718Z`.
- Purpose: answer whether the B=4 corruption gate is eager-reproducible or CUDA-graph-only before trusting eager substate localization.
- Arms: TREE forked-FA2 `TREE_ATTN/tree_mtp` seed `1313`; native E5 `FLASH_ATTN/naive_mtp` seed `1313`; native-noise `FLASH_ATTN/naive_mtp` seed `2313`.
- All arms were B=4, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR10_METRICS=0`, one GPU sequential, with host-memory recovery between arms.
- Eager validation: all three logs show `Enforce eager set, disabling torch.compile and CUDAGraphs` and `Cudagraph is disabled under eager mode`; no CUDA graphs were used.
- Prompt guard: prompt token IDs byte-identical; prompt token counts `681`, `1080`, `829`, `1614`.
- Gate output: `fr13_eager_b4_corruption_gate.json`, exit code `2`, `valid=true`, `passed=false`, `verdict=FAIL`.
- Self-noise: native self-noise mask positions `137`; native self bag-TV `0.15234375`.
- Self-noise-corrected token surface: compared positions `256`; eligible positions `119`; outside-self-noise losses `44/119 = 0.3697`; first real loss prompt `1` position `11`, tree `12182` vs native `26622`; longest real-loss run `31`; depth-collapse detector fired.
- Bag-TV: tree/native emitted-token bag-TV `0.19140625` > budget `0.15234375`.
- Accept/event: tree `1.8021978022` vs native `3.1875`, delta `-1.3853021978`; native-noise `2.9701492537`.
- Bisection verdict: EAGER B=4 reproduces the real loss. This is not a captured-only CUDA-graph bug; proceed with eager B=4 co-residency / batched-verify localization.
- Captured-hook caveat: direct captured substate attempts failed before serving. Layer-hidden capture hit Dynamo shape specialization on `int(hidden_states.shape[0])`; GDN subkernel capture hit unsupported `torch.cuda.is_current_stream_capturing`.
