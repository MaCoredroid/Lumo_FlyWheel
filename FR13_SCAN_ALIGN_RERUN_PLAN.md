# FR13 — Scan-align Phase-2 RE-RUN plan: int-view STATE gate + the working no-spec oracle

Date 2026-06-15 (CPU-only fix+investigate workflow). Supersedes the two BROKEN instruments from
`FR13_SCAN_ALIGN_VERIFY_INCONCLUSIVE_BIND.md` (w77rygxwf holds=false). The committed scan-alignment
(5e56b7aa) ENGAGES and default-OFF is byte-identical, but whether it drives the lossless flips 21→~3
was UNMEASURED because BOTH binding instruments were vacuous (playbook **#9** *silent/vacuous instrument*).
This plan fixes Instrument 1 (the int-view STATE gate) and documents the exact, reproducible mechanism for
Instrument 2 (the no-spec oracle) so the re-run can compute the binding flips.

Playbook rows in force (FR13_BUG_CLASS_PLAYBOOK.md, quoted verbatim):
- **#9 Silent fallback / vacuous instrument** — "a run 'passes' while measuring nothing | engagement asserts
  ... BEFORE trusting any number | fail-loud on disengagement". The old gate compared the all-zeros output
  and "passed" a vacuous neg-control. The fixed gate compares the durable STATE and the neg-control must
  genuinely FLIP it (both states non-zero).
- **#10 Shared-source ≠ shared-SASS (codegen identity)** — "byte A/B on captured payloads, int-view equality
  (NEVER atol), SASS hash pin". OUR tree-scan and the native packed-decode kernel inline the same rank-1 GDN
  body; the A/B is int-view on captured payloads, never atol.

---

## ROOT CAUSE of the vacuous gate (Instrument 1) — exact, from the pinned image

`scripts/vllm_src.sh model_executor/layers/fla/ops/fused_recurrent.py`, kernel
`fused_recurrent_gated_delta_rule_packed_decode_kernel` (vLLM 0.19.2rc1.dev134):

```
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
p_o = o + (i_n * HV + i_hv) * V + o_v
# Skip if state index is invalid (NULL_BLOCK_ID=0)
if state_idx <= 0:
    zero = tl.zeros([BV], ...).to(p_o.dtype.element_ty)
    tl.store(p_o, zero, mask=mask_v)          # writes ZEROS to out
    return                                    # and NEVER reaches the state store
...
tl.store(p_o, b_o..., mask=mask_v)            # real output o = sum(b_h*b_q)
p_ht = ht + state_idx * stride_final_state_token
tl.store(p_ht, b_h..., mask=mask_h)           # the DURABLE STATE h (b_h)
```

The old `native_packed_decode_per_path` passed `ssm_state_indices = torch.zeros(1)` → `state_idx == 0` →
the kernel **short-circuited**: it wrote **zeros to `out`** (the all-zeros-output that made `native_out`
norm 0.0) **and never reached `tl.store(p_ht, b_h)`** — so the returned `state` was the *un-updated* cloned
h0, NOT the durable post-update STATE. Both comparands were vacuous: comparing `|serving_out|` vs zeros
(int-view trivially False) and the +0.5 neg-control "powered" vacuously (#9).

The live decode path does NOT pass 0: `gdn_linear_attn._forward_core_decode_non_spec` (read from the image,
L1085-1095) calls the kernel with `initial_state=ssm_state` (the full multi-slot cache `[num_slots,HV,V,K]`)
and `ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens]` (real slot ids, all ≥ 1). The
kernel reads slot `state_idx`, runs the rank-1 update, and writes the durable STATE back to that slot in
place (`h0 == ht == initial_state`).

---

## FIX 1 — the native-packed ref now exposes the DURABLE STATE (committed)

`scripts/fr13_native_packed_decode_ref.py::native_packed_decode_per_path`:
- Build a ≥2-slot fp32 state buffer `[2, HV, V, K]`; place the real h0 at **slot 1**; slot 0 stays zeros
  (NULL_BLOCK_ID, so a stray slot-0 read would surface loudly).
- Pass `ssm_state_indices = torch.full((1,), 1)` (slot 1). The kernel now runs the real update, writes a
  **real (non-zero) output** to `out`, and writes the **durable STATE** to slot 1 in place along the path.
- Extract `states.append(state[1].clone())` = the durable post-update STATE `[HV,V,K]` per node — the A/B
  comparand the gate needs. Outputs are also real now (secondary).

This is OBSERVE-ONLY: native is the A/B oracle, no served-path splice (reward-hack CLEAN). The packed-decode
math (`b_q/||b_q||·scale`, `b_h*=exp(g)`, `b_v-=Σ(b_h·b_k)`, `b_v*=β`, `b_h+=b_v·b_k`, `b_o=Σ(b_h·b_q)`)
is line-for-line identical to OUR `_gdn_node_step` in `fr10_gdn_tree_kernel.py`, except the two SCAN_ALIGN
seams: l2norm `rsqrt` (ours, OFF) vs `div-by-sqrt` (native), and β `fp32` (ours, OFF) vs `bf16` round-trip
(native, since payload `b` is bf16). OUTPUT_SCALE == native `scale == head_k_dim**-0.5` (confirmed: payload
`output_scale = 0.08838 = 128**-0.5`).

## FIX 2 — the gate verdict is keyed on the DURABLE STATE (committed)

`scripts/fr13_gdn_scan_warp_gate.py`:
- The per-arm record already carried `state_vs_native_packed` (OUR scan STATE `tree_state` vs native
  `native_state`). The **verdict** `negative_control_powered` now reads
  `deployed["state_vs_native_packed"]` (NOT the zeros-prone output). It is True iff (a) the STATE int-view
  comparison FLIPPED to mismatch (`int_view_equal is False`) AND (b) both states are non-zero
  (`0 < norm_ratio < inf`) — so it can never re-vacuum off a zeros tensor (#9).
- Neg-control = perturb `value_tree[0,0,0]` (+0.5), an INPUT that feeds OUR recurrent STATE update
  (`b_v -= Σ(state·k); b_v *= β; state += b_v·k`), so OUR root-and-descendant STATE MUST diverge from
  native's unperturbed STATE → the STATE int-view MUST be False. **Provably non-vacuous.**
- New surfaced carriers (the measurement we never got): `off_arm_spine_state_vs_native` (deployed-OFF scan
  STATE vs native packed STATE — the genuine scan-vs-native gap) and `recompute_arm_spine_state_vs_native`.
- Schema bumped `fr13.gdn_scan_packed_ab_gate.v2_state`. CPU AST/wiring tests added
  (`tests/test_fr13_scan_vs_native_packed.py`): `test_ref_uses_valid_state_slot_not_null_block`,
  `test_gate_verdict_is_keyed_on_durable_state_not_output`. All 27 CPU tests pass (no GPU).

### Expected re-run reads (GPU)
- **neg_control STATE int-view = False, norm_ratio finite>0** → `negative_control_powered = True` (the gate
  is PROVABLY live). If the neg-control does NOT flip the STATE, STOP — the comparator is still vacuous.
- **OFF arm spine STATE vs native packed STATE** = the carrier number. Likely NON-zero (rsqrt≠div +
  β fp32≠bf16 seams) = the genuine OFF scan-vs-native-STATE gap we never measured.
- **BODY_SEAMS / RECOMPUTE arms** STATE vs native packed STATE → if int-view 0.0 (or rel_err at the bf16-ULP
  floor), the SCAN_ALIGN seams close the scan-vs-native-STATE gap = the carrier is the seam.

Run (GPU, payload = the deployed N_PAD=16 MTP-5 capture with bf16 q/k/v + fp32 h0 `[48,128,128]`):
```
scripts/vllm_src.sh --sha   # confirm pinned image
# inside the pinned container (CUDA), with the deployed payload:
python3 scripts/fr13_gdn_scan_warp_gate.py \
  --payload <gdn_l*_scan_capture.pt> \
  --out output/fr13_scan_align_rerun/state_ab_gate.json
# returns 0 iff negative_control_powered (STATE-keyed); read off_arm_spine_state_vs_native.
```

---

## INSTRUMENT 2 — the no-spec oracle: how the ORIGINAL 21 was measured, and the two valid mechanisms

The binding flip number is a **CHANNEL-2 flip**: tree-served token at position i vs the no-spec oracle's
argmax conditioned on the IDENTICAL prefix (`prompt_ids + served_token_ids[:i]`), on THIS boot's own stream
(`feedback_no_cross_boot_byte_gate`). Two scripts implement it; the original 21/22 came from the HTTP one.

### The crash (why per-request non_mtp/naive_mtp dies)
`EagleProposer has no attribute 'positions'` in `propose_tree` fires only on the **spec-decode DECODE path**.
A spec-configured server (`--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":5,...}'`)
builds the EagleProposer; a per-request `fr10_decode_mode=non_mtp`/`naive_mtp` tries to take a non-spec route
through that engine on a DECODE step and the proposer is invoked without the spec state it expects → crash.
This is a per-request DECODE-mode switch on a spec engine — NOT reachable when the request is a PREFILL or
when the engine has no spec config.

### MECHANISM A — the ORIGINAL 21/22 mechanism (same boot, chunked re-prefill, NO crash)
The original 21 (`output/fr13_verify_bisect/c4_classify.json`, "C4_sample_row_off_all_speed_off", 21
[3,6,6,6]; banked 22 was c0..c3) was produced by **`scripts/fr13_verify_bisect_probe.py classify`** (and the
equivalent `scripts/fr13_oracle_stream_teacher_force.py run`), oracle **`--mode tree_mtp`** (the DEFAULT;
the bisect raw has ZERO non_mtp/naive_mtp tokens). For each served position i:
```
ctx = prompt_ids + served_ids[:i]
POST /v1/completions {"prompt": ctx, "max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                      "seed": 1313, "logprobs": 20, "vllm_xargs": {"fr10_decode_mode": "tree_mtp"}}
oracle_argmax_id = choice.token_ids[0]; flip = served_ids[i] != oracle_argmax_id
clear_margin = flip and (served outside oracle top-k OR oracle_argmax_lp - served_in_oracle_lp > 1.0 nat)
```
- **Does NOT crash**: a `max_tokens=1` request over a multi-token `ctx` has `query_len = len(ctx) >> 1` →
  `split_decodes_and_prefills(decode_threshold=1)` makes it a **PREFILL**, never a spec DECODE, so the
  EagleProposer/`propose_tree` is never reached. (Confirm in the artifact: `spec_metrics_delta` ≈ 0 during
  the oracle phase = no drafts advanced = no spec engaged.)
- **CAVEAT (frame, `FR13_ORACLE_FRAME_DECISION.md` ec342d86 / FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND.md):**
  Qwen3-Next lacks `SupportsMambaPrefixCaching` → `mamba_cache_mode="align"` → the partial-suffix SSM state
  is REBUILT via the **CHUNKED** `chunk_gated_delta_rule` every request (`gdn_linear_attn._forward_core`
  `if num_prefills>0`). So Mechanism A measures vs a **chunked-prefill** oracle, which VIOLATES the
  directive's anchor "oracle = no-spec NOT prefill" and inflates the near-tie band by a ~9× chunk-vs-
  recurrent bf16-ULP frame (q1: L0 GDN 0.0078 chunked vs 0.000854 recurrent). The ORIGINAL 21/22 is a
  chunked number; reproduce it with Mechanism A for continuity, but it is NOT the deployment-correct ref.

### MECHANISM B — the deployment-correct RECURRENT no-spec DECODE oracle (RECOMMENDED, in-process)
Already built: **`scripts/fr13_recurrent_decode_oracle.py`** (option b/c of FR13_ORACLE_FRAME_DECISION §5).
Loads the model ONCE in-process (`vllm.LLM`, **no speculative config** → EagleProposer never constructed →
**structurally cannot hit the crash**), FLASH_ATTN, eager; drives a forced single-step decode loop over
`served_ids` via a per-request AdapterLogitsProcessor that, at each decode step i: records the CLEAN
recurrent argmax + top-k (no streamed-array indexing → the HTTP off-by-one is structurally absent), then
forces `served[i]` so greedy commits it and the recurrent conv/ssm state advances. Every post-prefill step
is `query_len==1` → `num_decodes>0, num_prefills==0` → `_forward_core_decode_non_spec`
(`causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`) = the RECURRENT roll deployment
runs without spec. Class-9 engagement: a monkeypatch counter on `_forward_core_decode_non_spec` must
increment (FAIL LOUD if zero recurrent decode calls) + asserts no speculative config.
```
# smoke (prove the recurrent path fired + reproduce q1's pinned stream):
python3 scripts/fr13_recurrent_decode_oracle.py smoke \
  --prompts output/fr13_acceptance_ladder/prompts_swe4.json --pid 2 \
  --out output/fr13_scan_align_rerun/recur_smoke.json
# rescore (full-stream recurrent flip count for an arm's served stream):
python3 scripts/fr13_recurrent_decode_oracle.py rescore --arm <arm> \
  --src <arm_capture.json> --out output/fr13_scan_align_rerun/<arm>_recur_flips.json
```

### MECHANISM B' — separate non-spec SERVER boot (HTTP, if an HTTP path is required)
If an HTTP oracle is needed instead of the in-process harness, boot a SECOND server WITHOUT spec via
`scripts/fr10_launch_speed_server.sh` with **`FR12_NO_SPECULATIVE_CONFIG=1`** (line 285: drops
`--speculative-config` → no EagleProposer → no crash on non-spec requests). The per-position teacher-force
is STILL a multi-token prefill (still CHUNKED, same frame caveat as A) unless driven as single-token decode
steps — so for the deployment-correct recurrent frame, prefer Mechanism B (in-process). Only the locked
spec build is one boot; a no-spec oracle boot is a SECOND, separate boot and counts against the 2-concurrent
limit (run serialized after the spec arm's capture).

---

## RE-RUN SEQUENCE (binds OFF vs recompute to flips, both oracle frames)

All on THIS-boot streams (no cross-boot byte gate). Two boots serialized (spec arm, then the oracle); never
a third concurrent workflow.

1. **STATE gate (CPU/GPU, observe-only, no oracle needed)** — run `fr13_gdn_scan_warp_gate.py` on the
   deployed payload. CONFIRM `negative_control_powered=True` (STATE int-view flips, both states non-zero).
   Read `off_arm_spine_state_vs_native` (the genuine OFF scan-vs-native-STATE carrier) and the BODY/RECOMPUTE
   STATE results. This is the per-node carrier discriminator (no e2e needed).

2. **Boot the locked cat9 spec server** (`scripts/fr13_launch_locked.sh`); assert tok/draft==9 (#9) +
   within_boot_det [T,T,T,T] (#8). Capture the cat9 served stream for OFF AND for recompute:
   - OFF: `fr13_gold_margin_probe.py capture --mode tree_mtp` with FR13_SCAN_ALIGN UNSET.
   - RECOMPUTE: same capture with the recompute alignment engaged at the EngineCore (the worker-env
     curation drops bare FR13_SCAN_ALIGN — verify `/proc/<EngineCore-pid>/environ` carries it, or re-set it
     via the dist-packages sitecustomize channel as in the prior verify; #9 fail-loud if not engaged).

3. **flips_before(OFF) and flips_after(recompute)** vs the SAME oracle:
   - Mechanism B (deployment-correct, RECOMMENDED): `fr13_recurrent_decode_oracle.py rescore` on the OFF
     served stream → flips_before; on the recompute served stream → flips_after. Re-score BOTH arms (cat9
     OFF, cat9 recompute) AND native E5 vs the IDENTICAL recurrent oracle (the FR13_ORACLE_FRAME discipline:
     re-score BOTH arms; adopt regardless of whether our number improves; class-12).
   - Mechanism A (continuity / reproduce the original 21): `fr13_oracle_stream_teacher_force.py run --mode
     tree_mtp` on each stream → the chunked-frame flip counts comparable to the banked 21/22.

4. **Discriminator:** recompute STATE int-view 0.0-or-floor vs native-packed (step 1) AND flips_after <
   flips_before toward native vs the recurrent oracle (step 3, Mechanism B) + lossless gate (same-seed
   byte-identical OFF stream pristine; regular-decode unaffected) = the scan-align is the carrier win.
   Else quantify the residual. No bake/close/pass-fail until both instruments report (user's call).

---

## Reward-hack statement
Native packed-decode is the A/B ORACLE ONLY (FIX 1) — no served-path splice; the committed kernel
`fr10_gdn_tree_kernel.py` is unchanged (zero git diff from 5e56b7aa). The recurrent oracle (Mechanism B)
changes only WHAT WE MEASURE AGAINST, deployment-correctly, and the FR13_ORACLE_FRAME discipline applies
(re-score BOTH arms vs the same oracle; do not adopt selectively to soften our count; do not declare
"+2 was frame" from a single row). `feedback_no_reroute_reward_hacking`, playbook #9/#10/#12.

## Files (absolute)
- Gate: `/home/mark/shared/lumoFlyWheel/scripts/fr13_gdn_scan_warp_gate.py`
- Native-packed STATE ref (FIXED): `/home/mark/shared/lumoFlyWheel/scripts/fr13_native_packed_decode_ref.py`
- CPU wiring tests: `/home/mark/shared/lumoFlyWheel/tests/test_fr13_scan_vs_native_packed.py`
- Oracle A (HTTP chunked, original 21): `/home/mark/shared/lumoFlyWheel/scripts/fr13_oracle_stream_teacher_force.py`,
  `/home/mark/shared/lumoFlyWheel/scripts/fr13_verify_bisect_probe.py` (the c4 21-flip generator)
- Oracle B (in-process recurrent, deployment-correct): `/home/mark/shared/lumoFlyWheel/scripts/fr13_recurrent_decode_oracle.py`
- No-spec server boot: `/home/mark/shared/lumoFlyWheel/scripts/fr10_launch_speed_server.sh` (FR12_NO_SPECULATIVE_CONFIG=1)
- Pinned image src reader: `/home/mark/shared/lumoFlyWheel/scripts/vllm_src.sh`
- Origin: `/home/mark/shared/lumoFlyWheel/FR13_SCAN_ALIGN_VERIFY_INCONCLUSIVE_BIND.md`,
  `/home/mark/shared/lumoFlyWheel/FR13_ORACLE_FRAME_DECISION.md`
