# FR13 — Conv-State Cross-Event Investigation (CPU read-only, observe-only A/B design)

Date 2026-06-14. CPU-only, read-only investigation. Triggered by the replay-durable-state A/B
(`w2vaqcsmx`, output banked at `output/fr13_fa2_tree_e2e/live/logs/fr13_replay_durable_ab.jsonl`,
3264 records) **refuting the SSM-recurrent durable-state as the back-loaded carrier** of the 21 cat9
flips. The pivot hypothesis to test: is the **conv-state cross-event handoff** (the sliding-window conv
state handed between verify events via `FR13_TREE_CONV_FUSED=1`, baked ALWAYS-ON) the better-supported
carrier?

**HEADLINE VERDICT (skeptical, evidence-grounded): NO. The conv prior-window seam is FIXED and CLOSED in
the current locked build; the conv-state is NOT a back-loaded accumulating carrier; the 21-flip pivot
should go to the GDN recurrent SCAN STATE-FEED realization gap (chunk-vs-recurrent, depth-scaled,
diffuse within-floor) — which the same durable-AB data is also consistent with once you stop treating
its L0 4.17 max as a real durable-state divergence.** Detail + the requested A/B design below.

Playbook rows quoted (per directive): **#12 cross-event = class 4 (spine-only-valid column arithmetic on
branch winners — the c0b53f5d conv fix's own class), and the cross-step contract class 8**;
**#10 codegen-identity (shared-source ≠ shared-SASS)**; **#9 silent/vacuous instrument**.

---

## STEP 1 — CURRENT-BUILD CONV STATUS

### 1a. The conv-state cross-event handoff — OURS vs native, with file:line

**Native `causal_conv1d_update` (spec-decode sliding window)** — the cross-event conv state is a
**sliding window of the last `width-1` taps**, rolled forward at acceptance. From the live image
`/tmp/vllm_live_019/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:845-863`:

```
# After forward:   [history2, ..., historyM, draft1, ..., draftN]
# accept 2 tokens: [history3, ..., historyM, draft1, draft2]
conv_state_token_offset = num_accepted_tokens - 1   (:859-861, spec path)
state_len = width - 1                                 (:565; spec path adds (seqlen-1) -> see 1c)
```

i.e. native consumes the prior window at `conv_state_token_offset` and the result is **bounded to
`width-1` columns** (Qwen3-Next conv width=4 → 3 taps). The conv state is a **fixed-size FIFO of
width-1 tokens**; it does NOT grow with event count.

**OUR conv-fused replay** — `fused_tree_conv_layer` in
`src/lumo_flywheel_serving/fr13_tree_conv_fused.py:571-636`. The next-event conv state is written at the
tail `index_copy_` (`:631-635`) from `fused_tree_conv_state_rows` (`:252-272`), whose static gather table
(`build_tree_conv_state_src_indices`, `:108-172`) implements the legacy closed form (`:120-124`):

```
new_state[i][:, j] = (prior ++ x[path_i] ++ zeros)[path_len_i + j]
  j < width-1 : last (width-1) taps of (prior ++ committed-path tokens of node i)
  j >= width-1: exact zeros (state_len=12 > width-1=3 -> zero pad)
```

The prior window itself is gathered PER EVENT from the **committed path's accepted-leaf NODE column**
(`gather_committed_path_conv_prior`, `fr10_gdn_tree_kernel.py:281-334`, called BEFORE the in-place
`launch_tree_state_linear_remap` permutes the bank). Patcher wiring: imports at
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:779-782`; live conv-read at `~:1907-1916` (per
FR13_NODE5_LADDER_DIFFUSE_BIND:45) uses the same `clamp(num_accepted-1, 0)` column convention as the
forward-SSM-read, replay-publish, and replay-h0 touchpoints (one convention, all four agree — no
fill-vs-read mismatch, NODE5_LADDER:43-46).

**Conclusion of 1a:** OURS writes the SAME logical width-1 sliding-window state as native, derived from
the committed path tokens, in fp32 (native: bf16 in-place roll). The only realization differences are
(i) the bf16-tap MAC + ex2-silu vs native's fused CUDA `causal_conv1d_update`, and (ii) fp32-vs-bf16
state storage (intentionally higher precision; aligning to native bf16 would be the banned reward-hack,
NODE5_LADDER:47-51). The handoff is **per-event width-1-bounded — there is no across-event accumulating
conv buffer** (unlike a true recurrent state).

### 1b. The 18.375 root and what the fix changed

The `conv1d_out = 18.375` capture (`output/fr13_gdn_substate_prompt0_20260609T061732Z`, dated
**2026-06-09**) was **playbook class 4** (spine-only-valid column arithmetic on branch winners):
the legacy prior-window read gathered `spec_state_indices[:, accepted_len-1]` AFTER the in-place
`launch_tree_state_linear_remap` had permuted the bank — linear-column arithmetic valid only for the
SPINE, so a BRANCH winner ([0,2], [0,1,4]) at `num_accepted>1` read the **wrong bank row/cols**
(tree bank row 6 cols [0,1,2] vs native row 1 cols [5,6,7] rolled-tail) → 18.375.

The fix `c0b53f5d` (06-10, `FR13_CONV_COMMITTED_PATH` default ON) reads the **committed path's LEAF NODE
column pre-remap** instead (`gather_committed_path_conv_prior:289-310`), so branch winners commit a window
built from committed-path tokens only; spine winners are **byte-identical to the legacy post-remap read**
(tested, `tests/test_fr13_conv_committed_path.py` test 3; FR13_CONVFIX_AB_BIND §B caveat 5). Subsequent
fixes: `02b1627a` (page-safe conv remap, gather-then-scatter, playbook class 3) and FIX-3 `ef4d7514`
(the fused emulation, byte A/B 283/283).

### 1c. Residual conv divergence in the current build — MEASURED ≈ 0

The 18.375 is **STALE and the bug is FIXED** (FR13_DRIFT_LOCALIZE_BIND:50-61). Post-fix LIVE evidence
that conv is now at the bf16-ULP floor or exactly 0.0:

- **FR13_CONVFIX_AB_BIND §B (06-10, post-fix live):** the deterministic whole-forward corruption that
  was bound at prompt-0 gen_pos 16 (trigger = follows an acc=2 BRANCH commit, the 18.375 class)
  **NO LONGER EXISTS** — p0 now matches native BI=0 to pos 35 (was 16); the old branch-commit site
  accepts 3 (was 2); lockstep fp32 logits argmax-match. The branch-commit conv class is HEALED.
- **FR13_GATEA_DEEP_DIVERGENCE.md:204 ("the conv silu grind SUCCEEDED"):** spread sub-op capture at
  clean-input layers — at L8 `input_hidden=pre_conv=conv1d_out=0.0` (ex2 conv replica bit-exact); at
  L0/L4/L8 `conv1d_out` is **0.0 vs native** (`:189`). The remaining conv delta is at most 1 bf16 ULP at
  one onset edge (L45 fresh capture, `:98`), and even that is shadowed by the GDN scan state-feed which
  diverges FIRST at L8 (`h0_state_in = 0.0007`).
- **FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND.md:7-18 (verify holds=True, wouldFixReduceFlips=False):**
  the conv1d is the L0 first-nonzero ENTRY op but is **row-occupancy M-invariant by construction**
  (per-b Python loop, no cross-row reduction in tap-acc or window gather — branch co-residency does NOT
  perturb the deep-spine conv arithmetic); the num_accepted-driven bank/col selection is FIXED+STALE;
  the surviving conv delta is at most sub-ULP and is ALREADY the live path. A further conv fix is
  predicted **NOT** to move e2e flips.

**STEP 1 verdict: conv is CLOSED in the current locked build.** It is the first op to show a (1-bf16-ULP)
nonzero, but the wrong-bank-row carrier (18.375) is fixed, the branch-commit corruption is healed, and
the residual is at the bf16 floor and M-invariant. The kernel-valid SUBOP_MAB conv arm (patcher
`:1607-1660`) is *designed* to measure exactly this and is *predicted ≈ 0* (FR13_TOTAL_DRIFT_REANALYSIS
_LEADS_BIND:58).

---

## STEP 2 — BACK-LOADING FIT (does conv fit better than the refuted SSM-recurrent?)

The 21-flip fingerprint is **back-loaded** (norm-mean 0.696 = late-stream accumulation), near-disjoint
from native's 3 boundaries (1/18 overlap = a SUPERSET of crossings), corroborated by sglang #25587
(conv-state corruption after partial accept diverging after ~100 tokens).

**Does conv ACCUMULATE across events?** **NO.** The conv cross-event state is a **width-1 (=3) sliding
FIFO** (Step 1a). Each event REBUILDS the prior window from the committed path's leaf-node bank row
(`gather_committed_path_conv_prior`) and writes a fresh width-1 window — there is no buffer that grows
with event count, and any single-event conv error is overwritten within 3 tokens. Conv is therefore a
**flat per-event** state in exactly the sense the durable-AB found the SSM state to be — it CANNOT be the
mechanism that turns per-forward-bit-exact kernels into a back-loaded 21-flip ramp.

**What the durable-AB data actually shows** (re-derived this tick from the banked jsonl, 68 events,
L0 `layers.0.linear_attn`):

| metric | value | reading |
|---|---|---|
| L0 slope per event | **-0.01185** | DECREASING, not growing (matches the WHY note's -0.011) |
| L0 first-half mean / second-half mean | 1.66 / 1.27 | second half LOWER -> not back-loaded |
| L0 nonzero events / acc_len | 68/68, acc_len {5:30, 3:12, 2:11, 4:7, 0:6, 1:2} | every event has a value; magnitude tracks acc_len/ring-gather not event index |
| L0 max | 4.17 | the **harness ring-gather artifact** (per WHY note): a real 4.17 in the committed durable state would garbage the next event, but serving is coherent at 21 small flips |

So BOTH candidate "states" (SSM recurrent durable, conv sliding-window) are **flat/non-accumulating per
event**. Neither is the back-loaded carrier. The back-loading is instead consistent with
**within-forward depth accumulation that scales with accept DEPTH** (longer accepted rank-1 chain =
more chunk-vs-recurrent realization gap), which is deeper later in the stream as the model commits to
longer spines — the GDN scan STATE-FEED realization gap (FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND:20-36,
FR13_NODE5_LADDER_DIFFUSE_BIND verdict), amplified ~32x by gate 1/rms over the deep ~22 layers
(L41-L63), crystallizing at L60/L61. sglang #25587's "~100-token" onset matches stream-depth, not a
conv-FIFO accumulation (3 taps can't carry 100 tokens of corruption).

**STEP 2 verdict: conv does NOT fit the back-loading better than the SSM-recurrent — it fits it the
SAME (badly). Conv is a 3-tap FIFO, not an accumulating state. The back-loaded carrier is the
within-forward depth-scaled GDN scan state-feed realization gap, not a cross-event handed buffer at
all.**

---

## STEP 3 — CONV-STATE CROSS-EVENT A/B DESIGN (built anyway, per directive; observe-only)

Designed reusing the PROVEN `_fr13_replay_durable_ab` harness (patcher `:6475-6602`, call site
`:7647-7661`). This A/B is **low expected value given Steps 1-2** (conv is fixed + non-accumulating);
its purpose is to CLOSE the conv-cross-event question with a direct number rather than leave it inferred.
It distinguishes (i) grows-across-events from (ii) flat per-event from (iii) harness artifact.

### Geometry — PREVENT the device assert (do NOT catch it)

The conv/scan SUBOP A/B device-asserted 5x (`5943d05e`) because the reduced-row M5/M1 tree-slice geometry
over-ran `causal_conv1d_update`'s bank cols: with `num_accepted_tokens` set, native computes
`state_len = width-1 + (seqlen-1)` (live kernel `:831-832`, varlen revises state_len) and the
`conv_state_token_offset = num_accepted-1` read + the `width-1+(m-1)` store span ran past the committed
width-1 window — beyond the Front-B host `_guard_rows`. A device assert is **unrecoverable context
poison**; it must be prevented by construction.

The replay-durable-AB DODGED the analogous scan assert (`:6488-6491`) with a **linear B=1 varlen chain +
NO ssm_state_indices/num_accepted_tokens** (IS_CONTINUOUS_BATCHING=False, IS_SPEC_DECODING=False, the
non-spec/prefill-style path). **Design the conv A/B IDENTICALLY:**

> Native arm = `causal_conv1d_update(x_chain, cs_clone, conv_weights, bias, activation,
> conv_state_indices=<single committed bank>, num_accepted_tokens=None, query_start_loc=[0, M],
> max_query_len=M, validate_data=False)` over the LINEAR accepted chain `[root] ++ accepted-path tokens`
> (M = acc_len+1). With `num_accepted_tokens=None` the wrapper sets `state_len = width-1` and
> `conv_state_token_offset = 0` (live kernel `:862-863`) — the footprint is INDEPENDENT of
> nacc/max_path_len/the physical conv_state column count, so it **provably cannot drive an OOB
> read/store** (exactly the kernel-valid reduced-row reasoning already coded at patcher
> `:1627-1641`). This is the non-spec call; state_len fits by construction. PREVENT, never catch.

This makes the native arm compute the standard depthwise causal conv over the linear chain seeded from a
CLONED committed prior window — the faithful "what should the next-event conv window be" reference.

### Our arm

Per-event, snapshot (CLONE) the conv prior window OUR `gather_committed_path_conv_prior` selected
(`bank_rows`, `fr10_gdn_tree_kernel.py:330-333`) and the width-1 window OUR `fused_tree_conv_layer`
wrote back for the next event (`fr13_tree_conv_fused.py:631-635`, the `index_copy_` destination rows =
`spec_state_indices[b, :tree_n]`). `H_ours_conv` = the width-1 tail OUR replay published at the accepted
leaf's bank row (the column the next event's `gather_committed_path_conv_prior` will READ via
`clamp(num_accepted-1,0)`). Compare `H_ours_conv` vs `H_native_conv` = native's rolled width-1 window
after the linear-chain call.

### The 5 hard-won constraints (all carried verbatim)

- **(a) kernel-valid geometry** — native non-spec linear call, `num_accepted_tokens=None`, state_len fits
  (above). PREVENT the assert.
- **(b) observe-only cloned state** — `cs = conv_state_snapshot.detach().clone()` (mirror patcher
  `:1612`); served bank UNTOUCHED; both arms read-only on the live state.
- **(c) RECORD relative error + state-norm** (the replay A/B LACKED these — it logged only `max_abs`,
  so it could not disambiguate gross-vs-tiny, which is exactly why the 4.17 ring-gather artifact was
  ambiguous). Emit: `max_abs`, `rel_err = max_abs / (||H_native||_inf + eps)`, `norm_ours`,
  `norm_native`, `num_nonzero_taps`, per-tap-column max. FIX THIS HERE.
- **(d) sidecar env + loud stage markers + eager-only** — reuse `_fr13_rdab_emit` (eager-guard
  `:4414-4421`), `_fr13_rdab_stage` loud markers, the sidecar bridge
  `_fr13_write_replay_durable_ab_sidecar` (`:14137`). New flag `FR13_CONV_CROSSEVENT_AB` + its own
  sidecar flag-file + jsonl path. Engagement asserts (playbook #9): header records flag-state; stage
  marker `record-written` on the first record; fail-loud if 0 records.
- **(e) ring-gather convention MUST match the conv replay kernel exactly** — the replay A/B's 4.17 came
  from a ring-gather mismatch (per WHY note). For conv the analogous trap is the WINDOW-COLUMN
  convention: OUR write-back stores the window as `(prior ++ committed-path tokens)[path_len+j]` over the
  fused `source_z = cat(prior.T, x, zero_row)` (`fr13_tree_conv_fused.py:120-124, 180-197`); the native
  arm rolls `[history..., drafts...]` at offset `num_accepted-1`. **Both must be sliced to the SAME
  last-(width-1) physical tap columns** before diffing (native's `state_len=width-1` window vs OUR
  `new_state[:, :width-1]` — column j in `[0, width-1)`; OUR columns `>= width-1` are the zero pad and
  must be EXCLUDED from the diff or both will mismatch trivially). Align on physical tap index, not
  bank column id. This is the conv-equivalent of the ring-gather column convention.

### Where it hooks

Add a `_fr13_conv_crossevent_ab(...)` call ALONGSIDE the existing `_fr13_replay_durable_ab` at patcher
`:7647-7661` (same per-commit `event_index = _FR13_BOUNDARY_EVENT` for the back-loading axis), reading
the same `_fr13_layer._fr13_replay_*` rings/spec_idx/prev_lens + the layer's `conv1d.weight/bias`. Same
4 SWE pinned prompts (`prompts_swe4.json`), greedy seed 1313, B=1 sequential, ENFORCE_EAGER=1,
`FR13_TREE_CONV_FUSED=1` (the deployed path), `FR13_REPLAY_DURABLE_AB=0` (run conv-only to halve cost),
layer filter `FR13_CONV_CROSSEVENT_AB_LAYERS` (default all 48 GDN layers; L0 is the named first-nonzero).

---

## STEP 4 — THE CRITICAL DISCRIMINATOR (grows-across-events test, specified directly)

The A/B must separate (i) conv-state GROWS-across-events (back-loaded carrier) / (ii) flat per-event
(not the carrier) / (iii) harness artifact.

1. **Artifact filter FIRST (iii):** record `rel_err` and `norm_ours`/`norm_native` per event (constraint
   c). If `max_abs` is large but `rel_err` is ~bf16-ULP and `norm_ours ≈ norm_native`, the magnitude is a
   gather/convention artifact (the replay-AB's 4.17), NOT a real divergence. Align the tap-column
   convention (constraint e) and re-derive: a true conv divergence survives column-alignment with
   `rel_err >> bf16 floor`.

2. **GROWS-across-events test (the decisive (i)-vs-(ii)):** the conv window does NOT survive >width-1
   events on its own (3-tap FIFO), so naively per-event conv error cannot accumulate. To test whether
   it COULD be a carrier despite that, run a **fed-forward variant**: take OUR published conv window at
   event N and FEED IT as the native arm's `cs` (cloned) prior window at event N+1 (instead of cloning
   the live committed bank). Define `conv_gap(N) = max_abs(H_ours_conv(N) vs H_native_conv(N))` with the
   native arm seeded from OUR event-(N-1) window.
   - If `conv_gap(N)` is FLAT or shrinks over N (slope ≤ 0, like the durable-AB's -0.0118) **AND**
     `rel_err` stays at the bf16 floor → conv is **flat per-event, NOT the carrier** (predicted outcome,
     consistent with Steps 1-2 and the fact that any error is overwritten within 3 tokens).
   - If `conv_gap(N)` RAMPS with a positive slope tracking the 21-flip back-loading (norm-mean 0.696
     onset) **AND** `rel_err >> floor` → conv-state IS the accumulating carrier (would OVERTURN
     Steps 1-2; flag STOP+REPORT, this contradicts the FIFO mechanism and the GATEA conv-grind success).

   **Prediction (skeptical, evidence-grounded): FLAT, slope ≤ 0, rel_err at floor — conv is NOT the
   carrier.** The mechanism forbids accumulation (3-tap FIFO + per-event committed-path rebuild), the
   GATEA grind already drove `conv1d_out → 0.0` at clean layers, and the durable-AB shows the SAME flat
   signature for the co-resident GDN state.

3. **Cross-check vs SSM:** run the conv A/B and the (existing) SSM durable-AB on the SAME boot/events;
   if BOTH are flat (slope ≤ 0), the back-loaded carrier is NEITHER handed buffer — it is the
   within-forward depth-scaled scan state-feed (Step 2 verdict), and the pivot is settled.

---

## DISPOSITION — where the pivot should go

**Conv is FIXED and CLOSED (Step 1), is non-accumulating (Step 2), and the A/B is predicted to confirm
flat-not-carrier (Step 4).** The conv-state cross-event hypothesis does NOT fit the back-loading better
than the refuted SSM-recurrent — both are flat per-event. Running the conv A/B is worthwhile only to
put a direct number on the conv-cross-event question (it is currently CLOSED by inference, not by a
cross-event A/B vs native), and to demonstrate via `rel_err` that the durable-AB's 4.17-class maxes are
ring/convention artifacts.

**The 21-flip carrier is the within-forward, depth-scaled GDN recurrent SCAN STATE-FEED realization gap**
(rank-1 tree-scan over the co-resident accepted chain vs the chunked-prefill realization of the same
logical state; chunk-vs-recurrent ~1-ULP born at L0, amplified ~32x by gate 1/rms over L41-L63,
crystallizing at L60/L61) — DUAL-VERIFIED at the node-5 carrier event
(FR13_NODE5_LADDER_DIFFUSE_BIND, holds=True) and code-confirmed M-invariant in the bank/wiring
(NODE5_LADDER:40-51). This is **diffuse within-floor, not a single fixable seam**
(reference_diffuse_gdn_accumulation_explained: native same-model fp8 drifts ~7x less = existence proof
it's a realization diff). Levers (no escalation, per feedback_grind_all_fronts):
**(a) scan state-feed bit-exact align** (WY is PARKED — failed abs-0.0 not the within-floor bar; non-WY
sub-levers fp32 state accumulation / op-order / l2norm / raw-g alignment are open) and
**(b) tree-reshape** (shallower committed spine + root-sibling width = less depth-accumulation, the
directive's preferred lever, project_fr13_tree_reshape_unifying_lever).

The durable-AB 4.17 max being a harness ring-gather artifact (not a real durable-state divergence) is
ITSELF a finding: the SSM durable-AB's `max_abs`-only logging (no rel_err) made it ambiguous; the conv
A/B's constraint-(c) rel_err + state-norm fixes that class of ambiguity for any future cross-event A/B.

## Riders / caveats

1. The conv A/B is **observe-only, no reward-hack**: align OUR conv kernel if it diverges (it doesn't,
   per GATEA); do NOT splice native `causal_conv1d_update` into the served path (banned, FR-12
   feedback_no_reroute_reward_hacking; native is the A/B oracle only).
2. Playbook #9: the A/B must fail-loud on 0 records (engagement assert) and record flag-state in the
   jsonl header — the SUBOP conv arm's 5 device-asserted runs produced 0 records and were caught only by
   the loud stage markers; carry both.
3. Playbook #10: the native `causal_conv1d_update` is a DIFFERENT kernel from OUR fused torch-op emulation
   (codegen identity is not spec-guaranteed) — the A/B is the right instrument, but the bar is per-tap
   argmax/within-floor, NOT atol; record int-view rel_err.
4. Playbook #12: the durable-AB `max_abs`-only basis is exactly the "blind to small-rate per-token
   defects / scalar metric" trap — the rel_err + norm + grows-slope (not a raw max) is the binding
   reduce. The L0 4.17 is a measurement trap (ring-gather), not a state divergence.
5. The durable-AB jsonl re-derivation (L0 slope -0.0118, first-half 1.66 > second-half 1.27) is CPU
   reproducible from `output/fr13_fa2_tree_e2e/live/logs/fr13_replay_durable_ab.jsonl`; numbers banked
   here (output/ is gitignored).
6. Conv width for Qwen3-Next = 4 (width-1 = 3 taps); state_len padded to 12 in the tree emulation
   (`fr13_tree_conv_fused.py:124`, zero pad for j ≥ width-1). The FIFO depth that bounds non-accumulation
   is width-1 = 3.
