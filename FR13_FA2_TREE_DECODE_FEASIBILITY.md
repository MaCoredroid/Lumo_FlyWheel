# FR13 — FA2-fork for DECODE full-attn: feasibility + would-it-close-0.00195

Date 2026-06-14. CPU-only, read-only investigation (concurrent GPU replay-A/B from the same
patcher — no code edited; this doc is the only write). Pathspec commit.

## TL;DR — the premise needs correcting BEFORE the swap can even be proposed

**The forked FA2 (`FR13_FA2_TREE_BIAS=1`, additive `-inf` ancestry bias post-QK) is ALREADY the live
DECODE tree-verify kernel in the locked build. It is NOT prefill-only.** The user hypothesis ("route the
decode full-attn through the FA2-fork instead of TREE_ATTN → close the 0.00195") describes a swap that
is, for the full-attn DECODE tree-verify forward, *already deployed*. There is no TREE_ATTN-Triton
`unified_attention` running on the decode tree rows when `FR13_FA2_TREE_BIAS=1` + `use_tree_bias` (which
the locked launcher sets, with `max_query_len = tree_len = 9 > 1`). So:

- **WHY-PREFILL-ONLY is a false premise** as stated. What IS prefill-only is `FR13_FA2_PREFILL_NATIVE`
  (native FLASH, NO bias) — a *different* flag that aligns the **prefill** path to native. The
  *tree-bias decode* path is the FA2-fork.
- **The 0.00195 TREE_ATTN-vs-FLASH residual is a measurement from the EXP2-Triton TREE_ATTN path**
  (`FR13_TREE_ATTN_EXP2_SOFTMAX`, the `unified_attention` log2 kernel) — the path that runs when
  `FR13_FA2_TREE_BIAS=0`. The locked build does NOT use that path for the tree decode rows. So the
  0.00195 number does not bound the live decode kernel; it bounds the *fallback* Triton kernel.
- **Routing decode through the FA2-fork therefore cannot "close" a residual that the live decode path
  already does not incur via TREE_ATTN-Triton.** The live decode full-attn already runs native-FLASH +
  additive bias, whose floor is the **14/16 whole-tree byte-exact / 2-single-ULP** grouping floor
  (`project_fr13_fa2_fork_nocopy_floor`), which is **~15x below** the E5 self-noise floor.

The honest restatement of the user's question: *"is there still a TREE_ATTN-Triton kernel on any live
full-attn decode rows, and if so should we move it to the fork?"* — answer below. There is one residual
ambiguity (the EXP2 patch coexists with the FA2-bias patch in `tree_attn.py`), resolved in §1.

---

## 1. WHY "PREFILL-ONLY" IS THE WRONG FRAME (root cause, file:line)

### The two patchers and their order (`scripts/fr13_launch_forked_fa2_tree_server.sh:399-400`)
The locked launcher (`scripts/fr13_launch_locked.sh` → `fr13_launch_forked_fa2_tree_server.sh`) copies a
prebuilt forked FA2 `.so` (`/tmp/fr13_fork_fa2.so` → `_vllm_fa2_C.abi3.so`, sha pinned), then runs:
1. `fr10_phase4_patch_vllm_tree_gdn.py` — installs the GDN tree kernel + the
   `FR13_TREE_ATTN_EXP2_SOFTMAX` Triton-softmax patch + the diagnostic op-capture wrapper.
2. `fr13_patch_fa2_tree_bias.py --skip-source` — Python-only (no recompile; the fork `.so` is already
   in place); wires `tree_attn.py` / `flash_attn.py` to call the fork.

### The DECODE dispatch inside `TreeAttentionImpl` (`fr13_patch_fa2_tree_bias.py:569-604`, `_patch_tree_attn`)
The second patcher rewrites the decode branch of `tree_attn.py` to:
```
if decode_meta := attn_metadata.decode_metadata:
    tree_bias = decode_meta.tree_attn_bias
    use_tree_bias = (tree_bias is not None and tree_bias.numel() > 0
                     and decode_meta.max_query_len > 1)          # patcher L611-620
    if os.environ.get("FR13_FA2_TREE_BIAS","0")=="1" and use_tree_bias:
        flash_attn_varlen_func(... tree_bias=tree_bias ...)       # <-- THE FORK, on DECODE
    else:
        unified_attention(... qq_bias=decode_meta.tree_attn_bias ...)  # TREE_ATTN-Triton fallback
```
Locked launcher sets `FR13_FA2_TREE_BIAS=1` (`fr13_launch_locked.sh:24`), and the deployed cat9 tree has
`max_query_len = tree_len = 9 > 1`, with a non-empty ancestry bias. **⇒ the `if` branch is taken; the
DECODE tree rows go through the fork.** The `unified_attention` (`+ EXP2`) branch is the **fallback**,
reached only when `FR13_FA2_TREE_BIAS=0` OR `max_query_len==1` (a pure single-token decode segment).

### Why the EXP2_SOFTMAX patch does not contradict this
`FR13_TREE_ATTN_EXP2_SOFTMAX` (`fr10_phase4...py:12002-12040`) patches `triton_unified_attention.py`'s
softmax to `tl.exp2` + reversed KV iteration. That only affects `unified_attention`. With
`FR13_FA2_TREE_BIAS=1`, `unified_attention` is **not called for the tree decode rows**, so the EXP2 patch
is **inert on the deployed tree-verify path**. It is live only on the fallback. (This is exactly the kind
of "two flags both ON, one shadows the other" trap that `FR13_BUG_CLASS_PLAYBOOK.md` row 9 warns about —
"a run 'passes' while measuring nothing" — here the trap is *the inverse*: the EXP2/0.00195 number is
attributed to a path that is shadowed off. Engagement assert: the boot echoes `TREE_ATTN` as the
*backend name*, which is true — but the backend's decode *impl* dispatches to the fork.)

### The real root of the "prefill-only" wording
- `FR13_FA2_PREFILL_NATIVE` (`fr13_patch_fa2_tree_bias.py:471-560`, `_patch_tree_attn` prefill anchor) is
  the flag that is **prefill-specific**: it routes the *prefill* full-attn through
  `flash_attn_varlen_func(... NO tree_bias ...)` = native FLASH, so prefill is byte-identical to E5.
  Its name ("PREFILL_NATIVE") is almost certainly the source of the "FA2 fork = prefill-only" slip in
  `FR13_TOTAL_DRIFT_REANALYSIS_LEADS_BIND.md:7-9`.
- That bind's claim "Decode backend = TREE_ATTN, FA2 fork = PREFILL-only" conflates the **backend name**
  (TREE_ATTN, true) with the **decode impl** (the fork, when `FR13_FA2_TREE_BIAS=1`). The reanalysis doc
  itself partially self-corrects at `FR13_TOTAL_DRIFT_REANALYSIS.md:166`: *"TREE_ATTN is the live decode
  kernel; the FA2 fork QPAD was on the prefill kernel"* — but that statement is **also imprecise**: the
  QPAD experiment (`9ad6793f`, `FR13_FA2_MDEPENDENT_BIND.md`) re-called the fork's `apply_tree_bias` on
  **decode-event** captured K/V at the p3 deep-accept tree-verify forward (M=10 vs M=5 spine slice). That
  is the **decode** tree-verify path, not prefill. So the QPAD M-dependence was measured ON the
  fork-as-decode-kernel — which only makes sense because the fork *is* the decode kernel.

**Verdict for step 1:** there is no code path that restricts the FA2-fork to prefill. The fork is the
deployed full-attn DECODE tree-verify kernel. "Prefill-only" is a naming/attribution slip
(`FR13_FA2_PREFILL_NATIVE` is the prefill flag) that has propagated into the recent bind. Citations:
`fr13_patch_fa2_tree_bias.py:569-644` (decode fork wiring), `:471-560` (prefill-native wiring),
`fr13_launch_locked.sh:24-26` (both flags ON), `fr13_launch_forked_fa2_tree_server.sh:399-400`
(patch order), `git fe21cb73` ("route TREE_ATTN **decode** through forked FA2 with -inf tree_bias").

---

## 2. CAN THE FORK SERVE DECODE TREE-VERIFY (it already does — itemized)

(a) **Tree-depth positions + tree-ancestry additive bias in the DECODE path** — YES. The bias is
`decode_meta.tree_attn_bias` (`bias[q,k]=0` if k is an ancestor of q else `-inf`), threaded as the fork's
`tree_bias=` arg and added to `acc_s` post-QK pre-softmax (`apply_tree_bias`,
`fr13_patch_fa2_tree_bias.py:26-74`; `set_params_tree_bias`, `:181-214`). Verified true `-inf`
(`FR13_LADDER_LOG.md:56`: `uniq_nonzero=[-inf]`, `exp2(-inf)=0` exact). Depth positions reach RoPE via the
scheduler's depth `position_ids` (the L3 full-attn depth-RoPE wiring fix in MEMORY; not part of this
kernel).

(b) **Paged KV-cache block layout** — YES, and this is the load-bearing answer to step 3. The tree-verify
forward is a **varlen** call against the paged KV: `flash_attn_varlen_func(... block_table=
decode_meta.block_table, seqused_k=decode_meta.seq_lens ...)`. In the live FA2 build,
`flash_attn_varlen_func` dispatches the paged case through the **same** `torch.ops._vllm_fa2_C.varlen_fwd`
op, passing `block_table` + `seqused_k` directly into it (`vllm_flash_attn/flash_attn_interface.py:300-311`,
guard `block_table is None or seqused_k is not None` at `:268`). **There is no separate
"decode paged-attention kernel" vs "prefill varlen kernel" for this path** — the spec-decode tree-verify
is a varlen forward with M=6-10 query rows over paged KV, the same op that prefill uses. The fork's
`varlen_fwd_tree_bias` wraps `mha_varlen_fwd_impl` — the **identical impl** the stock `varlen_fwd` now
calls (`fr13_patch_fa2_tree_bias.py:233-314`: `varlen_fwd` is renamed to `..._impl` and BOTH the
no-bias and tree-bias wrappers call it). So the fork covers exactly the paged block_table layout the
decode tree-verify uses; the tree-bias add is the only delta. (Note: `vllm_flash_attn/...:372` shows the
FA3 paged path is gated on `device_capability_family(90)`; GB10 is family 120, so it takes the FA2
`varlen_fwd` path — the one the fork patches.)

(c) **CUDA-graph capture** — CONFIRMED for this exact build. `FR13_LADDER_LOG.md:164-166`: the forked-FA2
TREE_ATTN tree arm logged `Profiling CUDA graph memory: PIECEWISE=8 (largest=80), FULL=4 (largest=40)`
and `Capturing CUDA graphs (decode, FULL)` completed before startup. The static `tree_bias` buffer is a
fixed-shape input, capture-friendly. So the "TREE_ATTN was chosen partly for capture" concern is moot:
the fork-as-decode-kernel **also** FULL-captures.

(d) **B=4 serve** — CONFIRMED it boots, captures, and serves at B=4
(`FR13_LADDER_LOG.md:28` `MAX_NUM_SEQS=4` FULL decode capture `PIECEWISE=8 FULL=4`; `:162` a real B=4
CUDA-graph e2e ran with `MAX_NUM_SEQS=4`, GPU_UTIL=0.86, returned 1855 tokens). **Caveat (the honest
part):** that B=4 e2e *FAILED the deliverable gates* — accept/event 1.1134 vs E5 3.076, bag-TV 0.50 vs
0.059 floor (`FR13_LADDER_LOG.md:162, 182`). BUT that run **predates `FR13_FA2_PREFILL_NATIVE`** (added
later @ `ea7e4eb0`/`f319be1e`); its failure was the *prefill* full-attn drift (layers 0-11 not byte-exact
to native FLASH), a structural break, NOT the fork's decode tree-bias call (which was 14/16 byte-exact at
the single-event level). A clean B=4 CUDA-graph e2e with `FR13_FA2_PREFILL_NATIVE=1` on this fork-decode
build **has never been measured to a PASS** (the chase moved to the GDN L0 carrier instead).

---

## 3. WOULD ROUTING DECODE THROUGH THE FORK CLOSE 0.00195

**No — because the 0.00195 is not the live decode residual.** Two distinct cases:

- **If "TREE_ATTN" in the question means the EXP2-Triton `unified_attention` decode path** (the path that
  produced 0.00195 = TREE_ATTN-vs-FLASH): that path is **already shadowed off** when
  `FR13_FA2_TREE_BIAS=1` (locked). Swapping to the fork there is a no-op on the deployed config — it is
  *already* the fork. The 0.00195 number characterizes the *fallback* kernel, not the live one.

- **The live decode full-attn residual is the FA2-fork's own floor**, not 0.00195. Per
  `project_fr13_fa2_fork_nocopy_floor` + `FR13_NOCOPY_GROUPING_FLOOR.md`: **14/16 calls whole-tree
  byte-exact 0.0; 15/16 spine byte-exact; exactly 2 single-bf16-ULP elements in ~983k comparisons**
  (max 0.0039), root-caused to **irreducible MMA fp32 fragment-grouping over the scattered no-copy KV**
  (`flash_fwd_kernel.h:367` gemm_rs). This floor is **~15x below the E5 self-noise floor (~0.059)** ⇒
  argmax/distributionally lossless on the full-attn full-attn layers (the theorem-backed branch gate,
  `reference_gdn_tree_branch_oracle_losslessness`). So the deliverable-losslessness residual on the 16
  full-attn layers is **already at the fork floor, already within the E5 floor** — there is nothing to
  "close" by swapping the decode kernel.

**The deliverable miss (the 0.42-0.50 bag-TV vs E5) does NOT live in the full-attn decode kernel.** The
top-down ladder (`FR13_CARRIER_OVERTURNED_BIND`, node7 ladder) puts **first-nonzero at L0 GDN
`linear_attention` (0.0078, 2 bf16-ULP), upstream of the first full-attn layer L3 (0.00409)**. A fix to
the full-attn kernel (FA2-fork or Triton) is **structurally incapable** of removing an L0-GDN-born
divergence (`FR13_BUG_CLASS_PLAYBOOK.md` row: locate WIRING vs KERNEL; here the carrier is upstream of
the full-attn kernel entirely). The FA2-fork query-tile *is* M-dependent (`FR13_FA2_MDEPENDENT_BIND`:
L31 1-ULP, 26/224 sweep cells) but QPAD-fixing it left e2e flips unmoved (24 vs 22,
`FR13_FA2_CARRIER_OVERTURNED_BIND`) = downstream correlate, not the carrier.

**Skeptical caveat (this session's pattern of single-carrier overstatement):** the FA2-tile/BV-warps/
width-H1 hypotheses were each presented as *the* carrier and each OVERTURNED (`8b7684dd`, `8d01ac6d`,
`4842818a`). Routing-decode-through-the-fork would be the same class of mistake — it targets a full-attn
kernel residual that (a) is already the fork, (b) is already within floor, (c) is downstream of the real
L0-GDN carrier. **There is no plausibly-cheap correct path here that closes the deliverable**, so by
`feedback_speed_is_the_goal_cost_gate` this swap should not be built.

---

## 4. PRIOR HISTORY + STANDING RULING (git)

- **FA2-fork-for-DECODE was the ORIGINAL design**, not a later add. `git fe21cb73` (codex_fr14): *"route
  TREE_ATTN **decode** through forked FA2 with -inf tree_bias + per-row/spine FA2 path oracle"*.
  `FR13_FA2_TREE_BIAS_FORK.md:20` build-step 4: *"in tree_attn.py (forward), replace the
  unified_attention(qq_bias=...) Triton call with the forked flash_attn_varlen_func(tree_bias=...)"*.
  The 14/16 byte-exact floor (`2264dd4b`, workflow `w86uygp1x`) was measured on the **decode** strict
  tree run `output/fr13_verify_strict_tree_20260607T091935Z` (`TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`).
- **`FR13_FA2_PREFILL_NATIVE` came LATER** (`ea7e4eb0` 2026-06-08+, `f319be1e`) to fix the prefill drift
  the first B=4 e2e exposed; it is the prefill-specific flag whose name caused the "prefill-only" slip.
- **The FA2-QPAD branch** (`9ad6793f` build, `030a1c22` fix) measured the **fork's decode query-tile**
  M-dependent, then `8b7684dd`/`FR13_FA2_CARRIER_OVERTURNED_BIND` OVERTURNED it as the carrier (QPAD →
  GATE-1 L31 0.0 but GATE-2 e2e flips 24, unmoved; first-nonzero is L0 GDN upstream). Branch
  `fr13-fa2-qpad` stays **UNMERGED/archived**.
- **The standing ruling** (MEMORY `feedback_fr12_subkernel_zero_gate`): *"do NOT patch FLASH_ATTN until
  TREE_ATTN confirmed dead; first check TREE_ATTN cuda-captures+serves at B=4; 0.00195 within E5 floor →
  TREE_ATTN deploy wins, beyond → FLASH_ATTN+tree-mask."* Status against that ruling:
  - **Is TREE_ATTN(=the deployed decode kernel) capturing+serving at B=4?** YES (the *fork* is the
    deployed decode kernel; FULL capture + B=4 serve confirmed, `FR13_LADDER_LOG.md:164-166, 28, 162`).
    A clean B=4 deliverable PASS has NOT been achieved — but the blocker is the L0-GDN carrier, not the
    full-attn kernel.
  - **Is 0.00195 within or beyond the E5 self-noise floor?** 0.00195 (the EXP2-Triton fallback residual)
    is **within** the ~0.059 E5 floor numerically — but it is *moot* for the deployed config because the
    fork (floor 0.0039, also within) is what runs. The actual e2e deliverable miss is bag-TV 0.42-0.50
    (`FR13_LADDER_LOG.md:31, 172`), which is **beyond** the floor and is carried by **L0 GDN**, not the
    full-attn kernel. So "0.00195 within → TREE_ATTN deploy wins" is already satisfied for the full-attn
    sub-question; the deliverable is gated elsewhere.

---

## 5. REWARD-HACK CHECK

**The FA2-fork-for-decode is a LEGITIMATE deliverable, not a reroute/splice.** It is FLASH + an
additive tree-mask bias, computed *in the real serving path*, byte-exactness verified against the
native-FA2-on-path oracle with the **splice OFF** (the fork genuinely computes the tree attention)
(`FR13_FA2_TREE_BIAS_FORK.md:35`, `project_fr13_fa2_fork_nocopy_floor`). The reroute ban
(`feedback_no_reroute_reward_hacking`) is about routing OUR computation *through native to pass a metric
while our kernel stays unused* — that is NOT this: here the fork IS the deployed verifier kernel; nothing
is copied or re-streamed from a native call. **Confirmed legitimate.** The only thing to flag is that the
*question's framing* (swap decode to the fork) is moot, not reward-hacking — the fork is already the
decode kernel, so there is no new deliverable to ship by "swapping."

---

## 6. GPU TEST PLAN — IF a test is still wanted

The swap itself is a no-op on the locked config (the fork is the decode kernel), so a "route decode
through the fork" test would re-measure what is already deployed. Two tests ARE worth the GPU IF the
team wants to nail the full-attn deliverable sub-question independently of the L0-GDN carrier:

**Test A (cheap, confirmatory) — full-attn decode cat9-vs-E5 on the 16 full-attn layers, fork ON.**
Boot `fr13_launch_locked.sh` (fork decode, `FR13_FA2_PREFILL_NATIVE=1`, B=1 eager, hooks ON). Capture
each full-attn layer's `attn_out` for the cat9 tree decode forward vs the native-FLASH-on-path oracle
(spine vs native chain; each branch vs its native-on-path oracle, `reference_gdn_tree_branch_oracle_
losslessness`). EXPECT: 14/16 whole-tree 0.0, ≤2 single-ULP, all within ~0.059. This re-confirms the
full-attn decode residual is at the fork floor (closes the "is 0.00195 the live residual?" question
definitively — it is not; the fork floor 0.0039 is).

**Test B (the actual open gate) — clean B=4 CUDA-graph e2e, fork decode + `FR13_FA2_PREFILL_NATIVE=1`,
cat9 vs E5.** This is the deliverable gate that has NEVER passed cleanly on the prefill-native fork build
(the prior B=4 e2e `FR13_LADDER_LOG.md:162` predates prefill-native). Config: `TREE_ATTN`,
`FR13_FA2_TREE_BIAS=1`, `FR13_FA2_PREFILL_NATIVE=1`, `MAX_NUM_SEQS=4`, CUDA graph (`enforce_eager=False`),
hooks OFF, metrics OFF. Gates: bag-TV vs E5 (`output/fr10_native_mtp5_same8_20260604T210257Z`, accept/
event 3.076) ≤ 0.0593 AND accept/event ≥ 3.076. **Prediction: still FAILS bag-TV**, carried by the L0-GDN
co-residency divergence — so Test B is really a test of the GDN front, not the full-attn kernel. Do NOT
self-declare pass; bring the E5-vs-fork table to the user.

**Do NOT** patch native FLASH_ATTN for decode or build a new full-attn kernel: the full-attn decode
residual is already at the fork floor and within the E5 floor; the deliverable miss is upstream at L0
GDN. Patching the full-attn kernel is the overstated-single-carrier mistake this session keeps overturning.

---

## Bug-class playbook rows in play (`FR13_BUG_CLASS_PLAYBOOK.md`)

- **Row 9 — Silent fallback / vacuous instrument** ("launcher silent-OFF `FR13_FA2_PREFILL_NATIVE`"; "a
  run 'passes' while measuring nothing"; discriminator = "engagement asserts: sentinel in logs, backend
  line, flag in container env BEFORE trusting any number"). Directly relevant: the "FA2 fork = prefill-
  only" slip is exactly a backend-line-vs-impl attribution error; the boot echoes `TREE_ATTN` (true name)
  while the decode *impl* is the fork. Engagement assert = grep the container's patched `tree_attn.py` for
  the `flash_attn_varlen_func(... tree_bias=...)` decode branch + `FR13_FA2_TREE_BIAS=1` in env.
- **Row 10 — Shared-source ≠ shared-SASS** ("byte A/B on captured payloads, int-view equality, SASS hash
  pin"). The fork `.so` sha is pinned (`FR13_LADDER_LOG.md:162`); any "fork == native" claim must be the
  no-bias regular-decode byte-A/B (gate-2, `project_fr13_fa2_fork_nocopy_floor`), not a re-execution
  assumption.
- **Row 12 — Measurement traps** ("non-like-for-like trajectories after fixes"; "single-draw floors
  0.0593 vs measured 0.113"). The prior B=4 e2e fail is class-12 confounded (stream forked, served_lens
  differ) AND predates prefill-native; the 0.00195 attributed to the live decode path is a
  path-attribution trap (it is the *fallback* path's number).

## Files (absolute)
- `/home/mark/shared/lumoFlyWheel/scripts/fr13_patch_fa2_tree_bias.py` (decode fork wiring `:569-644`;
  prefill-native `:471-560`; kernel `apply_tree_bias` `:26-74`; op `set_params_tree_bias` `:181-214`)
- `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py` (EXP2 softmax `:12002-12040`;
  tree_attn decode anchor `:12459-12500`)
- `/home/mark/shared/lumoFlyWheel/scripts/fr13_launch_locked.sh` (`:24-26` both flags ON)
- `/home/mark/shared/lumoFlyWheel/scripts/fr13_launch_forked_fa2_tree_server.sh` (`:397-400` so-copy +
  patch order)
- `/home/mark/shared/lumoFlyWheel/FR13_LADDER_LOG.md` (`:162-166` B=4 capture + the failed e2e; `:31,172`
  bag-TV; `:56,109` fork floor)
- `/home/mark/shared/lumoFlyWheel/FR13_FA2_MDEPENDENT_BIND.md`, `FR13_FA2_CARRIER_OVERTURNED_BIND.md`
  (QPAD A/B on the fork-as-decode-kernel + its overturn)
- MEMORY: `project_fr13_fa2_fork_nocopy_floor`, `feedback_no_reroute_reward_hacking`,
  `reference_gdn_tree_branch_oracle_losslessness`, `feedback_fr12_subkernel_zero_gate`
