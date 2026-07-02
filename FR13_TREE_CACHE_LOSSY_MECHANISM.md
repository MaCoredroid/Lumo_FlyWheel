# FR13 — Is tree + cache lossy? Mechanism of the cache-on-tree accept penalty (workflow w9lr1zh0u)

**Date:** 2026-07-02. Read-only code + banked-data investigation (7 agents; 3 hit the StructuredOutput
schema cap but the two deepest — cache-seed mechanism + drafter/committer — plus synth + adversarial
verdict completed). Verdict: **undetermined-leaning-LOSSY** — mechanism confirmed, banked instruments
confounded, one decisive gate specified.

## The user's premise is CONFIRMED: the lossless proof is spine-5-only
- cat8 has **sibling-fork groups at depths 0/1/2** (parents `()`,`(0,)`,`(0,0)`, each with 2 children:
  `(0,)/(1,)`, `(0,0)/(0,1)`, `(0,0,0)/(0,0,1)`). Spine-5 has **ZERO** sibling groups (pure linear chain).
  `fr10_tree_has_sibling` (patch:320) = False on spine-5, True on cat8.
- The EXACT_SEED state-diff harness (`fr13_apc_exactseed_statediff.sh`, `ENFORCE_EAGER=1`,
  `FR13_PREFILL_GDN_CAPTURE`) captures the **num_prefills>0 prefill scan ONLY** — it **never** exercises
  the `num_spec_decodes>0` branching-tree verify. The per-path-exact recompute kernel is **OFF** in the
  banked cat8 run. **⇒ the branching sibling-fork under cache has no losslessness proof anywhere.**

## What EXACT_SEED actually restores — and it is NOT bit-exact
- Restores **only the SSM recurrent state** (the `[HV=48,V=128,K=128]` fp32
  `chunk_gated_delta_rule` `output_final_state`; patch:5942-5945, 6358-6364). **Not** conv-state (native
  APC restores that by physical block_id), **not** attention KV.
- Banked eager state-diff (`run_20260629T171132Z`): **6/48 layers exceed the 0.01 threshold** —
  including non-accumulator **deep layers 52–62** (mean_abs 0.011–0.017) — and it is **run-unstable**
  (layer0 max_abs 30.11 vs 50.84 in `run_20260630T181913Z`). The "bit-exact" claim survives only by
  dismissing layer-0 (30.11 = known position-pairing artifact) *and* ignoring the deep layers.

## Why the accept drop AND the lossy risk — the greedy-LCP committer is the crux
- Acceptance = does each drafted token equal the **verify-forward ARGMAX at temp 0** (greedy LCP:
  longest prefix where `draft==parent_target.argmax`; `_lumo_tree_path_lcp_max_greedy_sample` patch:8080,
  8450-8453, 8723). **Explicitly "not the rejection sampler" (patch:8079).**
- The verify argmax rows come from the **default co-resident `h_cache` scan** (`fr10_gdn_tree_kernel.py:
  624-648`), which carries the **measured 0.0289 leaf-co-residency state gap** (kernel:1893) + a
  `-0.0→+0.0` handoff flip (kernel:1004-1008). The bit-exact per-path **recompute** kernel (kernel:776-787)
  is a **separate kernel, OFF by default** (`FR13_SCAN_ALIGN_MODE=recompute` unset in the banked cat8 run).
- **THE KEY POINT:** because the committer is **greedy-LCP, not exact rejection sampling, the standard
  spec-decode output-losslessness theorem does NOT protect this path.** "Lossless" here means only
  `committed-token == target-argmax`; if that argmax is computed from a **perturbed co-resident SSM seed**,
  the committed token can differ from a clean per-path forward, **with no theorem to prevent it.**
- Cache-ON adds the residual-carrying SSM seed feeding the co-resident branching verify → shifts near-tie
  argmax at sibling depths → (i) drops lcp acceptance (slower) **and** (ii) can change the committed
  token (**lossy**). `FR13_REPLAY_ROUTE=1` re-seeds the next step from the accepted chain, so a changed
  accept-count **compounds** along the trajectory.
- **Signature match:** the per-depth continue-rate gap **widens with depth** exactly across cat8's
  sibling-fork depths — depth0 0.938/0.955, depth1 0.812/0.862, depth2 0.762/0.847, depth3 0.680/0.773
  (ON/OFF, `run_20260701T001109Z` per-pos counters).

## Why NOT a confident "lossy" yet (adversarial caveats — both banked instruments are confounded)
1. **Trajectory confound:** cat8_ON vs cat8_OFF free-generated *different* sequences on the same task
   (corrected numbers: ON **19040** draft-tok / **7060** accepted / rate **0.371** / **742.0s**; OFF **6208**
   / **2691** / **0.433** / **504.7s** — both resolved astropy-12907). Accept-rate isn't measured over the
   same token stream. *(The synth mis-stated ON as faster; the artifact shows ON slower — confound is real,
   numbers corrected here.)*
2. **Metastability confound:** the 75% argmax-flip is a **two-boot call-order row compare** on a temp-0
   metastable model that **autotune alone forks** (`fr13_apc_teacher_forced_logit_gate.py` docstring). Neither
   the accept drop nor the flip cleanly isolates a per-position state error.
3. The residual (6/48 layers, 75% flip) is measured on the **prefill** path, not the branching verify — it
   proves the SSM *restore* isn't bit-exact, but is not direct evidence about the branch-fork verify. **The
   branch-fork lossiness has literally zero direct measurement anywhere.**

## The decisive gate (buildable — most instruments already exist)
**Teacher-forced, same-sequence, served-vs-recompute argmax diff on the branching verify, cache-ON, single
boot.** Feed BOTH the served co-resident `h_cache` scan AND the per-path recompute kernel
(`FR13_SCAN_ALIGN_MODE=recompute`, bit-exact-by-construction) the **identical** restored SSM seed + **identical
fixed continuation tokens** (teacher-forced ⇒ a flip fires only on a real per-position state divergence, not
free-gen divergence). For every tree node, compare verify argmax served-vs-recompute, **split by spine vs
sibling-fork nodes** (`(1,)`,`(0,1)`,`(0,0,1)`).
- **FAIL/lossy:** sibling-node argmax flips the spine nodes don't show → branch-fork under cache commits a
  different token than the per-path-exact recurrence.
- **PASS/benign:** zero sibling flips → the accept drop is trajectory/speed only, cache-on-tree is
  committed-token-lossless despite lower acceptance.
- **Oracle-hole fix (verdict):** first validate the recompute kernel against an independent **native per-path
  packed-decode** forward on the identical seed, so a PASS is meaningful (not self-consistent). The recompute
  kernel PASS is currently only a *comment* claim (kernel:783-787), never measured vs native truth.
- Existing tools: recompute kernel (`scan_align_mode=recompute`), `FR13_FORK_MARGIN_DUMP` (sub-1-nat near-tie
  vs >1-nat genuine forks, patch:7398-7422), `fr13_apc_hit_first_divergence.py` (per-position argmax diff).
  Only the same-forward served-vs-recompute twin dump on the num_spec verify needs wiring. Secondary: the
  designed-but-never-run temp-0.6 q-vs-p TV gate on the tree path (FR13_TEMP06_DRIFT_GATE).

## Bottom line
The greedy-LCP committer breaks the spec-decode losslessness guarantee on the tree; EXACT_SEED restores a
non-bit-exact SSM state; the co-resident branching verify (never covered by the spine-5 proof) can commit a
different token under cache. **All arrows point to tree+cache being lossy on the branching fork** — but it is
**not yet proven** to the field's standard because the accept-rate and free-gen-logit-flip instruments are
trajectory/metastability-confounded. The teacher-forced served-vs-recompute sibling-node argmax diff is the
one experiment that settles it.
