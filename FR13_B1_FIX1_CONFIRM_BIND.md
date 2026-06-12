# FR13 B=1 FIX-1 Confirm Bind — BI substrate + in-process dual-path proof (DRAFT, not committed)

Date: 2026-06-12 UTC. Executor: confirm workflow agent (serialized GPU).
Purpose: close the 3 OPEN floor items flagged by the holds=True verify of
`FR13_B1_FIX1_GATE_BIND.md`: (a) cat9 t0.6 stream floor underpowered,
(b) chain5 greedy p1 ON fork 15 vs n=1 floor 45, (c) cat9 t0.6 accept ON
1.861 vs OFF band {2.081, 2.168}.

Artifacts root: `output/fr13_b1_fix1_confirm/bi/` (runners
`run_bi_arm.sh`/`run_bi_campaign.sh`/`run_selfcheck_arm.sh`/
`run_selfcheck_campaign.sh`, reducers `step1_substrate_check.py` ->
`step1_verdict.json`, `reduce_confirm_final.py` ->
`confirm_final_reduce.json`, per-arm subdirs with container_env, needles,
probe JSONs, selfcheck stage snapshots, docker_full.log,
`campaign_status.log`).

## STEP 0 — BI-on env set (class 11: identical on EVERY arm)

`BATCH_INVARIANT=1` (launcher -> `VLLM_BATCH_INVARIANT=1` +
`LUMO_BATCH_INVARIANT_VLLM=1`) + `FR13_BI_TREE_ATTN=1` (Method-A TREE_ATTN
BI allowlist + `num_splits=1` BI decode dispatch; launcher asserts both
patches in-container; guard requires `FR13_FA2_TREE_BIAS=1` +
`FR13_FA2_PREFILL_NATIVE=1`, both launcher defaults). Everything else =
the canonical FIX-1 gate regime (PORT=9950, GPU_UTIL=0.82, MAX_NUM_SEQS=1,
FR10_METRICS=0, FR13_REPLAY_ROUTE=1, pinned prompts
`output/fr13_acceptance_ladder/prompts_swe4.json` seed 1313, FULL CUDA
capture proven per boot). Only `FR13_DRAFTER_SINGLE_LOGITS` varies. BI
engagement (class 9) per arm: container env pins asserted + live
in-container `vllm.envs.VLLM_BATCH_INVARIANT == True` + patched-file grep
counts (`bi_engagement_asserts.txt`).

## STEP 1 — substrate proof: BI=1 is NOT cross-boot deterministic at B=1

Two boots, chain5 FLAG-ON BI=1, same seed, same env (zero code diff):
`chain5_on_a` vs `chain5_on_b`. Within-boot rep1==rep2 byte-identical in
BOTH boots, greedy AND t0.6 (class 8 PASS; each boot is an internal fixed
point — rep2 comparisons reproduce rep1 fork positions exactly). Cross-boot:

| probe | first divergence p0/p1/p2/p3 | accept/event a vs b |
|---|---|---|
| greedy (and rep2) | 35 / 11 / 27 / 71 | 2.946565 vs 2.810219 |
| t0.6 (and rep2) | 0 / 25 / 70 / 12 | 2.946565 vs 2.828358 |

Same character as the measured BI=0 cross-boot floor (forks at 11-91,
t0.6 forks at position 0). **BI=1 does NOT determinize cross-boot at B=1**
— the boot-to-boot channel (autotune/kernel-selection class, the L0c
cross-session family) is not covered by batch-invariance, which pins
batch-composition variance within a process. Consistent with
`FR13_METHOD_A_PARTIAL_BIND.md` (B=4 tree non-det under BI), now measured
at B=1 post-cc008587 with within-boot determinism intact. Consequence:
cross-boot EXACT A/B gates are unattainable on this substrate under either
BI flag state; step 2 of the plan (cross-boot exact OFF-vs-ON) is
substrate-impossible, and the fallback below is the decisive instrument.
Side product: the ON-vs-ON pair above is a SAME-FLAG cross-boot floor
measurement with ZERO code diff — every fork position and the 0.118-0.136
accept swing are pure floor.

## FALLBACK (decisive) — FR13_FIX1_SELFCHECK in-process dual-path proof

Implementation (working tree, this workflow):
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` (drafter template: selfcheck
setup + `_fr13_sc_check` after the engagement needle; call sites at the
root step and the loop step, gated `if _fr13_selfcheck:` inside the
single-logits branch), launcher passthrough
`scripts/fr13_launch_forked_fa2_tree_server.sh`
(`FR13_FIX1_SELFCHECK=${FR13_FIX1_SELFCHECK:-0}` + docker -e, plus
`FR13_FIX1_SELFCHECK_DUMP`), tests extended
`tests/test_fr13_drafter_single_logits_wiring.py` (5 passed; legacy-call
count updated 2 -> 4 with the two diagnostic sites accounted).

Semantics: default OFF; DIAGNOSTIC ONLY (never bind =1 into serving/speed).
With the single-logits path SERVING, every drafter step ALSO runs legacy
`self._greedy_sample` (live eagle.py:398-402 =
`compute_logits(hidden).argmax(-1)`, i.e. the second full-vocab lm-head
read) on the same hidden states and raises AssertionError on any token
mismatch (fail loud, class 9); counters {steps_checked, rows_checked,
mismatch_steps} dumped to `/logs/fr13_fix1_selfcheck.json` per check +
log needle every 50 steps. Needs no cross-boot anything.

Boots (canonical BI=0 gate regime — the substrate where the residuals were
observed; FLAG ON + SELFCHECK=1; FULL capture; both engagement needles
asserted in boot log; warmup + greedy x2 + t0.6 x2 on the pinned prompts):

| arm | steps checked | rows | mismatches | within-boot rep1==rep2 | accept greedy | accept t0.6 |
|---|---|---|---|---|---|---|
| chain5_sc | 2915 | 2915 | **0** | greedy True, t0.6 True | 2.659574 (141 ev) | 2.916667 (132 ev) |
| cat9_sc | 3320 | 3320 | **0** | greedy True, t0.6 True | 2.262411 (141 ev) | 1.982659 (173 ev) |

steps_checked covers every live propose call (5 checks/event: root + 4
loop) including boot/capture dummy batches — strictly more than the
metric-counted draft events (labeled per class 12); stage snapshots
(`selfcheck_after_*.json`) are monotone with zero mismatches throughout.

Decisiveness (the induction): at every drafter step, the ON spine token ==
what legacy OFF would select on identical state (proven by the zero-
mismatch dual-path compare, which includes _greedy_sample's second
compute_logits — the ONLY code difference between flag states); leaf
tokens are identical-by-construction (both branches take
`topk(_fr10_step_logits, 2)[:, 1]` from the SAME first-compute tensor);
the extra OFF compute is pure (no RNG/state). Hence by induction over
events, OFF and ON serve BYTE-IDENTICAL streams and EXACTLY equal
accept/event within any boot, greedy and t0.6, both topologies. Every
cross-boot OFF-vs-ON difference is therefore floor, not FIX-1.

## Residual resolutions (all three CLOSED as floor artifacts)

- (a) cat9 t0.6 stream floor (ON fork at 25 vs n=1 OFF-OFF fork at 71 on
  p2): **floor artifact.** In-process identity (cat9_sc, t0.6 trajectories
  included) excludes any OFF-vs-ON within-boot fork; the step-1 same-flag
  ON-vs-ON pair forks t0.6 at {0, 25, 70, 12} with zero code diff — the
  t0.6 cross-boot floor itself forks at position 0.
- (b) chain5 greedy p1 (ON fork 15 vs n=1 floor 45): **floor artifact.**
  chain5_sc zero-mismatch + the step-1 zero-diff ON-vs-ON pair forks
  greedy p1 at 11 — EARLIER than the flagged ON-vs-OFF fork at 15.
- (c) cat9 t0.6 accept 1.861 vs OFF band {2.081, 2.168}: **floor
  artifact.** Drafter semantics proven unchanged => accept/event is
  exactly equal OFF-vs-ON within any boot; the cross-boot accept swing is
  directly measured with zero code diff: ON-vs-ON BI=1 greedy 2.9466 vs
  2.8102 (delta 0.136), t0.6 2.9466 vs 2.8284; and the BI=0 selfcheck boot
  drew cat9 t0.6 accept 1.9827 — between the flagged 1.861 and the OFF
  band, on FLAG-ON code. The t0.6 accept draw range across boots
  (1.86-2.17) brackets the residual.

## Verdict + standing notes

- biDeterministic: **false** (B=1 BI=1 cross-boot — new measured fact).
- byteIdentityExact: proven IN-PROCESS (the only exact instrument this
  substrate admits): 6235 dual-path drafter-step compares across both
  topologies, greedy + t0.6, zero mismatches, fail-loud armed.
- Accept: exactly-equal-by-induction within boot; cross-boot values are
  floor draws (recorded above; class 12 raw counters in probe JSONs).
- Speed: NOT bound here — BI=1 numbers are non-representative by design;
  the speed verdict stays with the BI=0 gate (chain5 1.051x / cat9 1.088x,
  `FR13_B1_FIX1_GATE_BIND.md`).
- cat9 accept vs native (2.15-2.26 band vs 3.1613) remains the B1-2/S3
  superset blocker, unchanged in kind — tracked in FR13_TRAIL.md.
- Instrument banked: FR13_FIX1_SELFCHECK joins the playbook part-2 set as
  the in-process dual-path equivalence vehicle for any future
  "semantics-preserving" speed fix on a non-deterministic-cross-boot
  substrate (pattern: serve the new path, compute the legacy path
  alongside, assert per-step, count into a needle).

## SWE-regime verdict (640bef1c gate regime, early-paused) — WITHIN-FLOOR PASS

Date: 2026-06-12 UTC. Executor: SWE-regime confirm workflow (serialized GPU,
3 boots, zero boot failures, docker clean between arms).
Artifacts: `output/fr13_b1_fix1_confirm/swe/` (per-arm dirs `off/ on/ off_b/`
with container_env, boot/engagement needles, warmup probe + raw-counter
engagement bracket, proxy env + pair/request dumps, full /metrics brackets,
swe_out per-task artifacts incl. codex_trace + vllm_request_metrics,
docker_full.log; runner `run_swe_arm.sh` (+`continue_swe_arm_off.sh`),
wrapper `run_swe_arm_wrapper.py`, reducer `reduce_swe_compare.py` ->
`swe_compare_reduce.json`).

Regime: cat9 tree server, canonical FIX-1 gate env (PORT=9950 GPU_UTIL=0.82
MAX_NUM_SEQS=1 **BI=0 pinned on every arm** [class 11; BI=1 provides no
cross-boot exactness per step 1 above and is slow by design], FR10_METRICS=0,
FR13_REPLAY_ROUTE=1, FULL CUDA capture proven per boot, B=1). Workload =
`scripts/run_swe_bench_q36_a.py` on `astropy__astropy-12907` only, agent wall
EARLY-PAUSED at 540 s; eval + empty-patch retry skipped by wrapper (served
streams are the gate, not the SWE score — x86 offload not exercised).
Recorded sampling config: proxy `LUMO_PROXY_FORCE_TEMPERATURE=0.0` (greedy
byte-identity instrument; the canonical Q36-A forced-temp mechanism set to
the greedy point; every captured request row shows request_temperature=0.0;
no seed field — greedy draws none). Served-stream vehicle: NEW env-gated
proxy pair dump (`LUMO_PROXY_PAIR_DUMP_DIR`, src/lumo_flywheel_serving/
inference_proxy.py) capturing every upstream /v1/responses call (incl.
auto-continue retries) as exact request payload + parsed response.

Engagement (class 9, per arm): container env pins asserted
(FR13_DRAFTER_SINGLE_LOGITS=0/0/1, BI pins, SELFCHECK=0); "Graph capturing
finished" in every boot log; drafter needle with the right state
(off/off_b single_logits=False, on single_logits=True); tree engagement by
RAW counters (spec_drafts warmup delta 6/7/7 — the probe's
--require-tree-engagement vehicle needs trace logs that stay OFF in the
traceless window, so raw /metrics deltas + the needle, which lives inside
the caterpillar drafter block, are the engagement proof); proxy temp +
pair-dump pins asserted from /proc/<pid>/environ.

Result (reducer `swe_compare_reduce.json`, fork attribution per the trail's
design note):
- Request 0 (11,172-token agentic prefill): normalized prompt prefix
  IDENTICAL across all three arms (after stripping upstream-random
  id/call_id and the per-container `x-codex-installation-id` — environment
  identity, not content). **matched_requests=1** per pair; requests 1+2
  prefixes diverge only downstream of the response-0 fork
  (**env_forks=0**; ON tail 17 calls vs OFF 2 = trajectory divergence
  amplification downstream of the first fork, incl. 4 auto-continue
  retries on the ON path).
- OFF vs ON served stream, request 0: **forks at served-token ~22**
  (char 88: "...astropy's modeling " -> OFF "module. ... identify separable
  inputs/outputs" / ON "package. ... compute separability" — near-tie
  synonym flip). byte_identical 0/1.
- **Measured same-flag floor at the SAME regime** (class-11 mismatch rule,
  OFF vs OFF_B, zero code diff): **forks EARLIER, at served-token ~5**
  (char 20: "Let me analyze this " -> "task." vs "SWE-Bench task.").
  floor_rule_eval: on_fork_at_or_after_floor_fork=true => **WITHIN-FLOOR,
  not attributable to FIX-1**. Same structure as the pinned-probe gate
  (floor forks 11-91) and the BI-gate zero-diff pair; the in-process
  dual-path proof (6235 compares, 0 mismatches) independently excludes any
  within-boot OFF-vs-ON difference.
- Accept counters, matched request 0 (raw, class 12; non-like-for-like
  after token 22): off 106/40=2.650, on 102/42=2.429, off_b 134/49=2.735
  per event — ON inside the same-flag spread direction-free.
- s/fwd at the SWE regime (per-request /metrics deltas, B=1 clean rows,
  BI=0; CONTEXT — bound speed verdict stays with FR13_B1_FIX1_GATE_BIND):
  req0/req1 OFF 0.3216/0.3218, OFF_B 0.3206/0.3215, ON **0.2458/0.2464**
  — the FIX-1 ~75 ms/fwd drop REPRODUCES at 11k-token agentic context
  (gate pinned-probe: OFF 0.3118, ON 0.2373; native E5 0.2182).
  Window-aggregate decode_seconds/spec_drafts is INVALID here (named trap:
  request-level histograms only record at request COMPLETION, and the
  off/off_b third call was still mid-generation at the early-pause kill —
  drafts accumulate per step, decode_seconds does not).

Verdict: FIX-1 deployed-regime gate **PASS under class-11 floor semantics**
— at the CUDA-captured SWE-Verified agentic regime the FLAG-ON stream is
indistinguishable from legacy beyond the measured same-flag cross-boot
floor, and the speed win transfers. Note for the FINAL CALL gate: at this
substrate the cross-boot floor forks agentic greedy trajectories within
~5-22 tokens, so the full-30-min-task byte-identity reading MUST be
floor-bracketed the same way (or run pre/post arms within one boot).
