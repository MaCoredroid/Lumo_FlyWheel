# FR13 measure — RE-AIMED to the DEPLOYMENT regime (real SWE-Verified + codex)

**Date:** 2026-06-15. **Convention:** EXTENDS `scripts/fr13_measure.py` /
`scripts/fr13_measure_orchestrate.sh` (NEVER a new fork —
`feedback_canonical_speed_lossless_infra`). CPU re-aim (no GPU); the GPU stages
delegate to the already-proven big-denom machinery.

## The regime bug this fixes (the headline, confirmed in the data)

The previously-canonical `fr13_measure` regime sent the handrolled
`prompts_swe4.json` as a **RAW string to `/v1/completions` with NO chat
template**. That is **OFF-DISTRIBUTION** for this chat/thinking-trained model:

- native E5's served stream on prompt 0 REPEATS the block
  `[271,248068,271,248069,271,40]` = `"\n<think>\n</think>\nI"` at served
  positions 0/27/58 — a **degenerate empty-`<think></think>` repetition loop**
  (verified `output/fr13_measure/native_e5_q_temp06_on.json`;
  `served_tokens[:12] = ['\n\n','<think>','\n\n','</think>','\n\n','I',...]`).
  `prompts_swe4[0]` itself has NO `<think>` tags.
- This off-distribution degeneration (**NOT a kernel bug**) is what tanked native
  accept to ~1.589 and forked the stream cross-boot (the GB10 near-tie).
- The no-spec recurrent oracle ranks the **coherent** continuation correct by
  ~11 nats — so the real model decode is coherent; **only the off-distribution
  raw-prompt spec boots degenerate.**

## The fix (user): measure on the DEPLOYMENT regime

Measure on **real SWE-Verified + codex** = the codex agent loop on real
SWE-bench-Verified tasks, **chat-templated via `/v1/responses`, multi-turn, real
tool calls**. The big-denom ALREADY proved this regime **faithful +
representative**: codex on `astropy-12907` gave native ≈ cat9 (13.99% vs 13.55%
clear-margin flips, NO degenerate loop, spec-vs-non-spec CONFIRMED). So the
**deployment regime is the canonical one**; the raw-`/v1/completions` handrolled
path is **DEPRECATED**.

## The canonical path — the 4 deployment numbers on the deployment trajectory

| number | subcommand | basis | instrument |
|---|---|---|---|
| s/fwd | `deploy-speed` | `d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)` | OFF |
| accept/event | `deploy-speed` | `d(spec_decode_num_accepted_tokens_total)/d(drafts)` (B-dependent, deployment trajectory) | OFF |
| committed/event | `deploy-speed` | accept/event + 1 | OFF |
| derived TPS | `deploy-speed` | committed/event ÷ s/fwd (DERIVED, not measured) | OFF |
| clear-margin flip rate + Wilson CI | `deploy-lossless` | each arm vs its OWN no-spec RECURRENT decode oracle; native-E5 = the within-floor BAR | ON |

**The key plumbing realization:** `scripts/run_swe_bench_q36_a.py` (the codex
loop) ALREADY brackets `/metrics` per task into `vllm_metrics_pre.txt` /
`vllm_metrics_post.txt` (lines 609/627/669; the four counters
`request_decode_time_seconds_sum`, `spec_decode_num_drafts_total`,
`spec_decode_num_accepted_tokens_total`, `spec_decode_num_draft_tokens_total` are
all present). So the deployment-regime s/fwd + accept/event is the **same
raw-counter delta basis** as the deprecated `cmd_speed` — only the trajectory
differs (real codex loop, no degenerate fork). `cmd_deploy_speed` reduces those
brackets; nothing is re-measured.

### Smoke results on the proven big-denom run (this session, CPU reduce)

```
deploy-speed cat9      : s/fwd 0.2481, accept/event 3.685, committed 4.685, TPS 18.88  (tok/draft=9, engaged)
deploy-speed native_e5 : s/fwd 0.2334, accept/event 3.267, committed 4.267, TPS 18.28  (tok/draft=5, engaged)
deploy-lossless        : cat9 1181/8717=13.548% [12.846,14.283] vs native 1224/8752=13.985% [13.275,14.728]
                         -> LOSSLESS_within_floor (Wilson CIs overlap), within_proc_det both
```

Note the deployment accept (cat9 3.685 / native 3.267) is **healthy on the
deployment trajectory** — the off-distribution raw-prompt 1.589 was the
degenerate fork, not the real model.

## How the big-denom machinery is INTEGRATED (orchestrated, not re-invented)

`fr13_measure` (via `fr13_measure_orchestrate.sh`) ORCHESTRATES the proven
deployment harness; it does not re-implement any of it. The serialized GPU chain:

1. **`scripts/fr13_bigdenom_swe_serve.sh <arm_dir> <cat9|native> <subset>`** —
   boots the arm (cat9 via `fr13_launch_locked.sh` with the baked pad fix LIVE /
   native via `fr10_launch_speed_server.sh` `num_spec=5`) + runs
   `run_swe_bench_q36_a.py` (the codex agent loop) with the proxy pair-dump ON
   (`LUMO_PROXY_PAIR_DUMP_DIR` via `inference_proxy.py`) + raw-`/metrics`
   spec-engagement asserts + per-task `/metrics` brackets. Class-9 flag-live +
   worker-`/proc/environ` + CUDA-capture + pair-dump non-vacuity asserts.
2. **`fr13_measure deploy-speed`** — reduces the per-task `/metrics` brackets →
   the 4 OFF speed numbers on the real codex trajectory. CPU.
3. **`scripts/fr13_bigdenom_phase3_rescore.sh`** —
   `fr13_swe_stream_to_oracle_src.py` (byte-exact detok of the proxy pair-dump →
   oracle `--src`, class-#12 round-trip-validated denominator) →
   `fr13_recurrent_decode_oracle.py rescore` (the **no-spec RECURRENT decode
   oracle** = the lossless flip, `FR12_NO_SPECULATIVE_CONFIG=1` so spec counters
   cannot advance) → `fr13_bigdenom_rescore_consolidate.py` (Wilson 95% CI +
   non-vacuity gate).
4. **`fr13_measure deploy-lossless`** — reads the consolidation → the within-floor
   lossless verdict (cat9 vs native-E5, each vs its OWN no-spec oracle). ON.

Orchestrator entry points (in `fr13_measure_orchestrate.sh`):
`deploy-serve` (one arm, GPU), `deploy-speed` (reduce brackets, CPU),
`deploy-rescore` (Phase-3 + lossless, GPU), `deploy-full` (end-to-end serialized).

## Truthful accounting + ON/OFF + B=1/B=4 + cat-shapes (carried forward, unchanged)

- **s/fwd** = `d(request_decode_time_seconds_sum)/d(spec_drafts)` during the real
  codex workload (decode-only, per-event, ~B-invariant). NEVER TPS/accept/wall.
  Banned bases still blocked + asserted (`assert_speed_basis`).
- **accept** = `d(accepted)/d(drafts)`, B-DEPENDENT, now **on the deployment
  trajectory** (no degenerate fork). **committed** = accept+1; **TPS** DERIVED.
- **INSTRUMENT ON/OFF:** SPEED only from clean-OFF (`FR10_METRICS=0`, no
  q-capture/probe = the codex run as deployed); lossless from ON (the recurrent
  oracle rescore). `assert_no_mode_mix` enforces it per record
  (`deploy-speed` = OFF, `deploy-lossless` = ON).
- **B=1 + B=4:** `deploy-speed --batch-size` labels the co-residency regime the
  arm was booted in (`MAX_NUM_SEQS`); B=4 is a separate serve boot. accept/event
  is recorded with the batch_size (B-dependent).
- **Any cat shape:** the tree arm's `--expected-tok-per-draft = len(TREE)` is the
  class-9 engagement gate (cat9 = 9); a fail-loud raise if the served shape ≠
  expected (silent fallback / unbuilt shape).
- **The temp-0.6 (q,p) drift + q-by-token-id fix + paired teacher-forced accept**
  remain available (`capture-q` is id-keyed via `top_logprobs_ids`; `temp06-drift`
  aligns by token id; `paired-accept` is fork-immune on a common oracle
  trajectory). These are the deprecated raw-regime instruments; the deployment
  lossless verdict is the big-denom recurrent-oracle clear-margin flip rate.

## DEV-iteration vs FINAL-judgment split (user)

- **DEV-iteration** = the **cheapest deployment-faithful proxy**: a SHORT codex
  run / few turns (a single-task subset, e.g. `subset_astropy12907.json`), NOT
  raw prompts. Cheap deploy-speed + a small-denom flip check per change.
- **FINAL judgment** = **real SWE-Verified + codex, B=4 + CUDA-captured + 4
  tasks + ~30 min**, lossless **re-confirmed at B=4** (B=4 changes
  co-residency). The deployable gate the user set:
  `deploy-full <4-task-subset> 4`.

## The deprecated raw-`/v1/completions` path (off-distribution cautionary note)

The `speed` / `capture-q` / `temp06-drift` / `bag-tv` / `paired-accept` /
`reconcile` / `diag-residue` subcommands (and the orchestrator
`native`/`tree`/`drift`/`paired`/`reconcile` commands) drive the raw-string
`/v1/completions` regime. They are **DEPRECATED** — kept ONLY as:

1. a documented **off-distribution cautionary note** (the `<think></think>` loop
   above is the cautionary artifact), and
2. (flagged) for a **regime-robust s/fwd cross-check** — s/fwd is
   bandwidth-bound and ~trajectory-invariant, so a raw-regime s/fwd is a sanity
   cross-check for the deployment s/fwd; **never** the deployment accept or the
   lossless verdict.

The module docstring + `SUBCOMMANDS` section in `fr13_measure.py` label these
DEPRECATED; the deployment numbers come from `deploy-speed` / `deploy-lossless`
ONLY.

## Files

- `scripts/fr13_measure.py` — `cmd_deploy_speed` + `cmd_deploy_lossless` added;
  docstring + SUBCOMMANDS re-aimed; raw path labelled DEPRECATED.
- `scripts/fr13_measure_orchestrate.sh` — `deploy-serve` / `deploy-speed` /
  `deploy-rescore` / `deploy-full` orchestrate the big-denom machinery; raw
  commands labelled DEPRECATED.
- Reused (not re-invented): `fr13_bigdenom_swe_serve.sh`, `run_swe_bench_q36_a.py`,
  `inference_proxy.py` (proxy pair-dump), `fr13_swe_stream_to_oracle_src.py`,
  `fr13_recurrent_decode_oracle.py`, `fr13_bigdenom_phase3_rescore.sh`,
  `fr13_bigdenom_rescore_consolidate.py`.

## LIVE deployment validation (2026-06-15, HEAD `cab6c157` + serve `AGENT_WALL_S`)

The deployment harness was driven END-TO-END on the current HEAD to confirm the
served streams are **COHERENT** (no degenerate `<think></think>` loop) and that
the deployment-regime metrics reduce correctly + reconcile with the big-denom.

### What ran (GPU, this wf as the only GPU user)

- **Fresh bounded cat9 arm** `output/fr13_bigdenom_swe/cat9_dev` —
  `AGENT_WALL_S=360 fr13_bigdenom_swe_serve.sh cat9_dev cat9 subset_astropy12907.json`
  on HEAD `cab6c157`. Booted via `fr13_launch_locked.sh` (cat9 num_spec=9
  TREE_ATTN, **CUDA graph captured** — "Graph capturing finished in 7 secs"),
  healthy after 437s, **spec engagement OK: draft_tokens/drafts = 9.0** (cat9
  TREE live), proxy pair-dump ON (forced temp 0.0). Codex agent loop on the real
  SWE-Verified task `astropy__astropy-12907` via `/v1/responses`, multi-turn,
  real `exec_command` tool calls. `ARM_DONE swerc=0 health_rc=0 pair_nonempty=1`,
  wall 725s, **11 pair-dumps**, clean teardown + recover.
  - A 360 s bounded codex CANNOT finish the full astropy fix (the big-denom
    needed the full 1500 s), so the bounded arm's task verdict is `failed`
    (`patch_bytes 0`) — EXPECTED for a DEV-iteration proxy. The deliverable here
    is COHERENCE + healthy metrics, NOT a resolved patch.

### Served streams are COHERENT (the headline)

Full scan of all 11 cat9_dev pair-dumps: **33,132 served chars, 7 tool-calling
turns, ZERO empty `<think></think>` blocks**. The agent reasons about the real
bug (e.g. turn 2 = 13,117 chars: *"Now I understand the code. Let me trace
through the issue: The `_separable` function ... when it encounters a
CompoundMod..."*; later *"In the `_cstack` function, when handling the right
operand that's already a coord_matrix (ndarray) ..."*) and issues real
`exec_command` calls (`find`, `cat separable.py`). This is the deployment regime
working as intended — **NO degenerate loop**, in stark contrast to the
off-distribution raw-`/v1/completions` artifact (native E5 record[2] literally
re-opens the empty block: `[271,248068,271,248069,271,40, ... ,271,248068,271]`
= `\n\n<think>\n\n</think>\n\nI have read the task.\n\n<think>\n\n`).

### Deployment-regime metrics (measured + reconciled)

`deploy-speed` reductions (OFF, B=1, class-9 engagement asserted, tok/draft=9):

| arm | source | s/fwd | accept/event | committed | derived TPS |
|---|---|---|---|---|---|
| native_e5 | big-denom `native_a` (full 1500 s) | 0.2334 | 3.267 | 4.267 | 18.28 |
| cat9 | big-denom `cat9_a` (full 1500 s) | 0.2481 | 3.685 | 4.685 | 18.88 |
| cat9 | **fresh `cat9_dev` (bounded 360 s, HEAD cab6c157)** | **0.2404** | **3.240** | **4.240** | **17.64** |

**Reconciliation with the big-denom:**
- **s/fwd reconciles tightly** (0.2404 vs 0.2481, ~3%): s/fwd is bandwidth-bound
  and ~trajectory-invariant, exactly as the regime-robust-cross-check note
  predicts.
- **accept/event lands in the same HEALTHY 3–4 band** (3.240 vs 3.685, both far
  from the off-distribution 1.589 degenerate fork). The small delta is the
  B-/trajectory-dependence: the bounded arm hit its wall mid-task with more
  exploratory file-reading turns (different acceptance than the full task) —
  expected for a deployment-trajectory accept.
- **Engagement identical:** tok/draft = 9.0 on both (cat9 TREE), 5.0 native.

`deploy-lossless` (ON) on the big-denom consolidation is unchanged and holds:
native ≈ cat9 (**cat9 1181/8717 = 13.548 % [12.846, 14.283]** vs **native
1224/8752 = 13.985 % [13.275, 14.728]**), Wilson CIs OVERLAP →
**`LOSSLESS_within_floor`** (native-E5 = the BAR; cat9 NOT separated above),
`within_proc_determinism_both`. This is the per-token clear-margin instrument
vs each arm's OWN no-spec RECURRENT decode oracle, the binding lossless gate.

### One serve-script change (behavior-preserving)

`scripts/fr13_bigdenom_swe_serve.sh` now honours an optional `AGENT_WALL_S` env:
when UNSET it is the proven full 25-min deployment wall
(`run_swe_bench_q36_a.py` `DEFAULT_AGENT_WALL_S = 1500`); when set it passes
`--agent-wall-s` for a DEV-iteration BOUNDED deployment run (the /metrics
brackets still wrap the real bounded codex trajectory → a deployment-faithful
`deploy-speed`). The FINAL gate is unchanged: `deploy-full <4-task-subset> 4`
(B=4 + CUDA-captured + 4 tasks + ~30 min, lossless re-confirmed at B=4).

### Bottom line

The deployment regime is the canonical one and it is **coherent + faithful** on
the current HEAD. The four deployment numbers reduce correctly from the real
codex trajectory; the off-distribution raw-prompt 1.589 degenerate fork is gone.
