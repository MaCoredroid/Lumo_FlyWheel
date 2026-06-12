# FR13 Superset-Chase STEP-1 Bind — H1 ROWBUG CONFIRMED IN-PROCESS (Verdict A)

Date: 2026-06-12 UTC. Repo @ HEAD 41629b07 (branch main; instrument edits +
this bind left UNCOMMITTED for the monitor). Plan = wf_c43b084f Step 1
(`research/fr13_workflows/superset_chase_plan_wf_c43b084f.raw.json`); prep =
the CPU prep report (instruments i-v, FR13_CHASE_DIAG flag, rescued reducers
in `scripts/fr13_chase/`).

## Verdict (decision rule applied mechanically)

**(A) H1 CONFIRMED**: per-event INTEGER records show the drafter is seeded
from the stock linear row (`sampled_row_offset == prev_accepted_len`,
164/164 decode events) which differs from the committer's published accepted
LEAF flat verify row on **84/84** ROWBUG-class decode events (0/80 mismatches
on rowOK events) — while the GDN state handoff is **byte-equal as-read vs
as-written on 160/160** joined events (incl. all 82 events immediately after
a ROWBUG commit); 0 BYTE_DIFF, 0 READ_ROW_NOT_WRITTEN, 4 NO_PRIOR_WRITE (the
first verify of each request — expected, nothing written yet). Branch (B)
state-handoff is **excluded** for the h0 carrier on this evidence; the plan's
"state equal => drafter topology/position mapping" branch is now proven with
the precise mechanism. Next step = **FIX-A (Step 2)**.

- S2-on-current (the 06-11 gate's declined item 2b, closed here): **near-tie
  floor-class**, not gross corruption.
- H6 conv-prior byte check (MANDATORY per planVerify caveat 1): **clean**
  (dual-path selfcheck mismatch=0 across 8 stages x 16000 checks).

## Campaign (3 boots, serialized, launcher-only; artifacts `output/fr13_chase_step1/`)

Regime: canonical FIX2/FIX3-bind regime — PORT 9950, GPU_UTIL 0.82,
MAX_NUM_SEQS=1, BATCH_INVARIANT=0, FR13_BI_TREE_ATTN=0, FR10_METRICS=0,
FR13_REPLAY_ROUTE=1, FIX-1/2/3 at committed defaults on every arm; pinned
prompts `output/fr13_acceptance_ladder/prompts_swe4.json`, seed 1313, B=1,
greedy t0.0/top_p1.0, 128 tok (warmup 1x16), probes x2 reps; docker rm -f
between arms; per-probe window snapshots (copy+truncate O_APPEND jsonl, the
s1s2s3 `snapshot_window.sh` pattern). Runner
`output/fr13_chase_step1/run_chase_arm.sh`; reducer
`output/fr13_chase_step1/reduce_chase_step1.py` ->
`chase_step1_reduce.json`. The reducer's event-walk/fingerprint machinery was
validated against the BANKED gate boot first (reproduces 2.151515
accept/event cell-exact and the 51%/0.53-vs-0.897 fingerprint).

| arm | mode | boot | needles (class 9) | gates |
|---|---|---|---|---|
| cat9_clean | captured ("Graph capturing finished" asserted) | healthy 448s | FIX-1/2/3 + chase=0 inert, both chase needles fired | same-seed repeat PASS (4/4 prompts byte-identical greedy vs rep2) |
| chain5_clean | captured | healthy 442s | same | same-seed repeat PASS (4/4) |
| cat9_diag | EAGER (ENFORCE_EAGER=1; "Graph capturing finished" ABSENT asserted) + FR13_CHASE_DIAG=1 + FR13_TCF_SELFCHECK=1 + FR13_FINAL_LOGIT_CAPTURE (fp32, NUM_TOKENS=10) | healthy 377s (attempt 2) | chase=1 armed both needles; instruments emitting asserted post-warmup | diag-only; accept recorded never gated |

Boot-failure record (same-reason-twice rule): cat9_diag attempt 1 died at the
first tree event — **a bug in the NEW CV tap** (instrument v):
`gather_committed_path_conv_prior{,_prepared}` return `read_node_cols`/
`bank_rows` as **[B,1]** (kernel-lib contract), my tap walked `int(_x)` over
a nested `tolist()`. Fixed with `.reshape(-1)` (patcher CV tap, comment
pinned to the crash); wiring tests 24/24; attempt 2 clean. Not a serving bug
— clean arms unaffected (tap is chase-gated).

## 1) Fresh re-baselines (clean captured boots; accept = DRAWS, never gates)

| arm | accept/event (greedy, x2 reps identical) | events | root-reject% | accepted_len hist 0..5 |
|---|---|---|---|---|
| cat9_clean | **2.0274** (296/146) | 146 | **39.7%** | 58/13/15/17/13/30 |
| chain5_clean | **2.9612** (382/129) | 129 | **12.4%** | 16/20/20/18/7/48 |

cat9 draw 2.0274 sits in the standing 2.01-2.26 band; chain5 2.9612 inside
the post-FIX-2 same-flag band [2.6596, 3.0078]. Native E5 bar = 3.16
(not re-run; banked).

Fingerprint (transition table, fresh CLEAN cat9 boot, greedy window, 142
transitions; observational/class-12):

| class | root-match | rate | mean next spine LCP |
|---|---|---|---|
| ROWBUG transitions (prev L=2-alt or L>=3) | 28/67 | **0.418** | 1.12 |
| rowOK transitions (prev L in {0,1,2-spine}) | 60/75 | **0.800** | 2.53 |
| chain5 control (all rowOK) | 113/125 | **0.904** | 3.06 |

ROWBUG transition share fresh = **47.2%** (clean) / **50.3%** (diag boot) vs
banked 51% — the banked fingerprint REPRODUCES on today's defaults
(FIX-1/2/3 ON). Worst cells fresh: L3-alt 0/9, L3-spine 2/8, L4-spine 2/4,
L5-spine 10/24. In-process rowbug share of decode events (diag) = 84/164 =
51.2%.

Fresh pre-fork counters (cat9_clean vs chain5_clean, same served prefix
before first output fork — the bind-table refresh; denominator problem
reproduces):

| prompt | fork pos | cat events before | chain events before | cat accepted before | chain accepted before |
|---:|---:|---:|---:|---:|---:|
| 0 | 35 | 10 | 9 | 19 | 25 |
| 1 | 15 | 6 | 6 | 6 | 6 |
| 2 | 57 | 17 | 15 | 39 | 39 |
| 3 | 68 | 21 | 18 | 44 | 49 |

## 2) Instrument (i) — per-event INTEGER row records (in-process, diag boot, greedy window)

168 propose events; 164 decode + 4 prefill proposes. The 4 prefill rows are
excluded as **instrument-vacuous** (sample row = last prompt token,
offset 680/1079/828/1613 = prompt_len-1; `_LUMO_FA_SAMPLER_ROW_REQ_IDS` only
updates on decode steps so the REQKEY join is stale there — class-12 trap,
recorded as `prefill_stale_join=4`, not a serving signal).

- stock math confirmed: `sampled_row_offset == prev_accepted_len` **164/164**.
- ROWBUG events (published leaf flat row != prev_accepted_len): **84**;
  `h1_row_mismatch` true **84/84**; false on **0/80** rowOK events.
- example (the H1 mechanism verbatim): prev accepted `[1,2,4]` (= node path
  [0,1,3], L=3) -> stock samples flat row 3 = the REJECTED (0,1)-alt; the
  published leaf row is 4.

## 3) Instrument (ii) — GDN state parity B_JOIN (in-process byte verdicts, layer 0 linear_attn h0)

| verdict | count | of which prev_rowbug=True |
|---|---:|---:|
| BYTE_EQUAL | **160** | 82 |
| BYTE_DIFF | 0 | — |
| READ_ROW_NOT_WRITTEN | 0 | — |
| NO_PRIOR_WRITE | 4 (first verify per request, expected) | — |

The committer/replay-published GDN h0 the next verify READS is byte-identical
to what was WRITTEN, including immediately after every ROWBUG commit — the
deficit carrier is NOT the GDN state handoff (h0 carrier; classes 4/7/8
excluded on this evidence for this seam).

## 4) S2-on-current piggyback (instrument iii + fp32 final-logit capture)

Capture/event alignment: 164 captures = 164 lcp rows; alignment audit 40/40
(argmax(row0) == parent_target_ids[0]). Baseline root top1-top2 fp32 gap:
median 5.5, p10 1.125 (n=161).

First-fork events vs chain5_clean (diag boot streams; cross-boot fork
position observational, margins in-process):

| prompt | fork pos | event winner/L | fork row | served vs chain token | fp32 gap top1->chain | chain rank |
|---:|---:|---|---:|---|---:|---:|
| 0 | 67 | [0,2] L=1 | 1 | 9764 vs 71093 | 0.625 | 1 |
| 1 | 11 | spine L=5 | 8 | 12182 vs 26622 | 0.125 | 1 |
| 2 | 21 | [0,1,3,5,8] L=5 | 4 | 1970 vs 3425 | 1.375 | 1 |
| 3 | 61 | [0,2] L=2 | 1 | 20049 vs 1901 | 0.125 | 1 |

All four: argmax==served (self-consistent verify), chain token is the
IMMEDIATE runner-up (rank 1), gaps 0.125-1.375 — at or below the baseline
p10-median — **near-tie floor-class. The episodic-corruption front stays
closed.** (Gross would be: chain token far down-rank with >5x-baseline gaps.)

## 5) H6 — conv-prior byte check (MANDATORY; FIX-3 adjudication hook)

`FR13_TCF_SELFCHECK=1` on the diag boot: **mismatch=0** at 16000 checks in
EVERY stage (committed_read_cols / committed_bank_rows / committed_prior_bank
/ remap_dst_permutation / conv_window / conv_acc / conv_out / conv_new_state);
zero `FR13_TCF_SELFCHECK MISMATCH` lines in the full docker log. CV tap
recorded 164 rows (read_col/bank_row/sha4096 of the prior window AS-READ).
**H6 clean under FIX-3-ON live.**

## 6) Instrument (iv) — drafter-KV row hashes (banked for FIX-A2 decision)

168 records (event_idx, layers, rows_window, seq_lens) in
`cat9_diag/win_greedy/logs/fr13_chase_drafter_kv.jsonl`. No chain5 diag
boot was spent (arm 4 conditional; verdict A did not need the control's
integer records). The H2 cat-vs-chain hash comparison is DEFERRED to the
FIX-A gate campaign (the plan's FIX-A2 decision input), where a chain5 diag
boot can ride along.

## Next step — FIX-A (plan Step 2), on fresh working-tree coords

FIX-A1 (core): flag `FR13_TREE_SAMPLE_ROW` (default OFF until gated):
`token_indices_to_sample = base + (len>0 ? paths_buf[b, len-1] : 0)` — the
+1-shifted published node id IS the flat verify row, already in the
persistent device buffers the REQKEY pre-forward rewrite installs
(`_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR`/`_LENS_TENSOR`, committer refill WT
:5723-5821; rewrite `_patch_gpu_model_runner_tree_reqkey` WT :8015, anchor
`num_decode_draft_tokens.copy_to_gpu`) => one gather, no CPU sync,
capture-legal, drafter-agnostic. Patch vehicle: the eagle consumption patch
(`_patch_eagle_tree_consumption_verify` WT :8997; stock-selected
`sample_hidden_states` consumed at root logits WT :9179/:9188; trace wrapper
:9957/:9964 stays untouched). CHAIN-NEUTRAL BY CONSTRUCTION (chain leaf row
== L == stock) — assert byte-identity on the chain5 regression arm +
class-9 needle. Gates: dual-path selfcheck (sampled row == published-leaf
row every event), same-seed repeat, lockstep cat9-vs-chain5 => SUCCESS =
ROWBUG fingerprint rows rise to ~0.85-0.9 rowOK-class, root-reject% falls
toward chain-class 10-15%, next-spine-LCP at L>=3 rises ~1.2 -> 3+,
pre-fork event counts converge; accept/event recorded as multi-draw
consequence only (never the gate). Fix-class: WIRING (playbook class 5) —
do NOT build a kernel. FIX-A2 (drafter context re-pack) decided from the
banked instrument-(iv) data during the FIX-A gate campaign.

## Working-tree state (for the monitor)

Modified (uncommitted): `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (chase
instruments + the CV-tap [B,1] reshape fix), 
`scripts/fr13_launch_forked_fa2_tree_server.sh` (chase env defaults+passthrough).
Untracked: `scripts/fr13_chase/` (7 rescued reducers),
`tests/test_fr13_chase_diag_wiring.py` (12 tests),
`output/fr13_chase_step1/` (campaign artifacts, gitignored), this bind.
Wiring suites re-run post-fix: test_fr13_chase_diag_wiring +
test_fr13_tree_conv_fused_wiring = 24/24.
