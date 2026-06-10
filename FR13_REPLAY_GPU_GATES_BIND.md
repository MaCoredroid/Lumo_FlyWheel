# FR13 replay-route GPU gates — BIND (main doc of record, 2026-06-10)

**Branch under test:** `fr13-replay-route` @ `9d4d22e3` (NOT merged; the branch
carries its own copy of this bind — this main copy is the doc of record).
Branch history: rebase head `607f6bdd` (onto main `c3766eb0`, conv fix
`c0b53f5d` + S1 `4d45be27` inherited) + `761862cd` (launcher
`FR13_REPLAY_ROUTE` passthrough, the only campaign code change) + `9d4d22e3`
(branch bind). Run dir `output/fr13_replay_gpu_gates/` (gitignored,
machine-local). Governing doc: `FR13_REPLAY_GATE_TRANSFER_MATRIX.md` (+
ADDENDUM riders). NO merge, NO close — user decision class.

**Headline:** ALL offline byte gates PASS — Gate A codegen identity 126/126,
old-vs-new scan binary byte-equal, Gate D 0-spill, zero-accept/−0.0/dst==h0/
two-event synthetics, 48-layer sweep — the replay kernel and its conventions
are byte-correct, including on live captured bytes. But the route is **BROKEN
LIVE (gate-4 class)**: accept/event collapses 2.02→1.58, within-boot same-seed
non-determinism appears, and the served stream forks from the native oracle at
pos 11–17 (vs 21–35+ flag-OFF), in BOTH the CUDA-captured and eager regimes.
Offline-bit-identical ≠ live multi-step — exactly the
`FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md` lesson, recurring on a byte-proven
kernel. The live wiring seam is unlocalized (R1/R6g/R8 candidates below).

**LINEAGE-TABLE IMPLICATION (FLAGGED — reported to the user):**
1. The `_tree_gdn_replay_kernel` row's "bit-exact conditional on codegen
   identity (byte A/B pending)" is **discharged**: codegen identity PROVEN
   (R4 retired) — but the row gains **LIVE FAIL, gate-4 class** (kernel
   exonerated; live wiring owns the failure). Row updated in
   `FR13_GDN_KERNEL_LINEAGE.md`.
2. The **p0-pos-35 "legacy spine-commit flip" premise FALLS**: it is a
   cross-boot NEAR-TIE coin flip on the legacy path (this campaign's flag-OFF
   captured boot emitted the NATIVE token 44675 at pos 35), not a stable
   legacy-kernel defect. Every doc citing it as a stable legacy indictment
   (`FR13_CONVFIX_AB_BIND.md` residual class, the replay-route discriminator
   premise) must be read with this re-interpretation.
3. Live-kernel row gains a factual datum: legacy handoff log (boot1a, layer0,
   113 events) shows **2/113 nonzero ssm next-read deltas** (21.8/23.6,
   prev_len 2/4, branch-commit class) — the legacy path is not proven fully
   clean either; it is just far better live than the replay route today.

## Headers — boots (ONE GPU serial; teardown + recover_host_memory between; >=100G free before each; docker ps empty)
| boot | regime | FR13_REPLAY_ROUTE | purpose | windows |
|---|---|---|---|---|
| (failed) | captured + in-forward captures | 0 | crashed in CUDA-graph capture: COMMIT_HANDOFF `.cpu()` diagnostics are capture-unsafe (standing policy: diagnostics run EAGER) | — |
| 1a | **EAGER**, captures ON (CAPTURE_PAYLOAD layer62, SRC_NATIVE layer0, COMMIT_HANDOFF L0 limit300) | 0 | A/B payloads + eager flag-OFF battery | b1a_eager_tree_greedy |
| — | offline (in 1a container, server idle; then throwaway CPU container) | — | byte A/Bs, cubin hashes, spill, gate-2 rider | ab/ |
| 1b | captured, FINAL_LOGIT only | 0 | LEGACY reference battery (live binary-level inertness) | b1_tree_greedy (naive_mtp window killed the engine — pre-existing, §7) |
| 1c | captured | 0 | non_mtp regular-decode reference | DIED on first non_mtp request (same pre-existing crash) |
| 2 | captured, FINAL_LOGIT only | **1** | replay-route battery + rep2 + discriminator | b2_tree_greedy, b2_tree_greedy_rep2 (+ non_mtp crash probe last) |
| 3 | **EAGER** | **1** | regime discriminator for the live failure | b3_eager_tree_greedy, b3_eager_tree_greedy_rep2 |

Battery everywhere: `fr10_quick_decode_tps_probe.py`, pinned
`output/fr13_acceptance_ladder/prompts_swe4.json`, B=1, spp=1, max_tokens=128,
greedy 0.0/1.0, seed 1313, warmup 0, --require-tree-engagement; env
GPU_UTIL=0.82 BI=1 FR13_BI_TREE_ATTN=1 FR10_METRICS=1 FR13_TREE_BONUS_SELF=1
FR13_CONV_COMMITTED_PATH=1, caterpillar TREE. Container verified to mount the
BRANCH working tree (sha256 match; patched module carries FR13_REPLAY_ROUTE).
Within the matrix 4–6 boot budget (5 serving boots + 1 failed + throwaway CPU
container). Boot-plan adaptation from the prescribed single boot-1: in-forward
captures crash FULL capture, so captures ran on the eager boot (1a).

## 0) CPU precondition (same campaign, earlier stage) — rebase + A/B-vehicle port + suites
- Rebase onto main `c3766eb0`: ONE conflict hunk
  (`scripts/fr10_phase4_patch_vllm_tree_gdn.py` pre-remap region ~:765),
  resolved as a union — conv-committed-path snapshot FIRST (runs under
  FR13_REPLAY_ROUTE=1 too; conv-bank-only, node-indexed, pre-remap), then the
  flag-gated remap launch with `ssm_state=(None if flag else ssm_state)`,
  conv half unconditional. Inherited + verified in HEAD: conv fix `c0b53f5d`,
  S1 `4d45be27`, FR13_FORCE_SPINE_COMMIT, e8a64eed-class input-ids fix
  (matrix item 13 monitor-lesson check).
- Rider-2 port TEST-ENFORCED (`tests/test_fr13_replay_ab_vehicle_port.py`, 5
  tests): STORE_NODE_STATES=True diagnostic mode retains the full
  COMMIT_HANDOFF capture chain (the exact `serving_tree_state` field
  `fr10_tree_kernel_h0_ab_replay.py:101` reads); flag-ON capture refusal
  covers all four scratch-consuming envs (fail-loud); CAPTURE_PAYLOAD saves
  ALL replay inputs (key_spec/value_tree/a/b/h0) + serving_out/serving_state.
- CPU suites on the pushed head: 91 passed / 5 GPU-skips (replay battery +
  vehicle port + serving wiring + main suites after the ONE matrix-predicted
  text-assertion co-update in `test_fr13_conv_committed_path`). Full-suite
  parity vs origin/main: failure sets IDENTICAL (the 2
  `test_fr10_lossless_equivalence` failures read gitignored GPU artifacts —
  environment-class, present on origin/main too).

## 1) Old-vs-new SCAN binary A/B (flag-OFF inertness recompile condition) — PASS
Real captured payloads (layer0 `src_native_payload.pt` accepted_len=3 path
[1,2,4]; layer62 `tree_gdn_capture_payload.pt`), × prev_len 0..5, serving call
form (bank-mode h0, raw gating): **out AND state int-view byte-equal** in all
12 case-pairs for old(main `c3766eb0`) vs new(branch) export-ON; new
export-OFF out also byte-equal (pure-export claim holds live-shaped). Offline
scan reproduces the LIVE serving exported states byte-exactly for both
payloads at every prev_len — the offline context is valid. Discharges matrix
open item 3 + the item-1 recompile condition: flag-OFF inertness is now
**binary-level**, and boot-1b flag-OFF captured serving ran clean on the
rebased branch (accept 2.024, within the historical cross-boot floor of
banked bootB 2.215).

## 2) Replay-vs-scan durable-bank byte A/B (Gate A) — PASS (126/126 + synthetics + sweep)
- All 10 root-to-node paths × prev_len 0..5 × 2 real payloads: published
  LINEAR columns byte-equal to the scan export (int-view), untouched rows
  untouched. 80 dst==h0-row cases, 12 zero-accept (root→col0) cases.
- REAL EVENT (live accepted path [1,2,4]): replay bytes == offline export ==
  **live `next_read_ssm_state`** == live `serving_tree_state[leaf]` — the
  durable row the next live event actually consumed.
- Synthetics: −0.0-in-h0 and −0.0-surviving-in-parent (non-vacuous, bit-exact
  incl. the per-edge +0.0 flip); zero-accept; dst==h0-row; **two-event
  sequence** (replay-chain vs legacy-chain h0 handoff) byte-equal; fp32-bank
  fail-loud raise confirmed.
- 48-GDN-layer A_log/dt_bias sweep at SERVED dtypes (cubins identical to the
  served family): all 48 layers byte-equal. (Caveat: layer-0 activations with
  per-layer A_log/dt_bias — per-layer activations would need 48 one-shot
  capture boots.)
- `tests/test_fr13_replay_gpu_byte_ab.py`: 5/5 pass on GPU.
**Verdict: every CONDITIONAL-ON-BYTE-A/B matrix item's evidence carries AT THE
KERNEL LEVEL. Lineage implication flagged above (item 1).**

## 3) Kernel binary hashes + Gate D spill — PASS
cubin sha256 (TRITON_CACHE_DIR-isolated; SASS text undumpable on GB10 — image
nvdisasm cannot decode sm_121a → cubin hashes + ptxas -v are the pinned
artifacts; **re-arm the A/B on any toolchain change**):
- scan OLD (main): `123ae9541acd9490…`
- scan NEW export-ON (diagnostic): `441b3697134b3b3e…`
- scan NEW export-OFF (replay-route serving): `327e1dbe173aaf95…`
- replay kernel (served family): `50fcd257e348a728…`
ptxas (CUDA 13.0) at deployed constexprs: **0 bytes spill stores/loads, 0
stack frame** on all (scan 128 regs, replay 48 regs). Recompile changed the
binary but not one output byte.

## 4) Mechanical gate-2 rider — CLEAN (no gate-2 re-run required)
Emitted `gdn_linear_attn.py` main-patcher vs branch-patcher (pristine
container): 10 hunks, ALL inside the two `num_spec_decodes > 0` guarded
regions. Caveat: the metadata-builder file also gains the env-gated init-time
ring allocation (allocation-only, FR13_REPLAY_ROUTE=1).

## 5) LIVE gates (flag ON) — **FAIL, gate-4 class**

### Accept progression (walked probe metric, pinned battery, B=1 greedy seed 1313)
| arm | regime | flag | accept/event |
|---|---|---|---|
| banked bootB (conv-fix campaign) | captured | OFF | 2.215 |
| b1 (this campaign) | captured | OFF | 2.024 |
| b1a | eager | OFF | 2.083 |
| **b2** | captured | **ON** | **1.583** |
| **b2 rep2** | captured | **ON** | **1.689** |
| **b3** | eager | **ON** | **1.710** |
| **b3 rep2** | eager | **ON** | **1.665** |
| chain ref | captured | OFF | 2.277 |
| native BI=1 ref | captured | — | 3.047 |

Historical progression for context (all trajectory-confounded): pre-fix 2.024
→ cc008587 2.082 → post-S1 1.819 → conv-fix 2.215 → replay flag-ON
**1.58–1.71**. COLLAPSE of ~0.4–0.6 accept/event under the flag, consistent
across regimes and reps — the gate-4-shaped signature (2.024→1.521
precedent). Accepted-len histogram shifts systematically (len5 40→18, len1
19→37, len2 22→44 at 174→202 events) = per-event acceptance deficit, NOT
occasional near-tie flips.

### Determinism (matrix item 5iv re-run) — FAIL flag-ON
Flag-OFF boots reproduce (banked bootB rep2 byte-identical). Flag-ON does
NOT: captured forks rep1-vs-rep2 at p1@42 p2@32 p3@34 (p0 token-equal but
verify-targets/drafts diverge at p0 event 35, immediately after an IDENTICAL
zero-accept event 34); eager forks p0@104 p1@22 (p2/p3 clean) — sporadic,
race-shaped, present in BOTH regimes ⇒ NOT a CUDA-graph-staleness-only effect
(R3 excluded as sole cause). Note: boot 2 DID survive FULL CUDA-graph capture
with the route on — the scratch-deletion capture-blocker fix works.

### Durable-diff vs legacy — classification table (per the interpretation rule)
| evidence | observation | classification |
|---|---|---|
| spine-commit, no-flip early events | b2 events 0–2 emitted/winner match flag-OFF exactly; ev1 drafts byte-identical | diff ZERO as the rule requires — consistent |
| first fork vs native oracle | flag-ON 15/12/15/13 (captured), 15/11/17/14 (eager) vs flag-OFF 35/21/21/57+ | strictly EARLIER, regime-independent; p0 fork identical across regimes |
| p0 pos-15 flip (1970→2313, after [0,1,4] branch-commit, bonus=reject_parent_target) | fp32 final logits, state-downstream row: flag-OFF logit(1970)=29.375 vs logit(2313)=26.375 (margin +3.0, native wins); flag-ON logit(2313)=28.625 vs logit(1970)=27.75 | MATERIAL ~3-logit displacement of the verify forward = real durable-state damage, not near-tie |
| accepted-len histogram | len5 40→18, len1 19→37, len2 22→44 | systematic acceptance deficit |
| legacy side | boot1a handoff log: 2/113 nonzero ssm next-read deltas (21.8/23.6, prev_len 2/4, branch-commit class) | legacy not proven clean — classification evidence, NOT a heal |
| **VERDICT** | flag-ON diffs make the stream WORSE vs native and collapse acceptance | **REPLAY-ROUTE-WRONG-LIVE (gate-4 class)** — not legacy-buggy-healed |

Since the kernel + conventions are byte-proven offline (incl. on live captured
bytes and two-event chains), the seam is **live-only wiring**. Candidates
(matrix R-classes, unresolved): **R1** (prev-lens/commit-index generation at
live event sequences), **R6g** (native mamba_utils `get_temporal_copy_spec`
cross-step state copy — never neutrality-verified, now load-bearing), **R8**
(REQKEY/ring staleness at request churn — non-determinism concentrated after
request succession).

### Zero-accept gate (Gate C)
Offline: PASS definitively — root state replayed into column 0 on every len=0
case (12 payload-driven + −0.0 + pytest), byte-exact; the next-event h0 read
(clamp col 0) consumes exactly those bytes (two-event chain proven). Live:
first zero-accept handoffs clean (after p0 ev0 len=0, the next event's
drafts/targets/emitted match flag-OFF exactly); no isolated zero-accept
corruption observed — BUT the captured-boot within-rep divergence first
appears at the event AFTER an identical zero-accept event, and the route's
live handoff is unsound overall ⇒ live zero-accept gate **NOT separately
passable**.

### p0-pos-35 discriminator — CONFOUNDED, and the banked premise itself falls (REPORT)
This campaign's flag-OFF CAPTURED boot (b1, same env/seed as banked bootB)
emitted **44675 = the NATIVE token** at p0 pos 35 (native lockstep through and
past 35, fork at 57+ elsewhere), while banked bootB emitted 8445 and the eager
flag-OFF boot emitted 58046: the pos-35 "legacy flip" is a **cross-boot
NEAR-TIE coin flip** on the legacy path, not a stable legacy-kernel defect —
the lineage/bind rows citing it must be re-interpreted (flagged, header item
2). Healed-vs-survives is unanswerable as posed: flag-ON never reaches pos 35
in native lockstep (forks at pos 15 with its own deterministic, material flip
1970→2313, margin analysis above). **The flag-ON pos-15 flip is the new,
reproducible discriminator target.**

## 6) Regular-decode quick check — NOT RUNNABLE (pre-existing main-side bug)
`naive_mtp` AND `non_mtp` requests against a tree-config boot kill the engine:
`AttributeError: 'EagleProposer' object has no attribute 'positions'`
(eagle.py:1430 propose_tree, vLLM 0.19.2rc1.dev134). Identical signature flag
OFF (boots 1b, 1c) and flag ON (boot 2 last act); zero `positions` hunks in
the branch diff ⇒ inherited from main, flag-independent. The non-tree modes on
tree boots are broken upstream of any GDN code; track as a separate main-side
bug (cost this campaign: boots 1b/1c each lost their engine to it).

## Verdict
- Gate A + scan-binary A/B + Gate D + gate-2 rider + Gate C offline: **PASS**
  — the route's kernels and conventions are byte-correct; R4 retired.
- Live (items 5iv / 9-class): **FAIL** — do NOT proceed to the corruption
  gate / final regime.
- NO merge to main (user decision class). The branch stays unmerged at
  `9d4d22e3`.

## What remains before the user merge decision
1. **Localize + fix the live wiring seam** (the route is dead until this
   lands): Gate B live ordering probe (publish-vs-next-h0-read), the R6g
   native `get_temporal_copy_spec` neutrality check (live-container source
   read first — read-vllm-source-first lesson), the R8 REQKEY/ring churn
   test. The flag-ON p0 pos-15 flip is the reproducible target.
2. Re-earn the live gates after the fix: determinism re-probe (item 5iv,
   flag-ON byte-identical reps), accept/event ≥ flag-OFF legacy (no collapse),
   native-fork positions ≥ legacy's.
3. Then the deferred chain in matrix order: corruption gate (item 9, 1–3
   boots), CUDA-graph capture gates (item 11 — partially de-risked: boot 2
   captured fine), final regime Gate F = B=4 + FULL capture + SWE-4 vs E5
   (within E5 self-noise floor + accept/event vs native 3.076 bar).
4. Separate main-side fix: the EagleProposer.positions crash (§6) so
   regular-decode references are runnable at all.
5. Re-arm the byte A/B on any toolchain/constexpr change (cubin-hash pin).

## Artifacts
Run dir `output/fr13_replay_gpu_gates/` (gitignored, machine-local):
boot1a_eager_capture_logs/{src_native_payload.pt,tree_gdn_capture_payload.pt,
commit_handoff.jsonl,ab/*.json,ab/sass_spill_report.txt}, battery windows
b1a/b1/b2/b2_rep2/b3/b3_rep2 (probe JSONs + lcp/sampler/logit logs),
emitted_gdn_main_vs_branch.diff, offline_byte_ab.py, run_battery.sh,
run_header.json, boot*_launch.log. CPU-stage logs in
`output/fr13_replay_rebase/` (incl. the disk-safe per-file pytest runner —
single-process full-suite runs write >300 GB of /tmp and are infeasible on
this host). Branch bind copy: `FR13_REPLAY_GPU_GATES_BIND.md` @ `9d4d22e3`
on `fr13-replay-route`.
