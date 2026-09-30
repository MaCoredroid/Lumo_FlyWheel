# FR13 replay-route GPU gates — BIND (2026-06-10)

**Branch:** `fr13-replay-route` @ `761862cd` (= rebase head `607f6bdd` + launcher
`FR13_REPLAY_ROUTE` passthrough). Run dir `output/fr13_replay_gpu_gates/`
(gitignored, machine-local). Governing doc: `FR13_REPLAY_GATE_TRANSFER_MATRIX.md`.

**Headline:** ALL offline byte gates PASS (Gate A codegen identity, old-vs-new
scan binary, Gate D spill, zero-accept/-0.0/dst==h0/two-event, 48-layer sweep)
— but the route is **BROKEN LIVE** (gate-4 class): accept/event collapses
2.02→1.58, within-boot same-seed non-determinism appears, and the serve stream
forks from the native oracle at pos 11–17 (vs 21–35+ flag-OFF), in BOTH the
CUDA-captured and eager regimes. Offline-bit-identical ≠ live multi-step,
exactly the `FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md` lesson. NO merge; the live
wiring seam is unlocalized (R1/R6g/R8 class candidates below).

## Boots (ONE GPU serial; teardown + recover_host_memory between; >=100G before each; docker ps empty)
| boot | regime | FR13_REPLAY_ROUTE | purpose | windows |
|---|---|---|---|---|
| (failed) | captured + in-forward captures | 0 | crashed in CUDA-graph capture: the COMMIT_HANDOFF `.cpu()` diagnostics are capture-unsafe (known: diagnostics run EAGER) | — |
| 1a | **EAGER**, captures ON (CAPTURE_PAYLOAD layer62, SRC_NATIVE layer0, COMMIT_HANDOFF L0 limit300) | 0 | A/B payloads + eager flag-OFF battery | b1a_eager_tree_greedy |
| — | offline (in 1a container, server idle; then throwaway CPU container) | — | byte A/Bs, hashes, spill, gate-2 rider | ab/ |
| 1b | captured, FINAL_LOGIT only | 0 | LEGACY reference battery (live inertness) | b1_tree_greedy (naive_mtp window killed the engine — pre-existing, see below) |
| 1c | captured | 0 | non_mtp regular-decode reference | DIED on first non_mtp request (same pre-existing crash) |
| 2 | captured, FINAL_LOGIT only | **1** | replay-route battery + rep2 + discriminator | b2_tree_greedy, b2_tree_greedy_rep2 (+ non_mtp crash probe last) |
| 3 | **EAGER** | **1** | regime discriminator for the live failure | b3_eager_tree_greedy, b3_eager_tree_greedy_rep2 |

Battery everywhere: `fr10_quick_decode_tps_probe.py` pinned prompts_swe4, B=1,
spp=1, max_tokens=128, greedy 0.0/1.0, seed 1313, warmup 0,
--require-tree-engagement; env GPU_UTIL=0.82 BI=1 FR13_BI_TREE_ATTN=1
FR10_METRICS=1 FR13_TREE_BONUS_SELF=1 FR13_CONV_COMMITTED_PATH=1 caterpillar
TREE. Container verified to mount the BRANCH working tree (sha256 match,
patched module carries FR13_REPLAY_ROUTE).

## 1) Old-vs-new SCAN binary A/B (flag-OFF inertness recompile condition) — PASS
Real captured payloads (layer0 `src_native_payload.pt` accepted_len=3 path
[1,2,4]; layer62 `tree_gdn_capture_payload.pt`), × prev_len 0..5, serving call
form (bank-mode h0, raw gating): **out AND state int-view byte-equal** for
old(main c3766eb0) vs new(branch) export-ON, and out equal for new export-OFF
(pure-export claim). Offline scan also reproduces the LIVE serving export
bytes exactly (both payloads, all prev_lens) — the offline context is valid.
Discharges matrix open item 3 + item-1 condition.

## 2) Replay-vs-scan durable-bank byte A/B (Gate A) — PASS (126/126 + synthetics + sweep)
- All 10 root-to-node paths × prev_len 0..5 × 2 real payloads: published
  LINEAR columns byte-equal to the scan export (int-view), untouched rows
  untouched. 80 dst==h0-row cases, 12 zero-accept (root→col0) cases.
- REAL EVENT (live accepted path [1,2,4]): replay bytes == offline export ==
  **live `next_read_ssm_state`** == live `serving_tree_state[leaf]` (the actual
  durable row the next live event consumed).
- Synthetics: −0.0-in-h0/−0.0-surviving-in-parent (non-vacuous) byte-exact;
  zero-accept; dst==h0-row; **two-event sequence** (replay-chain vs
  legacy-chain h0 handoff) byte-equal; fp32-bank fail-loud confirmed.
- 48-GDN-layer A_log/dt_bias sweep at SERVED dtypes (A_log fp32/dt_bias bf16;
  cubins identical to served family): all 48 layers byte-equal.
- `tests/test_fr13_replay_gpu_byte_ab.py`: 5/5 pass on GPU.
- Legacy live handoff log (boot1a, layer0, 113 events): ssm/conv next-read vs
  expected nonzero on 2/113 events (prev_len 2 and 4; ssm 21.8/23.6, conv
  59.08) — legacy-side classification evidence, branch-commit class.

## 3) Kernel binary hashes + Gate D spill — PASS
cubin sha256 (TRITON_CACHE_DIR-isolated; SASS text undumpable on this host:
image nvdisasm cannot decode sm_121a → cubin hashes + ptxas -v are the pinned
artifacts; re-arm the A/B on any toolchain change):
- scan OLD (main): `123ae9541acd9490…`
- scan NEW export-ON (served): `441b3697134b3b3e…`
- scan NEW export-OFF (replay-route serving): `327e1dbe173aaf95…`
- replay kernel (served family): `50fcd257e348a728…`
ptxas (CUDA 13.0) at deployed constexprs: **0 bytes spill stores/loads, 0
stack frame** on all four (scan 128 regs, replay 48 regs).

## 4) Mechanical gate-2 rider — CLEAN (no re-run required)
Emitted `gdn_linear_attn.py` from main patcher vs branch patcher (pristine
container): 10 hunks, ALL inside the two `num_spec_decodes > 0` guarded
regions (guard indent 16; every changed line ≥16 with the only =16 lines being
the scratch-alloc swap itself). Caveat: the metadata-builder file also gains
the env-gated init-time ring allocation (allocation-only, FR13_REPLAY_ROUTE=1).

## 5) LIVE gates (flag ON) — **FAIL, gate-4 class**
**Accept/event** (walked probe metric):
| arm | regime | flag | accept/event |
|---|---|---|---|
| banked bootB | captured | OFF | 2.215 |
| b1 (this campaign) | captured | OFF | 2.024 |
| b1a | eager | OFF | 2.083 |
| **b2** | captured | **ON** | **1.583** |
| **b2 rep2** | captured | **ON** | **1.689** |
| **b3** | eager | **ON** | **1.710** |
| **b3 rep2** | eager | **ON** | **1.665** |
| chain ref | captured | OFF | 2.277 | native BI=1: 3.047 |
Accepted-len histogram shifts systematically (len5 40→18, len1 19→37, len2
22→44 at 174→202 events) — not occasional near-tie flips.

**Determinism (same boot, same seed):** flag-OFF boots reproduce (banked
bootB rep2 identical). Flag-ON does NOT: captured forks p1@42 p2@32 p3@34
(p0 streams equal but verify-targets/drafts diverge at p0 event 35, following
an IDENTICAL zero-accept event 34); eager forks p0@104 p1@22 (p2/p3 clean) —
sporadic, race-shaped, present in BOTH regimes.

**Native-oracle lockstep (first fork vs native BI0; BI1 identical):**
flag-OFF: 35/21/21/57+ (b1, captured). flag-ON: **15/12/15/13** (b2) and
**15/11/17/14** (b3 eager) — strictly worse, regime-independent, p0 fork
identical across regimes (pos15: 1970→2313).

**Material, not near-tie:** at the p0 pos-15 flip (same served history,
branch-commit event [0,1,4], bonus=reject_parent_target), fp32 final logits at
the state-downstream row: flag-OFF logit(1970)=29.375 vs logit(2313)=26.375
(margin +3.0, native token wins); flag-ON logit(2313)=28.625 vs
logit(1970)=27.75. A ~3-logit displacement of the verify forward = material
durable-state damage.

**Durable-diff classification (interpretation rule):** the flag-ON diffs are
NOT the heal-the-legacy pattern — they make the stream worse vs the native
oracle and collapse acceptance ⇒ classified REPLAY-ROUTE-WRONG-LIVE (the
live wiring, not the kernel: the kernel is byte-proven offline incl. on live
captured bytes). Candidate seams (matrix R-classes, unresolved): R1
(prev-lens/commit-index generation at live event sequences), R6g (native
mamba_utils `get_temporal_copy_spec` cross-step state copy — reads
`block_ids[cur + accepted - 1]`; its run-time vs the commit-time replay and
vs the deleted remap is now load-bearing and was never neutral-verified), R8
(REQKEY/ring staleness at request churn — non-determinism concentrated after
request succession). CUDA-graph staleness (R3) is EXCLUDED as the sole cause
(eager reproduces the damage), though boot 2 proves the no-scratch forward
captures fine.

**Zero-accept gate:** offline = PASS (root→col0 byte-exact, 12 cases + −0.0 +
pytest). Live = the first zero-accept handoffs are clean (after p0 ev0 len=0,
the next event's drafts/targets match flag-OFF exactly), no isolated
zero-accept corruption observed; but the p0 within-boot rep divergence
(captured) first appears at the event AFTER an identical zero-accept event,
and the route's overall live handoff is unsound — live zero-accept gate
therefore NOT separately passable.

**p0-pos-35 discriminator: CONFOUNDED — and the banked premise itself falls.**
This campaign's flag-OFF captured boot (b1) emitted **44675 = the NATIVE
token** at pos 35 (lockstep with native through and past 35), while banked
bootB emitted 8445 and eager flag-OFF emitted 58046: the pos-35 "legacy flip"
is a cross-boot NEAR-TIE coin flip, not a stable legacy property. The
lineage-table row citing the pos-35 flip as a stable legacy-kernel defect must
be re-interpreted (REPORTED). Flag-ON never reaches pos 35 in native lockstep
(forks at 15), so healed-vs-survives is unanswerable as posed.

## 6) Regular-decode quick check — NOT RUNNABLE (pre-existing)
`naive_mtp` AND `non_mtp` requests against a tree-config boot kill the engine:
`AttributeError: 'EagleProposer' object has no attribute 'positions'`
(eagle.py:1430 propose_tree, vLLM 0.19.2rc1.dev134). Identical signature flag
OFF (boots 1b, 1c) and flag ON (boot 2 last act); `git diff c3766eb0..HEAD`
has ZERO `positions` hunks ⇒ inherited from main, flag-independent,
pre-existing. The non-tree modes on tree boots are broken upstream of any GDN
code; fix tracked separately.

## Verdict
- Gate A + scan-binary A/B + Gate D: **PASS** — the route's kernels and
  conventions are byte-correct; every CONDITIONAL-ON-BYTE-A/B matrix item's
  evidence carries AT THE KERNEL LEVEL.
- Live (items 5iv/9-class): **FAIL** — do NOT proceed to the corruption gate /
  final regime; build Gate B (live ordering probe) + the R6g native-copy-path
  neutrality check FIRST, on the post-mortem seams above.
- NO merge to main (user decision class). Launcher passthrough commit
  `761862cd` is the only code change from this campaign.

## Artifacts (run dir, gitignored)
boot1a_eager_capture_logs/{src_native_payload.pt,tree_gdn_capture_payload.pt,
commit_handoff.jsonl,ab/*.json,ab/sass_spill_report.txt}, b1a/b1/b2/b2_rep2/
b3/b3_rep2 windows (probe JSONs + lcp/sampler/logit logs),
emitted_gdn_main_vs_branch.diff, offline_byte_ab.py, run_battery.sh,
run_header.json, boot*_launch.log.

## ADDENDUM 2026-06-11 — root cause FOUND + FIXED; live gates now PASS (both regimes)

**Root cause (boundary-trace, run dir `output/fr13_replay_boundary_trace`):**
R6g-CLASS but OUR OWN wiring, not vLLM's align-mode machinery — the
"conv-only" linear remap (`launch_tree_state_linear_remap(ssm_state=None,
conv_state=...)` at the next forward's conv branch) launches the frozen
Triton remap on the conv kv-cache VIEW, whose `stride(0) ==
num_element_per_page` (conv kv[0] and ssm kv[1] are as_strided views over the
SAME mamba page). The kernel copies `stride(0)` elements per row ⇒ every
len≥1 commit copied WHOLE PAGES node→linear, dragging never-written
node-column ssm bytes over the replay's just-published linear-column ssm
states (byte prediction B.window[c] == A.post.window[node path[c]]: 581/581,
0 mismatches, both probed layers; len=0 commits 71/71 clean = remap writes
nothing). Legacy immunity: the all-rows ssm publish made the page-wide copy
semantically identical to the intended ssm remap. Flag-ON non-determinism:
wiped-in bytes are history-dependent page garbage (boot-fresh zeros vs
reused-block leftovers).

**Fix (PURE WIRING, kernel file untouched), commit `02b1627a`:** under
FR13_REPLAY_ROUTE=1 the conv half runs through
`lumo_flywheel_serving.fr13_replay_conv_remap.replay_conv_state_linear_remap`
— identical gather-then-scatter permutation in plain tensor ops
(index_select/index_copy_) that copy ONLY the conv view's logical elements,
never the page remainder. Flag OFF keeps the legacy whole-page launch
verbatim. CPU tests: page-shared as_strided fixture (mirrors
_reshape_kv_cache_tensors MambaSpec branch), whole-raw-tensor byte compare vs
a kernel-semantics reference, ssm-slice byte-frozen assertion, capture-safety
AST lint.

**Re-gate (run dir `output/fr13_replay_fix_regate`, branch @ 02b1627a, pinned
battery prompts_swe4 B=1 greedy seed1313 BI=1, FR13_REPLAY_ROUTE=1, boundary
instrument OFF, one eager boot + one captured boot, battery x2 each,
RestartCount=0 both):**
| gate | eager (e1/e1r2) | captured (c1/c1r2) | broken (b3/b2) | flag-OFF ref |
|---|---|---|---|---|
| same-seed determinism | **4/4 identical** | **4/4 identical** | 2/4 / 1/4 | 4/4 |
| accept/event | **2.0833 / 2.0833** | **2.1467 / 2.1467** | 1.665-1.710 / 1.583-1.689 | 2.083 (b1a eager) / 2.024 (b1 captured) |
| first fork vs native_greedy | **35/15/21/71** | **70/21/21/71** | 15/11/17/14 / 15/12/15/13 | 35/15/21/71 (b1a) / 54/24/25/57 (b1) |

- EAGER flag-ON is TOKEN-IDENTICAL to flag-OFF eager (b1a) on all 4 prompts
  (first-fork = None everywhere) — the route now reproduces legacy exactly.
- CAPTURED flag-ON forks from flag-OFF captured (b1) only at 54/21/21/57 =
  the known captured cross-boot near-tie band (b1 vs banked bootB themselves
  differ at 35), and diverges from broken b2 exactly at b2's corruption
  positions (15/12/15/13). On p0/p3 it holds native lockstep LONGER than b1
  (70/71 vs 54/57).
- Class-6 check: the page-safe torch remap CUDA-graph-captures and replays
  correctly (captured boot healthy, gates pass, no eager fallback).

**Verdict: gate-4-class live failure RESOLVED. Replay-route live gates PASS
in both regimes.** R1/R8 excluded by the trace; align-mode actors untested
because this serving config never runs them (enable_prefix_caching=False).
