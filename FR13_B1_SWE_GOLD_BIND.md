# FR13 B=1 SWE Gold-Standard Gate — SALVAGED bind

Date: 2026-06-13 UTC. Workflow `wf_d8a86320-7e7` (gold gate). **SALVAGE NOTE:** the
campaign + verify subagents ERRORED mid-run on a `claude-fable-5` model-access failure
(session model lost), but the prep agent finished and **all 6 GPU arms actually ran and
left complete artifacts on disk**. task1's reduce had already completed (00:24); task2's
reduce + the adversarial verify never ran. The monitor (on Opus 4.8) ran the task2 reduce
from the on-disk pair-dumps and red-teamed both reduces by hand. Raw reduces banked:
`research/fr13_workflows/gold_gate_task{1,2}_reduce_wf_d8a86320.json`.

## Build under test
HEAD b7887c89: FIX-1 (single-logits) + FIX-2 (eager-pack) + FIX-3 (conv-fusion) + FIX-A
(tree-sample-row) ALL default ON. Tree = cat9 / TREE_ATTN / num_spec 9; native = E5 /
FLASH_ATTN / num_spec 5 via `fr10_launch_speed_server.sh`.

## Arms (6, all health-clean)
| arm | task | verdict | wall | rc | draft-shape |
|---|---|---|---|---|---|
| tree_a / tree_b | astropy-12907 | resolved / resolved | 1567 / 1565 s | 0 / 0 | 9.0 ✓ |
| native_a / native_b | astropy-12907 | resolved / resolved | 1564 / 1565 s | 0 / 0 | 5.0 ✓ |
| tree | astropy-13033 | failed | 3058 s (retry) | 0 | 9.0 ✓ |
| native | astropy-13033 | failed | 1595 s | 0 | 5.0 ✓ |

Health: **6/6 rc=0, full wall, zero health flags.** task2 BOTH arms failed the SWE task
(symmetric tree+native — a hard instance, not a tree regression). Resolved/failed are
draws, not gate criteria.

## BINDING VERDICT — greedy SWE served-stream lossless (the final-call criterion)
**WITHIN-FLOOR PASS, both tasks**, even under the strict GAP-3 native-native-only floor:
- task1: tree-vs-native served fork at **char 80**; native-native self-floor forks at
  **char 20**; tree-tree self-floor at char 20. Verdict fork is far AFTER the floor ⇒ not
  attributable to the tree build. env_forks=0.
- task2: tree-vs-native served fork at **char 78**; bracketed by the task1 floor (char 20).
- Within-boot determinism: rep1==rep2 byte-identical on ALL arms, greedy AND t0.6, both
  tasks (the substrate is deterministic within a boot).

## RED-TEAM YELLOW FLAG (honest, NOT hand-waved) — 128-tok probe riders
The rider probes (pinned `prompts_swe4.json`, the tighter secondary signal) show
tree-vs-native greedy forking EARLIER than the native self-floor on prompt 0 (and slightly
on 2/3), reproducibly across both tasks:
- task1 greedy: native-vs-tree `[17,15,21,61]`; native-native floor `[35,15,25,71]`;
  tree-tree floor `[34,21,21,61]`. Prompt 0: tree-native 17 < both self-floors ~34-35.
- task2 greedy: native-vs-tree `[17,31,21,61]` (prompt 0 again 17).
**Interpretation (open, requires per-token margin to close):** tree-vs-native compares TWO
backends (FLASH_ATTN/5 vs TREE_ATTN/9 → different GEMM batch shapes/roundings), so its
near-tie floor is INHERENTLY WIDER than the single-backend cross-boot self-floor. A fork
earlier than the self-floor is EXPECTED for two lossless-but-numerically-different impls
IFF every early fork is a near-tie (target top-1/top-2 within float noise). It is a real
loss only if the fork is at a CLEAR margin (tree served a token the target would clearly
reject). **This cannot be classified from served tokens alone** — needs per-position top-k
logit/margin capture at the fork.

## RESOLVED 2026-06-13 (close workflow wf_3c6f5c0a, margin agent + independent verify BOTH = FAIL):
**The probe forks are NOT all near-ties — there is a REAL greedy-lossless gap in the tree
serving path.** Measured via CLEAN teacher-forced distributions (the streamed tree
top_logprobs are off-by-one at spec-decode positions — a vLLM logprob-reporting bug — so
they were discarded; teacher-forcing on the byte-identical shared prefix is the valid
reference, validated by ASYMMETRY: native teacher-forced argmax == native served at 4/4
forks, but tree teacher-forced argmax != tree served at 3/4):
| prompt | fork | tree served | clean argmax (both backends agree) | dev_gap | verdict |
|---|---|---|---|---|---|
| 0 | 17 | ` and` | ` structure` | 0.375 | near-tie (same defect, small margin) |
| 1 | 11 | ` workspace` | genuine ` workspace`/` repository` tie | 0.000 | NEAR-TIE (lossless) |
| 2 | 21 | ` code` | ` files` (8.4×) | 2.125 | **CLEAR-MARGIN real loss** |
| 3 | 68 | `Let` (0.59%) | ` ``` ` (98.88%, 168×) | 5.125 | **CLEAR-MARGIN real loss** |
A 168× argmax flip is NOT cross-backend numerics — it is a logic/commit-path defect: the
tree spec-decode COMMITTER serves a non-argmax token at a subset of accept boundaries
(clustered at structural/template boundaries: code fences, stop regions). Reconciles with
codex SOLVING task1 on the tree: the within-floor agentic verdict is a COARSE first-fork
check (blind to per-position argmax misses); a ~6%-of-positions non-argmax commit stays
COHERENT (task resolves, stream within-floor) yet is not greedy-lossless.

**VERDICT: B=1 does NOT cleanly graduate. Binding agentic within-floor HOLDS but is too
coarse; the tight probe is the binding lossless instrument and it FAILS.** B4-1 NOT launched.
DECISIVE NEXT (in-process, confound-free): per-accept-boundary committer-row gate asserting
committed_id == argmax(verify-forward logits at that row) over the pinned probes; first
divergent boundary names the seam. Candidate seams (agent): FIX-A tree-sample-row bonus/self
at accept-run ends, eager-pack replay row mapping, conv-fusion committed-path row at
num_accepted>1. Also split channel-1 (commit != verify-argmax = committer row bug) from
channel-2 (verify-argmax != clean-argmax = verify-forward losslessness gap).
See [[feedback_math_correct_vs_bitexact]] ("don't hand-wave backend nature, locate the op").

## Speed (task1, per-request medians, DRAWS — class 12)
| arm | s/fwd | accept/event | (TPS noisy/prefill-confounded) |
|---|---:|---:|---|
| tree_a / tree_b | 0.2489 / 0.2541 | 3.40 / 3.97 | — |
| native_a / native_b | 0.2361 / 0.2364 | 3.46 / 3.64 | — |
Tree ~1.05-1.08x native s/fwd at 11k-ctx agentic (consistent with the 64-tok 1.02-1.03x +
the residue audit's "gap grows with context = kernel/launch-shape, not flag plumbing").
accept/event overlapping bands (trajectory-confounded draws, NOT a superset verdict here).

## STATUS (2026-06-13): gate did NOT cleanly pass. Coarse agentic within-floor verdict HOLDS
(full-health, det-clean, env_forks=0), but the close workflow's tight teacher-forced probe
+ independent verify found a REAL greedy-lossless gap in the tree serving path (committer
serves non-argmax at clear margin, 2/4 forks). **B=1 NOT graduated; B4-1 NOT launched.**
NEXT = in-process committer-row argmax localization (then fix). This is the gate working as
designed — the within-floor SWE check was too coarse; the probe caught what it missed.
