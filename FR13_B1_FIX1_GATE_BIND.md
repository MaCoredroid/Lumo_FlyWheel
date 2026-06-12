# FR13 B=1 FIX-1 Lossless Gate Bind — FR13_DRAFTER_SINGLE_LOGITS (DRAFT, not committed)

Date: 2026-06-12 UTC. Executor: gate workflow agent (serialized GPU).
Fix under test: FIX-1 from `FR13_B1_SPEED_ATTRIBUTION_BIND.md` — drafter
single-logits (`FR13_DRAFTER_SINGLE_LOGITS`, default ON; OFF = exact legacy
double-lm-head path, the A/B instrument). Working-tree diff (uncommitted):
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` (root :7288-7320, loop
:7427-7452, diagnostics field), launcher passthrough in
`scripts/fr13_launch_forked_fa2_tree_server.sh:38,:158` and
`scripts/fr10_launch_speed_server.sh:85,:188`, plus
`tests/test_fr13_drafter_single_logits_wiring.py` (4 passed; 90 existing
CPU tests across 8 files passed).

Artifacts root: `output/fr13_b1_fix1_gate/` (per-arm subdirs with
`container_env.txt`, engagement needles, full /metrics snapshots,
probe JSONs, `docker_full.log`). Reducers:
`reduce_fix1_gate.py` -> `fix1_gate_reduce.json`;
`reduce_fix1_nsys.py` -> `nsys_chain5_on/engagement_reduce.json`.

## Regime (canonical, mirrored from FR13_B1_SUPERSET_PRECONDITION_BIND.md)

PORT=9950, GPU_UTIL=0.82, MAX_NUM_SEQS=1, BATCH_INVARIANT=0,
FR13_BI_TREE_ATTN=0, FR10_METRICS=0, FR13_REPLAY_ROUTE=1; pinned prompts
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 prompts), seed 1313,
samples/prompt 1, client B=1, max_tokens 128 (warmup probe: 1 prompt x 16);
greedy = temp 0.0 top_p 1.0; temp-0.6 probes = temp 0.6 top_p 1.0 (fresh
PRE-change capture per the bind's step 0 — no banked t0.6 refs existed).
chain5 topology per FR13_B1_CHAIN_SPEED_DISCRIMINATOR.md
(`TREE="[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"`); cat9 = launcher
default 9-node caterpillar. FULL CUDA capture proven in every arm
("Graph capturing finished" in each boot log; ENFORCE_EAGER unset).
docker ps empty + free -g recorded before/after each arm;
`docker rm -f` between arms; launcher host-memory recovery engaged each boot.

## Engagement proof (class 9) — needle fires in BOTH states

`logger.info_once` at live eagle.py:557, present in every tree arm's
docker log and absent (as required) in the no-spec arms:

- cat9_off / cat9_off_b / chain5_off:
  `FR13_DRAFTER_SINGLE_LOGITS drafter path engaged: single_logits=False (env=0, use_local_argmax_reduction=False)`
- cat9_on / chain5_on / nsys_chain5_on:
  `... single_logits=True (env=1, use_local_argmax_reduction=False)`
- container_env.txt records `FR13_DRAFTER_SINGLE_LOGITS=<0|1>` per arm
  (asserted by the runner before any probe).

## Gate (1): within-boot same-seed repeat byte-identity (class 8) — PASS

rep1 == rep2 byte-identical (all 4 prompts x 128 tokens), greedy AND t0.6,
in ALL FIVE tree boots (cat9_off, cat9_off_b, cat9_on, chain5_off,
chain5_on). Accept/event identical within boot in every repeat. B=1
fixed-point determinism holds post-fix in both flag states.

## Gate (2): post-vs-pre served-stream byte-identity — NOT ATTRIBUTABLE MISMATCH (floor-bracketed)

Streams are NOT byte-identical OFF-vs-ON — but the bind's class-11 mismatch
rule was executed: the FLAG-OFF arm was re-booted (cat9_off_b) and the
OFF-vs-OFF cross-boot floor measured FIRST. The OFF-vs-ON divergence does
NOT exceed the OFF-vs-OFF floor, so it is not attributable to the diff.

cat9 greedy first-divergence matrix (3 OFF boots incl. banked
`output/fr13_b1_current_gate/tree`, 1 ON boot; None = byte-identical):

| pair | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| banked_off vs off_a | 17 | 27 | 21 | 61 |
| banked_off vs off_b | 17 | 11 | 84 | 61 |
| off_a vs off_b | 53 | 11 | 21 | 91 |
| banked_off vs on | 17 | 11 | 25 | 84 |
| off_a vs on | 53 | 11 | 21 | 61 |
| off_b vs on | 68 | 24 | 25 | 61 |

Per-prompt earliest ON-vs-OFF fork >= earliest OFF-vs-OFF fork on every
prompt (`floor_rule_eval` in fix1_gate_reduce.json: on_exceeds_floor=false
x4). ON forks LATER than the OFF-vs-OFF floor on p0/p1/p2 in at least one
pairing.

chain5 greedy: banked_off vs on -> p0 **byte-identical (all 128)** while
the two OFF boots fork at p0:90; off vs on p2 byte-identical while
banked-vs-off forks at p2:25. The ON arm interleaves INSIDE the OFF
cross-boot equivalence class — the strongest available evidence that the
single-logits argmax equals the legacy second-compute_logits argmax and
the residual forks are boot-level GEMM/autotune near-tie flips
(BATCH_INVARIANT=0 by regime; the known L0c cross-session drift class).

t0.6: OFF-vs-OFF floor itself forks at position 0 on 2/4 prompts
(cat9_t06 off_a vs off_b: {0:0, 1:12, 2:71, 3:0}); ON-vs-OFF same
character ({0:0/12, 1:12/15, 2:25, 3:0/11}) — within floor.

## Gate (3): accept/event — EXACT within-boot; cross-boot within OFF band (greedy); one flagged t0.6 residual

Within-boot: accept/event exactly equal rep1 vs rep2 everywhere.
Cross-boot exact equality is unattainable on this substrate: the OFF arm
itself moved 2.013245 -> 2.236220 (cat9 greedy) across two same-seed
boots (banked OFF = 2.151515 in between; chain5 historical flip
2.8824 vs 3.1875 per the bind's caveat).

| family | OFF values (this campaign) | banked OFF | ON | verdict |
|---|---|---|---|---|
| cat9 greedy | 2.013245, 2.236220 | 2.151515 | 2.184932 | inside OFF band |
| cat9 t0.6 | 2.081481, 2.168421 | — | 1.861111 | **below 2-sample OFF band — FLAGGED, not bound** (see caveat) |
| chain5 greedy | 3.000000 | 3.256198 (flips 2.88-3.19) | 3.007812 | inside OFF cross-boot band |
| chain5 t0.6 | 3.186992 | — | 3.126984 | within band magnitude |

cat9-t0.6 caveat: t0.6 trajectories fork at position 0 within the
OFF-vs-OFF floor itself, so accept/event is computed over non-like-for-like
token paths (playbook class 12). The demonstrated same-topology boot-to-boot
accept swing at greedy is 0.223 wide (2.013-2.236); the ON t0.6 value sits
0.220 below the lower of only two OFF t0.6 draws — within the demonstrated
swing magnitude but outside the (underpowered) 2-sample band. Attribution
NOT established; if a binding t0.6-accept verdict is needed, add OFF t0.6
boots to power the band. cat9 accept remains the pre-existing B1-2/S3
superset blocker (2.15 vs native 3.16) tracked in FR13_TRAIL.md — unchanged
in kind by FIX-1 (greedy accept moved UP on ON boots: 2.185 vs 2.013/2.236
band).

## Gate (4): regular decode == pristine — PASS BY CONSTRUCTION + floor-bracketed live check

The protocol's primary vehicle (non_mtp probe on the tree server) is
IMPOSSIBLE on this substrate: a non_mtp request on the tree server crashes
in STOCK vLLM `propose_tree` (live eagle.py:1476
`AttributeError: 'EagleProposer' object has no attribute 'positions'`) —
a pre-existing latent incompatibility, demonstrated in the FLAG-OFF arm
(cat9_off, legacy path) so not attributable to FIX-1. Evidence:
`output/fr13_b1_fix1_gate/cat9_off/docker_full.log` (EngineCore traceback;
crash AFTER all tree probes completed, so arm data unaffected).

Fallback executed (no banked pristine 128-tok no-spec reference exists):
two no-spec boots via `scripts/fr10_launch_speed_server.sh`
`FR12_NO_SPECULATIVE_CONFIG=1` (no `--speculative-config`; FLASH_ATTN
default), flag OFF vs ON, same pinned greedy probe, mode non_mtp,
spec_drafts=0 both:

- Inverse engagement (class 9): no `SpeculativeConfig` in engine-init line;
  drafter needle ABSENT in both boots (the flag's only consumer is inside
  the EagleProposer caterpillar block, which is never constructed without
  spec config — the diff is code-path-unreachable in regular decode).
- 3-boot no-spec matrix (mismatch rule executed with floor boot
  nospec_off_b), greedy first-divergence (None = byte-identical):

  | pair | p0 | p1 | p2 | p3 |
  |---|---|---|---|---|
  | nospec_off vs nospec_off_b (FLOOR) | 35 | 11 | 25 | 71 |
  | nospec_off vs nospec_on | None | 25 | 25 | 71 |
  | nospec_off_b vs nospec_on | 35 | 11 | 25 | None |

  The plain-decode cross-boot floor itself forks on ALL FOUR prompts; the
  ON boot byte-matches one OFF boot on p0 and the OTHER OFF boot on p3, and
  never forks earlier than the floor on any prompt
  (`nospec_floor_rule_attributable=false`). Regular decode is unchanged by
  FIX-1 within the measured substrate floor — consistent with the
  code-path-unreachability proof above.

## Gate (5): speed verdict (clean non-nsys FLAG-ON boots, decode_seconds/spec_drafts) — PASS

Native reference 0.2182 s/fwd (E5, FR13_B1_SPEED_ATTRIBUTION_BIND.md).

| arm | probe | s/fwd | ratio vs native | warm TPS |
|---|---|---|---|---|
| chain5 OFF | greedy / rep2 | 0.307007 / 0.306538 | 1.407x / 1.405x | 12.93 |
| **chain5 ON** | greedy / rep2 | **0.229393 / 0.229222** | **1.0513x / 1.0505x** | **17.44** |
| cat9 OFF | greedy / rep2 | 0.311795 / 0.311729 | 1.429x / 1.429x | 9.54 |
| **cat9 ON** | greedy / rep2 | **0.237306 / 0.236342** | **1.0876x / 1.0831x** | **13.28** |
| chain5 ON | t0.6 / rep2 | 0.245746 / 0.245691 | 1.126x | 16.54 |
| cat9 ON | t0.6 / rep2 | 0.307010 / 0.315938 | 1.407-1.448x | 9.27 |

chain5 ON landed INSIDE the bind's expected window (~0.222-0.235 s/fwd).
Recorded, not tuned. Remaining gap to B1-3's <=1.0x bar: ~0.011 s/fwd
(chain5) — next queue items per the bind (eager-op storm, committer packed
DtoH, conv-emulation fusion, then the joint-lm-head sub-native lever).

## ENGAGEMENT capture (step 5) — PASS

One nsys boot (chain5 FLAG-ON, LUMO_NSYS_WRAP_VLLM=1,
LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx, delay 420s, duration 240s, 3 in-window
64-tok probes; reducer based on `output/fr13_b1_kernel_attrib/analyze_kernels.py`):

- cuBLAS `internal::gemvx` bf16 (lm-head class): **5.913 calls/draft**
  (pre-fix chain5 11.10, native 5.65) — drop of ~5.19/draft, matching the
  predicted root 2->1 + 4 loop steps 2->1 (verify call untouched).
- gemvx ms/draft 89.59 (pre-fix 167.10; capture-overhead caveat on
  absolutes, the count is the gate).
- `engagement_reduce.json`; sqlite kernel rows present (zero-drop check
  asserted in reducer).

## Verdict summary

- (1) within-boot byte-identity: **PASS** (10/10 probe pairs).
- (2) post-vs-pre byte-identity: mismatch present but **bracketed by the
  measured OFF-vs-OFF cross-boot floor** on every prompt (mismatch-rule
  re-boot executed); chain5 ON byte-matches a banked OFF boot on a full
  prompt where two OFF boots fork. **Not attributable to FIX-1.**
- (3) accept/event: within-boot exact; greedy inside OFF cross-boot band
  both topologies; cat9-t0.6 flagged (underpowered band, non-like-for-like
  trajectories), not bound as a regression.
- (4) regular decode: drafter code-path unreachable without spec config
  (needle-absent proof) + live ON-vs-OFF within cross-boot floor class;
  tree-server non_mtp vehicle dead from a PRE-EXISTING stock propose_tree
  bug (documented, FLAG-OFF arm).
- (5) speed: chain5 **0.2293 s/fwd = 1.051x native** (expected window hit);
  cat9 0.2368 = 1.085x; engagement gemvx 11.10 -> **5.91**/draft.

GATE: **PASS** under the bind's own class-11 floor semantics, with the
cat9-t0.6 accept/event residual explicitly flagged for the B1-2 lossless
re-chase (which resumes after speed settles per FR13_TRAIL.md).
