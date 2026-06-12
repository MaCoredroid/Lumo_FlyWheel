# FR13 B=1 FIX-3 Lossless Gate Bind — FR13_TREE_CONV_FUSED (DRAFT, not committed)

Date: 2026-06-12 UTC. Executor: gate workflow agent (serialized GPU).
Fix under test: FIX-3 from the conv-fusion design workflow
(`research/fr13_workflows/fix3_conv_fusion_impl_wf_4f871a76.raw.json`) —
`FR13_TREE_CONV_FUSED` (default OFF; OFF = verbatim legacy emulation),
committed 1f5f37f0: the tree causal-conv emulation's per-node state
write-back loop / per-col tap loop / remap + committed-prior row math fused
into vectorized torch ops over init-time static index tables (~3.4k/5.1k
captured nodes saved per fwd; bit-exact by construction — same per-element
ops, same order; tree-only, native causal_conv1d_update untouched). This
gate decides default-ON and whether chain5 reaches <=1.0x native.

STATUS: COMPLETE — GATE PASS (with the cat9 accept-band flag recorded
below); 2 live fixes landed en route. 19 boots total this campaign
(byte-A/B preboot + 10 first-campaign arms + 1 failed cat9_on + 2 diag
captures + 1 failed diag + 2 selfcheck boots + byte-A/B postfix +
byte-A/B regate + 4 regate arms; final CPU regression 495 passed /
25 skipped).

## LIVE FIX during the gate: capture-time static-table build vs cu130 profiling capture (class 6, wiring)

First campaign attempt HALTED on the first ON boot (cat9_on, 13:38): the
container died during EngineCore init at
`determine_available_memory -> Profiling CUDA graph memory ->` (stream
capture) `-> gdn_linear_attn._forward_core:1907 ->
build_tree_conv_state_src_indices -> torch.tensor(flat, dtype=long,
device=cuda)`: "Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned" (full log
`output/fr13_b1_fix3_gate/cat9_on.fail_1338/docker_full.log`). ROOT: the
design's lazy table cache assumed "eager warmup precedes capture", but the
cu130 nightly's cudagraph-MEMORY profiling captures a dummy run BEFORE any
eager tree forward (the clean cat9_off boot confirms the one-shot TCF
needle fires AFTER the "Profiling CUDA graph memory" line) — so the first
fused forward hits the capture-time-miss path, whose fallback built the
int64 index table via a PAGEABLE host->device copy = illegal during
capture. This is the exact class-6 "init-time tables" risk named in the
plan; FIX-2 never hit it because its lazy tables are `torch.arange(...,
device=...)` (device-side fill, capture-legal). WIRING fix (fused-path
only — `build_tree_conv_state_src_indices` has no OFF-path caller, so
cat9_off arm 1 REMAINS VALID): stage through PINNED host memory and RETAIN
the staging tensor module-globally (`_CAPTURE_STAGING_RETAIN`,
`src/lumo_flywheel_serving/fr13_tree_conv_fused.py`) — a pinned-source H2D
is capture-legal but gets baked into the graph as a copy node that
re-reads the staging buffer on every replay, so the staging must outlive
the graph; the DEVICE-side table stays unretained per the capture-pool
license (the graph owns its pool address; the baked copy refills it per
replay). Values byte-identical (exact int64, no numerics). New T8
capture-replay byte test (capture the build into a graph, trash the out
buffer, replay x2, exact-equal vs eager build) added to the byte A/B suite
(now 284 cases); CPU suite 269 passed / 15 GPU-skipped. Consequence for
the needle: on cu130 the one-shot ON needle fires during the profiling
capture where the table cache is CORRECTLY not retained, so
`static_tables`/`zero_row_cached` are RECORDED not asserted
(`prepared_rows=1` — init-time prep buffers — still asserted). Campaign
resumed from cat9_on (`run_fix3_campaign_resume.sh`) after a fresh
in-container 284-case byte A/B.

## Pre-gate: GPU byte A/B (class 10) — PASS (with one test-side fix)

In the serving container (vllm/vllm-openai:cu130-nightly, torch
2.11.0+cu130, triton 3.6.0, GB10), 3 independent boots:

- Suite `tests/test_fr13_tree_conv_fused_byte_ab.py`, 283 cases. Boot 1:
  282/1 — the single failure was T6 payload SELECTION (sorted glob accepted
  an FR12-era capture whose schema predates `window_selected`; KeyError,
  NOT a byte divergence). Test-side fix (key-presence predicate; working
  tree, uncommitted per monitor-commits rule). Boot 2 post-fix: 283/283.
  Boot 3 fresh-JIT GPU-subset repeat (class 8): 14/14 incl. T5 full
  per-layer pipeline with real `triton_ex2_silu_bf16` at dim=10240, T6
  live-FR13-capture anchor, T7 B=2 disjoint-window batching. Zero int-view
  mismatches in any A/B assert across all three boots.
- Compile identity (class 10): zero NEW Triton compiles in the fused path,
  proven bidirectionally (legacy-first and fused-first orders), identical
  3-specialization `_fr13_ex2_silu_kernel` set with the same 'D'
  divisible-16 pointer specialization, identical on-disk cache census —
  the FIX-2 tt.int_to_ptr AxisInfo lesson does not recur (FIX-3 adds no
  kernel). Logs: `output/fr13_fix3_gpu_byte_ab/`.
- This campaign re-runs the FULL suite pre-boot in a fresh container
  (`output/fr13_b1_fix3_gate/byte_ab_preboot_rerun.log`) before any live
  ON boot.

## LIVE FIX 2 during the gate: PER-GROUP prep buffers vs group-local spec_state_indices (class 3/9, wiring — THE first-campaign gate-(b) failure)

The first full campaign FAILED gate (b): ON-vs-OFF streams forked at
positions 12-22 on 3 of 4 prompts in BOTH families, beyond even the
same-flag pooled floor — with the SAME fork position across all four
ON-vs-OFF pairs per prompt (p0:12, p3:13 chain5; deterministic value
change), accept/event down ~0.06-0.21, while gate (a) within-boot repeat
stayed 14/14 and ALL needles were green. Reduced evidence:
`fix3_gate_reduce.first_campaign.json`. Dual-arm EAGER capture boots
(diag_off/diag_on, FR12_SUBKERNEL_CAPTURE via the new
FR13_TCF_DIAG_OVERRIDE=1 license) showed all INT metadata identical but
float values diverging from call0 — inconclusive on its own (eager
boot-noise confound), so the decisive instrument was built: an IN-PROCESS
DUAL-PATH SELFCHECK (`FR13_TCF_SELFCHECK=1`, eager-only, the FIX-1
pattern applied to the conv section) — every fused value bitwise-compared
against a same-forward legacy recompute, boot-noise-immune. ONE boot
localized it exactly (4000 checks/stage): conv_window / conv_acc /
conv_out / conv_new_state / remap_dst_permutation / committed_read_cols
all **0 mismatches**, `committed_bank_rows` (and its dependent
committed_prior_bank) **2499/4000 = exactly 30/48 layers per forward**.
ROOT CAUSE: the design premise "row math value-identical across kv-cache
groups" is FALSE — `spec_state_indices_tensor` is kv-cache-GROUP-LOCAL
(bank rows differ per group on the cu130 3-group split). All three group
owners wrote the SAME single global prep buffer; the last writer (group
C's owner, layer 2) won, so the 30 layers of groups A/B (minus owners
0,1 which consumed their own writes, plus group C's 16 = 18 correct)
gathered the WRONG bank rows for the committed-prior window AND remapped
the WRONG rows of their banks — boot-seeded wrongness (group row
assignments vary per boot), explaining both the deterministic ON-vs-OFF
forks and the elevated ON-ON cross-boot instability (chain5 ON-ON p0:16
vs OFF-OFF 90). WIRING FIX (fused-path only; OFF arms remain valid):
**PER-GROUP prep buffers** — `_FR13_TCF_PREP` becomes
group_key->buffers, builder init exports a `_FR13_TCF_LAYER_GROUP`
layer->group map (union semantics + fail-loud on cross-init group moves),
each owner writes ITS group's slot, every layer reads its OWN group's
slot (write gate = equality with the group's owner). New fail-loud:
layer missing from the group map; legacy single-buffer export rejected.
Wiring test updated to pin the per-group semantics. POST-FIX PROOF:
selfcheck boot = **0/4000 mismatches on ALL 8 stages**, and the fixed ON
boot's p3 stream is **byte-identical (64/64) to the OFF reference**.
Re-gate: byte A/B + wiring rerun in-container (296 passed) + the 4 ON
tree arms re-run (`run_fix3_regate.sh`); contaminated arms archived as
`*.pergroup_bug`.

## Regime (canonical, mirrored from FR13_B1_FIX2_GATE_BIND.md)

PORT=9950, GPU_UTIL=0.82, MAX_NUM_SEQS=1, BATCH_INVARIANT=0,
FR13_BI_TREE_ATTN=0, FR10_METRICS=0, FR13_REPLAY_ROUTE=1;
FIX-1 + FIX-2 pinned at their committed defaults
(FR13_DRAFTER_SINGLE_LOGITS=1, FR13_EAGER_PACK=1) on EVERY arm — the ONLY
varying flag is FR13_TREE_CONV_FUSED; pinned prompts
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 prompts), seed 1313,
B=1, max_tokens 128 (warmup 1x16); greedy = t0.0 top_p 1.0; t0.6 probes x2
reps each. chain5 = 5-node spine TREE (tree_n=6 with root); cat9 = 9-node
caterpillar (tree_n=10). FULL CUDA capture proven per boot ("Graph
capturing finished"). docker ps empty + free -g before/after each arm;
docker rm -f between; launcher host-memory recovery each boot.

Arms (serialized): cat9_off -> cat9_on (FR13_FIX1_SELFCHECK=1, the free
FIX-1 drafter dual-path regression guard; its s/fwd is diagnostic-loaded =
non-binding for speed per FR13_B1_FIX1_CONFIRM_BIND.md) -> cat9_off_b ->
chain5_off -> chain5_on (clean) -> cat9_on_clean (selfcheck=0, the gate-(e)
cat9 speed boot) -> chain5_on_b (clean; the same-flag ON-ON floor pair,
planned up-front this campaign — the FIX-2 supplementary-arm lesson) ->
nospec_off -> nospec_on -> nospec_off_b (fr10 launcher,
FR12_NO_SPECULATIVE_CONFIG=1; non_mtp probe — the tree-server non_mtp
vehicle is dead from the pre-existing stock propose_tree bug documented in
FR13_B1_FIX1_GATE_BIND.md gate 4).

OFF-class banked draws used as extra floor samples (EXACTLY the fresh
OFF-arm serving config: single_logits=1 + eager_pack=1 committed defaults,
TREE_CONV_FUSED absent == OFF-verbatim, BI=0): the FIX-2 gate's FLAG-ON
boots fix2 cat9_on (selfcheck, value-pure) / cat9_on_clean / chain5_on /
chain5_on_b. fix1-era boots (EAGER_PACK=0 builds) excluded from gate
(b)/(c) banking; the fix1+fix2 no-spec boots remain in the gate-(d) pool
(flags have no consumer without a spec config — inverse-needle proven in
all campaigns).

Artifacts: `output/fr13_b1_fix3_gate/` (runners `run_fix3_arm.sh`,
`run_fix3_nospec_arm.sh`, `run_fix3_campaign.sh`; reducer
`reduce_fix3_gate.py` -> `fix3_gate_reduce.json`; per-arm dirs with
container_env, needles, metrics brackets, probe JSONs, docker_full.log).

## Engagement (class 9) — per arm — PASS

(i) drafter `single_logits=True` on all tree arms; (ii) eager-pack
`pack=1 cuda=1 layers=48 packed_dtoh_elems=133/117 replay_batched=1
stacked_rings=1 boundary_legacy_loop=0` everywhere (committed default ON);
(iii) FIX-3 needle in BOTH states — OFF: `fused=0 tree_n=10|6 width=4
state_len=12|8 prepared_rows=0 static_tables=0 zero_row_cached=0`; ON:
`fused=1 ... prepared_rows=1 static_tables=0 zero_row_cached=0`
(static_tables/zero_row RECORDED not asserted: on cu130 the one-shot
needle fires inside the profiling capture where the table cache is
correctly not retained — see live fix 1); (iv) prep re-export needle
count == builder-init count on EVERY ON boot (6 = stacked-rings-rebuilt
5 + 1; groups sequence 1,2,3,3,3,3, last at the full 3-group union; OFF
boots: 0 prep needles); (v) FIX-1 selfcheck on cat9_on only:
first-campaign 3560/0, regate 3370/0 mismatch — the FIX-1 guard stays
green under FIX-3 ON. No-spec arms: forward-path needles
(drafter/committer/conv-emulation) ALL absent + no SpeculativeConfig;
NOTE the prep re-export and stacked-rings needles are BUILDER-INIT
needles and legitimately fire on no-spec boots (degenerate b_max=1
path_cols=1, no forward consumer) — the first nospec_on run failed only
this over-strict harness assert; data valid, assert corrected in
`run_fix3_nospec_arm.sh`. Fail-loud scans clean in every arm log.

## Gate (a): within-boot same-seed repeat byte-identity (class 8) — PASS

rep1 == rep2 byte-identical on all 4 prompts x 128 tokens, greedy AND
t0.6, in ALL SEVEN tree boots (14/14 probe pairs), first campaign and
regate alike (the per-group bug was within-boot deterministic — exactly
why gate (b)'s cross-boot floor + the dual-path selfcheck were needed).

## Gate (b): OFF-vs-ON floor-bracketed stream identity (class 11) — PASS after live fix 2 (one strict-rule residual, pooled-bracketed)

FIRST CAMPAIGN: FAILED — attributable under BOTH the strict OFF-only rule
(cat9_greedy, cat9_t06, chain5_greedy, chain5_t06) and the same-flag
pooled floor (3 of 4 families), with identical fork positions across all
ON-vs-OFF pairs (deterministic value change) — the per-group prep root
cause above. Evidence banked in `fix3_gate_reduce.first_campaign.json` +
archived `*.pergroup_bug` arms.

REGATE (per-group fix): `floor_rule_attributable_same_flag_pooled` =
false x4 families. Strict rule flags ONE cell — chain5_greedy p3: ON min
69 vs OFF-OFF floor 82 — and the chain5 ON-ON pair itself forks at
exactly 69 (pooled floor 69 == ON min 69): a same-build substrate fork
site, the established FIX-2-bind pattern. cat9 regate ON-vs-OFF forks
(17/15-27/25-61/61-117) sit AT or BEYOND the OFF-OFF floor
(17/15/25/61) on every prompt. Decisive post-fix byte coups: the eager
selfcheck ON boot's p3 stream is byte-identical 64/64 to the OFF
reference (which is itself byte-identical across eager/captured boots:
diag_off ≡ cat9_off_b 64/64), and the in-process dual-path selfcheck is
0/4000 on all 8 stages. NOT attributable to FIX-3 post-fix.

## Gate (c): accept/event within same-flag band — chain5 PASS; cat9 FLAGGED (below 4-draw band, inside the demonstrated same-flag swing)

| family | OFF band (fresh + banked, 3-4 draws) | ON values | verdict |
|---|---|---|---|
| cat9 greedy | [2.1515, 2.2482] | on 2.1282, on_clean 2.1032 | -0.023/-0.048 below band; BOTH inside the FIX-1-demonstrated same-flag greedy swing [2.0132, 2.2362] |
| cat9 t0.6 | [2.0671, 2.2206] | on 1.9565, on_clean 1.9536 | -0.11 below the 4-draw band; inside the FIX-1-campaign same-flag t0.6 band [1.8611, 2.1491]; class-12 caveat (t0.6 accept over non-like-for-like trajectories) |
| chain5 greedy | [2.6596, 2.9922] | on 3.0391, on_b 2.6596 | on +0.047 ABOVE band (favorable), on_b at band edge (inside) |
| chain5 t0.6 | [2.9313, 3.0154] | on 2.9618 (inside), on_b 2.9167 | on_b -0.0146 below (1/6 band width) |

cat9 flag honestly recorded: both fresh ON draws sit below the 4-draw
band in both temperatures (unfavorable direction), but within the
historically demonstrated same-flag swings of this exact substrate, and
the gate-(b) stream evidence (forks at/beyond the OFF floor, byte
coups) shows no value change. Not bound as a FIX-3 regression; flagged
for the next campaign's banked-draw pool to watch.

## Gate (d): regular-decode pristine (no-spec matrix) — PASS (pooled floor exact)

3 fresh boots (fr10 launcher, FR12_NO_SPECULATIVE_CONFIG=1, greedy
128-tok pinned probe, spec_drafts=0): the strict fresh-floor rule flags
(3 draws are a class-12 trap, as in FIX-2); pooling the fix1 + fix2
campaigns' banked no-spec boots (8-draw OFF-class pool; flags have no
consumer without a spec config — inverse forward-needles proven in all
three campaigns) gives `nospec_pooled_floor_rule_attributable` = false
(pooled floor brackets the ON minima exactly). Regular decode unchanged
by FIX-3 within the measured substrate floor.

## Gate (e): speed (raw counters, decode_seconds/spec_drafts) — cat9 BEATS window; chain5 in/near window; <=1.0x NOT reached

Native ref 0.2182 s/fwd (E5). Greedy clean-boot regate numbers
(rep1/rep2):

| arm | s/fwd | ratio vs native | warm TPS |
|---|---|---|---|
| chain5 OFF | 0.226061 / 0.225691 | 1.0360x / 1.0343x | 17.7 |
| **chain5 ON** | **0.224696 / 0.224632** | **1.0298x / 1.0295x** | 17.8 |
| **chain5 ON_b** | **0.222562 / 0.222296** | **1.0200x / 1.0188x** | 17.9 |
| cat9 OFF | 0.232775 / 0.232674 | 1.0668x / 1.0663x | 13.9 |
| cat9 OFF_b | 0.232254 / 0.232045 | 1.0644x / 1.0634x | 14.0 |
| **cat9 ON (clean)** | **0.224901 / 0.224730** | **1.0307x / 1.0299x** | 14.4 |
| cat9 ON (FIX-1 selfcheck) | 0.306394-0.306509 | diagnostic-loaded, non-binding | — |

cat9: ON_clean 0.2247-0.2249 — BELOW the design window (0.227-0.232),
~7.5-8.0 ms/fwd saved vs the fresh OFF pool (0.23205-0.23278), fully
separated from the OFF spread; cat9 ratio 1.077x (post-FIX-2) -> 1.030x.
chain5: ON_b 0.2223-0.2226 inside the design window (0.218-0.223), ON
0.2246-0.2247 just above it; saving vs fresh OFF ~1.0-3.4 ms; best
chain5 ratio 1.033x (post-FIX-2) -> **1.0188x**. THE <=1.0x QUESTION:
**NOT reached** — chain5 ON draws are 1.019-1.030x, i.e. ~4.1-6.5 ms/fwd
above native; the conv-emulation reclaim landed at the modest end of the
design estimate on chain5 (its conv share was the smaller ~7 ms) while
cat9 (the larger ~17 ms residual) over-delivered. Recorded honestly, no
tuning. t0.6 s/fwd carries wall jitter between identical-stream reps —
greedy is the speed basis (class 12).

## Verdict

- (a) within-boot repeat: **PASS** 14/14.
- (b) streams: **PASS post live-fix-2** — same-flag pooled floor clean
  x4 families; single strict-rule residual (chain5_greedy p3:69)
  bracketed exactly by the ON-ON pooled floor; post-fix dual-path
  selfcheck 0/4000 x8 stages + ON≡OFF p3 byte-identity. The FIRST
  campaign's attributable failure was real, root-caused (group-local
  spec_state_indices vs shared prep buffer) and fixed (per-group prep).
- (c) accept/event: chain5 within/above band; **cat9 flagged** (below
  the 4-draw band both temperatures, inside the demonstrated same-flag
  swings) — recorded, not bound.
- (d) regular decode: pooled-floor exact bracketing + inverse
  engagement. **PASS.**
- (e) speed: cat9 0.2247-0.2249 (**1.030x**, window beaten, ~7.7 ms
  saved); chain5 0.2223-0.2247 (**1.019-1.030x**, ~1-3.4 ms saved).
  **chain5 <=1.0x NOT reached** — the remaining ~4-6.5 ms/fwd gap is no
  longer conv-emulation-dominated (FIX-3 banked); next ranked residual
  per the census = remaining eager committer serialization + tree-mode
  graph/scheduler path (FR13_B1_TRACELESS_SPEED_BIND.md).
- (f) engagement: all needles correct in both states incl. the per-init
  prep re-export count (6 = rebuilds+1, last at groups=3); FIX-1
  selfcheck green under FIX-3 ON (3370/0); inverse needles clean; FULL
  capture every gate boot.

GATE: **PASS** under the established FIX-1/FIX-2 floor semantics — with
TWO live fixes that are prerequisites for any default-ON flip:
(1) capture-time pinned-staging table build (cu130 profiling-capture
crash, `src/lumo_flywheel_serving/fr13_tree_conv_fused.py` + byte-A/B T8);
(2) PER-GROUP prep buffers (group-local spec_state_indices — the
first-campaign lossless failure, `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
builder init + forward consume + `_FR13_TCF_LAYER_GROUP` map + fail-louds,
wiring test updated). Plus the FR13_TCF_SELFCHECK dual-path instrument
(eager-only, log-only) and the FR13_TCF_DIAG_OVERRIDE capture license —
both default-OFF diagnostics. Default-ON decision = monitor/user (this
workflow does not commit); note the cat9 accept-band flag and the
unmet <=1.0x bar in that decision.
