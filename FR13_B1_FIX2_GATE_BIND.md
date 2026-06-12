# FR13 B=1 FIX-2 Lossless Gate Bind — FR13_EAGER_PACK (DRAFT, not committed)

Date: 2026-06-12 UTC. Executor: gate workflow agent (serialized GPU).
Fix under test: FIX-2 from the eager-storm design workflow
(`research/fr13_workflows/fix2_eager_pack_impl_wf_511ed73b.raw.json`) —
`FR13_EAGER_PACK` (default OFF; OFF = verbatim legacy), committed 6c2f46d6:
packed committer DtoH (102->1), batched all-layer replay launch (96->2),
pinned HtoD staging, runner metadata cache, sampler quick wins, verified
trivials. This gate decides default-ON.

STATUS: CAMPAIGN IN FLIGHT — sections below fill as arms land.

## LIVE FIX during the gate: single-group guard vs builder RE-INIT (class 9 guard, wiring)

First campaign attempt HALTED on the first ON boot (cat9_on, 08:56): the
container died during EngineCore init,
`RuntimeError: FR13_EAGER_PACK requires a single GDN replay group; a second
metadata-builder group cannot join the stacked rings (fail-loud, class 9)`.
Reproduced once with full-log capture
(`output/fr13_b1_fix2_gate/diag_cat9_on/docker_full.log`); traceback pins the
raise at the cu130-nightly's
`profile_cudagraph_memory -> _init_minimal_kv_cache_for_profiling ->
initialize_kv_cache(minimal_config, is_profiling=True) ->
initialize_metadata_builders` — i.e. the nightly constructs the GDN
metadata builder SEVERAL times per boot for the SAME single GDN group
(profile run, cudagraph-memory profiling minimal-KV init, real KV init;
container source: `gpu_model_runner.py` `_init_minimal_kv_cache_for_profiling`
builds the FULL layer-set groups with minimal block count, and
`_cleanup_profiling_kv_cache` clears `attn_groups` so the real init
re-creates builders). A first guard relaxation (allow same-LAYER-SET re-init only) failed the
SECOND cat9_on boot at the same site with the layer-set check firing —
proving the incoming group's layer set genuinely differs. Container source
(`kv_cache_utils.py _get_kv_cache_groups_uniform_page_size`) confirms the
real topology: hybrid models are grouped by repeating pattern with equal
layers per group, so Qwen3-Next's 48 GDN + 16 full-attn = **3 separate GDN
kv-cache groups of 16 layers each**, each with its OWN metadata builder —
the FIX-2 design's "requires a single GDN replay group" premise was wrong
for this model (the legacy merge comment even says "a second builder for
another GDN group must not drop the first group's layers"). The implVerify
minor finding (b) ("publishes stacks before the guard raise —
overwrite-then-die") was smoke from this seam. WIRING fix (eager-pack
branch only, no OFF-path execution change, no computed-value change in
either state): **GROUP-UNION + RE-INIT semantics** — every GDN builder
init rebuilds the stacked rings from the UNION of all GDN replay layers
registered so far (sorted by name = the committer's iteration order),
re-binds EVERY union layer's layer-slice views and REPLACES
`_FR13_EAGER_PACK_STACKS`. Commits and the real CUDA capture happen only
after the LAST init and the committer re-reads the module attr per commit,
so replacement is capture-safe (class 6). Fail-loud retained: non-GDN
layer in a group, non-uniform dims/dtype across the UNION, and a new
union invariant (stacks may never drop a registered layer); the
commit-time `stacked layer order != sorted(_FR13_REPLAY_LAYERS)`
validation still binds end-to-end consistency. An info needle
"FR13_EAGER_PACK stacked rings rebuilt at metadata-builder init: union N
layers (group adds M)" fires on every subsequent init. Files:
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` (builder init block + union
invariant), `tests/test_fr13_eager_pack_wiring.py` (union semantics
pinned; single-group raise asserted ABSENT). CPU regression after fix:
214 passed / 10 skipped (unchanged). cat9_off (arm 1, completed pre-fix)
REMAINS VALID: the diff is init-time logic inside the `_fr13_eager_pack`
branch; the OFF arm's serving path is untouched. Campaign resumed from
cat9_on (`run_fix2_campaign_resume.sh`).

## Pre-boot gate: 2b batched-replay byte A/B (class 10) — PASS (with fix)

Run BEFORE any FR13_EAGER_PACK=1 boot, in the serving container image
(vllm/vllm-openai:cu130-nightly, torch 2.11.0+cu130, triton 3.6.0, GB10):

- INITIAL (as-committed 6c2f46d6): **3 FAILED / 1 passed** — the batched
  kernel's `tl.load(bank_ptrs+pid_l).to(tl.pointer_type(tl.float32))` loses
  AxisInfo divisibility through `tt.int_to_ptr` on the container Triton
  3.6.0, scalarizing the whole kernel layout (sizePerThread [1,1] vs [1,4],
  88 scalar st.global.b32 vs 22 st.global.v4.b32) => different `tl.sum`
  reduction trees in `_gdn_node_step` => fp32 rounding-level divergence
  (max_abs 2.38e-7..3.58e-7, ~63% of diffs <=1 ULP) on exactly the published
  path rows. Case 1515 additionally exposed a TEST-construction write-write
  race (both batch rows shared spec_state_indices rows — playbook class 3),
  fixed in the test.
- FIX (working tree, 4 files): kernel params `bank_ptrs` ->
  `bank_anchor` (layer-0 bank as real pointer arg = divisibility-16 anchor)
  + `bank_off16` (int64 table of (data_ptr-anchor)//16);
  `state_bank = bank_anchor + tl.load(bank_off16+pid_l)*4` (tt.addptr path,
  AxisInfo exact). Post-fix: identical fp32 instruction histograms, identical
  TTGIR layouts vs legacy; sole delta = 1x ld.global.b64 (offset-table read).
- FINAL: `test_fr13_eager_pack_replay_byte_ab.py` **4 passed**, repeated in a
  second independent container boot (fresh JIT), and re-confirmed a THIRD
  time by this gate workflow pre-boot
  (`output/fr13_b1_fix2_gate/byte_ab_preboot_rerun.log`: 4 passed in 7.59s).
- CPU regression: 214 passed / 10 skipped across tests/test_fr10*.py +
  test_fr13*.py.

## Regime (canonical, mirrored from FR13_B1_FIX1_GATE_BIND.md)

PORT=9950, GPU_UTIL=0.82, MAX_NUM_SEQS=1, BATCH_INVARIANT=0,
FR13_BI_TREE_ATTN=0, FR10_METRICS=0, FR13_REPLAY_ROUTE=1;
FR13_DRAFTER_SINGLE_LOGITS pinned at its committed default (1) on EVERY arm
— the ONLY varying flag is FR13_EAGER_PACK; pinned prompts
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 prompts), seed 1313,
B=1, max_tokens 128 (warmup 1x16); greedy = t0.0 top_p 1.0; t0.6 probes x2
reps each. chain5 = 5-node spine TREE; cat9 = 9-node caterpillar. FULL CUDA
capture proven per boot ("Graph capturing finished"). docker ps empty +
free -g before/after each arm; docker rm -f between; launcher host-memory
recovery each boot.

Arms (serialized): cat9_off -> cat9_on (FR13_FIX1_SELFCHECK=1, the free
FIX-1 drafter dual-path regression guard; its s/fwd is diagnostic-loaded =
non-binding for speed per FR13_B1_FIX1_CONFIRM_BIND.md) -> cat9_off_b ->
chain5_off -> chain5_on (clean) -> cat9_on_clean (selfcheck=0, the gate-(e)
cat9 speed boot) -> nospec_off -> nospec_on -> nospec_off_b
(fr10 launcher, FR12_NO_SPECULATIVE_CONFIG=1; non_mtp probe — the tree-server
non_mtp vehicle is dead from the pre-existing stock propose_tree bug
documented in FR13_B1_FIX1_GATE_BIND.md gate 4).

OFF-class banked draws used as extra floor samples (same serving config:
single_logits=1 default, EAGER_PACK absent == OFF-verbatim, BI=0):
fix1_gate cat9_on / chain5_on, fix1_confirm cat9_sc / chain5_sc.

Artifacts: `output/fr13_b1_fix2_gate/` (runners `run_fix2_arm.sh`,
`run_fix2_nospec_arm.sh`, `run_fix2_campaign.sh`; reducer
`reduce_fix2_gate.py` -> `fix2_gate_reduce.json`; per-arm dirs with
container_env, needles, metrics brackets, probe JSONs, docker_full.log).

## Supplementary arm

`chain5_on_b` (clean ON, EDC=5, selfcheck=0) was added after the planned
arms to supply the missing chain5 same-flag ON-ON pair for the t0.6 floor
question (below). Same runner, same regime. It also doubles the chain5
gate-(e) draw. The campaign driver was killed externally once mid-probe
(harness background-task stop, 10:38); the nospec_off boot was VALID and
healthy, so the probe + post-steps were finished by a detached takeover
driver (`run_fix2_nospec_takeover.sh`; artifacts identical in form) and
the remaining arms ran detached. 13 boots total this campaign (incl. the
2 archived cat9_on failures + 1 diag boot).

## Engagement (class 9) — per arm — PASS

Eager-pack needle (rejection_sampler.py:759) fires in BOTH states:

| arm | needle state | rebuilt needle |
|---|---|---|
| cat9_off / cat9_off_b / chain5_off | `pack=0 cuda=1 layers=0 packed_dtoh_elems=0 replay_batched=0 stacked_rings=0 boundary_legacy_loop=0` | 0 (OFF: union code not active) |
| cat9_on / cat9_on_clean | `pack=1 cuda=1 layers=48 packed_dtoh_elems=133 replay_batched=1 stacked_rings=1 boundary_legacy_loop=0` | 5x "stacked rings rebuilt" |
| chain5_on / chain5_on_b | `pack=1 cuda=1 layers=48 packed_dtoh_elems=117 replay_batched=1 stacked_rings=1 boundary_legacy_loop=0` | 5x |

(133 vs 117 packed elems = 9-node vs 5-node tree pack size; the 5 rebuilds
per ON boot = profiling-init groups B,C + real-init groups A,B,C — exactly
the predicted multi-group/multi-init topology.) Drafter needle
`single_logits=True` on all 7 tree arms (committed default pinned).
Selfcheck (cat9_on only): needle present; final dump
`{"steps_checked": 3290, "rows_checked": 3290, "mismatch_steps": 0}` —
the FIX-1 dual-path guard stayed green under FIX-2 ON. No-spec arms:
BOTH needles ABSENT (inverse engagement) + no `SpeculativeConfig` in
engine init. Container env flags asserted per arm before any probe.
FULL capture proven per boot (`cudagraph_mode FULL_AND_PIECEWISE`,
"Graph capturing finished" asserted). Fail-loud scan (EAGER_PACK
RuntimeError / SELFCHECK MISMATCH) clean in every arm log.

## Gate (a): within-boot same-seed repeat byte-identity (class 8) — PASS

rep1 == rep2 byte-identical on all 4 prompts x 128 tokens, greedy AND
t0.6, in ALL SEVEN tree boots (14/14 probe pairs; `gate_a_within_boot_repeat`
all true). Accept/event exactly equal within boot everywhere.

## Gate (b): OFF-vs-ON floor-bracketed stream identity (class 11) — PASS (not attributable)

GREEDY: clean under the strict OFF-only floor rule on every prompt, both
families (`floor_rule_attributable`: cat9_greedy=false,
chain5_greedy=false). Earliest ON-vs-OFF fork == earliest OFF-vs-OFF fork
position on every prompt. Byte-identity coups: `chain5_on_b` is
**byte-identical on ALL FOUR prompts** to banked OFF-class boot
fix1_confirm `chain5_sc` (ONOFF pair = None x4) while OFF-vs-OFF pairs
fork at 11-116; cat9 ON boots interleave inside the OFF equivalence class
(e.g. off_a_vs_on p2=84 vs OFF floor 21).

t0.6: the strict OFF-only rule flags 3 cells (cat9 p1: ON 11 vs floor 12;
cat9 p3: ON 12 vs floor 57; chain5 p1: ON 13 vs floor 25) — all THREE are
bracketed by the SAME-FLAG pooled floor (OFF-OFF ∪ ON-ON pairs, the
class-11-correct substrate floor): the cat9 ON-ON pair itself forks at
p1:11 / p3:12 and the chain5 ON-ON pair at p1:13 — the exact flagged
positions are same-build cross-boot fork sites
(`floor_rule_attributable_same_flag_pooled` = false x4 families).
Decisive: `chain5_on_b` t0.6 is **byte-identical on ALL FOUR prompts** to
banked OFF boot `chain5_sc` (and to banked fix1on on p1/p3) — an ON boot
landing exactly inside an OFF trajectory equivalence class at temperature
0.6 is incompatible with any deterministic ON value change. Mechanism
check: the sampled committer (`_lumo_tree_canonical_multidraft_sample`)
has zero flag references (implVerify pass 6) and 2g is greedy-gated — no
t0.6-only mechanism exists. NOT attributable to FIX-2.

## Gate (c): accept/event within same-flag band — PASS (one flagged residual, favorable)

| family | OFF band (fresh + banked, 3-4 draws) | ON values | verdict |
|---|---|---|---|
| cat9 greedy | [2.0551, 2.2624] | on 2.1515, on_clean 2.1898 | inside |
| cat9 t0.6 | [1.8611, 2.1491] | on 2.0671, **on_clean 2.2206** | on inside; on_clean **+0.0715 ABOVE band — FLAGGED, not bound** |
| chain5 greedy | [2.6596, 3.0078] | on 2.8358, on_b 2.6596 | inside |
| chain5 t0.6 | [2.9098, 3.1270] | on 3.0000, on_b 2.9313 | inside |

cat9-t0.6 caveat (same class-12 caveat as the FIX-1 bind, opposite and
favorable direction): t0.6 accept is computed over non-like-for-like
sampled trajectories; the 4-draw band is 0.288 wide and the residual is
0.0715 above its top. The cat9_on (selfcheck) t0.6 draw sits mid-band.
Not bound as a regression; accept moved UP if anything.

## Gate (d): regular-decode pristine (no-spec matrix) — PASS (pooled floor, exact bracketing)

3 fresh boots (fr10 launcher, `FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN,
greedy 128-tok pinned probe, spec_drafts=0 all): fresh-floor rule flags
ON minima 11/25/71 vs the single fresh OFF pair's 13/28/78 — but pooling
the FIX-1 campaign's 3 banked no-spec boots (same launcher/config class;
drafter/committer flags have NO consumer without spec config —
inverse-needle proven in both campaigns) gives a 5-draw OFF-class pool
whose floor is **exactly** the ON minima: pooled floor p0:35 p1:11 p2:25
p3:71 == ON min 35/11/25/71 (`nospec_pooled_floor_rule_attributable` =
false). The flagged positions are prompt-intrinsic near-tie flip sites
recurring across campaigns (11/25/35/71/90 appear in OFF-OFF pairs of
both). Byte-identity coups: fix1:nospec_off ≡ fix2:nospec_off_b on ALL
FOUR prompts (cross-campaign OFF pair fully identical);
fix2:nospec_on ≡ fix2:nospec_off_b on p0, ≡ fix1:nospec_on on p0+p3.
Regular decode is unchanged by FIX-2 within the measured substrate floor,
consistent with code-path unreachability. Warm TPS 7.833/7.831/7.823
(off/on/off_b) — indistinguishable.

## Gate (e): speed (raw counters, decode_seconds/spec_drafts) — chain5 IN WINDOW; cat9 0.5-0.9 ms above window top

Native ref 0.2182 s/fwd (E5). Post-FIX-1 baselines: chain5 ON 0.2294
(1.051x), cat9 ON 0.2373 (1.088x). Design windows: chain5 0.225-0.227,
cat9 0.232-0.234 (est 3-5 ms; residual bulk = FIX-3 conv-emulation, not
chased here). Greedy clean-boot numbers (rep1/rep2):

| arm | s/fwd | ratio vs native | warm TPS |
|---|---|---|---|
| chain5 OFF | 0.229193 / 0.229308 | 1.0504x / 1.0509x | 16.66 |
| **chain5 ON** | **0.225708 / 0.225429** | **1.0344x / 1.0331x** | **16.93** |
| **chain5 ON_b** | **0.226499 / 0.226204** | **1.0380x / 1.0367x** | 16.87 |
| cat9 OFF | 0.234519 / 0.234263 | 1.0748x / 1.0736x | 13.32 |
| cat9 OFF_b | 0.237294 / 0.237134 | 1.0875x / 1.0868x | 13.17 |
| **cat9 ON (clean)** | **0.234884 / 0.234743** | **1.0765x / 1.0758x** | 13.31 |
| cat9 ON (selfcheck) | 0.305703 / 0.305564 | diagnostic-loaded, non-binding | — |

chain5: BOTH fresh ON draws inside the design window; saving vs the
OFF-class pool (fresh 0.2292-0.2293 + FIX-1-class 0.2292-0.2294) =
~3.0-3.9 ms/fwd, fully separated from boot noise. New best tree ratio:
**1.033x native**. cat9: ON 0.23474-0.23488 vs window top 0.234 — missed
by 0.5-0.9 ms. The cat9 OFF-class pool spans 0.23452-0.23731 (fresh OFF_a
draw 0.2345 at the LOW edge), so the cat9 saving (vs 3 of 4 OFF draws
~2.4-2.6 ms) is NOT separable from the widest OFF boot-to-boot spread.
Recorded, not tuned (per the bind contract; the residual is FIX-3
territory). t0.6 s/fwd carries wall jitter between identical-stream reps
(e.g. cat9_off 0.310 vs 0.294) — greedy is the speed basis (class 12).

## Verdict

- (a) within-boot repeat: **PASS** 14/14.
- (b) streams: greedy floor-bracketed clean; t0.6 strict-rule residuals
  all bracketed by the same-flag pooled floor + a full 4-prompt ON≡OFF
  byte-identity at both temperatures. **Not attributable to FIX-2.**
- (c) accept/event: within band 7/8; cat9-t0.6 on_clean +0.0715 above an
  0.288-wide band — flagged, not bound (favorable direction).
- (d) regular decode: pooled-floor exact bracketing + cross-campaign
  OFF≡OFF and ON≡OFF byte identities + inverse engagement. **PASS.**
- (e) speed: chain5 0.2254-0.2265 (window hit, 1.033x native, ~3-4 ms
  saved); cat9 0.2347-0.2349 (window missed by 0.5-0.9 ms; saving inside
  OFF spread). Honest miss recorded.
- (f) engagement: all needles correct in both states; selfcheck 3290/0;
  inverse needles clean; FULL capture every boot.

GATE: **PASS** under the established FIX-1 floor semantics — FIX-2
(FR13_EAGER_PACK) is lossless within the measured substrate floors with
full engagement proof; speed delivers in-window on chain5 and a marginal
window miss on cat9. The two LIVE FIXES this campaign (byte-A/B
anchor+offset 834bab16; builder group-union/re-init, working tree) are
prerequisites for any default-ON flip. Default-ON decision = monitor/user
(this workflow does not commit).
