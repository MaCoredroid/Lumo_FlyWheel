# FR13 frontier branch disposition — 2026-08-09

Offline, read-only disposition review of the six outstanding 20260805-era
frontier branches. **No GPU was touched. No serving code was changed by this
artifact.** This is a record of merge/archive decisions, not a performance
result and not an acceptance run.

Baseline: `origin/main` at `7263c134d` (2026-08-08T22:45:37Z).

Archive here means **recorded, not deleted**. Every branch stays on `origin`
at the tip SHA named below and git preserves the full history. Nothing in this
artifact removes a ref.

---

## 0. Outcome

**All six branches are ARCHIVED. Zero merges to main.**

Raw ahead-counts overstate how much of this work is actually outstanding. Once
patch-equivalence is taken into account (`git cherry -v origin/main`), **only
eight commits across all six branches are genuinely absent from main**, and five
of those eight are on the one branch the drafter lock rejects.

| # | Branch | Tip | Ahead / behind | Genuinely new | Verdict | Governing evidence |
|---:|---|---|---:|---:|---|---|
| 1 | `codex/b1-bm8-composed-main-20260805` | `e703dcb58` | 7 / 232 | 5 of 5 non-merge | **ARCHIVE** | Drafter-side (drafter lock); tip self-declares unqualified; depends on three purge-list families |
| 2 | `codex/verifier-head-m32-sm121a-20260805` | `f770ab642` | 6 / 233 | **2 of 6** | **ARCHIVE** | Verifier-head near-worthless — 2.8 ms/step removable ceiling |
| 3 | `codex/integrate-gdn-bv16-main-20260805` | `bb7580e9c` | 3 / 82 | **0 of 3** | **ARCHIVE (absorbed)** | BV16 byte-rejected; already cherry-picked onto main |
| 4 | `codex/integrate-dfwd-u8-exact-taw-20260805` | `87344abdc` | **0** / 90 | **0** | **ARCHIVE (no-op)** | Strict ancestor of `origin/main`; fully duplicated |
| 5 | `codex/b4-physical32-next-kernel-20260805` | `3ff9c7b2e` | 2 / 97 | **1 of 2** | **ARCHIVE** | Target-GEMM scheduling levers measured dead (ladder rows 7–9) |
| 6 | `agent/fixed32-floor-publish-20260730` | `53256ac84` | **0** / 876 | **0** | **ARCHIVE (no-op)** | Strict ancestor of `origin/main`; fully duplicated |

Branches 3, 4 and 6 carry **no unduplicated work whatsoever**; archiving them is
bookkeeping, not rejection. Branches 2 and 5 are mostly absorbed already, with
only the residue named in their sections below. Branch 1 is the only one whose
content is substantially outstanding, and it is the one the drafter lock bars.

---

## 1. `codex/b1-bm8-composed-main-20260805` — ARCHIVE

7 commits ahead (2 of them merge commits from main), 232 behind.

The disposition question was whether BM8 is verify-side (mergeable) or
drafter-side (archive per the drafter lock: BF16 + K64 + suffix). **BM8 is
drafter-side.** Four independent lines of evidence converge:

1. **The kernel is installed through the drafter's replay hook.** The banked
   byte PASS artifact `results/fr13_fixed32_bm8_b1_live_pass_20260731T180804Z`
   states the candidate for `kernel_unified_attention_2d` used
   `BLOCK_M=8, BLOCK_Q=1` against stock `BLOCK_M=16, BLOCK_Q=2`, and that the
   candidate identity "binds the executed source commit plus the patcher,
   emitted unified-attention source, and **Eagle replay-hook** hashes". Eagle is
   the speculative drafter.
2. **The patcher arms it inside the drafter graph capture.** The helper block in
   `scripts/fr10_phase4_patch_vllm_tree_gdn.py` is headed
   `# FR13_DFWD_UNIFIED_BM8_LIVE_GATE: diagnostic-only B1 **MTP** attention gate.`
   `_fr13_dfwd_unified_bm8_production_begin` is docstring'd *"Arm BM8 only for
   the attested final fixed32 B1 **drafter** capture"* and gates on
   `_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT` and
   `_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT`.
3. **The acronym is fixed by the repo's own decomposition.**
   `FR13_PIPELINE_SPEED_BREAKDOWN.md` defines `DFWD=propose drafter` and
   `DFWD_GPU_TIMER: the MTP head forwards`.
4. **The branch's added accounting is drafter accounting.** The new
   `_fr13_dfwd_unified_bm8_production_finalize` reconciles
   `drafter_graph_signature`, `drafter_runtime`, `graph_replays`, and
   `mtp_forward_calls == 4`.

So the premise that BM8 is the unified-attention kernel *on the verify side* does
not hold. It is the unified-attention kernel *inside the drafter's MTP forward*.
Under the drafter lock this is drafter-quality work and is archived.

Three further facts make merging actively undesirable even setting the lock
aside:

- **The tip is explicitly unqualified.** `results/fr13_fixed32_bm8_final_flush_qrow_v3_ready_20260805/README.md`
  reads `Status: SOURCE READY; LIVE QUALIFICATION BLOCKED BY QROW SPLIT2
  WRAPPER` and closes: *"No GPU, Docker, synthetic probe, real task, byte
  comparison, timing, TPS, acceptance, or hardware-floor measurement was run for
  this source artifact. It makes no kernel-qualification or performance claim."*
- **It hard-depends on three families being purged.** The branch's own composed
  tuple is documented in
  `results/fr13_b1_bm8_composed_source_ready_20260805/README.md` as admitting
  nonstock composition only for *"the exact B1 K64/root1 **Qrow32 split2**,
  GQA3, DFWD top3, **wide256 target GEMM**, SFWD, **TAW**, CFWD, and BM8
  production tuple"*. Qrow32 split2, wide256, and fused floating TAW are all on
  the standing purge list. Merging BM8 would re-entrench exactly the code the
  companion purge removes, and its stated blocker is the split2 wrapper itself.
- **The composed stack it feeds measured null.** See §7.

The banked byte PASS (`fr13_fixed32_bm8_b1_live_pass_20260731T180804Z`, status
`KERNEL_LIVE_PASS_WITH_POSTVALIDATOR_PERMISSION_DEFECT`, *"Stock output was
served throughout; the candidate was not production-enabled"*) predates the
branch and is **already on main**. Archiving the branch does not discard it.

The existing BM8 plumbing on main is untouched by this disposition and remains
default-OFF (canonical env pins `FR13_DFWD_UNIFIED_BM8_LIVE_AB=0` and
`FR13_DFWD_UNIFIED_BM8_PRODUCTION=0`).

## 2. `codex/verifier-head-m32-sm121a-20260805` — ARCHIVE

6 commits ahead, 233 behind — but **4 of the 6 are already patch-equivalent on
main**. `git cherry -v origin/main` marks these as upstream:

- `1c7189f97` results: next-kernel evidence audit
- `3d80fcf7b` feat: stage fixed32 verifier-head M32 kernel
- `ffbc2c7f1` fix: correct verifier-head CUTLASS orientation
- `981b31dba` docs: correct verifier-head tile description

The kernel itself is therefore **already on main** —
`csrc/fr13_bf16_verifier_head_m32_sm121a.cu` landed via `b3cd2c5a9`, together
with `scripts/fr13_build_bf16_verifier_head_m32_sm121a.py` and
`tests/test_fr13_bf16_verifier_head_m32_sm121a.py`.

Only two commits are genuinely outstanding (`7261648e9`, `f770ab642`), and what
they add on top of the merged kernel is the **shadow-gate wiring**:
`scripts/fr13_verifier_head_m32_gate.py`,
`scripts/fr13_run_b1_verifier_head_m32_live_gate.sh`,
`tests/test_fr13_verifier_head_m32_gate.py`, and the
`results/fr13_fixed32_b1_verifier_head_m32_sm121a_build_20260805/` dir. Archiving
declines to add gate wiring for a target measured not worth attacking; the
already-merged kernel is a separate matter and is handled as an M32 purge
candidate.

Archived on the measured efficiency verdict: the verifier head is near-worthless
as a target, with **2.8 ms/step removable**. Corroborated by
`results/fr13_attack_ladder_analysis_20260808`, which measures the POSTPROCESS
phase (the bf16 LM head, one `nvjet_sm121_tst_mma_128x208x64`) at
**12.339 ms/step GPU-busy with 0.000 ms idle** at **206 GB/s** against a
273 GB/s LPDDR5X roofline. The phase is already memory-bound and gapless, so
even a perfect head recovers only the roofline shortfall — a low-single-digit
ms/step ceiling on a 237.248 ms/step envelope, under ~1.2%.

Documented, not merged. The kernel source stays on the branch at `f770ab642`.

## 3. `codex/integrate-gdn-bv16-main-20260805` — ARCHIVE (already absorbed)

3 commits ahead by `rev-list`, 82 behind — but `git cherry -v origin/main` marks
**all three as already upstream**. The branch is fully absorbed:
`4933b4f56`, `901e39bbd` and `bb7580e9c` are each patch-equivalent to work
already cherry-picked onto main.

Verified directly rather than trusting the mark: the GQA3 BV16 candidate is live
on main today — `gqa_group3_bv16` appears in the BV allowlist at
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:8441` and in the admission tuples at
lines 5195 / 5412 / 5704, with the candidate module at
`src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py` and the artifact test at
`tests/test_fr13_fixed32_gdn_gqa_group3_bv16_artifact.py`.

So there is nothing here to merge or to decline. The standing BV16 byte
rejection is not enforced by refusing this branch — it is enforced by **removing
the already-merged BV16 code from main**, which is the companion purge's job.
This entry records that the branch is closed and that the live code, not the
branch, is the actual object of the rejection.

Checked for salvage anyway: `bb7580e9c` "Preserve fixed32 ordered admission
diagnostic" (3 insertions / 2 deletions in
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`) only rewords a
`RuntimeError` string to name the "admitted GQA3 BV16" contract. It has no value
once BV16 is gone.

## 4. `codex/integrate-dfwd-u8-exact-taw-20260805` — ARCHIVE (no-op)

**0 commits ahead**, 90 behind. `git merge-base --is-ancestor
origin/codex/integrate-dfwd-u8-exact-taw-20260805 origin/main` returns true: the
branch tip is a strict ancestor of `origin/main`.

The dedupe requested for this branch resolves to the empty set — main's
cherry-picks already absorbed all of it. A merge would report "Already up to
date". There are no unduplicated commits, and therefore no question of
drafter-quality work to filter. Recorded and closed.

## 5. `codex/b4-physical32-next-kernel-20260805` — ARCHIVE

2 commits ahead, 97 behind. `git cherry -v origin/main` marks the results commit
(`3ff9c7b2e`) as already upstream, leaving **one genuinely new commit**,
`1e07e0c24` "perf(fr13): specialize exact B4 projection schedules", which touches
`scripts/fr13_patch_cutlass_fixed32_wave.py` and
`tests/test_fr13_cutlass_fixed32_wave_patch.py`.

Archived against the measured target-GEMM verdict. `results/fr13_attack_ladder_analysis_20260808`
headlines *"The target GEMM has no legal scheduler lever"* and rates the three
relevant ladder rows:

| # | Lever | Modelled ms/step | Verdict |
|---:|---|---:|---|
| 7 | Target GEMM persistent / megakernel / graph-node fusion | 0.009 | **DEAD** |
| 8 | Target GEMM wave / tile quantization | ≈0 | **DEAD** |
| 9 | Target GEMM L2 policy | 0 | **DEAD** |

The branch is precisely a lever of this class — it *"moves the logical tile
bound, launch width, and N stride from runtime scheduler state to compile-time
constants"*. The measurement says the ceiling for the whole class is
0.009 ms/step, because SFWD is already one CUDA graph whose 256 GEMM instances
carry a mean inter-node gap of 35.8 ns, there is no wave-quantization penalty
(40-CTA shapes match or beat 272-CTA shapes on GB/s), and the GEMM already runs
at 85.8% of roofline at p5.

The branch's own artifact agrees it has no runtime evidence: *"This is an
offline code-generation win, not a performance or acceptance measurement. No GPU
kernel, Docker service, synthetic workload, or real task was run."* Its measured
delta is 1,024 SASS slots vs 1,032, six fewer `LDCU` and one fewer `LDC`, with
branch/QMMA/FFMA/LDSM/STSM counts unchanged — a codegen nicety against a lever
class measured dead.

## 6. `agent/fixed32-floor-publish-20260730` — ARCHIVE (no-op)

**0 commits ahead**, 876 behind. Also a strict ancestor of `origin/main`
(`git merge-base --is-ancestor` returns true). Fully duplicated; recorded and
closed.

---

## 7. Standing measured verdicts cited

These governed the decisions above and are reproduced for traceability.

| Verdict | Value | Source |
|---|---|---|
| Composed M128-coop + SFWD-fusion B1 | **null — 1.0016× stock step-wall** | `timing_summary.json` at `output/fr13_b1_target_sfwd_exact4_timing_1c5c4c1d5_20260808T170248Z` |
| Two-M B4 | −3.0% per-request | alignment study |
| Verifier head | near-worthless, 2.8 ms removable | `results/fr13_attack_ladder_analysis_20260808` |
| Target-GEMM scheduling levers | dead (0.009 / ≈0 / 0 ms/step) | `results/fr13_attack_ladder_analysis_20260808` rows 7–9 |
| BV16/32/64 | byte-rejected | historical |
| conv | already removed (0.5 ms) | historical |

The composed-stack figure is quoted from the run's own reducer:
`ratios.candidate_to_stock_step_wall = 1.0015932402588816`, `status = MEASURED`,
`performance_claim = false`, `production_default_enabled = false`,
`source_commit = 1c5c4c1d5c26affb5dce83da59b53f8e74947fdf`, `task_set = exact4`
over 4 real SWE-Verified tasks. The same file reports
`candidate_to_stock_full_wall_tps = 1.0714`, but that ratio moves with
acceptance (`accepted_drafts_per_event` 4.676 vs 4.289), not with kernel speed;
the step-wall ratio is the kernel-honest number and it is null.

## 8. Method

- `git rev-list --left-right --count origin/main...origin/<branch>` for
  ahead/behind.
- `git merge-base --is-ancestor` for containment.
- **`git cherry -v origin/main origin/<branch>` for patch-equivalence.** This is
  the load-bearing check: raw ahead-counts credit cherry-picked work twice. It
  reduced 18 nominally-outstanding commits to 8 genuinely new ones and revealed
  that branch 3 is fully absorbed and that branch 2's kernel is already on main.
  Spot-verified against the tree rather than taken on trust.
- `git log --name-status origin/main..origin/<branch>` for the change surface.
- Branch-side artifacts read with `git show <branch>:<path>` (this worktree is a
  sparse checkout; unmaterialised `results/` paths are still readable from the
  index).

No branch was deleted. No ref was moved. No measurement was run.
