# E1 candidate methods: bounded independent review

2026-09-22 11:24 UTC. Candidate `notes/e1-methods-for-final.tex` SHA-256 **`db5002468282a167514c5c7c0bf38b6438e45603097f06889f95c73fdaca66de`**. Read-only review against the immutable campaign snapshot, current CPU aggregation source, completed native preflight records, and the prior seed-scope adjudication. No inference, tests, aggregate execution, manuscript/campaign edits, or review of unfinished results.

**Disposition: two small methods clarifications are required; no experiment is necessary.** The configuration description, workload, warm-up, completed native preflights, statistical rule, and seed-deviation scope otherwise agree with the evidence. This review does not approve a final result table or numerical conclusion in advance.

## M1 — state the timed-only support in the equation (candidate line 14)

The definition of `I` currently includes consecutive pure-decode steps with the same active set and occupancy, then excludes “phase breaks.” That does not explicitly exclude whole preflight/warm-up intervals, which can meet those conditions. It also leaves `t_j` undefined and can read as allowing adjacency after filtering out intervening mixed forwards. These distinctions determine the numerator and denominator; they should be explicit in the equation's definition rather than inferred from the earlier warm-up prose.

Minimum correction: define `t_j` as the recorder's `physical_step.t_start` timestamp at its wall-mark hook, and restrict `I` to steps in the **timed phase**, with adjacent physical-step and full-forward indices, no intervening mixed/prefill forward or chain break, the same active request set, and the declared occupancy. Explicitly exclude preflight and warm-up intervals while retaining their outputs in reconciliation. Existing first-step, terminal-step, slow-interval, and B4 single-denominator language is correct.

Evidence: snapshot `E1_FREEZE.md:23,27–28,44`; `e1_cell_summary.py:30–38` maps every non-timed phase into the excluded warm-up set; `e1_join.py:96–110` requires successor physical ID +1, excludes warm-up/overlap, refuses bridging a forward-sequence gap or chain break, and requires unchanged cohort/occupancy (also excludes discarded rows). `e1_recorder.py:103` writes `t_start=time.perf_counter()`; `e1_event_recorder_shim.py:61` inserts this call immediately before the runner's wall-mark hook. No measurement repair is indicated.

## M2 — distinguish primary reporting from numeric eligibility (candidate line 16)

“The initial eighteen cells remain the primary set, including insufficient-support outcomes” correctly preserves the design, but does not state what happens to means/intervals when a cell cannot supply an eligible rate. The actual implementation does not average the remaining rates or replace a missing rate with zero.

Minimum correction: say that only sealed VALID cells contribute rates; an arm mean requires all three blocks, and a contrast is not estimable unless all three paired blocks and the complete native-5 normalization baseline are available. Insufficient-support and invalid outcomes remain listed without imputation or unpaired replacement. Also call the reported interval an interval for the **mean paired rate difference**, the statistic actually bootstrapped.

Evidence: snapshot `E1_FREEZE.md:29–34,46,56`; current `experiments/e1/e1_aggregate.py:25–29` enforces the seal/VALID gate, 35–41 requires complete arm means and contrast pairs plus the native-5 baseline, and 42–46 resamples three differences with replacement and takes their mean. This is a reporting clarification, not a request to change the frozen analysis.

## Checked and accepted

- Lines 5–7 correctly distinguish cat10 maximum draft depth five from native-11's longer chain, patched-runner native paths from an unpatched baseline, and the two attention backends. Eager/synchronous/cache-off settings, equal recorder policy, eight pilot prefixes, 128-token budget with EOS, B1 sequential/B4 fixed-cohort submission, and the two different warm-up schedules match the snapshot. The finite warm-up caveat avoids claiming all shapes are warm.
- The line-7 preflight claims are already factual. Cells 1–4 each have nine passing live checks and nine passing final checks, a VALID terminal seal, and a `driver_trace.txt` live-PASS entry before the main workload starts. First-cell seals are 10:08:59, 10:16:31, 10:24:14, and 10:33:21 UTC. Qualification reuse matches the campaign runner. These saved records were checked here; the raw model/output qualification was independently reviewed earlier, not rerun in this pass.
- Line 16's three paired blocks, separate B1/B4 analysis, 10,000 percentile-bootstrap replicates, fixed analysis seed 20260921, and half-width divided by mean native-5 rate match the freeze and aggregator. The analysis RNG seed is distinct from the engine-seed discrepancy. The coarse uncertainty and absent between-boot pilot-variance caveat are appropriate; no extra replication is required by the methods as stated.
- Line 19 faithfully reports T1: discovery during cell 5's boot, preserved original campaign, common API seed but distinct native/tree engine seeds, narrow selector-source observation, and no inference of whole-engine invariance or a seed-controlled algorithm effect. The dated addendum explicitly records that the coordinator's continue-unchanged disposition occurred after cell 5 completed; the candidate does not misstate that as a prespecified exception. The prior no-extra-boot adjudication stands.
- Final integration must still report actual per-prompt/cohort coverage, exclusions and uncertainty from completed cells under the existing plan. This is already required reporting, not a new experimental axis or an additional finding.

## Identities and evidence locations

Remote paper root: `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2`. `S` below is `experiments/out-20260922T100028Z-e1-18cells/campaign_snapshot/`. Native evidence is under that run's `cell_01_b1_native-5_B1_a1`, `cell_02_b1_native-5_B4_a1`, `cell_03_b1_native-11_B4_a1`, and `cell_04_b1_native-11_B1_a1` directories. Hashes are SHA-256.

| Source | SHA-256 |
|---|---|
| `S/SHA256SUMS` | `d1ab57c53464ef814c7eed3b87ebd21a670cf8278541b8adc4864e5cf6dc2ecd` |
| `S/E1_FREEZE.md` | `11f5dd00b8d8b7073575f10cd9a00a4282760746be05f16bb9c7d188fa027a4e` |
| `S/e1_cells.json` | `d5aa5b6eb3da19bec9955f57162b2508c85b4d6413eedf93cfdd1f3157a75363` |
| `S/e1_workload.py` | `bcc5af170fd16599223ad7286f0367c82f978faf640ac4acebe0e9c36d001594` |
| `S/e1_join.py` | `cbee2ae70947f1adb4073bd151a7148fcb05cbc02d1c5dff37b381ffadf1e766` |
| `S/e1_cell_summary.py` | `04a119321a2aa95aec826c56f229800ea4f2bf2e5126b8bfdce4875939759dff` |
| `S/e1_recorder.py` | `1cf3f53552e008c44f7b7e5e6187d8052bcfd4c75bd6f81e2e07d1f70fbf65a0` |
| `S/e1_event_recorder_shim.py` | `06344718f035949ea7f1cea49e4677e9acc2a71a460bbbc7b94e01a1bbbbc1df` |
| `S/e1_native_preflight.py` | `6dd79084a8a7dcec0a5a84432bd768ddc64cd73c1a7da96d6da480a732ba7562` |
| `S/e1_run_cells.v2.sh` | `910e2dac38a8661361da7f011f8a6a48563b84f6799eb5b1257fc558fa765d72` |
| current `experiments/e1/e1_aggregate.py` (local = remote) | `98beec9efac26f560783d9d77af3e2154edbd6b24ebadfe141b0cafce6e48a68` |
| local `notes/e1-protocol-deviations.md` (includes later independent adjudication) | `72c10d8c04e5bb966cdf658273180c8745f06a267d3744164ad320d1139996f1` |
| local `p0/monitor/e1-seed-scope-paper-review.md` | `73aa0821433fecf735f3d5808ce2703be1f23a870ec324eb203ca65fc23b6e03` |
| cell 1 `native_preflight_final.json` | `270f5c5c67feea3a76bbfc8e4d12aff9176aef441e13cf9dd4aeeed1006849a6` |
| cell 2 `native_preflight_final.json` | `e4a09046123b32dd959d5488cd8f0431089139a74ef1f2331fe8c71a24364e29` |
| cell 3 `native_preflight_final.json` | `db139eece0c801cc8ac824bfb7690e550a0a374b551e5650afd9fe5d2cb4004d` |
| cell 4 `native_preflight_final.json` | `a2f43981aafc9de63f08dc0f34c7a6c79268c5133679dba781363f5bb8191fc8` |

## Wording closure — 2026-09-22 11:36 UTC

Rechecked only the revised candidate, SHA-256 **`61aa07c880f40935245cf3c663e1995578a891d460fe48cc75680b41e906f26e`**. **M1 and M2 are closed.** Line 14 now defines the wall-mark timestamp and timed-only support, adjacent physical/full-forward indices, and explicit preflight/warm-up exclusion. Line 16 identifies the mean paired rate difference and limits numeric estimates to sealed VALID cells with complete three-block support and the native-5 normalization baseline; other outcomes remain in the reporting set without imputation or unpaired replacement. These corrections match the previously reviewed joiner and aggregator rules. No further methods finding or experiment is required from this bounded review. Final 18-cell results, manuscript integration, and PDF verification remain pending and are not pre-approved by this closure.
