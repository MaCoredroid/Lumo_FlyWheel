# PILOT_FREEZE — retrospective completeness erratum (append-only; the frozen record is NOT modified)

Written 2026-09-22T06:46:10Z from EXISTING data only (no new inference). Frozen record: `e7a/PILOT_FREEZE.json` sha256 `21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce` (status FROZEN, frozen 2026-09-22T01:36:16Z). Audit outputs: `out-20260922T064200Z-e7a-freeze-retrospective-audit/` (`pilot_v2_retrospective.json`, `confirmation_retrospective.json`, `pilot_v1_superseded_retrospective.json`), produced by `e7a/check_confirmation_vs_freeze.v2.py` (85b9703feb724851…), controls 17/17 incl. an exact cross-check against the parent's independent recomputation (`e7a/tests_out/freeze_checker_v2_controls.json`). Trigger: review 14 (paper red-team P1, parent-reproduced): the v1 checker `check_confirmation_vs_freeze.py` evaluated only 12 tolerance classes + 9 decision checks and OMITTED three rules that the frozen record declares: padding-control bitwise invariance (D6c), compact-state B == C ordering (D7b), and the verify-kernel timing half-width target (P1). Earlier verdict files (`freeze_check.json` in the harness roots, `PILOT_FREEZE.md`) are preserved unchanged; this file maps each original freeze claim to the source-bound value.

**This is a retrospective completeness audit. It does not constitute a corrected preregistration, does not relax any threshold, and does not promote any candidate.** The confirmation remains descriptive failed-rule evidence.

## A. Claims in `criteria.candidate_selection.rationale` (pilot, 8 VERIFIED-FRESH sets) → audited values (pilot-v2 harness `out-20260922T013313Z-e7a-fresh-pilot-v2/`)

| # | Original claim (verbatim fragment) | Audited value (pilot-v2) | Status |
|---|---|---|---|
| 1 | "A_prod fp32 out bit-identical to the pinned native spec-update kernel on 8/8 (integer view, fp32 and bf16 stores)" | D1 PASS 8/8 (fp32 and bf16 bitwise frac 1.0 on every set) | confirmed |
| 2 | "B/C fp32-ieee out vs fp64 oracle ≤ 4.447e-08 (A ≤ 2.724e-08)" | T3 worst 4.447e-08 (tolerance 8.893e-08) | confirmed |
| 3 | "compact-commit state vs oracle ≤ 7.867e-07" | T4 worst 7.867e-07 | confirmed |
| 4 | "bf16-store B/C agreement with native ≥ 0.999788" | D4 minimum agreement 0.9997884115 | confirmed |
| 5a | "sibling-reorder … controls bitwise-invariant on 8/8" | D6b PASS 8/8 (out and U bitwise frac 1.0) | confirmed |
| 5b | "… and padding controls bitwise-invariant on 8/8" | **FALSE.** D6c FAIL 8/8: B_fs npad32-vs-npad16 out bitwise frac 0.8728–0.9280, U 0.8589–0.9126; C_nm out 0.8869–0.9472, U 0.8921–0.9444; max abs difference 7.451e-09. Same on the superseded first pilot run (D6c FAIL 8/8). The statement was written without evaluating the control's bitwise fraction. | **erratum** |
| 6 | "replay repeat-launch bitwise-equal on 8/8" | D6d PASS 8/8 | confirmed |
| 7 | "Served output (live server, tree step 1) byte-identical to the production scan and native spec-update bf16 outputs on 8/8 (fresh-pilot-v2)" | D2/T12 PASS 8/8 on pilot-v2 (the superseded first run has no served-output row: D2/T12 report missing → FAIL, as expected for that run) | confirmed (pilot-v2 only) |
| 8 | ordering "A <= B == C for out vs oracle; B == C for compact state" (decision list) | D7a PASS 8/8; D7b PASS 8/8 on the pilot | confirmed (pilot) |

## B. Claims in `criteria.timing_precision_target.basis` → audited values

| # | Original claim | Audited value | Status |
|---|---|---|---|
| 9 | VERIFY-class half-width "pilot max 0.0623 (met)" | The figure 0.0623 reproduces on the SUPERSEDED first pilot harness run `out-20260922T012554Z-e7a-fresh-pilot/` (P1 max 0.0623, 2 cells on 2 prefixes), not on the pilot-v2 run that the freeze cites as its evidence root. On pilot-v2: P1 max **0.2023** (7 cells on 6 prefixes > 0.05: p015, p058, p072, p083, p085, p095). In either run the value exceeds the 0.05 target, so "(met)" was false as written. | **erratum** |
| 10 | COMMIT-class "pilot half-width max 0.216 on 3/8 sets exceeded 0.05" | first run: max 0.216, 3 sets (matches the text); pilot-v2: max 0.159, 4 sets, 5 cells labeled imprecise | text derived from the first run; pilot-v2 values differ |
| 11 | "Probe drift pilot max 0.0370" | first run: max 0.0370 (matches); pilot-v2: max 0.0457 (P3 PASS 8/8) | text derived from the first run |

Evidence-root note: `pilot_evidence` in the freeze = [{"prefix_id": "p072", "run_dir": "/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T002242Z-e7a-step2-captures-v4/capture_01_p072", "payload_sha256": "02b37fff1f856247b4b7f9a694061bca1b2500c76afc7bad12b401c1c36d8c54", "provenance_sha256": "37a3902923db83ce387d52b83e48c67a1d6cdd89af5b8d223b585906d74e8f8a", "inspector_sha256": "59c5b2b92d0d1387009fdc3ff1751cdb… ; `pilot_harness_summary.run_dir` = None. The rationale text mixes numbers from the first harness run (timing) with the pilot-v2 run (served-output row); the padding claim is false on both.

## C. Confirmation (31 harness sets; p087 provenance FAIL, hung boot 8 without payload) under the exhaustive rule set

`confirmation_retrospective.json`: 12 of 33 rules FAIL — T2, T3, D1, D2, D3, D6c, D7a, D7b, D8b, D8d, P1, P3. Of the three rules omitted by the v1 checker: D6c padding FAIL 31/31 (minimum bitwise frac 0.8563); D7b compact-state B == C FAIL on p031 (B 2.809754e-07 vs C 3.499136e-07); P1 verify-class half-width FAIL on 12 prefixes (14 cells, max 0.3209). Also P3 probe drift FAIL on p059, p067, p088 (max 0.0664); commit-class imprecise cells 23 (max 0.472). Minimum bf16 B/C agreement with native over the 31 sets: **0.9997721354** (= 99.97721354 %; the manuscript's 99.980 % is superseded — parent owns the manuscript fix). The previously reported misses (T2 scan-bf16 bitwise on p049/p005; T3 on p086; D7a ordering on p031/p038/p086; coverage; p087) are unchanged.

## D. Confirmation launch ledger (reconstructed from run directories; corrects "34 confirmation boots" in STATUS)

| Root | Boots (capture dirs) | Note |
|---|---|---|
| `out-20260922T013616Z-e7a-step2-captures-v5-confirmation-refused-md-not-frozen` | 0 | gate refusal (PILOT_FREEZE.md not FROZEN): 0 containers launched |
| `out-20260922T013708Z-e7a-step2-captures-v5-confirmation` | 8 | part 1: prefixes 1–7 verified; boot 8 (p031) hung after model load (container ran; health timeout; no payload) |
| `out-20260922T024543Z-e7a-step2-captures-v5-confirmation-part2` | 23 | part 2: prefixes 8–30 (p031 retried once; p087 provenance FAIL kept) |
| `out-20260922T050330Z-e7a-step2-captures-v5-confirmation-part3` | 2 | part 3: prefixes 31–32 |

Executed confirmation boots: **33** = 32 prefixes + 1 hung retry (p031 appears in part 1 as the hung boot 8 and in part 2 as boot 8). The gate refusal root launched no container (only `PILOT_FREEZE.check.json`). The earlier "34" was a miscount.

## E. What is NOT changed
The immutable freeze (`PILOT_FREEZE.json`/`.md`, `.criteria.json`), the v1 checker and its verdict files, the per-capture provenance verdicts, and all harness result files are untouched. No threshold, margin or class definition was altered; the audit adds rules that were declared but not evaluated.
