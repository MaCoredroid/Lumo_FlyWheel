# Pinned-native partial adverse result and supported disposition

Independent bounded read-only review, 2026-09-30. The reviewed snapshot is partial; this note neither terminates B nor replaces the queued full terminal reducer. No tests, new inference, numerical criteria, gates or candidate outcomes were introduced.

## Authenticated evidence

The local and remote saved result both hash to `5398ab2d8fd05916d794829cd45c1723cd80a661b06fb4f6eeb6b14b2d35180c`: `p0/monitor/review-response-20260927/NATIVE-RETRY1-AB-PARTIAL-ADVERSE-20260930T0526Z.json`.

I read exactly the declared complete-line prefixes, not the growing logs' later tails:

| Process | Bytes | Rows | Prefix SHA256 |
|---|---:|---:|---|
| A | 734872 | 168 | `99df75e102366d3b3f949dd04c6d5f1b3648ae842ae10f05df7b7a3dfea2600d` |
| B | 551280 | 126 | `3d43404e22d965294c283f8058c22e892b69ba778c1523b00bc6cb555b81b351` |

Independent reduction of those rows verifies 294 unique case/process/repeat observations, 63 complete A/B×R2 cases, 60 categorically stable complete cases and exactly three adverse complete cases. All 63 have equal recorded target O0, MTP-initial and joint-source digests across their four observations. The other 21 of the frozen84 cases remain incomplete at this snapshot, not passed or removed. Corpus SHA is `8b1ab3e055292801e1e40b3f6372b49bd48377f30d60f6eb5df01e74115159ae`; the actual categorical projection source SHA was independently checked remotely as `45187dfa879c79bbdcb2bffe963cce3ca913702628086c235bb98062ce3f40f4`.

For the three adverse cases I authenticated all12 actual case JSON files against their row-log file hashes and sealed record digests, then read 16 full-vocabulary MTP score vectors (eight unique vectors, 248320 FP32-exported values each). Each raw SHA, finite value population, recorded smallest-ID greedy and actual captured top3 matched. The stored FP32 exports retain the declared BF16 native-head semantics. This does not rerun a GPU top-k: it validates the captured exact-k IDs against full saved scores, preserving tie ambiguity.

| Long-prefix case / phase | A ordered top3 | B ordered top3 | Raw interpretation |
|---|---|---|---|
| n01 / after-z-first | 628,599,2908 | 628,599,1381 | Strict third/fourth gap in both processes |
| n04 / after-z-first | 248069,510,760 | 248069,510,248058 | Strict third/fourth gap in both processes |
| n05 / before-z-first | 353,11,40 | 353,11,18770 | A has a third-place cutoff tie; B does not |
| n05 / before-z-follow | 628,635,2688 | 628,2688,635 | Unequal second/third scores exchange order |

For example n01's A top3 scores are 15.5625/13.9375/13.8125 (fourth13.75); B's are 15.875/14.0/13.75 (fourth13.625). n05 follow's A second/third scores are13.75/13.625, versus B13.8125/13.625 assigned in the reverse ID order. Thus tie handling alone cannot remove the adverse finding. Within each process both repeats have identical raw hashes for these phases. The scalar token, position, extent, dtype and operator inputs match across A/B, while captured target-hidden hashes differ. This is therefore end-to-end native-route repeatability failure from a common imported O0; it is not evidence of an isolated MTP kernel's behavior on identical hidden/cache operands.

## Meaning and disposition

The unchanged joint categorical contract (`Q1-JOINT-MTP-CATEGORICAL-CONTRACT-v1.json:195–216`) requires every case/observation/phase and mandatory conjunction. These verified differences already prevent that aggregate from passing if confirmed in the full terminal cohort. They do **not** demonstrate a candidate failure: no corresponding candidate cohort has been admitted, and the failure occurs in the native reference's eligibility prerequisite. Nor do stable greedy winners erase ordered-top3 failures. Keep the84/336 denominator and all raw adverse witnesses; do not retain only60 favorable cases or change the criterion after seeing them.

Supported narrowing can close *work on the campaign* after complete terminal review without closing its scientific gates. The approved response plan :7,33,77,99,105 permits removal/narrowing of unsupported claims, and :56 requires repair before a speed comparison. Under the user's finish-or-supported-narrowing instruction, a truthful final disposition may be:

- Native calibration: executed, reviewed adverse outcome under its unchanged criterion.
- Downstream candidate joint comparison: not executed because the reference prerequisite failed; equivalence unestablished.
- Continuous, lifecycle/APC and held-out route stages: unexecuted/unqualified, explicitly deferred or withdrawn from the supported claim. Source readiness is retained as preparation, not evidence of correctness.
- Timed mechanism and four-attempt workload stages: not admitted, zero new timed results/attempts, corresponding performance and task-equivalence claims removed. Preserve already completed component studies strictly within their own observed populations and original criteria.

This requires an explicit parent evidence/disposition record naming each unrun stage, its dependency and the manuscript claim removed. Do not synthesize PASS/waiver receipts into the workload assembler, mark Q1 qualified, increment experiments, or alter preserved gates. `workload-case-study-v1/SCOPE.json:64–78` and `q1-f5-l3-workload-routing-check-20260930.md` remain consistent: qualification is mandatory **if timing is pursued**; an honest decision not to pursue unsupported timing does not require pretending qualification succeeded.

No scientific obligation requires unlimited repairs merely to establish a negative result. Repair remains necessary if the paper retains current-route end-to-end equivalence, continuous/lifecycle correctness, or new speed/agent claims dependent on this reference. If a concrete source/instrumentation bug is discovered, its effect must be resolved before attributing the failure to deployed native behavior; otherwise the bounded statement is the observed implementation/reference failure, with cause unestablished. No current evidence supports a broader algorithmic impossibility or intrinsic TreeWY/Weaver/LumoTree conclusion.

Finish the unchanged full B population, terminal provenance/raw reduction and adverse-result review first; only then bind final counts and disposition. This note authenticates the partial evidence, not final336 completion, full cache raw re-audit, candidate qualification or workload admission.
