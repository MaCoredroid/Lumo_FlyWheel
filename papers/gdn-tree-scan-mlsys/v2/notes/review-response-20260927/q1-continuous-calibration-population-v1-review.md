# Continuous calibration population — bounded prospective review

2026-09-29. **Initial disposition: counts/scope PASS; one native admission wording clarification pending.** Reviewed proposal SHA `b54409265928b0c11bae91e9295549edb0386722816c37dd97509559e81096e1`. This is a design review only: no source implementation, runtime/GPU call, gate, qualification or workload attempt.

The exact ordered `record_ids` equal every calibration row of fixture `e9683d2f…`: 24 records, 54 cycle descriptions, no omissions/duplicates/reordering, no held-out rows. There are 39 F2 cycles, six F3 cycles, and nine F4 cycles. Multiplying by two repeats in each of two fresh processes is arithmetically correct:

| Quantity | Per route/process | Per route, A/B × two repeats |
|---|---:|---:|
| Sequence requests | 48 | 96 |
| Declared cycles | 108 | 216 |
| Terminal flushes | 48 | 96 |

Two routes give 192 sequence requests total. Each route additionally requires 216 target-O2 observations and 528 applicable MTP phase observations (216 first + 216 follow + 96 terminal first), derived from the unchanged phase contract. These are prospective denominators, not measured completion counts, extra workload tasks, or counts of all native token-by-token forwards.

The input/source selection remains finite and predetermined. The exact source inventory and prepared source plan contain twelve calibration prefix/root groups: three existing exact sources plus nine missing padded F4 sources, each selected by the first fixed representative, not by outcomes. Those nine source requests are not sequence qualification observations. All 24 sequence IDs remain required even if an observation fails, is missing, or becomes baseline-unqualified; no reference replacement or favorable subset is authorized. Initial union import occurs once per sequence and existing no-interior-rehydration applies. The A/r0 source/reference choice is prospective. Current declaration says no continuous candidate outcomes have been observed; this review does not independently certify runtime absence beyond the presented source-only stage.

The phase pin `a7c93219…` remains unchanged and binds cycle0 categorical contract `4dad3bf5…`, including exact smallest-ID spine, actual ordered top3, zero disagreements, finite raw heads, correct phase/input identity, unmatched self-fed-follow handling, mandatory state/publication/source obligations, and diagnostic-only MTP score/state numerical differences. It transitively binds the operational decision and paired GDN numerical contracts (`3e4fd09b…` for the latter). Two-process × two-repeat full-model characterization is the existing joint design; it does not reduce the separate component calibration repetitions or relax any numerical rule. No MTP numerical-equivalence tolerance is invented.

Held-out 24 evaluation records remain outside this denominator and require their separate gate. APC, request retirement/reallocation/same-ID lifecycle, later proposal levels, full-Q1 and workload/timing admission remain unqualified. A finite calibration pass cannot substitute for these separately required surfaces.

**R1 (pending wording closure):** `native_to_candidate_gate` says only “Complete and independently reviewed native cohort”. A fully reviewed failed cohort could satisfy that phrase. Require all 24 records × four native observations categorically qualified with the unchanged mandatory obligations; any invalid/missing/native-unqualified record must block cohort candidate admission rather than merely remove an interpretable record. Do not turn diagnostic full-score/MTP-state byte differences into a new bitwise numerical rule. This is clarification of the existing all-record/no-subset requirement, not a changed criterion or new experiment.

The note and seal bind a direct fixture-derived per-record census. No broader campaign or additional experiment is required for this population-design review. Executable job/reducer/source bindings and a fresh scientific/operational gate remain subsequent obligations.

## Final bounded recheck — R1 closed, 2026-09-29

**PASS for prospective population design** at proposal SHA `e5cdb7dad8fb1a07c660abc7e9e4f3e0fed30959470de9afb2d39430c7aab648`. The only changes from the initial `b5440926…` bytes are `native_to_candidate_gate` and the clarification timestamp. All record IDs, counts, source choices, scope boundaries, phase/numerical/categorical policy bindings and non-authorization flags remain unchanged.

The gate now explicitly requires all 24 records × all four native observations to qualify with every mandatory state/publication/source/raw obligation independently reviewed. Any missing or unqualified native record blocks the entire candidate cohort; exclusions and alternate references are prohibited. Full-score/state byte equality remains a separate diagnostic, not a new threshold. This closes R1 without relaxing or adding a numerical criterion.

This approval concerns only the prospective population document. It does not approve a launch, satisfy the separate candidate cycle0 prerequisite, override an adverse native result, or grant continuous/full-Q1/workload qualification. No GPU, remote execution or additional experiment was performed. The initial manifest, initial review text and initial seal are preserved alongside the final seal.
