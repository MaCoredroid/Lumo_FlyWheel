# Review14: omitted frozen checks (paper red-team finding, independently reproduced)
UTC: 2026-09-22T06:33:32.925034+00:00

P1: check_confirmation_vs_freeze.py omits frozen padding bitwise, compact-state B==C ordering, and verify-kernel half-width <=5% checks. These were declared in PILOT_FREEZE.json, so absence is not an optional extra experiment. Parent independent raw recomputation is review-14-frozen-audit.json.

Pilot v2: B and C padding output/factor checks fail on8/8 prefixes;7 verify timing cells on6 prefixes exceed5% (max20.225%). Confirmation: padding fails31/31;14 verify timing cells on12 prefixes exceed5% (max32.086%); compact-state B==C differs onp031. Original frozen rationale falsely says padding bitwise invariant8/8; timing basis says .0623 met .05 and reports older run while root is pilot-v2. These preexisting pilot inconsistencies invalidate a statement that every frozen gate was met before confirmation. Do NOT modify the immutable freeze or replace old verdict files.

Please repair the checker to exhaustively enumerate every machine-evaluable frozen decision rule, strict equality where specified, finite values/missing data fail closed, and both probe drift and half-width. Write new named posthoc reanalysis files and an append-only frozen-record erratum mapping each original claim to correct source-bound data. Reanalyze pilot and confirmation, preserve failure tallies, source hashes, and original reports. New output must explicitly say retrospective completeness audit; do not imply corrected preregistration or relaxed thresholds. No extra inference is needed for this correction. Parent will update manuscript and claims; confirmation remains descriptive failed-rule evidence, no promotion.

Also confirmation B/C minimum agreement is99.97721354%, not manuscript99.980%; parent will fix. Old34-boot claim needs launch ledger reconstruction (32prefixes+one hung retry=33 executed; distinguish preflight refusal). State the count only once verified.
