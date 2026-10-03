# Candidate joint pair reducer — independent source review

**PASS for the bounded source/CPU review at SHA256 `9107882f64a8e7773c6b6f021fe835de4247a63b883193491c0804951f7a7360`.** No remaining material blocker found in this reducer's fixed-population, terminal, or reference joins. No candidate run was executed or qualified by this review. Parent launch, run-provenance, state-numerical, and scientific conjunction decisions remain separate.

Reviewed `experiments/review-response-20260927/tools/q1_candidate_joint_pair_reduce_v1.py`. The unchanged dependency freeze is SHA `6ad7cc5f8e9872b3f79abce8134cee7d351c0f00525f8335d9d2b8e8fdb4b7c6`; its complete 62-member `source_check()` passed locally. The new reducer itself requires its own prospective caller/freeze binding; it is not retroactively a member of that older freeze.

## Findings and closure

Initial source `bfd5143251062412fdd6634f5dbd8e24110d8d87c455d8c1062afa7fa2bd3398` and its source context are preserved in the review directory. The first 31 independent controls found four accepted negatives, preserved verbatim in `initial-controls.json`:

1. **Terminal inventory omission** (initial lines 48–70): a receipt containing only the job passed despite omitting all raw case seals. Final code requires the exact 168 case-member names and hashes every listed member before raw auditing.
2. **Accepted-native join missing** (initial lines 151–173): consistently substituting another native record in all candidate repeats escaped the cross-repeat hash check. Final code joins the selected native A job and terminal hashes to the accepted reduction's run binding, checks complete frozen fixture case fields, and checks every returned native record hash and winner against the accepted same-case A/r0 row. The unchanged prelaunch job builder already enforced this selection; this repair closes the reducer replay boundary.
3. **Narrowed helper phase scope** (initial lines 73–126): supplying only `target_o2` and removing all three MTP phases returned complete agreement. Final `summarize` requires the exact four-phase tuple. The original CLI already supplied the frozen tuple; this was a helper-level gap, not a demonstrated original CLI bypass.
4. **Container assertions not joined to saved inspect** (initial lines 54–70): a hashed inspect with a foreign CID, running state, and restart count 1 was ignored. Final code requires the CID and clock member files, binds exact image/CID and stopped non-OOM zero-restart/zero-exit inspect state, and checks both wrapper and actual container intervals across A/B.

The clock repair was further corrected to respect the actual producer's whole-second `date +%FT%TZ` precision. Docker's fractional finish must be below the recorded stop plus one second, exclusive; actual finish/start ordering additionally prevents same-second A/B overlap. The completed native A receipt supplies a concrete precision witness: `18:20:46.059618378Z` versus wrapper `18:20:46Z`. This is timestamp representation handling, not an enlarged experimental tolerance.

## Independent controls and semantics

`final_controls.py` passes **35/35** focused controls. It calls the real `terminal`, `summarize`, and outer `reduce` logic on temporary synthetic files. Connected reduction controls inject only the expensive prior/raw-audit boundaries and job validation; the real categorical reducer and frozen population are used. These fixtures are not raw scientific evidence.

Coverage includes a valid 84-case × two-repeat × two-process × four-phase positive; each of the 12 individual phase/field disagreement types retained as 83/84 complete cases and 3/4 decisions for the affected case; missing process/row, duplicate substitution, foreign case, boolean repeat, integer truth values, wrong observation ID, missing phase, inconsistent aggregate, switched single-repeat native reference, consistently switched accepted native reference, omitted terminal inventory, conflicting inspect, reused container, overlapping processes, and fractional overlap with apparently nonoverlapping whole-second wrapper timestamps. A valid fractional stop passes; a finish outside its one-second representation interval refuses. No mismatch is discarded or replaced.

CLI controls additionally confirm `--help` and refusal to overwrite an existing output before any dependency/raw audit. The original source and all failed controls remain preserved. A preliminary local Python 3.9 test could not parse Docker's nine-digit fractional ISO timestamp; it is retained in `clock-repair-controls.json` as an auditor-interpreter limitation. The final standard-library controls ran on Python 3.14.2; the intended remote CPU environment is Python 3.12. No engine module, GPU query, container, HTTP/model request, remote mutation, or real reduction was run.

Output semantics remain bounded: 84 cases and 336 observations, all four observations required per case, accepted native A/r0 fixed for each case, and explicit false flags for candidate/full-Q1/launch authority. Raw disagreement rows remain in `raw_audits`. The reducer does not claim MTP numerical equivalence, continuous-cycle/lifecycle/held-out qualification, timing, or workload completion. The already accepted full raw modules were inspected only at their row/CLI seams; their scientific predicates were not reopened.

## Reproduction and artifact identities

Review artifacts: `p0/monitor/review-response-20260927/candidate-joint-pair-reducer-review/`.

Run `PYTHONDONTWRITEBYTECODE=1 python3 final_controls.py` from that directory using Python 3.12 or newer. The script imports the preserved `FINAL-SOURCE.py`, binds its SHA through `FINAL-SNAPSHOT.json`, and verifies the actual existing dependency freeze for the source-check control. `INITIAL-SNAPSHOT.json`, original and repair source copies, initial/final controls, CLI receipt, `MANIFEST.json`, and `REVIEW-SEAL.json` preserve the review chain. No frozen experiment source or gate was edited.
