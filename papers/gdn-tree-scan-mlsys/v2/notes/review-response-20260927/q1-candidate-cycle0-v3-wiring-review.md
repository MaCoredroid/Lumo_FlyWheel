# Cycle0 v3 collector wiring review

**PASS for the bounded source/CPU wiring delta; no new blocker found.** This review does not open a gate or authorize a model boot. Parent approval, accepted native/root results, readiness, and exact final source binding remain separate prerequisites.

Reviewed immutable snapshot: `p0/monitor/review-response-20260927/candidate-cycle0-v3-wiring-review/snapshot/`.

| Source | SHA256 |
| --- | --- |
| `run_q1_candidate_cycle0_v2.sh` | `7fbb1f5f34eb8b5c08fef9e695db1d5cec66e45ce082ac92be7dcad7ee6e2874` |
| `q1_candidate_cycle0_gate_v2.py` | `7ebfa84e33f81fa2c440416e62086b6c57a731d433ca495c98ad23fad12e4637` |
| `q1_make_diag_launcher_v3_4.py` | `cb233133fbd68398c546fb211402279258a3098ac4c39c1210ba3280ab89ae84` |
| Generated `fr14_leg3_launch_nomiddleware.q1diag.v3_4.sh` | `93cbef1f8ccd19e697492bff1eebf41b5ce9f8271803e5dc6bce84d94d268d0e` |

The wrapper selects hooks/job/patcher v3, gate v2, the unchanged driver v2, and the accepted raw auditor v2. All 69 declared dependency keys are unique. The new job-builder-v2 import, layer registry, and KV witness are explicitly pinned alongside their callers. The wrapper's actual pre-import heredoc checks every declared dependency hash before importing the gate. Runtime patcher hash export, host validation, in-container existence/invocation, and final provenance all select patcher v3; that patcher imports hooks v3. The hooks bind the registry/witness sources through the rebuilt job. No stale hooks-v2/patcher-v2 selection remains in the new wrapper.

I independently regenerated the launcher without executing it. Replacing its five `q1_patch_candidate_v3.py` occurrences with `q1_patch_candidate_v1.py` recovers the pinned v3.2 base exactly (`db89386fb76db8594c00b575bddf369f6f8b05a6183f425734d1679f81de5352`). It also equals the previous v3.3 bytes after only the five v2-to-v3 path replacements. Scientific flags, model, memory settings, diagnostic exceptions, and operational script bodies are unchanged by this generator.

Gate v2 adds the strict integer revision-3 requirement and rebuilds the prepared job through builder v3. Existing same-case native A/r0 acceptance and root-instrumentation acceptance checks remain. The admitted scope remains one A or B process, 84 calibration cases × two repeats, cycle0, 168 requests, temperature/top-p/seed 1/1/0, and candidate memory utilization 0.7. The raw auditor is hash-pinned for subsequent offline analysis; the wrapper does not automatically run that analysis or turn a collection receipt into qualification.

Independent CPU controls passed:

- All four supplied gate test methods, including source/rebuilt-job/root-result/scope refusal subcases. The existing native builder is mocked in this suite; no native result is created or approved.
- Six explicit revision refusals: absent, boolean, float `3.0`, string `"3"`, old revision 2, and revision 4.
- The exact wrapper pre-import Python heredoc accepts a temporary fake positive module, but refuses a closed temporary gate or wrong hashes for job-builder-v2, registry, witness, and reducer before that module's import sentinel is created. These are isolated CPU fixtures, not operational authorities.

The first snapshot test attempt omitted the token fixture and failed during reviewer setup; after copying the exact fixture, all four methods passed. Both logs are preserved. Local Bash 3.2 parses the wrapper, but cannot parse the generated launcher's inherited `[[ -v ... ]]` syntax; the byte-inverse proof establishes that this is unchanged. Parent subsequently reported synchronized SHA checks and successful target-machine `bash -n` for both new shell files. That target-shell evidence is parent-performed, not an independently repeated remote action.

Evidence: `SNAPSHOT.json`, `check_wiring.py`, `WIRING-CONTROLS.json`, gate-test logs, and `LOCAL-BASH-PARSE.json` beside the snapshot. No remote command, GPU/container/cache operation, launcher execution, implementation edit, or gate change was performed. Previously accepted helper semantics were not reopened.
