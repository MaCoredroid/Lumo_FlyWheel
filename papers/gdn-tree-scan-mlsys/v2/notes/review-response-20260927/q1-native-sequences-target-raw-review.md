# Native continuous target raw auditor — bounded independent review

**PASS for the reviewed source integration; no launch or numerical-qualification approval.** Reviewed `tools/q1_native_sequences_raw_audit_v1.py` SHA-256 `dae9f3a752bc9098d7d718f767efadbbdfdc0fa914085e4cca3694f5a63c765f`. No concrete blocking defect found in the new target cycle/map/prefix joins.

## Evidence and scope

Executed **32 independent CPU controls: 3 valid controls accepted and 29 malformed/corrupt controls refused**. The reviewer extracted the four new functions by AST to avoid importing runtime Torch hooks; their bodies are unchanged. The accepted `q1_native_corpus_v1` implementations of `observation`, `snapshot`, `Objects`, and `logits` ran directly against content-addressed full-geometry payloads. The tests reused retained synthetic target/MTP fixture objects and created reviewer-owned finite KV/logit witnesses. No GPU, model, remote command, cache action, gate, experiment, or held-out numerical result was involved.

The initial-source/union proof remains the accepted joint auditor's exact AST except the deliberate substitution of the old overall manifest predicate with `initial_import_only is True`. The new entry path calls `Hooks.fixed_inputs` and `Hooks.source_record` before the source snapshot and observation audit (lines 22–28); it checks exact job/case/prefix identity, frozen input bytes, bound source file/record/terminal receipt/job, actual root token, and target/MTP/history digests. The parent separately reviewed those callbacks. This review reads their integration and reuses that prior acceptance; it does not claim to have rerun their 24 controls. The caller still owns source-run eligibility, authenticated job/launch/boot/gate provenance and exact corpus population.

## New checks verified

- **Each checkpoint, including intermediate ones:** exact cycle-key population; full O1 tag/observation/extent/layer coverage and raw finite state; O2 cycle, consumed pending token, computed position, row zero and trace forward ID; authenticated complete 248,320-element logits and recomputed smallest-ID maximum/ties/margin (lines 174–197). Missing intermediate records/layers/logits, foreign observation, duplicate forward IDs, wrong token/position/row/decision and nonfinite logits refuse. Final scalar O1/O2 must exactly equal the final checkpoint aliases.
- **Actual map and ownership:** original request/runner/context/config/batch/group/owner identity, original prompt, stable MTP allocation identity, native single-decode metadata, current GDN row and complete bank geometry, attention table owner/units/range/uniqueness, and logical-to-physical snapshot mapping are joined (lines 142–171). Wrong owner, table, layer, bank pointer, prepared extent, selected state row, decode metadata, and duplicate/out-of-range attention allocation refuse.
- **KV immutable prefix:** both K and V compare authenticated bytes over the complete captured materialized prefix (lines 124–139). The positive sequence 63→64→65 crosses the native 64-token page boundary. Coherent physical remapping, a partial-to-full first-byte corruption and a full-block prefix corruption refuse even when each edited snapshot is internally resealed and finite.
- **GDN inclusive 1024 boundary:** 1023→1024→1025 is accepted with the appropriate selected row transition. Coherently changing the already-selected column 1 after the 1024 checkpoint refuses through the `prior_extent // 1024 + 1` rule (line 154). The old attention `ceil(extent/64)` rule is retained for attention only.
- **Numerical findings stay visible:** a valid two-cycle raw fixture returns winners **7 then 8**, with distinct complete-vector hashes. The auditor authenticates and exposes these observations; it does not hide them by requiring within-sequence equality or turn them into a numerical pass/fail verdict.

## Boundaries retained

O1 is bound by its authenticated state extent, the actual prepared live-map boundary and fixed flat trace; O2 additionally carries the explicit real forward-sequence join. All target O1/O2 raw payloads are checked; this does not observe all inactive cache bytes. MTP sequence history/phases are delegated to the separate MTP auditor, which is called before any successful result. This target-only control harness does not substitute for that independent MTP review or a complete future connected runtime test. Interior `after_z_first` remains explicitly unobserved. No new population, comparison criterion, repeat denominator or qualification count is introduced.

## Reproducer and seal

- `p0/monitor/review-response-20260927/native-sequences-target-raw-review/independent_controls.py`
- `RESULTS.json`, `controls.log`, `INTEGRATION-AST.json`, `SOURCE-MANIFEST.json`, exact target/native-reader source snapshots, reviewer-owned `objects/`, and `REVIEW-SEAL.json` in that directory.
- Run locally with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -B <absolute path to independent_controls.py>`.

The prospective caller/freeze must bind the reviewed target source and its separately reviewed dependencies. This note grants no gate authority.
