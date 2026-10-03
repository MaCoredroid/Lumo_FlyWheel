# M1-Q C0 reducer v1.1 — bounded independent delta review

2026-09-28. **PASS, source delta only.** No numerical criterion, arithmetic, population, repeat, classification, failure-demotion, output, or control-flow change was found after the declared filename substitution. This does not approve the lifecycle helper, a retry, a reduction, or candidate qualification; those remain separate reviews/gates.

The canonical v1 remains byte-identical to the previously closed reducer:

- `experiments/review-response-20260927/tools/m1/m1_q_c0_reduce_v1.py`: SHA256 `4e2aa7dab16e385ddee610d85b69ef9e23f0ce0d0cdf51fc3538578117f736de`, 13,918 bytes.
- New `m1_q_c0_reduce_v1_1.py`: SHA256 `d41cf4249b382d2d7250768f0bece9c5fa00848f9a850b934eb9ad944c497191`, 13,924 bytes.

Independent stdlib checks confirmed exact **byte equality** after replacing `m1_q_c0_collect_v1.py` with `m1_q_c0_collect_v1_1.py` twice and `run_m1_q_c0_owned_v1.py` with `run_m1_q_c0_owned_v1_1.py` once. Parsed ASTs are also identical after those substitutions. `cycle_domain`, `load_tensor`, and `classify_cells` have identical ASTs without any substitution. All changes are confined to dependency filenames at lines 74 and 118.

The compatibility boundary remains correct: the reduction authorization must name this reducer's actual hash (`:70–77`), all required v1.1 collector/launcher dependency bytes are checked, and the actual pre-C0 gate must already contain this reducer hash (`:100–105`). The authenticated raw collector receipt must match the v1.1 collector hash (`:115–118`). The new collector/launcher retain the producer status and stage spellings consumed by the scorer. Thus no compatibility exemption admits the old v1 run under this new reducer: a newly approved launch/reduction must bind the v1.1 chain explicitly.

Executed four tiny extracted-AST authority controls: matching reducer authorization + pre-C0 gate + collector identity passed; stale reduction-authority hash, stale pre-C0 reducer hash, and old collector receipt identity each refused. The unchanged failed-collection barrier, both-repeat requirements, RMS/max rule, and missing/nonfinite handling inherit the earlier source closure; no unchanged suite was rerun.

Both exact source files and machine-readable results are preserved at `p0/monitor/review-response-20260927/m1-q-c0-reducer-v11-delta-review/REVIEW.json`. This audit imported no reducer/helper/backend, generated no held-out inputs or candidate outputs, and performed no GPU, container, remote operation, or actual reduction. No material blocker within this filename-only delta.
