# M1-Q reference pack: partial-repeat retention addendum

2026-09-28. Source reviewed: `m1_q_reference_pack_v1.py` SHA256 **`dfc0be5814781ba3aa8fa6a5cc11519934a59368407091384e30b443e1e87632`**. No qualification seed, RNG, operand, Torch/NumPy object, reference arithmetic, engine, remote operation or GPU work was executed.

The latest source closes the original normal-path omissions: both C1/C2 prepared-operand hashes are recorded, exact-identical value bytes are stored once with both observed digest aliases, divergent repeat1 values are retained, and Lumo C2 finite/repeat failure is recorded rather than raised before the layer is saved. Exact eight surfaces, mandatory dependency bindings and actual reference extents remain closed.

**One exception-path retention defect remains.** The author loop computes both C1 repetitions and both C2 repetitions before assigning `references[key]`. A failure in a later call loses already completed current-surface raw values from the failure handler's `partial_references`. The Lumo loop likewise finishes both calls before storing the first reference.

The retained AST-only reproducer uses opaque strings and injected exceptions:

| Injected failure | Completed fake raw records | Records available in `partial_references` |
| --- | ---: | ---: |
| C1 repeat1 | 1 | 0 |
| C2 repeat1 | 3 | 0 |

Smallest repair: add each completed repeat to a current-surface raw evidence structure immediately, before issuing the next call. After both repetitions complete, compute identity checks and deduplicate identical bytes; retain divergent repeats. The existing failure handler can then serialize that partial structure. This preserves the prior one-use input opening and does not change any numerical rule, source comparator, experiment count or seed.

Evidence: `p0/monitor/review-response-20260927/m1q-reference-pack-partial-repair-20260928T214438Z/`, including reviewed source, `reproduce_partial_loss.py` and `reproduce-log-attempt2.txt`. The first reviewer AST selector accidentally chose the summary loop and raised `NameError: acc is not defined`; its script and explanation are preserved. Correcting the selector reproduced both cases without running any generator.

This addendum supersedes the first note's normal-path retention findings, but it does not approve CPU generation. The parent-owned incremental retention repair is the only additional source change requested by this bounded pass; calibration adoption, final authority/source freeze and later native/reference/C0 phase gates remain parent decisions.
