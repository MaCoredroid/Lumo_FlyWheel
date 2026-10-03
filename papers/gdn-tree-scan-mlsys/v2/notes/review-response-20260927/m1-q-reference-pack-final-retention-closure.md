# M1-Q reference preparation: final retention repair

Bounded independent source/control review, 2026-09-28. The remaining exception-path finding is closed for source SHA-256 `f1566c7c0a9301945f0223bae626bc076e4c9c3857eb9549cf9ffd84e5075fd2` in `experiments/review-response-20260927/tools/m1/m1_q_reference_pack_v1.py`.

The author C1/C2 loops now register their accumulating lists in `references[key]` before any reference call (lines 76–85). The Lumo C2 loop does the same (lines 109–113). Each successful call becomes reachable by the existing `partial_references` failure serializer before the next call. Canonicalization removes those temporary lists only after retaining the first values and any divergent second values. No numerical method, seed, threshold, or workload scope changed in this repair.

The reviewed snapshot is `p0/monitor/review-response-20260927/m1q-reference-pack-final-repair-reviewed-20260928T215025Z/`. It preserves the exact source, the three bound local dependencies, independent control scripts, this note, and the raw test log. Prior review snapshots and the original reproduced failure remain unchanged.

Fourteen independent CPU controls passed: six existing surface/dependency/observed-extent guards and eight focused retention controls. The focused controls execute only AST-extracted collection prefixes with fixed opaque strings. They cover failures at the second author C1 call, first and second author C2 calls, second Lumo C2 call, and hashing; each verifies that all completed records remain reachable by the mapping passed to the failure serializer. Positive controls verify exact-repeat canonicalization and divergent second-value retention. One source control checks the actual failure serializer's mapping.

Reproduce from the snapshot directory:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest -v reviewer_stdlib_controls reviewer_partial_retention_controls
```

`test-log-attempt1.txt`: 14 tests, 0 failures/errors. The control suite never imports the reviewed module, Torch, NumPy, or the baseline; it does not execute a generator, a seed, any reference arithmetic, or any GPU operation. No held-out inputs were generated.

Together with the prior bounded review and partial-retention addendum, this closes the requested source changes: exact eight-surface resolution, mandatory dependency bindings, observed reference extents, explicit C2-only/native-C1-pending labels, prepared-operand hashes, divergence retention, and incremental exception-path retention. This is source/control closure for prospective parent authorization, not an issued authorization or a numerical qualification result. Actual held-out reference generation, finite/repeat and frozen-rule eligibility, native C1 completion, and the later reference-eligibility seal remain unobserved. The parent owns those stage decisions; no gate is opened by this note.
