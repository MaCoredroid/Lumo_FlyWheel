# Held-out reducer v2.2.1: preliminary changed-scope review

2026-09-28 UTC. **Two concrete draft blockers; no launch/reduction approval.** This is a review of the first copied draft only. The author's launcher, targeted test completion and final freeze are still separate review inputs. No production source, gate, original output or tensor was edited. No GPU, Docker, model, Torch or real reduction was executed.

Snapshot: `notes/review-response-20260927/heldout-reducer-repair-draft-snapshot/`. The draft was absent from locally synced campaign tools when review started, so it was read once from the remote author's path into this isolated snapshot. It has SHA `eb117260ba4f55d47137dbe70db6dd335d566e434b26c16d66150b48448a8a24`; original v2.2 comparator SHA `6475182d9e0996dd4e7c228340c57299665cd1ea641ed0b61bf291d631af9d73`. Snapshot manifest SHA `21995e376532eb3d1bdb25fbbeaba76498d35823e86b32bc6e42e3058258fbf3`. Line numbers below refer to the snapshotted `q1_2b_reduce_v2_2_1.py`, not a subsequently edited draft.

## F1 — The inherited reducer-self check always refuses the original run

At lines 325–326, `exp['reducer']` from the immutable original `LAUNCH-BINDING.json` is still compared with `sha256_file(__file__)`. The launch correctly binds original v2.2 `6475182d…`; the repair source is `eb117260…`. Those values necessarily differ. Even with a valid new CPU authorization, this check returns malformed before the intended helper reconciliation.

Minimal correction: check the original launch expectation against pinned `ORIGINAL_RUN['original_reducer_sha256']`; keep the **new** CPU authorization's `repaired_reducer_sha256` equal to the repair's actual self-hash. Do not edit the original binding or allow either hash interchangeably. A tiny regression should demonstrate that the exact original binding passes this identity stage, while wrong original reducer and wrong repair authorization hashes refuse.

## F2 — The CLI writes to the original run even when its guard refuses that path

`main()` creates `a.out` at line 792, calls the reducer at 793, then writes `summary.v2_2_1.json` at 794–795 regardless of the returned refusal. `bind_original_run()` correctly identifies output inside the original tree as invalid, but the CLI still writes that refusal into that tree. This violates the explicit preservation contract. A read-only Docker mount would contain it in the intended launcher, but the CLI must also preserve the original on an invalid path.

Independent local AST reproduction executed the **actual** draft `main`, `reduce_run`, `bind_original_run` and `sha256_file` functions against a temporary metadata-only run directory. With `--out` equal to `--run`, it exited 2, included `reduction output directory must be OUTSIDE the original run directory`, and still created `<temporary-original>/summary.v2_2_1.json`. No actual experiment tree or production module was touched.

Minimal correction: resolve/check output versus input path **before any mkdir or output write**, reject same/descendant paths without writing inside the original tree, and preserve refusal evidence outside it or on stderr. Include a small same-path/descendant-path CLI negative with an unchanged-directory assertion; do not replace this with a unit assertion that the guard merely returns a message.

## Remaining changed logic inspected

The compatibility rule is otherwise narrowly framed: it requires the exact old v2.1 helper key/value and absence of the new key, requires exactly the expected defective key set, independently checks actual `attestation.runner_sha256` against the bound v2.2 source, retains exact checks for all other helpers and adds per-metric executed-runner binding. The known original run ID, gate, launch binding, terminal receipt, failed summary/findings, A/B results and inventories are pinned. The original run's failed outcome is explicitly carried into the new summary. Separate CPU authorization binds the original evidence, repair hash and new reduction ID.

Function-AST comparison confirms `_recompute_metrics`, `verify_record`, inventory loading, process independence, canonical hashing, runtime identity and other original helpers unchanged. Only `reduce_run`/`main` change, with new `bind_original_run`. Numerical equations, evaluator calls and tensor/C2 recomputation are unchanged in the reviewed diff. No further material defect was identified in this bounded draft pass. This does not substitute for the final launcher read-only mounts/CPU-only settings/output-path review and source/test freeze equality.

## Reproducer

```sh
python3 -B notes/review-response-20260927/heldout-reducer-repair-draft-snapshot/repro_draft.py > notes/review-response-20260927/heldout-reducer-repair-draft-snapshot/repro-results.json
```

The command completed exit 0 with both draft defects reproduced. Files and hashes are listed in the snapshot manifest. The isolated temporary directory was removed after the check. Keep this original failing draft snapshot and probe results; review corrected bytes as a new snapshot rather than rewriting this record.
