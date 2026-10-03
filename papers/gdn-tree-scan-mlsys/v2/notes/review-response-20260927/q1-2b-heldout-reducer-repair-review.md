# Held-out reducer v2.2.1 repair: F1/F2 closure

2026-09-28 UTC. **Bounded PASS: both prior concrete blockers are closed; no remaining material issue found in the exact compatibility/auth/output scope reviewed.** This is source-review acceptance conditional on the parent's final freeze equality and separate CPU authorization. It is not a completed numerical reduction or a held-out qualification verdict. The original held-out result remains unaccepted until the separately authorized all-tensor reduction completes.

No real reducer, GPU, Docker, Torch, model or candidate was executed by this reviewer. Only local standard-library AST/path/metadata controls ran. Original experiment outputs and the failing draft snapshot remain unchanged.

## Exact reviewed bytes

Files below are under `experiments/review-response-20260927/`:

| File | SHA-256 |
|---|---|
| `tools/q1_2b_reduce_v2_2_1.py` | `a7decd00863f69b6970f7e1015a1d725681f046ebc65c63a535d5bc42da00cc8` |
| `tools/run_q1_2b_reduction_repair_v2_2_1.sh` | `6aa6a9c3f374840b728d64f565c73bbd445e8843105d0bc06259fbe121c383c5` |
| `tools/tests/test_q1_2b_v2_2_1_reduction_repair.py` | `4404061bcc4c24c1274b3e4542fb47ab4a760b92691e90e7e1267027da494dc2` |
| completed `tools/test_log.q1_2b.v2_2_1_reduction_repair.attempt2.txt` | `9dd9d4c3de9141eb5dfc0ce563f785afc832a99affa5880dca7c3b0ecd843a8d` |

Review snapshot: `notes/review-response-20260927/heldout-reducer-repair-closure-snapshot/`. The initial in-progress log remains in its original copied location; `completed-test-log.txt` preserves the completed version separately. The failing draft `eb117260…`, its exact failure probes and `q1-2b-heldout-reducer-repair-draft-review.md` are retained in the separate earlier snapshot.

## Prior findings closed

**F1, original versus repaired reducer identity: closed.** The corrected comparison at reducer lines 338–341 binds immutable `LAUNCH-BINDING.expect.reducer` to `original['original_reducer_sha256']` (`6475182d…`). `bind_original_run` separately requires the CPU authorization's `repaired_reducer_sha256` to match this repaired source's actual SHA. The original launch binding is never rewritten. Executing the actual corrected predicate confirms the old bound reducer passes, while either the new repair hash substituted into the original binding or an arbitrary wrong original hash refuses.

**F2, writes inside the original on refusal: closed.** `output_guard` at lines 100–116 compares resolved paths, including symlinks, rejects same/descendant/ancestor relationships and existing files/summaries, and is invoked by `main` **before mkdir or any write** at 807–810. It remains checked in `bind_original_run` as well. Independent execution of the actual CLI AST verifies exit 2 and an unchanged temporary input tree for same path, descendant, symlink to input, symlink into input, ancestor and existing-file outputs. The numerical reducer is replaced by an assertion that it must never be called in these tests; every invalid-output test returns before that assertion. A distinct new output path remains eligible.

## Compatibility, authorization and launcher checks

The exception still accepts only the exact known defective self-helper shape from the immutable v2.2 runner: old v2.1 key/value present, new v2.2 helper key absent, exact key set, all other helper values unchanged. Actual runner identity remains bound through its separate attestation and each metric's execution identity. Original run/gate/launch binding/terminal/failed-summary/process results/inventories are pinned; the original failure is carried explicitly in the repaired summary. No generic missing-helper fallback was introduced.

The separate authorization must be approved, explicitly `CPU_REDUCTION_ONLY`, name the exact original run/evidence, name the repaired source hash and match the reduction output ID. I exercised the actual binding function against the existing local **metadata receipts only**, with an explicitly synthetic in-memory test authority: valid shape binds, while missing/not-approved/wrong-purpose/wrong-run/wrong-repair/wrong-output-ID/wrong-original-receipt variants refuse. This did not create authorization files or approve/execute anything.

The launcher lines 20–22 use the pinned image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, **no `--gpus`**, `CUDA_VISIBLE_DEVICES=''`, OMP/MKL 1 and `--recompute all`. Both repository and original run mounts are read-only; the sole writable run mount is the new reduction directory. The input is fixed to `q12b-heldout-20260928T013539Z`. It refuses an existing output directory, checks authorization/evidence before Docker launch, snapshots the separate authority into the output and emits separately named binding, summary, log and receipt files. The reducer rechecks authorization inside the CPU container. No command writes a replacement original summary/receipt or invokes the GPU runner.

The numerical helper ASTs were already established unchanged in the preceding bounded review; the closure diff consists of the old/new reducer identity correction and the pre-write output guard. The original numerical criterion, evaluator and raw-tensor/C2 recomputation path remain unchanged. No extra numerical experiment or full model work is requested by this review.

## Independent controls and author test receipt

Executed:

```sh
python3 -B notes/review-response-20260927/heldout-reducer-repair-closure-snapshot/check_closure.py > notes/review-response-20260927/heldout-reducer-repair-closure-snapshot/independent-checks.json
```

**18 controls passed**, plus shell syntax validation. These are targeted standard-library checks of the actual F1 predicate, actual F2 CLI entry path, output separation and authorization binding; no original tensor was loaded and no numerical reduction was called.

The completed source-bound author log reports **43 tests passed in 76.77 seconds**, ending `2026-09-28T02:01:20Z`. I independently checked all **22 start/end/current source/evidence hashes**. The tests now call the actual frozen runner's `helper_hashes()` (instead of hand-constructing a corrected map), reproduce the original v2.2 refusal, test the exact compatibility shape and malformed alternatives, per-record wrong runner identity, separate CPU authorization, real CLI output refusal and unchanged original metadata. Local Python lacks Torch/pytest, so I did not rerun that full synthetic suite; its source/log binding is verified rather than misrepresented as my own execution.

The source snapshot and controls are bound in `heldout-reducer-repair-closure-snapshot/AUDIT-MANIFEST.json`. Final author freeze was not yet present at the last check. The parent must compare its exact reducer/launcher/test/log and dependency identities with the above before issuing the separate CPU authorization. No unresolved code correction is requested by this closure review.
