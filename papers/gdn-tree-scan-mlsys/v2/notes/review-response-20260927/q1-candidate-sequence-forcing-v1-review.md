# Candidate continuous forcing controller: repaired closure

2026-09-29. **PASS: grouped F1 is closed on source SHA `6c33eab8d4b65e14c62e093c2aa226691340f56b0aedbeb4314df29ebcfa3622`.** This is source/CPU preparation only. No GPU, container, model, remote runtime, implementation or gate operation was performed.

The exact source delta adds only the final live-state checks inside `finish()` (current lines175–186): strict integer complete counter equal to the last saved seal, actual census list with the exact final length, pending event absent, and each retained census entry rehashed and passed through the unchanged HE auditor with its saved owner/proof. The existing fail-stop decorator is unchanged. Token forcing, schedule, draft staging and acceptance criteria are unchanged.

**14/14 independent controls pass** using the same preserved reproducer and fixtures. The valid positive still executes all24 fixed calibration sequences,54 cycles and24 root-only terminal flushes with actual CPU tensors. Each original false acceptance—extra event/counter, counter-only change, new pending event, and earlier census rewrite—is now refused. The previous results remain **10/14**, failing exactly those four negatives. No broader suite or live execution was added.

Exact source copy: `p0/monitor/review-response-20260927/candidate-sequence-forcing-review/source.repaired.py`. Replay output: `results.repaired.json` and `attempt3.repaired.log`. All 14 original manifest payload members were independently rehashed unchanged, as were the unchanged dependency/test source pins. Original note/source/failures are retained in `REVIEW.initial.md`, `initial-source/`, `results.initial.json`, `MANIFEST.initial.json` and `REVIEW-SEAL.initial.json`.

Command (repository root):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/candidate-sequence-forcing-review/controls.py papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/candidate-sequence-forcing-review/source.repaired.py papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/candidate-sequence-forcing-review/results.repaired.json
```

No remaining material issue was found within this repaired terminal-closure scope. The controller is still unconnected preparation. Actual O2 greedy provenance, live ownership/witness bindings, runtime hooks, complete raw audit and launch admission remain separate; no candidate/state/numerical qualification follows from these CPU controls.

---

## Preserved original review

# Candidate continuous forcing controller: independent source review

2026-09-29. **BLOCKED by F1 (terminal live-event closure) on original source SHA `ede17850f44671426ddd8a09a1652c09f2ab785e6cf00124a3be1d26aa0bb636`.** Source/CPU preparation only; no GPU/model/container/remote operation, runtime admission or numerical qualification. Implementation remains parent-owned.

## Exact reviewed scope

`tools/q1_candidate_sequence_forcing_v1.py` and parent test SHA `e88256eaed446d920941f721d4aa03e9ed77ceed443edd45244e31060e55b532`; accepted dependency bytes are in `INITIAL-SNAPSHOT.json`. The controller invokes the accepted immutable Schedule/Cursor, FA in-place tensor helpers and HE census reader. It does not import state, call a model, reset a live counter or perform a CUDA operation in these tests. Snapshot and independent controls/results are under `p0/monitor/review-response-20260927/candidate-sequence-forcing-review`.

## F1: finish accepts a changed or unfinished terminal event population

`finish` (lines170–176) only checks cursor completion, absence of its local frame, and the saved cycle-label list. `check` (lines60–64) authenticates only the most recent saved census element. Neither verifies the final live counter/list extent/pending state nor all earlier retained census elements.

Using real CPU forcing tensors, the unchanged HE auditor and the full valid multi-cycle progression, each following mutation after the final genuine `seal` is independently **accepted** by `finish()`:

1. Append an additional census event and increment `_FR13_FIXED32_COMPLETE_EVENTS`.
2. Increment the complete counter without appending an event.
3. Set `_FR13_FIXED32_PENDING_EVENT` to a non-None unfinished event.
4. Rewrite the first retained event's `forward_step_index`, leaving the most recent event unchanged.

These are exactly the final event/census continuity claims requested of this controller. A future raw auditor might catch them, but this method currently declares its own completion without establishing them. The narrow repair is to require pending None, exact final counter and census length, and verify every retained event against its frozen saved hash/owner using existing HE checks. Keep the existing failure latch. No numerical or scheduling rule needs to change.

## Passing changed-scope checks

Independent controls return **10/14 expected outcomes**: four invalid terminal mutations above fail the negative expectation; all other checks pass. The positive traverses **all24 calibration sequences,54 prescribed cycles and24 final flushes** with actual Torch CPU tensors. Forcing preserves tensor storage identity; the terminal root-only output carries the supplied greedy token and accepted length0; final live counters advance exactly once per sealed event in the valid fixture. Initialroot and31draft forcing, current-frame ownership during next-draft staging, and the no-next-forward-before-seal order use the accepted FA/Cursor semantics.

Independently refused: wrong nonroot token, wrong nonroot mRoPE axis, different request identity, pre-existing pending event at forward start, foreign seal observation owner, next forward before actual seal, held-out record, missing final flush and absent terminal O2 greedy input. Failure latching is checked after tensor/phase invalidity.

The terminal flush deliberately preserves natural proposals and validates the root token/position on every supplied position axis; it does not impose a frozen interior draft-token stream on that flush. Actual O2 greedy provenance and live proposal/pending/payload ownership must still be supplied by the eventual accepted runtime hook and raw witnesses. This unconnected controller's synthetic event tests do not prove GPU execution, graph replay, state publication, current native/candidate equivalence or live source selection.

## Reproducer and limits

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /Users/zhiyuanma/miniforge3/bin/python papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/candidate-sequence-forcing-review/controls.py
```

Local Python3.9.7/Torch2.8.0 executes the exact accepted topology source with a module-local equivalent implementation of `zip(strict=True)`; no global builtins or canonical source files are changed. All other reviewed/helper functions are unchanged. The first auditor harness attempt resolved the parent fixture relative to the snapshot directory and failed before test execution; `attempt1.log` and the initial harness are preserved. The repaired harness only points that fixture lookup at the original frozen fixture; `attempt2.log` and `results.initial.json` contain the actual findings.

No runtime-hook connection/launcher is approved by this review. This is a bounded source defect in new preparation code, not an observed candidate/GPU result.
