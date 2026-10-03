# Memory v2.3: reservation namespace closure review

**One narrow namespace correction remains before full closure.** The four prior scanner counterexamples are fixed, and safe idle/terminal cases still pass. A directory occupying a reserved marker name is refused by the producer but silently ignored by the scanner. Earlier F1–F5 remain closed; this review neither changes nor delays the already-approved M1 use of unchanged v2.2 in its parent-verified legacy-only state. No recovery is authorized.

Reviewed snapshot: `p0/monitor/review-response-20260927/memory-v23-reviewed-20260928T0532Z`, relative to paper v2. This is a source snapshot, not a final freeze. Read the parent-derived v2.2→v2.3 diff and independently verified both listed file sizes/hashes, zero mismatches.

| Item | SHA-256 |
|---|---|
| `PARENT-SNAPSHOT.json` | `45c6112c339e68e02de6c954ff8bb72174f98a00717682969b0357dddea59d46` |
| `repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/memory_recovery_v2_3.py` | `4d4e6893cf015e97e9515fe147b7f3f1f6fcac227037330f5e4f666cade84881` |
| Corresponding `tests/test_memory_recovery_v2_3.py` | `34a9645a3e5ee594e58489834fcdf9c5793321cb510f4c1e321028f8cfa1301f` |

## Confirmed closure

Producer authentication **822–827** enforces `AUTH-<run_id>.consumed.json` directly under the canonical authority and refuses a nonregular existing marker. Independent execution of only that AST validation slice accepted the canonical path and refused a nested path, missing `.json`, a different run ID's filename, and a directory at the canonical path. No authentication/operation flow was executed.

The new `authority_state` rejects malformed regular marker files and the old ignored cases. Standard-library temporary-file controls called the actual verifier methods:

| Fixture | Observed result |
|---|---|
| Empty authority; legacy directory plus `LATEST_DIR.txt` | idle true |
| Actual campaign lock; arbitrary hold file; hold subdirectory | idle false |
| `holds` is an ordinary file | idle false, malformed namespace |
| Nested regular consumed marker; non-`.json` regular marker | idle false, malformed namespace |
| Truncated canonical marker; wrong-schema marker; missing `run_dir` | idle false, malformed namespace |
| Canonical marker with existing run directory but no terminal receipt | idle false, unfinished reservation |
| Well-formed terminal v2.3 receipt with canonical reservation | idle true; terminal admission true |
| Well-formed frozen-v2.2-shaped terminal receipt with canonical reservation | idle true; terminal admission true |

Terminal positive fixtures used the producer's field layout and every current boot condition set true; they are verifier controls, not evidence of an actual recovery. F5's finalization ordering was not rerun or reopened: the reviewed diff leaves its behavior unchanged apart from diagnostic naming. Existing cleanup and scientific criteria were outside this review.

## Remaining namespace mismatch

At **1790–1795**, `authority_state` handles directories before testing the reserved name or requiring a regular marker file. It descends through such a directory, only examining `files`, then continues. Consequently:

- an empty top-level `AUTH-x.consumed.json/` yields `idle=true`, with no malformed entry;
- a nested `nested/AUTH-x.consumed.json/` also yields `idle=true`;
- adding the first directory to an otherwise valid terminal v2.3 or v2.2 authority leaves `verify_terminal_receipt(...).admit=true`.

This contradicts the fixed-layout regular-file contract at **1747–1750** and the producer's actual refusal at **826–827**. It is a scanner omission in the changed namespace code, not evidence that the runner created such a directory or that the approved M1 authority contains one.

Minimal independently executed reproduction, after importing the source under a non-main name:

```python
with tempfile.TemporaryDirectory() as tmp:
    authority = Path(tmp)
    (authority / 'AUTH-x.consumed.json').mkdir()
    idle, verdict = module.verify_authority_idle(str(authority))
    assert idle is True                     # current defect
    assert verdict['checks']['malformed'] == []
```

**Minimal fix:** classify a reserved/marker-like top-level directory as malformed before the generic directory traversal; likewise inspect marker-like names in the nested `dirs` list, not only `files`. Add the top-level directory refusal and valid-terminal-plus-stray-directory refusal controls. Keep ordinary legacy/run directories allowed. No expanded operational test or prior-suite rerun is required for this correction.

## Verification limits

Only source/diff reads, hash checks, standard-library imports, disposable local fixture I/O, actual pure verifier calls and an isolated authentication AST slice were used. No remote command, Docker/container operation, GPU/model call, operation-mode entry point, cleanup/reclaim, source edit or gate change occurred. An initial reviewer-only AST selection accidentally included two later `self` assignments and raised `NameError`; narrowing it to the actual validation statements at **822, 823, 826** produced the reported five path results. That harness error is not a campaign defect. Source hashes remained unchanged after the controls. Final freeze and future recovery authorization remain parent-owned.
