# Independent two-pilot evaluator preflight review

Reviewed September 27, 2026. Scope is exactly `scikit-learn__scikit-learn-9288` and `psf__requests-2317` in `experiments/review-response-20260927/evaluator-preflight/`. No GPU, model, task/test execution, network call, remote mutation, or manuscript change was performed by this review. Only this report was written.

**Disposition: approve the bounded environment/import/runner-startup preflight. The workload-launch gate remains closed.** The evidence establishes that the two pinned images start and import their repository packages and test runners on native x86_64. It supplies no task result or assurance that all evaluator tests will execute correctly.

## Independent verification

- All **62 payload files** matched manifest sizes and SHA-256 hashes during the full review. After the single documentation correction below, the final manifest SHA-256 is `77369bf3467ce8ea0a40719dafaa286c751eef6c4af60931b639e4a8c786ce97`; the corrected report SHA-256 is `1e05a1aaa6dcce22e78ab5c24ca23617c950e1ae47f0cd4577c055a2a8196267`. Both final hashes and the report's manifest entry were verified in the narrow closeout.
- All **ten numbered receipts** match the SHA-256 of their saved command scripts. The original nonzero exits in receipts 04, 07, and 08 remain preserved.
- All **six CPU identity/reduction tests pass**. Independently invoked the reducer with writes intercepted in memory: all **23 regenerated outputs**, including `PREFLIGHT-SUMMARY.json`, are byte-identical to the delivered files. No preflight artifact was overwritten.
- Both task metadata records exactly match the previously pinned neutral metadata projection and pilot selection. The dataset pin is `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`.
- The copied six harness source files match the installed-source manifest. The inspection reports swebench **4.1.0**, with **73** hash-bearing distribution RECORD entries checked and no mismatch; package import and CLI `--help` return zero. The original upstream Git/wheel and historical registry build recipe remain unverified, as disclosed.

## Image and source identity

| Task | Immutable manifest | Image configuration ID | Tracked repository result |
| --- | --- | --- | --- |
| scikit-learn 9288 | `702434646ba19a69bf216770efdbc2010f64aad9f1162be14abec1bb52399662` | `fae30c4fa5ad6588291ec3e3e4c33a36879f234d50ff5d00eb145e8b1ccc46ce` | All 1,209 paths, full Git object IDs, types, and modes match the pinned base. |
| Requests 2317 | `a0ce096d4dfa27ca8ea80ae4b38103e970d17b19066d886a550936394827c9fb` | `3c49e39da2143fc51599f096a7c959ea5f3c9fd97d22899d977ac92d6dc360b6` | All 128 paths, full Git object IDs, and types match; 125 mode changes are exclusively `100644` to `100755`. |

Registry-derived locks, pull receipts, RepoDigests, configuration IDs, root filesystem layer IDs, and `linux/amd64` architecture agree. Pulls and startup commands use the immutable references, not the discovery `latest` tag. Runtime reports `x86_64`, successful imports from `/testbed`, and zero exits for the startup commands.

The scikit-learn base `3eacf948e0f95ef957862568d87ce082f378e186` and image child commit share tree `5e4b2c1edc83832a86541dd45baeec2af5ba3b65`; rejecting it solely because HEAD differs would be incorrect. Requests is a disclosed permission variant of base `091991be0da19de9108dbe5e3752917fea3d7fdc`, not a content modification or fully identical Git tree. Its exact image should be retained rather than silently normalizing permissions. Both tracked working trees are clean. The installed recipe's permission adjustment and final setup commit explain this class of packaging difference, without proving the original builder revision.

Scikit-learn's separate `environment_setup_commit` is absent from retained Git objects; the required task base is present and its source is verified. Requests' environment setup commit equals its base and is present. Missing environment-setup history is not, by itself, a task-source mismatch.

## Documentation precision fix — resolved

The final `PREFLIGHT-REPORT.md` now distinguishes the direct-package recipe definitions from the preceding `load_cached_environment_yml(instance_id)` branch and explicitly states that the historical executed branch is not established by these receipts. It also separates absent environment-setup history from missing runtime dependencies. This resolves the sole requested documentation correction. **Review closed with bounded startup approval and no remaining review finding; the workload-launch requirements below remain unchanged.** No expanded checks or remote execution occurred during closeout.

## Scope and remaining launch requirements

The receipts support two pulls and six temporary startup/identity containers, with no GPU access, no task tests or patches, and no model requests. `pytest --version` runs from `/tmp`; it is a startup check, not test collection or a benchmark evaluation. The three preserved failures were respectively an overly strict HEAD check, absent environment-setup history, and the Requests mode-only tree difference. The later full-object audit resolves those specific checks without erasing them. Final Docker inventory is empty.

Before any pilot workload starts, the campaign adapter must enforce the per-task immutable image lock and verify the loaded configuration ID; the existing evaluator still uses floating `latest`. It must also bind the exact dataset/revision instead of loading by name. The remaining campaign controls—effective seeds, serving identities and precision, measurement closure, and final budget—are not satisfied by this CPU preflight. Approval is restricted to the recorded startup feasibility and disclosed identity classification.
