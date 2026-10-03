# Backup publisher v3 — bounded transport review

**PASS, no material blocker in the changed batching path.** Reviewed `p0/monitor/review-response-20260927/push_remote_archive_backup_v3.py`, SHA256 `d8e81e5e857d55cd25e56ba3b4a5093877f0857b26e281370cbb2105705ab908`. This is source and mocked-boundary validation, not an upload/restore result or operational authorization. The active v2 retry was not inspected, interrupted, or modified.

The v1→v2 change is only the SSH-controller timeout, 1500→3600 seconds. The v2→v3 change moves authentication and LFS action acquisition into each eight-object batch (lines 46–82). Eight matches the unchanged pinned uploader's eight workers (`remote_lfs_upload_v4.py`, SHA `cff5f5f5153ac8f094bcf13180dcffe82886bc3c426594c7f9093911927fe8f6`). A later batch no longer waits behind earlier batches using its original capabilities. This does not establish that capability expiry caused the previous SSL EOFs, or guarantee that a single slow upload finishes before an action expires.

The source verifies the remote uploader hash before any batch, obtains fresh SSH LFS authorization and batch actions immediately before each uploader invocation, checks each returned batch's object population, and persists only allowlisted results and a boolean stderr indicator. Normal completed-batch progress is flushed and fsynced. Aggregate success still requires every return code zero, no stderr, every object uploaded, and the complete `(oid,bytes)` multiset. Metadata must remain unchanged before any alternate-index write. The complete suffix starting at `git read-tree`—alternate-index construction, commit/ref update, push, and created-commit/remote verification—is byte-identical to v2.

## Independent controls

Eight scenarios passed in `p0/monitor/review-response-20260927/backup-publisher-v3-review/controls.json`. Every Git, SSH, subprocess, and URL-open boundary was replaced with a mock; only temporary local fixture files were created.

- Positive 18-object archive: batches 8/8/2, three fresh authorizations, exact auth→batch→upload order, three progress records, and publication only after all uploads succeed.
- Failed upload, missing result, unexpected result field, wrong returned byte size, stderr, second-batch authorization failure, and controller timeout: all refuse before `read-tree`, ref update, push, or verifier invocation.
- Prior successful batch records survive later failure; synthetic capability URLs/headers never appear in retained progress, aggregate logs, or stdout.
- Timeout calls only the mocked owned controller's terminate/wait path and stops scheduling. The existing instruction to verify remote uploader quiescence before retry remains necessary; local SSH termination is not asserted to prove remote termination.

The normal progress records are fsynced. The timeout-only marker at line 69 is closed without an explicit fsync; therefore it should not be described as crash-durable. This does not permit a success/publication false positive, and already completed normal batch records remain fsynced. No broader durability or inherited publication behavior was reopened.

The review directory retains exact v1/v2/v3/uploader bytes, the executable mock controls, results, a member manifest, and a review seal. No actual network, push, GPU, container, cleanup, cache eviction, or production source edit occurred.
