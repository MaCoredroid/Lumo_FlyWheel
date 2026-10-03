# Remote archive pointer-only publisher: bounded independent review

Disposition: the repaired publisher is accepted within this source/CPU review. Both reproduced binding defects are closed. This is not a live upload, push, archive verification, cleanup authorization, or claim that update 12 has been published.

Reviewed final source: `p0/monitor/review-response-20260927/push_remote_archive_backup_v1.py`, SHA256 `a15d58f7bb5317aaf9c59eae56425e4e0c7c6f5ceae506086f2d1ac50b3c0116`. Initial reviewed source `b58f800a73ec55c04eb970640d1cbc91f93ad70fea085e2d15bd4f46640f2739` and its failing controls are preserved. Companion sources inspected: `remote_lfs_upload_v4.py` SHA256 `cff5f5f5153ac8f094bcf13180dcffe82886bc3c426594c7f9093911927fe8f6`; `artifacts/machine-backup-20260929/verify_lfs_remote.py` SHA256 `5c5d9fc9ad8a5dd0729cf33f258fac18780a0f60ad19d76126df1cd7fc22d36b`.

## Preserved findings and closure

1. **Checked metadata could differ from published metadata.** Initial lines 35–42 checked the manifest, restore receipt and chunk list before upload, but line 72 later staged the mutable paths with `git add`. The independent injected upload changed a manifest member hash; the original publisher completed with `verified: true` although its published restore receipt no longer authenticated the published manifest. Final lines 42–44 retain and cross-check the exact bytes; lines 74–78 require unchanged files before staging and create Git blobs from those retained bytes. The same mutation now refuses before any commit/push. A separate mutation after the recheck still publishes the original authenticated bytes, so correctness does not rely solely on a last-moment path check.

2. **Push and receipt could refer to a different local branch commit.** Initial line 79 pushed the mutable backup branch name after its compare-and-swap update, and lines 80–82 never compared the verifier's commit to the commit just created. A synthetic concurrent local backup-ref advancement caused the original publisher to claim completion after selecting that other commit. Final lines 85–88 push the created commit ID to the explicit backup ref, check the local ref again, and require the verifier's exact commit and remote-match flag. The same race now selects only the intended commit and refuses false completion. A separate foreign-commit verifier receipt also refuses.

The local ref update is a compare-and-swap against the supplied parent. The remote push is explicit and non-forcing; it is not described here as a remote compare-and-swap. An ordinary remote branch conflict can leave a local commit and uploaded immutable objects requiring a deliberate retry; the script does not erase those recoverable artifacts or claim success after that failure.

## CPU evidence

Evidence directory: `p0/monitor/review-response-20260927/remote-archive-pointer-publisher-review-20260929/`.

The tests execute the actual captured publisher `main`, with real Git operations only inside temporary repositories. SSH, HTTP, upload, push and verifier transport are injected; the verifier stub mirrors the existing local/remote commit and chunk-manifest/pointer joins and assumes synthetic successful object availability. No credentials, actual chunks, real pushes, archive operations, or remote processes are used.

- Initial three controls: clean success; manifest mutation incorrectly accepted; mutable-branch race incorrectly accepted. The latter two are retained reproductions of defects, not passing safety behavior.
- Repaired five controls: clean success; metadata mutation before staging refused; local-ref race refused without selecting the foreign commit; foreign verifier commit refused; post-recheck metadata mutation safely stages retained bytes.
- Both positive controls produce exact canonical LFS pointer blobs without a local ciphertext file. The normal index bytes, `main` ref and pre-existing tracked working files remain unchanged. The backup parent's actual `.gitattributes` has the expected `machine-backup-20260929/*.enc` LFS rule.

Reproduce locally with `python3 .../remote-archive-pointer-publisher-review-20260929/check_publisher.py` and `check_publisher_repaired.py`. These use only temporary Git repositories and synthetic transports; their JSON results identify the captured publisher SHA.

## Recoverability boundary

The publisher consumes the existing remote archive/decrypt/member verification and its three local metadata files; it does not create or decrypt an archive. The final all-object verifier checks committed pointer bytes against available local chunk manifests and checks authenticated remote LFS availability/size. It does not download and rehash remote ciphertext. Actual remote archive integrity, baseline-first hardlink restore order, key custody, terminal-run selection, and successful publication remain the existing archive/operational workflow's responsibilities. No original raw data, ciphertext or keys are deleted here.

No further blocker was found within the requested pointer construction, metadata binding, explicit backup-branch safety and recovery-check integration scope. Operational failure still requires inspecting the retained state before retry; CPU acceptance is not evidence of an actual successful backup.
