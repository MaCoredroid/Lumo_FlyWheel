# Restoring the final native evidence and manuscript

30 September 2026 UTC. This is operational recovery documentation, not scientific qualification or proof that a push has finished. Verify the corresponding `GIT-BACKUP-VERIFIED` receipt before treating a checkpoint as remotely recoverable.

The standalone manuscript source archive can be extracted into a fresh directory and built using its `BUILD.txt`; it contains the actual manuscript dependency closure, not raw experiment tensors. The encrypted campaign backup preserves those larger raw objects and earlier failures separately on branch `codex/lumotree-recovery-backup-20260929`.

## Final native raw cohort

The latest native archives are deltas containing hard links to authenticated earlier archive members. They cannot be restored independently. Fetch the required Git LFS objects, check every chunk SHA/length against its `.chunks.json` and the Git pointer OID, then restore the following sets **in this order under the same fresh repository root**:

| Archive manifest | SHA256 |
| --- | --- |
| native-aligned-A.manifest.json | `8a81bfad1eca1f512f1127388af4a33fc14520ec949719ff14f533b8240836e6` |
| native-common-o0-repair2-A.manifest.json | `5cfb481ae8f0518f8f296f5cff12cb3de4f588739ac5d9c1bf08f4094fd0ae3d` |
| native-joint-source-A.manifest.json | `1b4b1091ec6ed8bba4c9031c65bd7149158d435fa51892c64ec5417cc14ae8b2` |
| native-joint-common-o0-A.manifest.json | `d1c9381da51ea34b54b6ce941f85a77b8864263650464f2f18207a01159c03f9` |
| native-joint-common-o0-B.manifest.json | `eef7ea64d174b741dbedec46a57d038a1b8dd0839cc87badf95953889daf8ead` |
| native-joint-kernel-pin-retry1-A.manifest.json | `e9407709866a84188b88a3a4a4b6823394eda2d388fb27a631cf0803b2a63da2` |
| native-joint-kernel-pin-retry1-B.manifest.json | `5d41e0eab8d73fb33347b74d6c7d44e8e1149f961fc4134cbbea6f07d3e78797` |

Each manifest's `baseline_manifest` and `baseline_manifest_sha256` must match the preceding dependency. Each set has a full decrypt/decompress/member-hash restore check; the check also authenticates cross-archive hard-link targets. Keep all failed-attempt archives and original raw directories. There was no raw deletion as part of final delivery.

Use the decryption instructions in `artifacts/machine-backup-20260929/README.md`, including manifest-ordered concatenation, the recorded salt length and a separately preserved owner key. Never put the key in Git or logs. Extract only after verifying the archive identity, into a new restoration directory rather than overwriting the working experiment tree.

## Code and paper changes

Select the desired verified backup commit. Restore the appropriate initial local or remote source snapshot into separate fresh checkouts, then apply the applicable `source-update` deltas in their recorded chronological order; later snapshots win. Do not sort numbered suffixes lexicographically. The final delivery receipt binds the paper, source bundle, claim disposition and result reviews. Final backup receipts bind the uploaded encrypted archives and verified Git/LFS objects; neither compilation nor an initial push alone proves those backups complete.

Historical operational-recovery records and consumed one-use gates remain preserved. Restoring evidence does not grant a scientific launch, count an experiment, or reopen Q1/WP.

Final process-B raw backup was verified at06:38:24Z: commit `6088b6e349f8bd00a881addae114c368bc8e4d9c`, 63 complete archives and410 remote LFS objects. The exact receipt is `p0/monitor/review-response-20260927/GIT-BACKUP-VERIFIED-20260930-update40.json`. Subsequent manuscript/source deltas have their own later receipts.
