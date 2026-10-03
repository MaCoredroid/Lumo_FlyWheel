# Native M1 local mirror: complete byte verification

**PASS for the completed mirror.** All **311 regular files /10,239,547,998 bytes** are accounted for and hash-matched. The reported320 rsync items reconcile as311 regular files +8 subdirectories +the root directory; they are not320 experiment records.

Local root: `experiments/review-response-20260927/runs/m1-native-c1/` under paper v2. The audit ran on the local Mac at23:57:36–23:57:40 UTC on September28, taking4.443 seconds. Every file had stable local inode/size/mtime/ctime across its SHA-256 read. There were no symlinks, missing members, extra members, size differences, or hash mismatches against the applicable inventories.

## Evidence closure

- The prior independent remote `RAW-AUDIT.json` (SHA`58a9dfb2485ed732dee3809f2bad1eb877a3b9078168c3835b5c9b841d454aad`) covers302 files /10,239,539,794 bytes. **Every local copy matches its saved hash and size**, including all292 payload files /10,239,107,065 bytes. No remote tensor/payload was reread.
- The complete prior remote stat ledger contains311 native-root files, and current local membership/lengths match it exactly. Nine launch/supervisor JSON/log sidecars totaling8,204 bytes were outside the302-member content-hash inventory. The initial qualified audit remains preserved; it did not pretend a full prior content inventory existed.
- With the parent's additional explicit read-only authorization, I read and hashed **only those exact nine small remote sidecars**, using pinned names/lengths, no-follow opens and stable before/after identities. All nine match local hashes. The exact command, script, output and empty stderr are preserved. This closes the metadata-only gap without reading remote payloads.

The resulting independent local archive therefore preserves the full previously reviewed native M1 payload population plus its metadata. It authenticates equality to the prior reviewed remote payload bytes and current explicitly checked metadata, not a fresh remote tensor hash at mirror completion. The raw ledger retains each local path/hash/stat, source inventory binding, and the exact nine current remote metadata identities.

## Population and scientific scope

The292 payload files comprise96 native-record files,192 operand/C2 reference-pack files, and4 reduction-array files. The original scientific population remains **96 seed/layer records with two repeats (192 raw repeat records)** from the two calibration seeds across48 layers. The same preserved review reports29,760 native calls and142,848 unique cells. File transfer and SHA verification add **zero experiments, cycles, fixtures, model boots, timing measurements or numerical qualification results**. No tensor deserialization, numerical recomputation, GPU/container execution or source/gate change occurred here.

The parent's mirror-completion notification and this complete archive verification close the prior *local-copy* prerequisite. They do not certify present remote memory residency or absence of all readers, and they do not authorize cache eviction/reclaim or a new query/model boot. Those require their own prospective action-time checks and authority. Waiting for a mirror to end does not itself imply its cached pages were released.

## Reproducible SHA summary

All evidence is under `p0/monitor/review-response-20260927/m1-native-c1-local-mirror-review/`.

| Artifact | SHA-256 |
|---|---|
| Local full-file audit | `3bbf4b5191c06cba9e572242b17792d9026e4c3bffe923693d5af2ee43a121f8` |
| Exact nine-sidecar remote closure | `7538c26af89b3b5a02dbe9be1c9012def0882e5235846c9a39eb003e11a3256e` |
| Final mirror summary | `002f4cc9502f2849e482c510e6415ad0b070bd8341337033fce401c65e14603c` |
| Local audit source | `b6374ac030a3abade45dbd57f1fd8bccebea339b0fc9f787181a91df211c0331` |
| Nine-sidecar read-only source | `2d232d933ca510210f8791bb2ae53e7632283330c164104b5a9ce53d0ef15482` |

`REVIEW-SEAL.json` binds this note, scripts, raw outputs, command and summary. No previous failed/qualified audit record was overwritten.
