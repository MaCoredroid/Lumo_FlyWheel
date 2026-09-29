# Recoverable campaign backup

This directory stores encrypted archives through Git LFS. The raw experiment records may contain local paths, execution environments and nonpublic logs; they are not exposed in plaintext by this backup. No encryption key is committed.

Key location on the owner's Mac: `/Users/zhiyuanma/.codex/private-backups/lumotree-20260929.key` (mode0600). Preserve this file separately. Without it these encrypted archives cannot be recovered.

Archive sets are independently encrypted streams split into ordered512MiB chunks. Each `.chunks.json` records exact names, lengths and SHA256 values. The Git LFS pointer OIDs must equal these SHA256 values.

- `remote-runs-direct`: the owned remote campaign's complete `runs/` and `fixtures/` directories, including failed attempts.
- `remote-source`: the rest of remote paperv2, workload wheelhouse and root checkpoint records. Reconstruct virtual environments from the recorded pinned wheels; the derived virtual-environment directories are not included.
- `local-worktree`: the complete local paperv2 snapshot, including its manuscript, historical artifacts and local evidence. This backup directory itself and Python test caches are excluded.
- Any later `source-update` set records source/evidence changes after those snapshots. Restore it last when selecting that checkpoint.

Baseline repository commits are recorded in `BACKUP-IN-PROGRESS.json`. The archive paths are repository-relative. Restore remote and local snapshots into **separate fresh checkout directories** to preserve their distinct provenance. Keep old snapshots intact.

To restore one set, fetch its LFS files (`git lfs pull`), verify every chunk against the manifest, concatenate chunks in manifest order, then decrypt and decompress:

```sh
cat remote-runs-direct.tar.zst.*.enc |
  openssl enc -d -aes-256-cbc -saltlen 8 -pbkdf2 -iter 200000 -md sha256 -pass file:/absolute/path/to/lumotree-20260929.key |
  zstd -d -c | tar -xf - -C /absolute/path/to/new/checkout
```

All four initial archive sets use an 8-byte salt, verified by complete explicit-8-byte decrypt/decompress/tar-list checks. Use `-saltlen 8` with the local OpenSSL3.6 executable; when decrypting on the remote OpenSSL3.0 executable, omit that unsupported option (8 is its observed default). Later source updates also specify8 explicitly. The same key file applies to all sets.

The key is a random256bit passphrase; the encryption is salted AES-256-CBC with PBKDF2/SHA256,200000iterations. Chunk hashes are authenticated through Git, and the compressed stream includes its checksum. This is an operational backup, not a scientific result or submission artifact.

`verify_lfs_remote.py` checks every complete archive's Git pointers and authenticated remote LFS object availability without writing credentials or signed URLs to its receipt. `BACKUP-VERIFIED.json`, when present, records completion; an initial or partial push alone is not a complete backup.

The incomplete `remote-runs` relay and experimental `remote-runs-compact` compression were operational backup attempts, superseded by the complete `remote-runs-direct` set. They are not required for restore and are not scientific experiments. Original evidence directories remain intact.
