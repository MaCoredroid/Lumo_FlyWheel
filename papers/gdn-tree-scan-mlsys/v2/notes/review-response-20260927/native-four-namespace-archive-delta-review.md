# Four-namespace archive helper: bounded delta review

Status: **unused preparation, superseded and not selected for B**. Parent elected to use the previously reviewed `archive_native_run_delta_v2.py` unchanged, SHA256 `6cddbbaf3a0cf535b7f0a9139d7f88dfa110137d5d4bc066a89f4406795a7585`.

The new, unused `archive_native_delta_v2.py` has SHA256 `3d56f991610321e1a79dedfd699a796e6cf6f87c8aaa09f67fd4eecdb3aac806`. It differs from accepted feeder snapshot `4d19882ccae347930b7c2238e744a8080c8eb6aef6e8fde9eccbe6a078305be1` only in `safe()`: the slash-delimited whitelist now includes corpus, common-O0, joint-source and joint-common-O0 run directories. Baseline membership, hashing, decryption verification and cross-archive link code are byte-identical to that older snapshot.

Thirty-one pure-predicate controls passed: valid run/object paths for all four prefixes; refusal of absolute paths, `..` traversal, similar-prefix directories, prefix-only strings without the required slash, foreign locations and empty input. No archive main, encryption, key read, remote or network action was executed.

The important lineage caveat is preserved: this older feeder base lacks the later `archive_native_run_delta_v2.py` safeguards against root/ancestor/directory symlinks and existing ciphertext chunks. The whitelist-only result does not carry those safeguards forward or approve their omission. Parent resolved this by retaining the new helper as unused and selecting the already reviewed newer helper for B. No implementation was edited during review.

Evidence and reproduction: `p0/monitor/review-response-20260927/native-four-namespace-delta-review-20260929/`, including both source snapshots, exact one-line diff, `RESULTS.json` and `check_whitelist.py`. This note is a source review and records the parent's selection; it does not claim that B was archived or published.
