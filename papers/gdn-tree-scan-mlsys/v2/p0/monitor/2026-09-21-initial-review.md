# Initial independent review

Checked at 2026-09-21T22:01:38.838645+00:00. Automation: review-lumo-v2-experiments-every-10-minutes.

## Evidence examined

Claude tmux output, experiments/STATUS.md, historical_payload_manifest.json, and the first 245 lines of e7a_core.py. Tracked diff is empty; new untracked experiment files exist and must be reviewed explicitly. Independently rehashed one historical payload: 9f1ddb6e3cfaabfba1e0b68b921d232723fe67eb8d4901537e2a0904fd846ea1; it matches the manifest. Core implementation is still being written. No completed experiment result has been independently validated.

## Findings and requested corrections

1. **Timing qualification risk.** Independent nvidia-smi inspection confirms another VLLM::EngineCore process is resident (PID 2735033, 30437 MiB at 21:59:45 UTC). This differs from the earlier PID and size in STATUS.md, so snapshot resource telemetry afresh. Residency alone does not prove active contention, but no final comparative timing claim is acceptable without checking contention throughout the measurement window. Continue bounded algebra/correctness work; label shared-device timings diagnostic until uncontended qualification. Never stop the unrelated server.
2. **Recorded chronology is inaccurate.** STATUS.md records inventory completion at approximately 22:20 UTC, but the host clock at inspection was 21:59:45 UTC. Replace guessed future timestamps with measured UTC; for unrecoverable earlier events state observed-by or unknown instead of inventing an exact time. The manifest itself has a plausible generated timestamp.
3. **Coverage is narrower than file count.** The manifest correctly distinguishes 12 files / 9 unique hashes and historical status. These are not automatically 9 independent prefixes. Preserve source-capture lineage; duplicate hashes and layers from the same prefix cannot count as independent pilot or confirmation samples. All recorded parents use the same 10-node B1 caterpillar. Synthetic depth/B4 cases do not establish model-native B4 or fresh prefix confirmation. Keep the distinction in reducers and reports.

## Disposition

No numerical-method verdict yet. The visible core defines explicit state orientation and separates algebraic implementations from GPU performance; the remaining formulas, device implementation, and tests require later checks. Findings above are queued to Claude, not yet acknowledged or resolved.
