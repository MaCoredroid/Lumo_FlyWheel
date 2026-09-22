# Parent-reviewed E8 qualification handoff

Issued at 2026-09-22T21:19:58.842662+00:00. User authorized the design reframe and targeted experiments, and the existing Claude tmux worker remains sole GPU executor.

The paper is sealed at main0d7bab38 / abstract887f36b6 / PDF0295c3f6, 14 pages. Do not edit the paper or any original E1/E7 evidence. Do not alter production source or the shared checkout. The new local and DGX stage is byte-verified at manifest `45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf` (21 hash-bound dependencies). Independent reviewer PASS covers TWO qualification boots only, ON then OFF, eight prefixes per arm, max32 tokens, B1. Parent also rehashed all66 model files against P0: PASS at21:18:26Z in p0/monitor/e8-weight-preflight-20260922.json.

Read experiments/e8-single-logits/README.md and the exact manifest. Recheck manifest hash, source gate, idle GPU/containers and no duplicate qualification process. Then execute exactly once from `/home/mark/lumo-paper-v2-20260921`:

```sh
python3 -u -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits/e8_qualify.py --repo /home/mark/lumo-paper-v2-20260921 --run /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T2119Z-e8-single-logits --execute
```

Capture stdout/stderr in a NEW p0/monitor/e8-qualification-worker-20260922.log and record the actual UTC start, process ID and final exit code in companion status files. The run root must not exist before execution; the driver creates it. Do not regenerate the manifest or edit stage/source files. Do not start another worker. The driver snapshots the immutable stage and owns only its newly labeled containers. It closes them after each boot; leave other sessions/processes/models alone.

Any source/API/head/state guard mismatch, insufficient coverage, timeout, or missing seal ends qualification. Preserve FAILED.json and all original artifacts; no automatic retry, threshold relaxation, favorable replacement or source repair inside the attempted run. Write a concise diagnosis and wait for the parent. If both gates pass, write the QUALIFICATION_PASS.json location and actual counts; parent will independently verify before any timing. Six clean B1 timing cells remain conditional and are NOT authorized by this handoff.

Parent and reviewer agents own implementation changes and the conditional timing stage. Do not implement new variants while waiting. Use p0/monitor/e8-worker-status.json for current stage/running jobs/counts/ETA, with timestamps from the host. Estimated two-boot qualification duration12–18minutes, plus unusual compile or diagnostic overhead; update from actual evidence. No publication, upload, external messages, repository push, model deletion, or unrelated cleanup.
