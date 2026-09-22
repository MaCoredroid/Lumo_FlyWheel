# Parent-reviewed E8 qualification-v2 infrastructure handoff

Issued 2026-09-22T21:31:42.475086+00:00. This authorizes exactly one new two-boot qualification attempt from the corrected stage; not a retry of the old snapshot and not timing.

Preserve original attempt `out-20260922T2119Z-e8-single-logits`, its manifest45da9147, FAILED.json and original worker logs. It produced zero containers, model boots, requests or scientific observations. The correction changes only launcher generation and the launcher among the original21payloads. Every scientific manifest field, numerical/API/coverage check, source pin and eagle variant is unchanged. Two files add the infrastructure diagnosis and execution test. New stage manifest SHA256: `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`.

Independent review PASS covers the exact new manifest. Both parent and reviewer compared all prior payloads/fields. Reviewer independently executed corrected ON/OFF outer+inner Linux shell dry runs and reproduced the exact original rc127 failure before Docker;27/27 CPU controls pass. Report is `p0/monitor/e8-qualification-v2-redteam.md` when delivered; exact approval is recorded in this parent handoff. Model hash preflight remains the all66-file PASS at21:18:26Z.

Recheck the new manifest/source gate, absent new output root, idle device/no containers/no duplicate drivers. Save the previous e8-worker-status.json as a dated original-failure status before updating it. From `/home/mark/lumo-paper-v2-20260921`, execute exactly once:

```sh
python3 -u -B papers/gdn-tree-scan-mlsys/v2/experiments/e8-single-logits-qualification-v2/e8_qualify.py --repo /home/mark/lumo-paper-v2-20260921 --run /home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T213142Z-e8-single-logits-v2 --execute
```

Capture stdout/stderr in NEW `p0/monitor/e8-qualification-v2-worker-20260922T213142Z.log`, and record actual UTC start, PID, exit code, roots and actual counts in companion status. Keep `p0/monitor/e8-worker-status.json` current. The driver will snapshot and own its labeled containers; leave other processes, sessions, models and shared checkout alone. Do not alter any attempted file or regenerate its manifest. Do not run another GPU worker.

On any failure, preserve all artifacts, stop, diagnose and wait; no retries or weakened gate. On two qualified arms, report actual counts and QUALIFICATION_PASS.json, then wait for independent parent review. No timing is authorized here. Expected qualification15–25minutes, update from observed boot and workload durations. No public submission/upload/contact/push.
