# K64/root1 SFWD B1 timing pre-task boot rejection

This reduced artifact records a failed timing launch that stopped before any
authenticated SWE-Verified task began. It is not a latency, TPS, acceptance,
quality, correctness, or hardware-floor result.

The stock arm failed while importing the generated worker source. The fixed32
flush module emitted its generation-zero ready acknowledgement before the
later-appended SFWD timer helper module had defined `_fr13_sfwd_timer`. The
timing-only ready path therefore raised `NameError` during engine startup.

Evidence classification:

- source commit: `30e2fd8f264413834640c8605a9b12c3b642ee7c`
- run tag: `k64root_20260802T081335Z`
- authenticated task directories: `0`
- timing summary: absent
- stock arm: boot failed before health
- candidate arm: not launched
- source, runtime, and external launch/end manifests: unchanged
- failed exited container: removed; Docker returned to zero containers

The required repair is limited to the timing-mode generation-zero ready
acknowledgement. Real pre/post task snapshots must continue to initialize,
drain, and reconcile all three GPU timers after module import is complete.

Raw run logs, task/model content, request/response data, patches, environment
values, process/container identities, and credentials are intentionally not
included.
