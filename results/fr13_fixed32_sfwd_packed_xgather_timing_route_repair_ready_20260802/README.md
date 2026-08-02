# Packed x-gather timing route repair

Status: **SOURCE_ROUTE_REPAIRED_READY_FOR_COORDINATED_RERUN**.

The matched real SWE-Verified B1 timing attempt completed its stock arm, but
the candidate stopped during model readiness before task ingress. The failure
was a source-route API mismatch: the served prior-reuse call still supplied the
removed source descriptor argument, while the descriptorless launcher requires
the validated fixed-tree parent.

Repair commit `3c6a25e9674ca041ec8430889211c596e40054ef` gives each
prior-reuse call the exact launcher keyword set and passes
`tree_parent=_fr10_parent`. The state-fusion call retains its separate
`source_flat` contract. Focused tests now parse the generated patch payload and
compare both prior-reuse callsites against the launcher signature.

The failed route closed with an all-zero host census. Runtime, external, and
candidate-source manifests were byte-identical from launch to exit. The stock
B1 diagnostic remains valid, but no candidate task or candidate timing sample
exists, so the attempt is not a paired timing result and is not acceptance
eligible.

No rerun was launched. GPU scheduling remains coordinated with the concurrent
CFWD gate.

This reduced package excludes raw logs, task/model/request/response/patch
content, task identifiers, environment values, credentials, process or
container identifiers, and raw sidecar paths.
